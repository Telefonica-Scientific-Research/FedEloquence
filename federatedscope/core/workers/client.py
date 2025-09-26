import copy
import logging
import sys
import pickle
import os

import hashlib

from federatedscope.core.message import Message
from federatedscope.core.communication import StandaloneCommManager, \
    StandaloneDDPCommManager, gRPCCommManager
from federatedscope.core.monitors.early_stopper import EarlyStopper
from federatedscope.core.auxiliaries.trainer_builder import get_trainer
from federatedscope.core.secret_sharing import AdditiveSecretSharing
from federatedscope.core.auxiliaries.utils import merge_dict_of_results, \
    calculate_time_cost, add_prefix_to_path, get_ds_rank
from federatedscope.core.workers.base_client import BaseClient

logger = logging.getLogger(__name__)
if get_ds_rank() == 0:
    logger.setLevel(logging.INFO)


class Client(BaseClient):
    """
    The Client class, which describes the behaviors of client in an FL \
    course. The behaviors are described by the handling functions (named as \
    ``callback_funcs_for_xxx``)

    Arguments:
        ID: The unique ID of the client, which is assigned by the server
        when joining the FL course
        server_id: (Default) 0
        state: The training round
        config: The configuration
        data: The data owned by the client
        model: The model maintained locally
        device: The device to run local training and evaluation

    Attributes:
        ID: ID of worker
        state: the training round index
        model: the model maintained locally
        cfg: the configuration of FL course, \
            see ``federatedscope.core.configs``
        mode: the run mode for FL, ``distributed`` or ``standalone``
        monitor: monite FL course and record metrics, \
            see ``federatedscope.core.monitors.monitor.Monitor``
        trainer: instantiated trainer, see ``federatedscope.core.trainers``
        best_results: best results ever seen
        history_results: all evaluation results
        early_stopper: determine when to early stop, \
            see ``federatedscope.core.monitors.early_stopper.EarlyStopper``
        ss_manager: secret sharing manager
        msg_buffer: dict buffer for storing message
        comm_manager: manager for communication, \
            see ``federatedscope.core.communication``
    """
    def __init__(self,
                 ID=-1,
                 server_id=None,
                 state=-1,
                 config=None,
                 data=None,
                 model=None,
                 device='cpu',
                 strategy=None,
                 is_unseen_client=False,
                 *args,
                 **kwargs):
        super(Client, self).__init__(ID, state, config, model, strategy)

        self.data = data

        # Register message handlers
        self._register_default_handlers()

        # Un-configured worker
        if config is None:
            return

        # the unseen_client indicates that whether this client contributes to
        # FL process by training on its local data and uploading the local
        # model update, which is useful for check the participation
        # generalization gap in
        # [ICLR'22, What Do We Mean by Generalization in Federated Learning?]
        self.is_unseen_client = is_unseen_client

        # Parse the attack_id since we support both 'int' (for single attack)
        # and 'list' (for multiple attacks) for config.attack.attack_id
        parsed_attack_ids = list()
        if isinstance(config.attack.attacker_id, int):
            parsed_attack_ids.append(config.attack.attacker_id)
        elif isinstance(config.attack.attacker_id, list):
            parsed_attack_ids = config.attack.attacker_id
        else:
            raise TypeError(f"The expected types of config.attack.attack_id "
                            f"include 'int' and 'list', but we got "
                            f"{type(config.attack.attacker_id)}")

        # Attack only support the stand alone model;
        # Check if is a attacker; a client is a attacker if the
        # config.attack.attack_method is provided
        self.is_attacker = ID in parsed_attack_ids and \
            config.attack.attack_method != '' and \
            config.federate.mode == 'standalone'

        # Build Trainer
        # trainer might need configurations other than those of trainer node
        self.trainer = get_trainer(model=model,
                                   data=data,
                                   device=device,
                                   config=self._cfg,
                                   is_attacker=self.is_attacker,
                                   monitor=self._monitor)
        self.device = device
        self.best_model_path = ""

        # For client-side evaluation
        self.best_results = dict()
        self.history_results = dict()
        self.history_results_with_bestloss = dict()
        # in local or global training mode, we do use the early stopper.
        # Otherwise, we set patience=0 to deactivate the local early-stopper
        patience = self._cfg.early_stop.patience if \
            self._cfg.federate.method in [
                "local", "global"
            ] else 0
        self.early_stopper = EarlyStopper(
            patience, self._cfg.early_stop.delta,
            self._cfg.early_stop.improve_indicator_mode,
            self._monitor.the_larger_the_better)
        
        # Configurations related to Local Early Stop
        self.local_early_stop = False
        self.was_early_stop = False
        self.local_early_stopper = EarlyStopper(
            self._cfg.early_stop.patience, self._cfg.early_stop.delta,
            self._cfg.early_stop.improve_indicator_mode,
            self._monitor.the_larger_the_better)
        self.best_client_model = {}
        self.round_in_which_best_client_model_is_saved = 0

        self.format_eval_res_LDES = {}

        # Secret Sharing Manager and message buffer
        self.ss_manager = AdditiveSecretSharing(
            shared_party_num=int(self._cfg.federate.sample_client_num
                                 )) if self._cfg.federate.use_ss else None
        self.msg_buffer = {'train': dict(), 'eval': dict()}

        # Communication and communication ability
        if 'resource_info' in kwargs and kwargs['resource_info'] is not None:
            self.comp_speed = float(
                kwargs['resource_info']['computation']) / 1000.  # (s/sample)
            self.comm_bandwidth = float(
                kwargs['resource_info']['communication'])  # (kbit/s)
        else:
            self.comp_speed = None
            self.comm_bandwidth = None

        if self._cfg.backend == 'torch':
            try:
                self.model_size = sys.getsizeof(pickle.dumps(
                    self.model)) / 1024.0 * 8.  # kbits
            except Exception as error:
                self.model_size = 1.0
                logger.warning(f'{error} in calculate model size.')
        else:
            # TODO: calculate model size for TF Model
            self.model_size = 1.0
            logger.warning(f'The calculation of model size in backend:'
                           f'{self._cfg.backend} is not provided.')

        # Initialize communication manager
        self.server_id = server_id
        if self.mode == 'standalone':
            comm_queue = kwargs['shared_comm_queue']
            if self._cfg.federate.process_num <= 1:
                self.comm_manager = StandaloneCommManager(
                    comm_queue=comm_queue, monitor=self._monitor)
            else:
                self.comm_manager = StandaloneDDPCommManager(
                    comm_queue=comm_queue, monitor=self._monitor)
            self.local_address = None
        elif self.mode == 'distributed':
            host = kwargs['host']
            port = kwargs['port']
            server_host = kwargs['server_host']
            server_port = kwargs['server_port']
            self.comm_manager = gRPCCommManager(
                host=host,
                port=port,
                client_num=self._cfg.federate.client_num,
                cfg=self._cfg.distribute)
            logger.info('Client: Listen to {}:{}...'.format(host, port))
            self.comm_manager.add_neighbors(neighbor_id=server_id,
                                            address={
                                                'host': server_host,
                                                'port': server_port
                                            })
            self.local_address = {
                'host': self.comm_manager.host,
                'port': self.comm_manager.port
            }

    def _gen_timestamp(self, init_timestamp, instance_number):
        if init_timestamp is None:
            return None

        comp_cost, comm_cost = calculate_time_cost(
            instance_number=instance_number,
            comm_size=self.model_size,
            comp_speed=self.comp_speed,
            comm_bandwidth=self.comm_bandwidth)
        return init_timestamp + comp_cost + comm_cost

    def _calculate_model_delta(self, init_model, updated_model):
        if not isinstance(init_model, list):
            init_model = [init_model]
            updated_model = [updated_model]

        model_deltas = list()
        for model_index in range(len(init_model)):
            model_delta = copy.deepcopy(init_model[model_index])
            for key in init_model[model_index].keys():
                model_delta[key] = updated_model[model_index][
                    key] - init_model[model_index][key]
            model_deltas.append(model_delta)

        if len(model_deltas) > 1:
            return model_deltas
        else:
            return model_deltas[0]

    def join_in(self):
        """
        To send ``join_in`` message to the server for joining in the FL course.
        """
        self.comm_manager.send(
            Message(msg_type='join_in',
                    sender=self.ID,
                    receiver=[self.server_id],
                    timestamp=0,
                    content=self.local_address))

    def run(self):
        """
        To listen to the message and handle them accordingly (used for \
        distributed mode)
        """
        while True:
            msg = self.comm_manager.receive()
            if self.state <= msg.state:
                self.msg_handlers[msg.msg_type](msg)

            if msg.msg_type == 'finish':
                break

    def run_standalone(self):
        """
        Run in standalone mode
        """
        self.join_in()
        self.run()

    def callback_funcs_for_model_para(self, message: Message):
        """
        The handling function for receiving model parameters, \
        which triggers the local training process. \
        This handling function is widely used in various FL courses.

        Arguments:
            message: The received message
        """
        if 'ss' in message.msg_type:
            # A fragment of the shared secret
            state, content, timestamp = message.state, message.content, \
                                        message.timestamp
            self.msg_buffer['train'][state].append(content)

            if len(self.msg_buffer['train']
                   [state]) == self._cfg.federate.client_num:
                # Check whether the received fragments are enough
                model_list = self.msg_buffer['train'][state]
                sample_size, first_aggregate_model_para = model_list[0]
                single_model_case = True
                if isinstance(first_aggregate_model_para, list):
                    assert isinstance(first_aggregate_model_para[0], dict), \
                        "aggregate_model_para should a list of multiple " \
                        "state_dict for multiple models"
                    single_model_case = False
                else:
                    assert isinstance(first_aggregate_model_para, dict), \
                        "aggregate_model_para should " \
                        "a state_dict for single model case"
                    first_aggregate_model_para = [first_aggregate_model_para]
                    model_list = [[model] for model in model_list]

                for sub_model_idx, aggregate_single_model_para in enumerate(
                        first_aggregate_model_para):
                    for key in aggregate_single_model_para:
                        for i in range(1, len(model_list)):
                            aggregate_single_model_para[key] += model_list[i][
                                sub_model_idx][key]

                self.comm_manager.send(
                    Message(msg_type='model_para',
                            sender=self.ID,
                            receiver=[self.server_id],
                            state=self.state,
                            timestamp=timestamp,
                            content=(sample_size, first_aggregate_model_para[0]
                                     if single_model_case else
                                     first_aggregate_model_para)))

        else:
            round = message.state
            sender = message.sender
            timestamp = message.timestamp
            content = message.content

            # dequantization
            if self._cfg.quantization.method == 'uniform':
                from federatedscope.core.compression import \
                    symmetric_uniform_dequantization
                if isinstance(content, list):  # multiple model
                    content = [
                        symmetric_uniform_dequantization(x) for x in content
                    ]
                else:
                    content = symmetric_uniform_dequantization(content)

            # When clients share the local model, we must set strict=True to
            # ensure all the model params (which might be updated by other
            # clients in the previous local training process) are overwritten
            # and synchronized with the received model
            if self._cfg.federate.process_num > 1:
                for k, v in content.items():
                    content[k] = v.to(self.device)
            self.trainer.update(content,
                                strict=self._cfg.federate.share_local_model)
            self.state = round
            skip_train_isolated_or_global_mode = \
                self.early_stopper.early_stopped and \
                self._cfg.federate.method in ["local", "global"]
            if self.is_unseen_client or skip_train_isolated_or_global_mode:
                # for these cases (1) unseen client (2) isolated_global_mode,
                # we do not local train and upload local model
                sample_size, model_para_all, results = \
                    0, self.trainer.get_model_para(), {}
                if skip_train_isolated_or_global_mode:
                    logger.info(
                        f"[Local/Global mode] Client #{self.ID} has been "
                        f"early stopped, we will skip the local training")
                    self._monitor.local_converged()
            else:
                if self.early_stopper.early_stopped and \
                        self._monitor.local_convergence_round == 0:
                    logger.info(
                        f"[Normal FL Mode] Client #{self.ID} has been locally "
                        f"early stopped. "
                        f"The next FL update may result in negative effect")
                    self._monitor.local_converged()

                # Client does local training only if it has not been early stopped
                if not self.local_early_stop:
                    # If the client is resuming training after early stopping, log this event
                    if self.was_early_stop:
                        # TODO: Update self.state so that it is only printed during the first resumed round
                        logger.info(f"[Client {self.ID}] Resuming training in round {self.state} after improvement in val_avg_loss due to contributions from other clients' models.")
                        self.was_early_stop = False  # Reset the flag after resuming CREC QUE SERIA AIXO PERO COMPROVAR-HO
                    sample_size, model_para_all, results = self.trainer.train()
                    if self._cfg.federate.share_local_model and not \
                            self._cfg.federate.online_aggr:
                        model_para_all = copy.deepcopy(model_para_all)
                    train_log_res = self._monitor.format_eval_res(
                        results,
                        rnd=self.state,
                        role='Client #{}'.format(self.ID),
                        return_raw=True)
                    logger.info(train_log_res)
                    if self._cfg.wandb.use and self._cfg.wandb.client_train_info:
                        self._monitor.save_formatted_results(train_log_res,
                                                            save_file_name="")                                      
                # If the client has been locally early stopped, it sends the best model it has seen so far
                else:
                    logger.info(
                        f"[Client {self.ID}] Local early stop active. Training halted. "
                        f"Sending best model from round {self.round_in_which_best_client_model_is_saved}."
                    )

                    # Load the best model seen so far
                    self.trainer.load_model(self.best_model_path)
                    model_to_send = self.trainer.get_model_para()
                    sample_size, model_para_all, results = 0, model_to_send, {}
                    
            # Return the feedbacks to the server after local update
            if self._cfg.federate.use_ss:
                assert not self.is_unseen_client, \
                    "Un-support using secret sharing for unseen clients." \
                    "i.e., you set cfg.federate.use_ss=True and " \
                    "cfg.federate.unseen_clients_rate in (0, 1)"
                single_model_case = True
                if isinstance(model_para_all, list):
                    assert isinstance(model_para_all[0], dict), \
                        "model_para should a list of " \
                        "multiple state_dict for multiple models"
                    single_model_case = False
                else:
                    assert isinstance(model_para_all, dict), \
                        "model_para should a state_dict for single model case"
                    model_para_all = [model_para_all]
                model_para_list_all = []
                for model_para in model_para_all:
                    for key in model_para:
                        model_para[key] = model_para[key] * sample_size
                    model_para_list = self.ss_manager.secret_split(model_para)
                    model_para_list_all.append(model_para_list)
                frame_idx = 0
                for neighbor in self.comm_manager.neighbors:
                    if neighbor != self.server_id:
                        content_frame = model_para_list_all[0][frame_idx] if \
                            single_model_case else \
                            [model_para_list[frame_idx] for model_para_list
                             in model_para_list_all]
                        self.comm_manager.send(
                            Message(msg_type='ss_model_para',
                                    sender=self.ID,
                                    receiver=[neighbor],
                                    state=self.state,
                                    timestamp=self._gen_timestamp(
                                        init_timestamp=timestamp,
                                        instance_number=sample_size),
                                    content=content_frame))
                        frame_idx += 1
                content_frame = model_para_list_all[0][frame_idx] if \
                    single_model_case else \
                    [model_para_list[frame_idx] for model_para_list in
                     model_para_list_all]
                self.msg_buffer['train'][self.state] = [(sample_size,
                                                         content_frame)]
            else:
                if self._cfg.asyn.use or self._cfg.aggregator.robust_rule in \
                        ['krum', 'normbounding', 'median', 'trimmedmean',
                         'bulyan']:
                    # Return the model delta when using asynchronous training
                    # protocol, because the staled updated might be discounted
                    # and cause that the sum of the aggregated weights might
                    # not be equal to 1
                    shared_model_para = self._calculate_model_delta(
                        init_model=content, updated_model=model_para_all)
                else:
                    shared_model_para = model_para_all

                # quantization
                if self._cfg.quantization.method == 'uniform':
                    from federatedscope.core.compression import \
                        symmetric_uniform_quantization
                    nbits = self._cfg.quantization.nbits
                    if isinstance(shared_model_para, list):
                        shared_model_para = [
                            symmetric_uniform_quantization(x, nbits)
                            for x in shared_model_para
                        ]
                    else:
                        shared_model_para = symmetric_uniform_quantization(
                            shared_model_para, nbits)
                
                # For loss-based aggregation, we send the best val loss seen so far to the server
                client_key = f"client #{self.ID}"
                # The training rounds before the first eval round the dict of best_results is empty.
                # Therefore, we set a default value of 1 so that all clients are equal and the first weights are FedAvg (same proportion)
                if not self.best_results:
                    client_best_val_loss = 1
                else:        
                    client_best_val_loss = self.best_results[client_key]['val_avg_loss']
              
                self.comm_manager.send(
                    Message(msg_type='model_para',
                            sender=self.ID,
                            receiver=[sender],
                            state=self.state,
                            timestamp=self._gen_timestamp(
                                init_timestamp=timestamp,
                                instance_number=sample_size),
                            content=(sample_size, shared_model_para, client_best_val_loss)))

    def callback_funcs_for_assign_id(self, message: Message):
        """
        The handling function for receiving the client_ID assigned by the \
        server (during the joining process), which is used in the \
        distributed mode.

        Arguments:
            message: The received message
        """
        content = message.content
        self.ID = int(content)
        logger.info('Client (address {}:{}) is assigned with #{:d}.'.format(
            self.comm_manager.host, self.comm_manager.port, self.ID))

    def callback_funcs_for_join_in_info(self, message: Message):
        """
        The handling function for receiving the request of join in \
        information (such as ``batch_size``, ``num_of_samples``) during \
        the joining process.

        Arguments:
            message: The received message
        """
        requirements = message.content
        timestamp = message.timestamp
        join_in_info = dict()
        for requirement in requirements:
            if requirement.lower() == 'num_sample':
                if self._cfg.train.batch_or_epoch == 'batch':
                    num_sample = self._cfg.train.local_update_steps * \
                                 self._cfg.dataloader.batch_size
                else:
                    num_sample = self._cfg.train.local_update_steps * \
                                 len(self.trainer.data.train_data)
                join_in_info['num_sample'] = num_sample
                if self._cfg.trainer.type == 'nodefullbatch_trainer':
                    join_in_info['num_sample'] = \
                        self.trainer.data.train_data.x.shape[0]
            elif requirement.lower() == 'client_resource':
                assert self.comm_bandwidth is not None and self.comp_speed \
                       is not None, "The requirement join_in_info " \
                                    "'client_resource' does not exist."
                join_in_info['client_resource'] = self.model_size / \
                    self.comm_bandwidth + self.comp_speed
            else:
                raise ValueError(
                    'Fail to get the join in information with type {}'.format(
                        requirement))
        self.comm_manager.send(
            Message(msg_type='join_in_info',
                    sender=self.ID,
                    receiver=[self.server_id],
                    state=self.state,
                    timestamp=timestamp,
                    content=join_in_info))

    def callback_funcs_for_address(self, message: Message):
        """
        The handling function for receiving other clients' IP addresses, \
        which is used for constructing a complex topology

        Arguments:
            message: The received message
        """
        content = message.content
        for neighbor_id, address in content.items():
            if int(neighbor_id) != self.ID:
                self.comm_manager.add_neighbors(neighbor_id, address)

    def callback_funcs_for_evaluate(self, message: Message):
        """
        The handling function for receiving the request of evaluating

        Arguments:
            message: The received message
        """
        sender, timestamp = message.sender, message.timestamp
        self.state = message.state
        if message.content is not None:
            # Carreguem els parameteres per evaluar
            self.trainer.update(message.content,
                                strict=self._cfg.federate.share_local_model)
            
        client_key = f"client #{self.ID}"

        use_LDES = self._cfg.federate.use_LDES

        if self.early_stopper.early_stopped and self._cfg.federate.method in ["local", "global"]:
            metrics = list(self.best_results.values())[0]
        # When no local or global methods are used
        else:
            metrics = {}
            metrics_LDES = {}

            if self._cfg.finetune.before_eval:
                logger.info("Finetune before eval")
                self.trainer.finetune()
                
            for split in self._cfg.eval.split:
                # TODO: The time cost of evaluation is not considered here
                eval_metrics = self.trainer.evaluate(
                    target_data_split_name=split)

                if self._cfg.federate.mode == 'distributed':
                    logger.info(
                        self._monitor.format_eval_res(eval_metrics,
                                                      rnd=self.state,
                                                      role='Client #{}'.format(
                                                          self.ID),
                                                      return_raw=True))

                metrics.update(**eval_metrics)

            formatted_eval_res = self._monitor.format_eval_res(
                metrics,
                rnd=self.state,
                role='Client #{}'.format(self.ID),
                forms=['raw'],
                return_raw=True)
            logger.info(formatted_eval_res)
            update_best_this_round = self._monitor.update_best_result(
                self.best_results,
                formatted_eval_res['Results_raw'],
                results_type=client_key,
            )
                    
            if update_best_this_round:
                logger.info(f"[Client {self.ID}] Validation improved in round {self.state}. New best val_avg_loss: {self.best_results[client_key]['val_avg_loss']}") 
                
                if self._cfg.federate.save_client_model:
                    path = add_prefix_to_path(f'client_{self.ID}_',
                                              self._cfg.federate.adapt_save_to)
                    if self.ds_rank == 0:
                        self.trainer.save_model(path, self.state)
                    logger.info(f"[Client {self.ID}] Saved improved model to {path}")
       
                # Client saves the best-performing model to disk and the round in which it was achieved
                self.best_model_path = add_prefix_to_path(f'client_{self.ID}_BEST_MODEL_', self._cfg.federate.adapt_save_to)
                self.trainer.save_model(self.best_model_path, self.state)
                self.round_in_which_best_client_model_is_saved = self.state
                # logger.info(f"[Client {self.ID}] Updated best model with new val_avg_loss: {self.best_results[client_key]['val_avg_loss']}")
            else:
                logger.info(f"[Client {self.ID}] No val_avg_loss improvement in this round.")
            
            """ TODO: Allow cases for early stop in local and global methods (case of 1 single client) [without FedAvg (not using the mean of clients in server)]
            self.early_stopper.track_and_check(self.history_results[
                self._cfg.eval.best_res_update_round_wise_key])
           
             """
            # History_results format from client X: {'test_loss': [44.047755, 43.070358, ...], 'test_total': [33, 33, ...], 'test_avg_loss': [1.33478, 1.305162, ...], 'val_loss': [48.858271, 49.094452, ...], 'val_total': [33, 33, ...], 'val_avg_loss': [1.480554, 1.487711, ...]}
            self.history_results = merge_dict_of_results(self.history_results, formatted_eval_res['Results_raw'])
            # This is the loss obtained by testing the latest aggregated model (downloaded from the server) on the client's local validation data.
            # The current validation loss should be worse than the best loss seen so far if the client is still within its patience window
            logger.info(f"[Client {self.ID}] History of CURRENT validation: {self.history_results['val_avg_loss']}")

            # If using Local Dynamic Early Stopping (LDES)
            if use_LDES:
                val_key = self._cfg.eval.best_res_update_round_wise_key
                # Save the previous state of local_early_stop to determine whether the client is initiating training for the first time or resuming after early stopping
                self.was_early_stop = self.local_early_stop
                self.local_early_stop = self.local_early_stopper.track_and_check(self.history_results[val_key])
                logger.info(f"[Client {self.ID}] Local early stop status: {self.local_early_stop}")
                
                # Best LDES-aware history
                val_avg_loss_to_use = self.best_results[client_key]['val_avg_loss'] if self.local_early_stop else formatted_eval_res['Results_raw']['val_avg_loss']
                self.format_eval_res_LDES['val_avg_loss'] = val_avg_loss_to_use
                self.history_results_with_bestloss = merge_dict_of_results(
                    self.history_results_with_bestloss, self.format_eval_res_LDES
                )
                logger.info(f"[Client {self.ID}] History of BEST validation loss using LDES: {self.history_results_with_bestloss['val_avg_loss']}")

                # Logging transitions
                if self.local_early_stop and not self.was_early_stop:
                    logger.info(f"[Client {self.ID}] Local early stopping triggered in round {self.state}. Client will now send best model.")
                elif self.local_early_stop and self.was_early_stop:
                    logger.info(f"[Client {self.ID}] Continuing in local early stopping mode.")

                # If the client is in local early stopping mode, it means that the patiente has been reached and 
                # the client will not train in the following traininig rounds. Instead, it will send the best model seen so far to the server.
                if self.local_early_stop:
                    # We save the current val_avg_loss as the "val_avg_loss_curr" to keep track of the current round's performance
                    metrics['val_avg_loss_curr'] = metrics['val_avg_loss']
                    # We override the current "val_avg_loss" value with the best valid loss seen so far. In LDES, "val_avg_loss" keeps track of the best valid loss seen so far (except in the window of patience)
                    metrics['val_avg_loss'] = self.best_results[client_key]['val_avg_loss']

                    # Load best model to get the best metrics (Valid and Test if needed)
                    self.trainer.load_model(self.best_model_path)
                    for split in self._cfg.eval.split:
                        # TODO: The time cost of evaluation is not considered here
                        eval_metrics_LDES = self.trainer.evaluate(
                            target_data_split_name=split)
                        if self._cfg.federate.mode == 'distributed':
                            logger.info(
                                self._monitor.format_eval_res(eval_metrics,
                                                            rnd=self.state,
                                                            role='Client #{}'.format(
                                                                self.ID),
                                                            return_raw=True))
                        metrics_LDES.update(**eval_metrics_LDES)
                # If client not in local early stop (window of patience), client keeps the current round's metrics
                else: 
                    metrics_LDES = copy.deepcopy(metrics) # fallback if no early stopping: keep current round's metrics
                    # We fill in the val_avg_loss_curr with the current round's val_avg_loss because 
                    # when the patience is not reached both metrics are the same (no best loss is sent to the server during the window of patience)
                    metrics['val_avg_loss_curr'] = metrics['val_avg_loss']
                
                # Log LDES-aware metrics (either current round's metrics (when in window of patience) or best-so-far if has reached early stop)
                formatted_eval_res_LDES = self._monitor.format_eval_res(
                    metrics_LDES,
                    rnd=self.state,
                    role='Client #{} - LDES'.format(self.ID),
                    forms=['raw'],
                    return_raw=True)
                logger.info(formatted_eval_res_LDES)
            
            # Include the local_early_stop flag in the metrics to keep the server informed of the client's training status
            metrics["local_early_stop"] = self.local_early_stop
            # In LDES, val_avg_loss sent to the server is either current round's metrics (when in window of patience) or best-so-far if client has reached early stop.
            # Besides, val_avg_loss_curr is the current round's performance (when not in early stop is the same as val_avg_loss)
            # Val_avg_loss and val_avg_loss_cur in non-LDES are both equal and is the loss obtained by testing the latest aggregated model
            logger.info(f"[Client {self.ID}] val_avg_loss sent to server: {metrics['val_avg_loss']}")
            logger.info(f"[Client {self.ID}] Track of best val_avg_loss: {self.best_results[client_key]['val_avg_loss']}")

            # If not using LDES
            if not use_LDES:
                # We don't use local_early_stop flag
                metrics['local_early_stop'] = False
                # We fill in the val_avg_loss_curr with the current round's val_avg_loss. 
                # In metrics we will have two equal lists to maintain the format, but we don't really use val_avg_loss_curr for anything when not using LDES
                metrics['val_avg_loss_curr'] = metrics['val_avg_loss']

        # Both when using LDES and when not, we send the same metrics dict format to the server, 
        # but in LDES val_avg_loss is LDES-aware (best-so-far sent if early stop reached) and local_early_stop flag indicates whether the client is in early stop mode
        # In non-LDES, val_avg_loss and val_avg_loss_curr are equal and local_early_stop is always False
        self.comm_manager.send(
            Message(msg_type='metrics',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    timestamp=timestamp,
                    content=metrics))

    def callback_funcs_for_finish(self, message: Message):
        """
        The handling function for receiving the signal of finishing the FL \
        course.

        Arguments:
            message: The received message
        """
        logger.info(
            f"================= client {self.ID} received finish message "
            f"=================")

        if message.content is not None:
            self.trainer.update(message.content,
                                strict=self._cfg.federate.share_local_model)

        self._monitor.finish_fl()

    def callback_funcs_for_converged(self, message: Message):
        """
        The handling function for receiving the signal that the FL course \
        converged

        Arguments:
            message: The received message
        """
        self._monitor.global_converged()

    @classmethod
    def get_msg_handler_dict(cls):
        return cls().msg_handlers_str
