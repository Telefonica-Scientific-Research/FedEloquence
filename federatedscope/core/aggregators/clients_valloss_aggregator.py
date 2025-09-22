import os
import torch
from federatedscope.core.aggregators import Aggregator
from federatedscope.core.auxiliaries.utils import param2tensor
import logging 

logger = logging.getLogger(__name__)

class ClientsValLossAggregator(Aggregator):
    """
    Implementation of FedValLoss aggregation, where the weights are
    proportional to the clients' validation losses (the higher the loss,
    the higher the weight).
    """
    def __init__(self, model=None, device='cpu', config=None):
        super(Aggregator, self).__init__()
        self.model = model
        self.device = device
        self.cfg = config

    def aggregate(self, agg_info):
        """
        To preform aggregation

        Arguments:
            agg_info (dict): the feedbacks from clients

        Returns:
            dict: the aggregated results
        """

        models = agg_info["client_feedback"]
        losses_dict = agg_info["losses_for_weighting"]

        recover_fun = agg_info['recover_fun'] if (
            'recover_fun' in agg_info and self.cfg.federate.use_ss) else None
        
        # FedValLoss aggregation
        avg_model = self._para_loss_weighted_avg(models, losses_dict, recover_fun=recover_fun)

        return avg_model
    
    def update(self, model_parameters):
        """
        Arguments:
            model_parameters (dict): PyTorch Module object's state_dict.
        """
        self.model.load_state_dict(model_parameters, strict=False)

    def save_model(self, path, cur_round=-1):
        assert self.model is not None

        ckpt = {'cur_round': cur_round, 'model': self.model.state_dict()}
        torch.save(ckpt, path)

    def load_model(self, path):
        assert self.model is not None

        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.device)
            self.model.load_state_dict(ckpt['model'])
            return ckpt['cur_round']
        else:
            raise ValueError("The file {} does NOT exist".format(path))
    
    def _para_loss_weighted_avg(self, models, losses, recover_fun=None):
        """
        Weighted average of models based on validation losses.
        Clients with higher loss receive larger weight.

        Args:
            models: list of tuples (sample_size, model_params)
                    The order of models corresponds to client IDs (1, 2, 3, ...).
            losses: dict mapping client_id -> validation loss (float)
            recover_fun: optional function for secret sharing recovery
        """
        assert len(models) == len(losses), \
            "losses must have the same length as models"

        # Step 1: Convert losses into proportional scores
        scores = [round(float(losses[i]), 6) for i in range(1, len(models) + 1)]

        # Step 2: Handle all-zero case (avoid division by zero)
        if sum(scores) == 0:
            scores = [1.0] * len(scores)

        # Step 3: Normalize scores to get weights
        total_score = sum(scores)
        normalized_weights = [s / total_score for s in scores]
        logger.info(f"Normalized weights: {normalized_weights}")
    
        # Step 4: Initialize with the first model
        _, avg_model, _ = models[0]

        # Step 5: Parameter-wise averaging
        for key in avg_model:
            for i in range(len(models)):
                _, local_model, _ = models[i]

                if not self.cfg.federate.use_ss:
                    local_model[key] = param2tensor(local_model[key])

                weight = normalized_weights[i]

                if i == 0:
                    avg_model[key] = local_model[key] * weight
                else:
                    avg_model[key] += local_model[key] * weight

            # Step 6: Secret sharing recovery if needed
            if self.cfg.federate.use_ss and recover_fun:
                avg_model[key] = recover_fun(avg_model[key])
                avg_model[key] /= total_score
                avg_model[key] = torch.FloatTensor(avg_model[key])

        return avg_model
    
class OnlineClientsValLossAggregator(ClientsValLossAggregator):
    """
    Implementation of online aggregation of FedAvg.
    """
    def __init__(self,
                 model=None,
                 device='cpu',
                 src_device='cpu',
                 config=None):
        super(OnlineClientsValLossAggregator, self).__init__(model, device, config)
        self.src_device = src_device

    def reset(self):
        """
        Reset the state of the model to its initial state
        """
        self.maintained = self.model.state_dict()
        for key in self.maintained:
            self.maintained[key].data = torch.zeros_like(
                self.maintained[key], device=self.src_device)
        self.cnt = 0

    def inc(self, content):
        """
        Increment the model weight by the given content.
        """
        if isinstance(content, tuple):
            sample_size, model_params = content
            for key in self.maintained:
                # if model_params[key].device != self.maintained[key].device:
                #    model_params[key].to(self.maintained[key].device)
                self.maintained[key] = (self.cnt * self.maintained[key] +
                                        sample_size * model_params[key]) / (
                                            self.cnt + sample_size)
            self.cnt += sample_size
        else:
            raise TypeError(
                "{} is not a tuple (sample_size, model_para)".format(content))

    def aggregate(self, agg_info):
        """
        Returns the aggregated value
        """
        return self.maintained
