import logging
import numpy as np
import torch
import json
import os

from federatedscope.core.auxiliaries.splitter_builder import get_splitter
from federatedscope.core.data import ClientData, StandaloneDataDict
from torch.utils.data import Dataset, Subset

# Uncomment this during training to verify that the subsets are correctly taken from the multilingual FL dataset
# from federatedscope.llm.dataloader.dataloader import get_tokenizer

logger = logging.getLogger(__name__)

class BaseDataTranslator:
    """
    Translator is a tool to convert a centralized dataset to \
    ``StandaloneDataDict``, which is the input data of runner.

    Notes:
        The ``Translator`` is consist of several stages:

        Dataset -> ML split (``split_train_val_test()``) -> \
        FL split (``split_to_client()``) -> ``StandaloneDataDict``

    """
    def __init__(self, global_cfg, client_cfgs=None):
        """
        Convert data to `StandaloneDataDict`.

        Args:
            global_cfg: global CfgNode
            client_cfgs: client cfg `Dict`
        """
        self.global_cfg = global_cfg
        self.client_cfgs = client_cfgs
        self.splitter = get_splitter(global_cfg)

    def __call__(self, dataset):
        """
        Args:
            dataset: `torch.utils.data.Dataset`, `List` of (feature, label)
                or split dataset tuple of (train, val, test) or Tuple of
                split dataset with [train, val, test]

        Returns:
            datadict: instance of `StandaloneDataDict`, which is a subclass of
            `dict`.
        """
        datadict = self.split(dataset)
        datadict = StandaloneDataDict(datadict, self.global_cfg)

        return datadict

    def split(self, dataset):
        """
        Perform ML split and FL split.

        Returns:
            dict of ``ClientData`` with client_idx as key to build \
            ``StandaloneDataDict``
        """
        s_val, s_test, train, val, test = self.split_sval_stest_train_val_test(dataset)
        datadict = self.split_to_client(s_val, s_test, train, val, test)
        return datadict

    def split_sval_stest_train_val_test(self, dataset, cfg=None):
        """
        Split dataset to server_val, server_test, train, val, test if not provided.

        Returns:
             List: List of split dataset, like ``[sval, stest, train, val, test]``
        """

        if cfg is not None:
            splits = cfg.data.splits
        else:
            splits = self.global_cfg.data.splits
        if isinstance(dataset, tuple):
            # No need to split train/val/test for tuple dataset.
            error_msg = 'If dataset is tuple, it must contains ' \
                        'train, valid and test split.'
            assert len(dataset) == len(['train', 'val', 'test']), error_msg
            return [None, None, dataset[0], dataset[1], dataset[2]]
        
        logger.info(f"Length input dataset: {len(dataset)}")

        if self.global_cfg.federate.shuffle_all_data == True:
            index = np.random.permutation(np.arange(len(dataset)))
        else:
            index = np.arange(len(dataset)) # NO SHUFFLE
        
        # Length for both the val and test sets in the server
        len_server_dataset = self.global_cfg.eval.len_server_dataset

        # Reamining length for clients
        len_clients_dataset = len(dataset)-2*len_server_dataset

        train_size = int(splits[0] * len_clients_dataset)
        logger.info(f"Length train set (before splitting it into clients): {train_size}")
        val_size = int(splits[1] * len_clients_dataset)
        logger.info(f"Length val set (before splitting it into clients): {val_size}")
        logger.info(f"For now length val set is the same as length test set (before splitting it into clients).")
        assert 2 * len_server_dataset + train_size + 2 * val_size <= len(dataset), \
            "Dataset is too small for the requested splits"

        if isinstance(dataset, Dataset):

            # To verify and consult the generated subsets
            model_type = self.global_cfg.model.type
            model_path = model_type.split("@")[0]
            path_to_subsets = "generated_FL_subsets"
            os.makedirs(path_to_subsets, exist_ok=True)

            # Uncomment this during training to verify that the subsets are correctly taken from the multilingual FL dataset
            # tokenizer, _ = get_tokenizer(model_path, "data/" , 1000, "huggingface_llm")
            
            val_server_dataset = Subset(dataset, index[:len_server_dataset])
            logger.info(f"Length of the server val set: {len(val_server_dataset)}")

            test_server_dataset = Subset(dataset, index[len_server_dataset : 2*len_server_dataset])
            logger.info(f"Length of the server test set: {len(test_server_dataset)}")
            
            train_dataset = Subset(dataset, 
                                    index[2*len_server_dataset : 2*len_server_dataset + train_size])

            """ Uncomment this during training to verify that the subsets are correctly taken from the multilingual FL dataset            
            output_path = f"{path_to_subsets}/server_VAL.jsonl"
            with open(output_path, 'w', encoding='utf-8') as f:
                for sample in val_server_dataset:
                    json_line = json.dumps(tokenizer.decode(sample["input_ids"], skip_special_tokens=True), ensure_ascii=False)
                    f.write(json_line + '\n')
            output_path = f"{path_to_subsets}/server_TEST.jsonl"
            with open(output_path, 'w', encoding='utf-8') as f:
                for sample in test_server_dataset:
                    json_line = json.dumps(tokenizer.decode(sample["input_ids"], skip_special_tokens=True), ensure_ascii=False)
                    f.write(json_line + '\n')
            """

            """
            if self.global_cfg.federate.monolingual_exp.enable == False:
                train_dataset = Subset(dataset, 
                                    index[2*len_server_dataset : 2*len_server_dataset + train_size])
            else:
                lang_pos_id = self.global_cfg.federate.monolingual_exp.lang_pos # Quan fem aquest experiment no fem servir el val i test dels clients, potser caldra adapar futur per val i test
                train_dataset = Subset(dataset, 
                                    index[2*len_server_dataset + train_size*lang_pos_id: 2*len_server_dataset + train_size*(lang_pos_id+1)])
            """
            logger.info(f"Length train set (before splitting it into clients): {len(train_dataset)}")

            val_dataset = Subset(dataset,
                                index[2*len_server_dataset + train_size : 2*len_server_dataset + train_size + val_size])
            logger.info(f"Length val set (before splitting it into clients): {val_dataset}")

            test_dataset = Subset(dataset, 
                                index[2*len_server_dataset + train_size + val_size : 2*len_server_dataset + train_size + 2*val_size])
            logger.info(f"Length test set (before splitting it into clients): {len(test_dataset)}")

            """ Uncomment this during training to verify that the subsets are correctly taken from the multilingual FL dataset
            output_path = f"{path_to_subsets}/clients_TRAIN.jsonl"
            with open(output_path, 'w', encoding='utf-8') as f:
                for sample in train_dataset:
                    json_line = json.dumps(tokenizer.decode(sample["input_ids"], skip_special_tokens=True), ensure_ascii=False)
                    f.write(json_line + '\n')
            
            output_path = f"{path_to_subsets}/clients_VAL.jsonl"
            with open(output_path, 'w', encoding='utf-8') as f:
                for sample in val_dataset:
                    json_line = json.dumps(tokenizer.decode(sample["input_ids"], skip_special_tokens=True), ensure_ascii=False)
                    f.write(json_line + '\n')

            output_path = f"{path_to_subsets}/clients_TEST.jsonl"
            with open(output_path, 'w', encoding='utf-8') as f:
                for sample in test_dataset:
                    json_line = json.dumps(tokenizer.decode(sample["input_ids"], skip_special_tokens=True), ensure_ascii=False)
                    f.write(json_line + '\n')
            """
        else:
            val_server_dataset = [dataset[x] for x in index[:len_server_dataset]]

            test_server_dataset = [dataset[x] for x in index[len_server_dataset : 2*len_server_dataset]]

            train_dataset = [dataset[x] for x in index[2*len_server_dataset : 2*len_server_dataset + train_size]]
     
            val_dataset = [
                dataset[x] for x in index[2*len_server_dataset + train_size : 2*len_server_dataset + train_size + val_size]
            ]

            test_dataset = [dataset[x] for x in index[2*len_server_dataset + train_size + val_size : 2*len_server_dataset + train_size + 2*val_size]]

        return val_server_dataset, test_server_dataset, train_dataset, val_dataset, test_dataset

    def split_to_client(self, s_val, s_test, train, val, test):
        """
        Split dataset to clients and build ``ClientData``.

        Returns:
            dict: dict of ``ClientData`` with ``client_idx`` as key.
        """

        # Initialization
        client_num = self.global_cfg.federate.client_num
        split_train = [None] * client_num
        split_val = [None] * client_num
        split_test = [None] * client_num

        train_label_distribution = None

        shuffle_clients_train = self.global_cfg.federate.shuffle_train_clients
        shuffle_clients_val = self.global_cfg.federate.shuffle_val_clients
        shuffle_clients_test = self.global_cfg.federate.shuffle_test_clients

        # Split train/val/test to n clients
        if len(train) > 0:
            split_train = self.splitter(train, shuffle_clients_train)
            logger.info(f"Number of splits of train set (number of train subsets): {len(split_train)}")

            if self.global_cfg.data.consistent_label_distribution:
                try:
                    train_label_distribution = [[j[1] for j in x]
                                                for x in split_train]
                except:
                    logger.warning(
                        'Cannot access train label distribution for '
                        'splitter, split dataset without considering train '
                        'label.')
        if len(val) > 0:
            split_val = self.splitter(val, shuffle_clients_val, prior=train_label_distribution)
            logger.info(f"Number of splits of val set (number of val subsets): {len(split_val)}")
            
        if len(test) > 0:
            split_test = self.splitter(test, shuffle_clients_test, prior=train_label_distribution)
            logger.info(f"Number of splits of test set (number of test subsets): {len(split_test)}")
        
        data_dict = {
            0: ClientData(self.global_cfg, train=None, val=s_val, test=s_test)
        }
        for client_id in range(1, client_num + 1):
            if self.client_cfgs is not None:
                client_cfg = self.global_cfg.clone()
                client_cfg.merge_from_other_cfg(
                    self.client_cfgs.get(f'client_{client_id}'))
            else:
                client_cfg = self.global_cfg
            data_dict[client_id] = ClientData(client_cfg,
                                              train=split_train[client_id - 1],
                                              val=split_val[client_id - 1],
                                              test=split_test[client_id - 1])        
        return data_dict