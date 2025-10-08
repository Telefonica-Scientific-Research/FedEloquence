import numpy as np
from federatedscope.core.splitters import BaseSplitter
import logging 

logger = logging.getLogger(__name__)

class IIDSplitter(BaseSplitter):
    """
    This splitter splits dataset following the independent and identically \
    distribution.

    Args:
        client_num: the dataset will be split into ``client_num`` pieces
    """
    def __init__(self, client_num):
        super(IIDSplitter, self).__init__(client_num)

    def __call__(self, dataset, shuffle_before_splitting=True, shuffle_per_client=True, prior=None):
        from torch.utils.data import Dataset, Subset

        length = len(dataset)
        logger.info("\n=== Starting Set Split ===")
        logger.info(f"Total set size: {length}")
        logger.info(f"Number of clients: {self.client_num}")
        logger.info(f"Shuffle before splitting: {shuffle_before_splitting}")
        logger.info(f"Shuffle per client: {shuffle_per_client}")

        index = [x for x in range(length)]
        logger.info(f"Index sample before shuffling: {index[:5]} ... {index[-5:]}")
        
        # Shuffle the indices of all the clients before splitting
        if shuffle_before_splitting:
            np.random.shuffle(index)
            logger.info(f"Index sample after global shuffle: {index[:5]} ... {index[-5:]}")

        idx_slice = np.array_split(np.array(index), self.client_num)
        logger.info(f"Client split sizes: {[len(idx) for idx in idx_slice]}")
        
        if shuffle_per_client:
            for i, idxs in enumerate(idx_slice):
                np.random.shuffle(idxs)
                logger.info(f"Client {i} sample after per-client shuffle: {idxs[:5]} ... {idxs[-5:]}")
        else:
            logger.info("No per-client shuffling applied.")

        total_after_split = sum(len(idxs) for idxs in idx_slice)
        logger.info(f"Sanity check — total samples after split: {total_after_split} (expected {length})")

        if isinstance(dataset, Dataset):
            data_list = [Subset(dataset, idxs) for idxs in idx_slice]
        else:
            data_list = [[dataset[idx] for idx in idxs] for idxs in idx_slice]
        
        logger.info("Dataset splitting complete.\n")
        return data_list
