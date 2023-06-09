from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def create_train_validation_loaders(
        dataset: Dataset,
        validation_ratio: float,
        batch_size: int = 100,
        num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    """
    Splits a dataset into a train and validation set, returning a
    DataLoader for each.
    :param dataset: The dataset to split.
    :param validation_ratio: Ratio (in range 0,1) of the validation set size to
        total dataset size.
    :param batch_size: Batch size the loaders will return from each set.
    :param num_workers: Number of workers to pass to dataloader init.
    :return: A tuple of train and validation DataLoader instances.
    """
    if not(0.0 < validation_ratio < 1.0):
        raise ValueError(validation_ratio)

    # TODO: Create two DataLoader instances, dl_train and dl_valid.
    # They should together represent a train/validation split of the given
    # dataset. Make sure that:
    # 1. Validation set size is validation_ratio * total number of samples.
    # 2. No sample is in both datasets. You can select samples at random
    #    from the dataset.

    # ====== YOUR CODE: ======
    dataset_idx = list(range(len(dataset)))
    np.random.shuffle(dataset_idx)
    split_train_val = int(validation_ratio * len(dataset))

    train_idx = dataset_idx[split_train_val:]
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
    dl_train = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)

    validation_idx = dataset_idx[:split_train_val]
    validation_sampler = torch.utils.data.sampler.SubsetRandomSampler(validation_idx)
    dl_valid = DataLoader(dataset, batch_size=batch_size, sampler=validation_sampler, num_workers=num_workers)
    # ========================

    return dl_train, dl_valid
