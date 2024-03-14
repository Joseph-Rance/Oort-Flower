"""Functions for CIFAR10 dataset download and processing."""

import logging
from pathlib import Path

import pandas as pd

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from flwr.common.logger import log

import fedscale.cloud.config_parser as parser
from fedscale.dataloaders.divide_data import DataPartitioner

from torch.utils.data import Dataset, Subset, random_split, ConcatDataset
from torchvision.datasets import CIFAR10
from torchvision import transforms

import numpy as np
from typing import cast
from collections.abc import Sequence

def _sort_by_class(
    trainset: Dataset,
) -> Dataset:
    """Sort dataset by class/label.

    Parameters
    ----------
    trainset : Dataset
        The training dataset that needs to be sorted.

    Returns
    -------
    Dataset
        The sorted training dataset.
    """
    class_counts = np.bincount(trainset.targets)
    idxs = np.array(trainset.targets).argsort()  # sort targets in ascending order

    tmp = []  # create subset of smallest class
    tmp_targets = []  # same for targets

    start = 0
    for count in np.cumsum(class_counts):
        tmp.append(
            Subset(
                trainset,
                cast(
                    Sequence[int],
                    idxs[start : int(count + start)],
                ),
            ),
        )  # add rest of classes
        tmp_targets.append(
            np.array(trainset.targets)[idxs[start : int(count + start)]],
        )
        start += count
    sorted_dataset = cast(
        Dataset,
        ConcatDataset(tmp),
    )  # concat dataset
    sorted_dataset.targets = np.concatenate(
        tmp_targets,
    ).tolist()  # concat targets
    return sorted_dataset

# pylint: disable=too-many-locals, too-many-arguments
def _power_law_split(
    sorted_trainset: Dataset,
    num_partitions: int,
    num_labels_per_partition: int = 2,
    min_data_per_partition: int = 10,
    mean: float = 0.0,
    sigma: float = 2.0,
) -> list[Subset]:
    """Partition the dataset following a power-law distribution. It follows the.

    implementation of Li et al 2020: https://arxiv.org/abs/1812.06127 with default
    values set accordingly.

    Parameters
    ----------
    sorted_trainset : Dataset
        The training dataset sorted by label/class.
    num_partitions: int
        Number of partitions to create
    num_labels_per_partition: int
        Number of labels to have in each dataset partition. For
        example if set to two, this means all training examples in
        a given partition will belong to the same two classes. default 2
    min_data_per_partition: int
        Minimum number of datapoints included in each partition, default 10
    mean: float
        Mean value for LogNormal distribution to construct power-law, default 0.0
    sigma: float
        Sigma value for LogNormal distribution to construct power-law, default 2.0

    Returns
    -------
    Dataset
        The partitioned training dataset.
    """
    targets = sorted_trainset.targets
    full_idx = list(range(len(targets)))

    class_counts = np.bincount(sorted_trainset.targets)
    labels_cs = np.cumsum(class_counts)
    labels_cs = [0] + labels_cs[:-1].tolist()

    partitions_idx: list[list[int]] = []
    num_classes = len(np.bincount(targets))
    hist = np.zeros(num_classes, dtype=np.int32)

    # assign min_data_per_partition
    min_data_per_class = int(
        min_data_per_partition / num_labels_per_partition,
    )
    for u_id in range(num_partitions):
        partitions_idx.append([])
        for cls_idx in range(num_labels_per_partition):
            # label for the u_id-th client
            cls = (u_id + cls_idx) % num_classes
            # record minimum data
            indices = list(
                full_idx[
                    labels_cs[cls]
                    + hist[cls] : labels_cs[cls]
                    + hist[cls]
                    + min_data_per_class
                ],
            )
            partitions_idx[-1].extend(indices)
            hist[cls] += min_data_per_class

    # add remaining images following power-law
    probs = np.random.lognormal(
        mean,
        sigma,
        (
            num_classes,
            int(num_partitions / num_classes),
            num_labels_per_partition,
        ),
    )
    remaining_per_class = class_counts - hist
    # obtain how many samples each partition should be assigned for each of the
    # labels it contains
    # pylint: disable=too-many-function-args
    probs = (
        remaining_per_class.reshape(-1, 1, 1)
        * probs
        / np.sum(probs, (1, 2), keepdims=True)
    )

    for u_id in range(num_partitions):
        for cls_idx in range(num_labels_per_partition):
            cls = (u_id + cls_idx) % num_classes
            count = int(
                probs[cls, u_id // num_classes, cls_idx],
            )

            # add count of specific class to partition
            indices = full_idx[
                labels_cs[cls] + hist[cls] : labels_cs[cls] + hist[cls] + count
            ]
            partitions_idx[u_id].extend(indices)
            hist[cls] += count

    # construct partition subsets
    return [Subset(sorted_trainset, p) for p in partitions_idx]

@hydra.main(
    config_path="../../conf",
    config_name="cifar10",
    version_base=None,
)
def download_and_preprocess(cfg: DictConfig) -> None:
    """Download and preprocess the dataset.

    Please include here all the logic
    Please use the Hydra config style as much as possible specially
    for parts that can be customized (e.g. how data is partitioned)

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train = CIFAR10("/datasets/CIFAR10", train=True, transform=train_transform, download=False)
    test = CIFAR10("/datasets/CIFAR10", train=False, transform=test_transform, download=False)

    #partitions = [int(len(train)/cfg.dataset.num_clients)]*cfg.dataset.num_clients
    #partitions[-1] += len(train) - sum(partitions)
    #train_sets = random_split(train, partitions)

    trainset_sorted = _sort_by_class(train)
    train_sets = _power_law_split(
        trainset_sorted,
        num_partitions=cfg.dataset.num_clients,
        num_labels_per_partition=5,
        min_data_per_partition=100,
        mean=0.0,
        sigma=2.0,
    )

    print([len(i) for i in train_sets])

    # 2. Save the datasets
    # unnecessary for this small dataset, but useful for large datasets
    partition_dir = Path(cfg.dataset.partition_dir)
    partition_dir.mkdir(parents=True, exist_ok=True)

    # Save the centralized test set
    # a centralized training set would also be possible
    # but is not used here
    torch.save(test, partition_dir / "test.pt")

    # Save the client datasets
    # validation is commented in order to follow setup in Oort paper
    for idx in range(cfg.dataset.num_clients):
        client_dataset = train_sets[idx]
        client_dir = partition_dir / f"client_{idx}"
        client_dir.mkdir(parents=True, exist_ok=True)

        #len_val = int(
        #    len(client_dataset) / (1 / cfg.dataset.val_ratio),
        #)
        #lengths = [len(client_dataset) - len_val, len_val]
        #ds_train, ds_val = random_split(
        #    client_dataset,
        #    lengths,
        #    torch.Generator().manual_seed(cfg.dataset.seed),
        #)
        ## Alternative would have been to create train/test split
        ## when the dataloader is instantiated

        torch.save(client_dataset, client_dir / "train.pt")
        #torch.save(ds_val, client_dir / "test.pt")


if __name__ == "__main__":
    download_and_preprocess()
