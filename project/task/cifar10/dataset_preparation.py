"""Functions for Google speech dataset download and processing."""

import logging
from pathlib import Path

import pandas as pd

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from flwr.common.logger import log

import fedscale.cloud.config_parser as parser
from fedscale.dataloaders.divide_data import DataPartitioner

from project.task.speech.fedscale_fixed import init_dataset

from torch.utils.data import Dataset, random_split
from torchvision.datasets import CIFAR10
from torchvision import transforms


@hydra.main(
    config_path="../../conf",
    config_name="speech",
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
    ]),

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train = CIFAR10("/datasets/CIFAR10", train=True, transform=train_transform, download=False)
    test = CIFAR10("/datasets/CIFAR10", train=False, transform=test_transform, download=False)

    train_sets = random_split(train, [1/cfg.dataset.num_clients]*cfg.dataset.num_clients)

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
        client_dataset = client_datasets.use(idx, istest=False)
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
