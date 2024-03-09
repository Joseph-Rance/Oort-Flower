"""Functions for Google speech dataset download and processing."""

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from flwr.common.logger import log

import fedscale.cloud.config_parser as parser
from fedscale.dataloaders.divide_data import DataPartitioner

from project.task.speech.fedscale_fixed import init_dataset

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
    # 1. print parsed config
    log(logging.INFO, OmegaConf.to_yaml(cfg))

    parser.args.task = "speech"
    parser.args.data_set = "google_speech"
    parser.args.num_class = 35
    parser.args.data_dir = Path(cfg.dataset.dataset_dir)

    trainset, testset = init_dataset()

    # use predetermined, "realistic" dataset split
    client_datasets = DataPartitioner(data=trainset, args=parser.args, numOfClass=parser.args.num_class)
    client_datasets.partition_data_helper(num_clients=None, data_map_file=cfg.dataset.dataset_dir+"/client_data_mapping/train.csv")

    fed_test_set = testset

    print(f'Total number of data samples: {client_datasets.getDataLen()}')
    print(f'Total number of clients: {client_datasets.getClientLen()}')

    # 2. Save the datasets
    # unnecessary for this small dataset, but useful for large datasets
    partition_dir = Path(cfg.dataset.partition_dir)
    partition_dir.mkdir(parents=True, exist_ok=True)

    # Save the centralized test set
    # a centralized training set would also be possible
    # but is not used here
    torch.save(fed_test_set, partition_dir / "test.pt")

    # Save the client datasets
    # validation is commented in order to follow setup in Oort paper
    for idx in range(client_datasets.getClientLen()):
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
