#!/bin/bash

#  1. download the data
# NOTE: must be run with slurm to access /datasets (remember to not run with GPUs!)

source /nfs-share/jr897/miniconda3/bin/activate oort
python -m project.task.cifar10.dataset_preparation

# 2. download the client traces

wget https://github.com/SymbioticLab/FedScale/raw/master/benchmark/dataset/data/device_info/client_behave_trace -O data/client_behave_trace.pkl
wget https://github.com/SymbioticLab/FedScale/raw/master/benchmark/dataset/data/device_info/client_device_capacity -O data/client_device_capacity.pkl

#RUN WITH:
#srun -c 1 --gres=gpu:0 -w ngongotaha bash run_scripts/download_cifar10.sh