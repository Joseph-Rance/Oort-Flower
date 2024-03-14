#!/bin/bash

#  1. download the data

source /nfs-share/jr897/miniconda3/bin/activate oort
python -m project.task.cifar10.dataset_preparation

# 2. download the client traces

wget https://github.com/SymbioticLab/FedScale/raw/master/benchmark/dataset/data/device_info/client_behave_trace -O data/client_behave_trace.pkl