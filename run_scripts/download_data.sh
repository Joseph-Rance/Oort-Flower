#!/bin/bash

#  1. download the data

wget https://raw.githubusercontent.com/SymbioticLab/FedScale/master/benchmark/dataset/download.sh
mkdir data
bash download.sh download speech

python -m project.task.speech.dataset_preparation

# 2. download the client traces

wget https://github.com/SymbioticLab/FedScale/raw/master/benchmark/dataset/data/device_info/client_behave_trace -O data/client_behave_trace.pkl