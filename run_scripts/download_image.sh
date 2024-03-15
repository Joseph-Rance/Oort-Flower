#!/bin/bash

#  1. download the data

wget https://raw.githubusercontent.com/SymbioticLab/FedScale/master/benchmark/dataset/download.sh
mkdir data
bash download.sh download open_images_detection

# below code required to join data and download

# import pandas as pd
# client_data_mapping_df = pd.read_csv("data/data/client_data_mapping/trainval.csv")
# client_data_mapping_df["name"] = client_data_mapping_df["data_path"].map(lambda x : x[:-4])
# names_labels_df = pd.read_csv("/datasets/FedScale/openImg/client_data_mapping/train.csv")
# names_labels_df["name"] = names_labels_df["sample_path"].map(lambda x : x[:-14])
# client_data_mapping_df = client_data_mapping_df.join(names_labels_df.set_index("name"), on="name", lsuffix="l", rsuffix="r")
# classes = client_data_mapping_df.groupby(["client_idl"]).count().sort_values(["count"], ascending=False).head(60)
# client_data_mapping_df = client_data_mapping_df.log[client_data_mapping_df["client_idl"] in classes]
# datasets = {}
# for i, row in client_data_mapping_df.iterrows():
#     datasets[row["client_idl"]] = datasets.get(row["client_idl"], "") + f"{row['sample_pathr']},{int(row['label_idr'])}\n"
#     if i == 100:
#         break

# 2. download the client traces

wget https://github.com/SymbioticLab/FedScale/raw/master/benchmark/dataset/data/device_info/client_behave_trace -O data/client_behave_trace.pkl
wget https://github.com/SymbioticLab/FedScale/raw/master/benchmark/dataset/data/device_info/client_behave_capacity -O data/client_device_capacity.pkl
