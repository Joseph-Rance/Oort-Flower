#!/bin/bash

#SBATCH --job-name=oort
#SBATCH -c 4
#SBATCH --gres=gpu:1

cd /nfs-share/jr897/Oort-Flower
source /nfs-share/jr897/miniconda3/bin/activate oort
bash run_scripts/launch.sh

#RUN WITH:
#srun -c 16 --gres=gpu:2 -w ngongotaha bash run_scripts/slurm.sh

# srun -c 1 --gres=gpu:0 -w ngongotaha bash run_scripts/download_cifar10.sh;srun -c 16 --gres=gpu:2 -w ngongotaha bash run_scripts/slurm.sh