#!/bin/bash

#SBATCH --job-name=oort
#SBATCH -c 4
#SBATCH --gres=gpu:1

cd /nfs-share/jr897/Oort-Flower
source ../miniconda3/bin/activate oort
bash run_scripts/launch.sh

#RUN WITH:
#srun -c 16 --gres=gpu:2 -w ngongotaha bash run_scripts/slurm.sh