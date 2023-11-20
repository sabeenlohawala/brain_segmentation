#!/bin/bash
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:2
##SBATCH --mem-per-cpu=32G # per node memory
#SBATCH --mem=40G # per node memory
#SBATCH -p gablab
#SBATCH -o ./slurm_outputs/test-med-512-20-dp.out

export PATH="/om2/user/sabeen/miniconda/bin:$PATH"
conda init bash

# -u ensures that the output is unbuffered, and written immediately to stdout.
# 24 batch size per A100 GPU
# For multi GPU training
cd src
srun python -u main.py $wandb_description

# to run:
# sbatch --export=ALL,wandb_description='testrun' jobs/job_multigpu.sh

# SBATCH -p multi-gpu
# SBATCH --constraint=high-capacity
# SBATCH --gres=gpu:a100:1
# SBATCH --constraint=any-A100
# SBATCH --constraint=high-capacity
# SBATCH --gres=gpu:1