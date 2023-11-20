#!/bin/bash
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -o ./jobs/job_pixel_count.out
#SBATCH --mem=5G # per node memory
#SBATCH -p gablab

export PATH="/om2/user/matth406/miniconda/bin:$PATH"
conda init bash

# -u ensures that the output is unbuffered, and written immediately to stdout.
# 24 batch size per A100 GPU
# For multi GPU training
cd src
srun python -u check_pixel_distr.py

# to run:
# sbatch --export=ALL,wandb_description='testrun' jobs/job.sh

# SBATCH -p multi-gpu
# SBATCH --constraint=high-capacity
# SBATCH --gres=gpu:a100:1
# SBATCH --constraint=any-A100
# SBATCH --constraint=high-capacity
# SBATCH --gres=gpu:1