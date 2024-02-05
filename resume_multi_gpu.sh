#!/bin/bash
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=40G # per node memory
#SBATCH -p multi-gpu
#SBATCH -o ./logs/med-50seg-multi-su.out
#SBATCH -e ./logs/med-50seg-multi-su.err
#SBATCH --mail-user=sabeen@mit.edu
#SBATCH --mail-type=FAIL

export PATH="/om2/user/sabeen/miniconda/bin:$PATH"
conda init bash

# -u ensures that the output is unbuffered, and written immediately to stdout.
# 24 batch size per A100 GPU
# For multi GPU training
srun python -u scripts/commands/main.py resume-train --logdir='20240204-multi-4gpu-Msegformer\Smed\Ldice\C51\B648\A0'
# srun python -u scripts/commands/main.py train --logdir='20240204-multi-4gpu-Msimple_unet\Smed\Ldice\C51\B352\A0' --num_epochs=1000 --batch_size=352 --model_name='simple_unet' --nr_of_classes=51 --lr=5e-5 --data_size='med'

# to run:
# sbatch --export=ALL,wandb_description='testrun' jobs/job.sh

# SBATCH -p multi-gpu
# SBATCH --constraint=high-capacity
# SBATCH --gres=gpu:a100:1
# SBATCH --constraint=any-A100
# SBATCH --constraint=high-capacity
# SBATCH --gres=gpu:1
