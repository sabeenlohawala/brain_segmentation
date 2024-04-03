#!/bin/bash
#SBATCH -t 1-00:00:00
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=40G # per node memory
#SBATCH -p gablab
#SBATCH -o ./logs/grid.out
#SBATCH -e ./logs/grid.err
#SBATCH --mail-user=sabeen@mit.edu
#SBATCH --mail-type=FAIL

export PATH="/om2/user/sabeen/miniconda/bin:$PATH"
conda init bash

# -u ensures that the output is unbuffered, and written immediately to stdout.
# 24 batch size per A100 GPU
# For multi GPU training
srun python -u scripts/commands/main.py train --logdir='/om2/scratch/Sat/sabeen/20240212-grid-Msegformer\\Smed\\C51\\B128\\LR0.0001\\A0/' --num_epochs=100 --batch_size=128 --model_name='segformer' --nr_of_classes=50 --lr=0.0001 --data_size='med'
# srun python -u scripts/commands/main.py train --logdir='20240205-single-4gpu-Msimple_unet\Ssmall\Ldice\C51\B370\A1' --num_epochs=1000 --batch_size=370 --model_name='simple_unet' --nr_of_classes=50 --lr=5e-5 --data_size='small' --augment=1

# to run:
# sbatch --export=ALL,wandb_description='testrun' jobs/job.sh

# SBATCH -p multi-gpu
# SBATCH --constraint=high-capacity
# SBATCH --gres=gpu:a100:1
# SBATCH --constraint=any-A100
# SBATCH --constraint=high-capacity
# SBATCH --gres=gpu:1
