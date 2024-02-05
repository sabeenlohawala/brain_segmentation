#!/bin/bash
#SBATCH -t 1-00:00:00
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=40G # per node memory
#SBATCH -p gablab
#SBATCH -o ./logs/small-51-aug-seg.out
#SBATCH -e ./logs/small-51-aug-seg.err
#SBATCH --mail-user=sabeen@mit.edu
#SBATCH --mail-type=FAIL

export PATH="/om2/user/sabeen/miniconda/bin:$PATH"
conda init bash

# -u ensures that the output is unbuffered, and written immediately to stdout.
# 24 batch size per A100 GPU
# For multi GPU training
srun python -u scripts/commands/main.py train --logdir='20240204-single-4gpu-Msegformer\Ssmall\Ldice\C51\B512\A1' --num_epochs=1000 --batch_size=512 --model_name='segformer' --nr_of_classes=51 --lr=5e-5 --data_size='small' --augment=1
# srun python -u scripts/commands/main.py train --logdir='20240204-single-aug-su' --num_epochs=3 --batch_size=374 --model_name='simple_unet' --nr_of_classes=51 --lr=5e-5 --data_size='small' --augment=1
# srun python -u scripts/commands/main.py train --logdir='20240204-single-4gpu-Msimple_unet\Ssmall\Ldice\C51\B374\A1' --num_epochs=1000 --batch_size=374 --model_name='simple_unet' --nr_of_classes=51 --lr=5e-5 --data_size='small' --augment=1

# to run:
# sbatch --export=ALL,wandb_description='testrun' jobs/job.sh

# SBATCH -p multi-gpu
# SBATCH --constraint=high-capacity
# SBATCH --gres=gpu:a100:1
# SBATCH --constraint=any-A100
# SBATCH --constraint=high-capacity
# SBATCH --gres=gpu:1
