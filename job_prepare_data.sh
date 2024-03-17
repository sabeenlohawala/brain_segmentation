#!/bin/bash
#SBATCH -t 1-00:00:00
#SBATCH -N 1
#SBATCH -n 10
#SBATCH -p gablab
#SBATCH -o ./logs/job_data_large.out
#SBATCH -e ./logs/job_data_large.err

export PATH="/om2/user/sabeen/miniconda/bin:$PATH"
conda init bash

# -u ensures that the output is unbuffered, and written immediately to stdout.
srun python -u scripts/prepare_data.py large new_large_no_aug_51