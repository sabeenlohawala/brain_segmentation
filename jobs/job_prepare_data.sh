#!/bin/bash
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH -n 5
#SBATCH -p gablab
#SBATCH -o ./jobs/job_data_small_2.out

export PATH="/om2/user/sabeen/miniconda/bin:$PATH"
conda init bash

# -u ensures that the output is unbuffered, and written immediately to stdout.
cd src
srun python -u prepare_data.py medium new_med_no_aug_51