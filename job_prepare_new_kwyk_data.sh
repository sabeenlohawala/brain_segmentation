#!/bin/bash
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -p gablab
#SBATCH -o ./logs/new_prepare_data.out
#SBATCH -e ./logs/new_prepare_data.err

export PATH="/om2/user/sabeen/miniconda/envs/tissue_labeling/bin/:$PATH"
conda init bash
# source activate tissue_labeling

# -u ensures that the output is unbuffered, and written immediately to stdout.
srun python -u scripts/mit_kwyk_data.py "/nese/mit/group/sig/data/kwyk_transform" "/nese/mit/group/sig/data/kwyk_slices" --rotate_vol=0