#!/bin/bash
#SBATCH -t 1-00:00:00
#SBATCH -c 48
#SBATCH --mem-per-cpu=16G # per node memory
#SBATCH -o ./logs/new_prepare_data_rot.out
#SBATCH -e ./logs/new_prepare_data_rot.err

export PATH="/om2/user/sabeen/miniconda/envs/tissue_labeling/bin/:$PATH"
conda init bash
# source activate tissue_labeling

# -u ensures that the output is unbuffered, and written immediately to stdout.
srun python -u scripts/mit_kwyk_data.py "/om2/scratch/tmp/sabeen/kwyk_data/kwyk/rawdata/" "/om2/scratch/tmp/sabeen/kwyk_data/kwyk_final_uncrop_rot/" --rotate_vol=1