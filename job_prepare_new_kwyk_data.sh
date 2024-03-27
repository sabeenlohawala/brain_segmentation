#!/bin/bash
#SBATCH -t 1-00:00:00
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=128G # per node memory
#SBATCH -p gablab
#SBATCH -o ./logs/new_prepare_data.out
#SBATCH -e ./logs/new_prepare_data.err

export PATH="/om2/user/sabeen/miniconda/envs/tissue_labeling/bin/:$PATH"
conda init bash
# source activate tissue_labeling

# -u ensures that the output is unbuffered, and written immediately to stdout.
srun python -u scripts/mit_kwyk_data.py "/om2/scratch/tmp/sabeen/kwyk/rawdata/" "/om2/scratch/tmp/sabeen/kwyk_final_uncrop/" --rotate_vol=0