#!/bin/bash
#SBATCH -t 02:00:00
#SBATCH -c 8
#SBATCH --mem-per-cpu=12G # per node memory
#SBATCH -w node115
#SBATCH -p normal
#SBATCH -o ./logs/new_prepare_data_8cpu_any.out
#SBATCH -e ./logs/new_prepare_data_8cpu_any.err

export PATH="/om2/user/sabeen/miniconda/envs/tissue_labeling/bin/:$PATH"
conda init bash
# source activate tissue_labeling
echo "Submitted Job: $SLURM_JOB_ID"

# -u ensures that the output is unbuffered, and written immediately to stdout.
# srun python -u scripts/mit_kwyk_data.py "/om2/scratch/tmp/sabeen-kwyk-data/kwyk-volumes/rawdata/" "/om2/scratch/tmp/sabeen-kwyk-data/kwyk_slice_uncrop_rot_new" --rotate_vol=1
srun python -u scripts/mit_kwyk_data.py "/om2/scratch/Mon/sabeen/kwyk-volumes/rawdata/" "/om2/user/sabeen/kwyk_data/0422_kwyk_slice_1000_8cpu" --rotate_vol=0
