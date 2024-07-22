#!/bin/bash
#SBATCH -t 1-00:00:00
#SBATCH -c 48
#SBATCH --mem-per-cpu=12G # per node memory
#SBATCH -p gablab
#SBATCH -o ./logs/new_kwyk_tests/kwyk_data_optimized.out
#SBATCH -e ./logs/new_kwyk_tests/kwyk_data_optimized.err

export PATH="/om2/user/sabeen/miniconda/envs/tissue_labeling/bin/:$PATH"
conda init bash
# source activate tissue_labeling
echo "Submitted Job: $SLURM_JOB_ID"

# VOLUME DATA
srun python -u scripts/mit_kwyk_data_optimized.py "/om2/scratch/Mon/sabeen/kwyk-volumes/rawdata/" "/om2/user/sabeen/kwyk_data/" "new_kwyk_full.npy" --n_vols 11479