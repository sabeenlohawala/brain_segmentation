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

# -u ensures that the output is unbuffered, and written immediately to stdout.
# srun python -u scripts/mit_kwyk_data.py "/om2/scratch/tmp/sabeen-kwyk-data/kwyk-volumes/rawdata/" "/om2/scratch/tmp/sabeen-kwyk-data/kwyk_slice_uncrop_rot_new" --rotate_vol=1
# srun python -u scripts/mit_kwyk_data.py "/om2/scratch/Mon/sabeen/kwyk-volumes/rawdata/" "/om2/scratch/Mon/sabeen/kwyk_slice_uncrop" --rotate_vol=0

# srun python -u scripts/mit_kwyk_data_2.py "/om2/scratch/Mon/sabeen/kwyk-volumes/rawdata/" "/om2/scratch/Mon/sabeen/kwyk_slice_split_250" --rotate_vol=0 --group_size=250 --group_num=47

# TESTING SCRIPT
# srun python -u scripts/mit_kwyk_data_2.py "/om2/scratch/Mon/sabeen/kwyk-volumes/rawdata/" "/om2/user/sabeen/kwyk_data/0423_split_vols_1000" --rotate_vol=0 --group_size=500 --group_num=1


# VOLUME DATA
srun python -u scripts/mit_kwyk_data_optimized.py