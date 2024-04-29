#!/bin/bash
#SBATCH -t 1-00:00:00
#SBATCH -c 10
#SBATCH --mem-per-cpu=5G # per node memory
#SBATCH -p gablab
#SBATCH -o ./logs/gen_dataset_nonbrain.out
#SBATCH -e ./logs/gen_dataset_nonbrain.err

export PATH="/om2/user/sabeen/miniconda/envs/tissue_labeling/bin/:$PATH"
conda init bash
# source activate tissue_labeling
echo "Submitted Job: $SLURM_JOB_ID"

srun python -u scripts/gen_dataset_nonbrain.py /om2/scratch/Sat/satra/ 10 /om2/user/sabeen/kwyk_hdf_nonbrains/