#!/bin/bash
#SBATCH --requeue
#SBATCH -t 1-00:00:00
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -p normal
#SBATCH -o ./logs/del_train_2_f.out
#SBATCH -e ./logs/del_train_2_f.err
#SBATCH --mail-user=sabeen@mit.edu
#SBATCH --mail-type=FAIL

echo "Submitted Job: $SLURM_JOB_ID"
# rsync -rav --delete /om2/scratch/Mon/sabeen/emptydir/ /om2/scratch/Mon/sabeen/kwyk_slice_split_250/train/labels/
rsync -rav --delete /om2/scratch/Mon/sabeen/emptydir/ /om2/scratch/Sat/sabeen/train/features/