#!/bin/bash
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -p gablab
#SBATCH -o ./logs/del_val_99_f.out
#SBATCH -e ./logs/del_val_99_f.err
#SBATCH --mail-user=sabeen@mit.edu
#SBATCH --mail-type=FAIL

echo "Submitted Job: $SLURM_JOB_ID"
rsync -rav --delete /om2/scratch/Sat/sabeen/emptydir/ /om2/scratch/Sat/sabeen/validation/features/

# rsync -v --remove-source-files --files-from=/om2/scratch/Mon/sabeen/kwyk_slice_split_250/train/file_to_delete.txt /om2/scratch/Mon/sabeen/kwyk_slice_split_250/train/ /om2/scratch/Sat/sabeen/train/