#!/bin/bash
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -p normal
#SBATCH -o ./logs/kwyk_transform_crop_10.out
#SBATCH -e ./logs/kwyk_transform_crop_10.err
#SBATCH --mail-user=sabeen@mit.edu
#SBATCH --mail-type=FAIL

echo "Submitted Job: $SLURM_JOB_ID"
# rsync -rav --delete /om2/scratch/Sat/sabeen/emptydir/ /om2/scratch/Sat/sabeen/validation/features/
rsync -rav --delete /om2/user/sabeen/emptydir/ /om2/user/sabeen/new_kwyk_data/kwyk_transform_crop_10/
# rsync -rav --delete /om/scratch/Fri/sabeen/emptydir/ /om/scratch/Fri/sabeen/to_delete/

# rsync -v --remove-source-files --files-from=/om2/scratch/Mon/sabeen/kwyk_slice_split_250/train/file_to_delete.txt /om2/scratch/Mon/sabeen/kwyk_slice_split_250/train/ /om2/scratch/Sat/sabeen/train/