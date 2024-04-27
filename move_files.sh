#!/bin/bash
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -p gablab
#SBATCH -o ./logs/mv_tl80.out
#SBATCH -e ./logs/mv_tl80.err
#SBATCH --mail-user=sabeen@mit.edu
#SBATCH --mail-type=FAIL

echo "Submitted Job: $SLURM_JOB_ID"

# rsync -v --remove-source-files --files-from=/om2/scratch/Mon/sabeen/kwyk_slice_split_250/train/features/files_to_copy.txt /om2/scratch/Mon/sabeen/kwyk_slice_split_250/train/features/ /om/scratch/Fri/sabeen/kwyk_slice_split_250/train/features/
rsync -v --remove-source-files --files-from=/om2/scratch/Mon/sabeen/kwyk_slice_split_250/train/labels/files_to_copy.txt /om2/scratch/Mon/sabeen/kwyk_slice_split_250/train/labels/ /om/scratch/Fri/sabeen/kwyk_slice_split_250/train/labels/

# rsync -v --remove-source-files --files-from=/om2/scratch/Mon/sabeen/kwyk_slice_split_250/validation/features/files_to_copy.txt /om2/scratch/Mon/sabeen/kwyk_slice_split_250/validation/features/ /om/scratch/Fri/sabeen/kwyk_slice_split_250/validation/features/
# rsync -v --remove-source-files --files-from=/om2/scratch/Mon/sabeen/kwyk_slice_split_250/validation/labels/files_to_copy.txt /om2/scratch/Mon/sabeen/kwyk_slice_split_250/validation/labels/ /om/scratch/Fri/sabeen/kwyk_slice_split_250/validation/labels/

# rsync -v --remove-source-files --files-from=/om2/scratch/Mon/sabeen/kwyk_slice_split_250/test/features/files_to_copy.txt /om2/scratch/Mon/sabeen/kwyk_slice_split_250/test/features/ /om/scratch/Fri/sabeen/kwyk_slice_split_250/test/features/
# rsync -v --remove-source-files --files-from=/om2/scratch/Mon/sabeen/kwyk_slice_split_250/test/labels/files_to_copy.txt /om2/scratch/Mon/sabeen/kwyk_slice_split_250/test/labels/ /om/scratch/Fri/sabeen/kwyk_slice_split_250/test/labels/