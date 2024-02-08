#!/bin/bash
#SBATCH -t 1-00:00:00
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=40G # per node memory
#SBATCH -p gablab
#SBATCH -o ./logs/grid.out
#SBATCH -e ./logs/grid.err
#SBATCH --mail-user=sabeen@mit.edu
#SBATCH --mail-type=FAIL

export PATH="/om2/user/sabeen/miniconda/bin:$PATH"
conda init bash

echo 'Start time:' `date`
echo 'Node:' $HOSTNAME
echo "$@"
start=$(date +%s)

"$@"

end=$(date +%s)
echo 'End time:' `date`
echo "Elapsed Time: $(($end-$start)) seconds"