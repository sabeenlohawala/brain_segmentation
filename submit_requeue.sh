#!/bin/bash
#SBATCH --requeue
#SBATCH -t 1-00:00:00
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=40G # per node memory
#SBATCH -p use-everything
#SBATCH -o ./logs/grid-test.out
#SBATCH -e ./logs/grid-test.err
#SBATCH --mail-user=sabeen@mit.edu
#SBATCH --mail-type=FAIL

export PATH="/om2/user/sabeen/miniconda/bin:$PATH"
conda init bash

# Set default values
BATCH_SIZE=256
LR=0.001
NUM_EPOCHS=200
MODEL_NAME="segformer"
DEBUG=0
NR_OF_CLASSES=51
DATA_SIZE="med"
LOG_IMAGES=0
PRETRAINED=1
AUGMENT=0

# LOGDIR="/om2/scratch/Sat/sabeen/20240215-grid-M$MODEL_NAME\S$DATA_SIZE\C$NR_OF_CLASSES\B$BATCH_SIZE\LR$LR\A0"
LOGDIR="/om2/scratch/Sat/sabeen/20240218-grid-M$MODEL_NAME\S$DATA_SIZE\C$NR_OF_CLASSES\B$BATCH_SIZE\LR$LR\PT$PRETRAINED\A$AUGMENT"
# CHECKPOINT_FILE="$LOGDIR/checkpoint_0001.ckpt"

# Check if checkpoint file exists
if ls "$LOGDIR"/*.ckpt 1> /dev/null 2>&1; then
    echo "Checkpoint file found. Resuming training..."
    echo $LOGDIR
    srun python -u scripts/commands/main.py resume-train \
        --logdir $LOGDIR
else
    echo "No checkpoint file found. Starting training..."
    echo $LOGDIR
    srun python -u scripts/commands/main.py train \
						--model_name $MODEL_NAME \
						--nr_of_classes $NR_OF_CLASSES \
						--logdir $LOGDIR \
						--num_epochs $NUM_EPOCHS \
						--batch_size $BATCH_SIZE \
						--lr $LR \
						--debug $DEBUG \
						--log_images $LOG_IMAGES \
						--data_size $DATA_SIZE \
						--pretrained $PRETRAINED \
						--augment $AUGMENT
fi