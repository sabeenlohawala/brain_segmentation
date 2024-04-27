#!/bin/bash
#SBATCH --requeue
#SBATCH -t 1-00:00:00
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=40G # per node memory
#SBATCH -p normal
#SBATCH -o ./logs/dice-debug-C2-L2.out
#SBATCH -e ./logs/dice-debug-C2-L2.err
#SBATCH --mail-user=sabeen@mit.edu
#SBATCH --mail-type=FAIL

export PATH="/om2/user/sabeen/miniconda/bin:$PATH"
conda init bash

# Set default values
BATCH_SIZE=512
LR=0.001
NUM_EPOCHS=100
MODEL_NAME="segformer"
LOSS_FN="dice"
DEBUG=0
NR_OF_CLASSES=2
DATA_SIZE="med"
LOG_IMAGES=0

PRETRAINED=0

AUGMENT=0
AUG_CUTOUT=0
CUTOUT_N_HOLES=1
CUTOUT_LENGTH=8
AUG_MASK=0
MASK_N_HOLES=1
MASK_LENGTH=64

INTENSITY_SCALE=0

# LOGDIR="/om2/scratch/tmp/sabeen/20240215-grid-M$MODEL_NAME\S$DATA_SIZE\C$NR_OF_CLASSES\B$BATCH_SIZE\LR$LR\A0"
# LOGDIR="/om2/scratch/tmp/sabeen/20240305-grid-M$MODEL_NAME\S$DATA_SIZE\C$NR_OF_CLASSES\B$BATCH_SIZE\LR$LR\PT$PRETRAINED\A$AUGMENT"
LOGDIR="/om2/scratch/tmp/sabeen/20240313-grid-M$MODEL_NAME\L$LOSS_FN\S$DATA_SIZE\C$NR_OF_CLASSES\B$BATCH_SIZE\LR$LR\PT$PRETRAINED\A$AUGMENT"

# LOGDIR="/om2/scratch/tmp/sabeen/20240227-aug-M$MODEL_NAME\S$DATA_SIZE\C$NR_OF_CLASSES\B$BATCH_SIZE\LR$LR\PT$PRETRAINED\A$AUGMENT"
# LOGDIR="/om2/scratch/tmp/sabeen/20240305-cut-$CUTOUT_LENGTH-$CUTOUT_N_HOLES-M$MODEL_NAME\S$DATA_SIZE\C$NR_OF_CLASSES\B$BATCH_SIZE\LR$LR\PT$PRETRAINED\A$AUGMENT"
# LOGDIR="/om2/scratch/tmp/sabeen/20240313-mask-$MASK_LENGTH-$MASK_N_HOLES-M$MODEL_NAME\S$DATA_SIZE\C$NR_OF_CLASSES\B$BATCH_SIZE\LR$LR\PT$PRETRAINED\A$AUGMENT"

# LOGDIR="/om2/scratch/tmp/sabeen/20240314-intensity-0.2-0.2-M$MODEL_NAME\L$LOSS_FN\S$DATA_SIZE\C$NR_OF_CLASSES\B$BATCH_SIZE\LR$LR\PT$PRETRAINED\A$AUGMENT"
# CHECKPOINT_FILE="$LOGDIR/checkpoint_0001.ckpt"

LOGDIR="/om2/scratch/tmp/sabeen/results/20240426-master-old-dice2loss-2metrics-segformer-2class-512-1e-3"

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
						--loss_fn $LOSS_FN \
						--nr_of_classes $NR_OF_CLASSES \
						--logdir $LOGDIR \
						--num_epochs $NUM_EPOCHS \
						--batch_size $BATCH_SIZE \
						--lr $LR \
						--debug $DEBUG \
						--log_images $LOG_IMAGES \
						--data_size $DATA_SIZE \
						--pretrained $PRETRAINED \
						--augment $AUGMENT \
						--aug_cutout $AUG_CUTOUT \
						--aug_mask $AUG_MASK \
						--cutout_n_holes $CUTOUT_N_HOLES \
						--cutout_length $CUTOUT_LENGTH \
						--mask_n_holes $MASK_N_HOLES \
						--mask_length $MASK_LENGTH \
						--intensity_scale $INTENSITY_SCALE
fi