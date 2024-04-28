#!/bin/bash
#SBATCH --requeue
#SBATCH -t 1-00:00:00
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=40G # per node memory
#SBATCH -p gablab
#SBATCH -o ./logs/old_pad_255.out
#SBATCH -e ./logs/old_pad_255.err
#SBATCH --mail-user=sabeen@mit.edu
#SBATCH --mail-type=FAIL

echo "Submitted Job: $SLURM_JOB_ID"

export PATH="/om2/user/sabeen/miniconda/bin:$PATH"
conda init bash

# General hyperparams
BATCH_SIZE=128
LR=0.001
NUM_EPOCHS=100
MODEL_NAME="segformer"
PRETRAINED=1
LOSS_FN="dice"
DEBUG=0
NR_OF_CLASSES=50
LOG_IMAGES=0
CLASS_SPECIFIC_SCORES=0
CHECKPOINT_FREQ=5

# Dataset params
NEW_KWYK_DATA=0
BACKGROUND_PERCENT_CUTOFF=0.8
ROTATE_VOL=0

PAD_OLD_DATA=1
USE_NORM_CONSTS=0
DATA_SIZE="med"

# Data augmentation params
AUGMENT=0
INTENSITY_SCALE=0
AUG_CUTOUT=0
CUTOUT_N_HOLES=1
CUTOUT_LENGTH=8
AUG_MASK=0
MASK_N_HOLES=1
MASK_LENGTH=64
AUG_NULL_HALF=0
AUG_BACKGROUND_MANIPULATION=0
AUG_SHAPES_BACKGROUND=0
AUG_GRID_BACKGROUND=1

LOGDIR="/om2/scratch/tmp/sabeen/results/20240428-oldPadded-noNorm-grid-M$MODEL_NAME\L$LOSS_FN\S$DATA_SIZE\RV$ROTATE_VOL\BC$BACKGROUND_PERCENT_CUTOFF\C$NR_OF_CLASSES\B$BATCH_SIZE\LR$LR\PT$PRETRAINED\A$AUGMENT"

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
						--intensity_scale $INTENSITY_SCALE \
						--aug_null_half $AUG_NULL_HALF \
						--new_kwyk_data $NEW_KWYK_DATA \
						--background_percent_cutoff $BACKGROUND_PERCENT_CUTOFF \
						--class_specific_scores $CLASS_SPECIFIC_SCORES \
						--checkpoint_freq $CHECKPOINT_FREQ \
						--aug_background_manipulation $AUG_BACKGROUND_MANIPULATION \
						--aug_shapes_background $AUG_SHAPES_BACKGROUND \
						--aug_grid_background $AUG_GRID_BACKGROUND \
						--pad_old_data $PAD_OLD_DATA \
						--use_norm_consts $USE_NORM_CONSTS
fi