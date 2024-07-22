#!/bin/bash
#SBATCH --requeue
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=40G # per node memory
#SBATCH -p gablab
#SBATCH -o ./logs/test.out
#SBATCH -e ./logs/test.err
#SBATCH --mail-user=sabeen@mit.edu
#SBATCH --mail-type=FAIL

export PATH="/om2/user/sabeen/miniconda/bin:$PATH"
conda init bash

# Set default values
DEFAULT_BATCH_SIZE=64
DEFAULT_LR=0.001
DEFAULT_NUM_EPOCHS=20
DEFAULT_MODEL_NAME="segformer"
DEFAULT_DEBUG=0
DEFAULT_NR_OF_CLASSES=50
DEFAULT_DATA_SIZE="small"
# aug_flip = 0 1 2 3
DEFAULT_LOG_IMAGES=0
DEFAULT_LOGDIR="/om2/scratch/Sat/sabeen/test"

# Parse arguments
for arg in "$@"
do
    case $arg in
        batch_size=*)
            BATCH_SIZE="${arg#*=}"
            ;;
        lr=*)
            LR="${arg#*=}"
            ;;
        num_epochs=*)
            NUM_EPOCHS="${arg#*=}"
            ;;
        model_name=*)
            MODEL_NAME="${arg#*=}"
            ;;
        debug=*)
            DEBUG="${arg#*=}"
            ;;
        nr_of_classes=*)
            NR_OF_CLASSES="${arg#*=}"
            ;;
        data_size=*)
            DATA_SIZE="${arg#*=}"
            ;;
        log_images=*)
            LOG_IMAGES="${arg#*=}"
            ;;
        logdir=*)
            LOGDIR="${arg#*=}"
            ;;
        *)
            echo "Unknown argument: $arg"
            exit 1
            ;;
    esac
done

# Use default values if variables are not set
: ${BATCH_SIZE:=$DEFAULT_BATCH_SIZE}
: ${LR:=$DEFAULT_LR}
: ${NUM_EPOCHS:=$DEFAULT_NUM_EPOCHS}
: ${MODEL_NAME:=$DEFAULT_MODEL_NAME}
: ${DEBUG:=$DEFAULT_DEBUG}
: ${NR_OF_CLASSES:=$DEFAULT_NR_OF_CLASSES}
: ${DATA_SIZE:=$DEFAULT_DATA_SIZE}
: ${LOG_IMAGES:=$DEFAULT_LOG_IMAGES}
: ${LOGDIR:=$DEFAULT_LOGDIR}

# Use the variables in your script
echo "Batch size is: $BATCH_SIZE"
echo "Learning rate is: $LR"
echo "NUM_EPOCHS is: $NUM_EPOCHS"
echo "MODEL_NAME is: $MODEL_NAME"
echo "DEBUG is: $DEBUG"
echo "NR_OF_CLASSES is $NR_OF_CLASSES"
echo "DATA_SIZE is $DATA_SIZE"
echo "LOG_IMAGES is $LOG_IMAGES"
echo "LOGDIR is $LOGDIR"

CHECKPOINT_FILE="$LOGDIR/checkpoint_0001.ckpt"

# srun python -u scripts/commands/main.py train --logdir='/om2/scratch/Sat/sabeen/20240212-grid-Msegformer\\Smed\\C51\\B128\\LR0.0001\\A0/' --num_epochs=100 --batch_size=128 --model_name='segformer' --nr_of_classes=51 --lr=0.0001 --data_size='med'
# srun python -u scripts/commands/main.py train --logdir='20240205-single-4gpu-Msimple_unet\Ssmall\Ldice\C51\B370\A1' --num_epochs=1000 --batch_size=370 --model_name='simple_unet' --nr_of_classes=51 --lr=5e-5 --data_size='small' --augment=1

# Check if checkpoint file exists
if [ -f "$CHECKPOINT_FILE" ]; then
    echo "Checkpoint file found. Resuming training..."
    echo $LOGDIR
    python -u scripts/commands/main.py resume-train \
        --logdir $LOGDIR
else
    echo "No checkpoint file found. Starting training..."
    echo $LOGDIR
    python -u scripts/commands/main.py train \
						--model_name $MODEL_NAME \
						--nr_of_classes $NR_OF_CLASSES \
						--logdir $LOGDIR \
						--num_epochs $NUM_EPOCHS \
						--batch_size $BATCH_SIZE \
						--lr $LR \
						--debug $DEBUG \
						--log_images $LOG_IMAGES \
						--data_size $DATA_SIZE
fi