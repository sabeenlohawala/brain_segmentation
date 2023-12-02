
import torch
import numpy as np
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import time
import argparse
import wandb
from torch.utils.data import Dataset, DataLoader
import glob

from data.dataset import get_data_loader
from utils import load_brains, set_seed, crop, init_cuda, init_fabric, init_wandb
from models.metrics import Dice
from models.segformer import Segformer
from training.trainer import Trainer

parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')

parser.add_argument('--wandb_description', help="Description add to the wandb run", type=str, required=False)
parser.add_argument('--batch_size', help="Batch size for training", type=int, required=False, default=64)
parser.add_argument('--learning_rate', help="Learning for training", type=float, required=False, default=6e-5)
parser.add_argument('--num_epochs', help="Number of epochs to train", type=int, required=False, default=20)
parser.add_argument('--seed', help="Random seed value", type=int, required=False, default=42)
parser.add_argument('--model_name', help="Name of model to use for segmentation", type=str, default='segformer')

args = parser.parse_args()

WANDB_ON = args.wandb_description is not None
WANDB_RUN_DESCRIPTION = args.wandb_description
WANDB_RUN_TITLE = "Brain Segmentation"

NR_OF_CLASSES = 107 # set to 2 for binary classification
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate # 3e-6
N_EPOCHS = args.num_epochs
DATASET = 'small'
MODEL_NAME = args.model_name
SEED = args.seed
SAVE_EVERY = "epoch"
PRECISION = '32-true' #"16-mixed"

def main():
    
    fabric = init_fabric(precision=PRECISION, devices=2, strategy='ddp') # accelerator="gpu", devices=2, num_nodes=1
    set_seed(SEED) # TODO: replace with seed_everything(SEED)?
    init_cuda()

    # model
    if MODEL_NAME == 'segformer':
        print('Model found!')
        model = Segformer(NR_OF_CLASSES, pretrained=True)
    else:
        raise Exception('Invalid model name provided')

    # TODO: loading model from checkpoint

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # loss function
    loss_fn = Dice(NR_OF_CLASSES, fabric)

    # get data loader
    train_loader, val_loader, _ = get_data_loader('/om2/user/sabeen/nobrainer_data_norm/data_prepared_segmentation_small', batch_size=BATCH_SIZE)

    # fabric setup
    train_loader, val_loader = fabric.setup_dataloaders(train_loader,val_loader)
    model, optimizer = fabric.setup(model, optimizer)

    # model params to track with wandb
    model_params = {
        'learning rate': LEARNING_RATE,
        '# epochs': N_EPOCHS,
        'batch size': BATCH_SIZE,
        'model': MODEL_NAME,
        'dataset': DATASET,
        'validation frequency': SAVE_EVERY,
        'precision': PRECISION
        }

    # init WandB
    if fabric.global_rank == 0:
        init_wandb(WANDB_ON, WANDB_RUN_TITLE, fabric, model_params, WANDB_RUN_DESCRIPTION) # comment to not save to wandb
        save_frequency = len(train_loader) if SAVE_EVERY == "epoch" else 1000
        if WANDB_ON:
            wandb.watch(model, log_freq=save_frequency) # comment to not save to wandb

    trainer = Trainer(
         model=model,
         nr_of_classes=NR_OF_CLASSES,
         train_loader=train_loader,
         val_loader=val_loader,
         loss_fn=loss_fn,
         optimizer=optimizer,
         fabric=fabric,
         batch_size=BATCH_SIZE,
         wandb_on=WANDB_ON
    )
    trainer.train(N_EPOCHS)
    print("Training Finished!")

if __name__ == "__main__":
    main()