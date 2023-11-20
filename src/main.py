
import torch
import numpy as np
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import time
import argparse
import wandb

from data.dataset import get_data_loader
from utils import load_brains, set_seed, crop, init_cuda, init_fabric, init_wandb
from models.metrics import Dice
from models.segformer import Segformer
from training.trainer import Trainer

parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
parser.add_argument('wandb_description', help="Description add to the wandb run", type=str)

args = parser.parse_args()

WANDB_RUN_DESCRIPTION = args.wandb_description
WANDB_RUN_TITLE = "Brain Segmentation"

NR_OF_CLASSES = 2 # set to 2 for binary classification
BATCH_SIZE = 64
LEARNING_RATE = 3e-6
N_EPOCHS = 5
DATASET = 'medium'
MODEL_NAME = "segformer"
SEED = 700
SAVE_EVERY = "epoch"
PRECISION = '32-true' #"16-mixed"

def main():
    
    set_seed(SEED)
    
    fabric = init_fabric(precision=PRECISION)#,devices=2,strategy='dp') # accelerator="gpu", devices=2, num_nodes=1
    init_cuda()

    # model
    model = Segformer(NR_OF_CLASSES)
    model = fabric.to_device(model)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # loss function
    loss_fn = Dice(NR_OF_CLASSES, fabric)

    # get data loader
    train_loader, val_loader, _ = get_data_loader(DATASET, batch_size=BATCH_SIZE)
    model, optimizer = fabric.setup(model, optimizer)
    # if A100 GPU, compile model for HUGE speed up
    # if "A100" in torch.cuda.get_device_name():
    #     print("Compile model...")
    #     model = torch.compile(model)

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
        init_wandb(WANDB_RUN_TITLE, fabric, model_params, WANDB_RUN_DESCRIPTION)
        save_frequency = train_loader.length if SAVE_EVERY == "epoch" else 1000
        wandb.watch(model, log_freq=save_frequency)

    trainer = Trainer(
         model=model,
         nr_of_classes=NR_OF_CLASSES,
         train_loader=train_loader,
         val_loader=val_loader,
         loss_fn=loss_fn,
         optimizer=optimizer,
         fabric=fabric,
    )
    trainer.train(N_EPOCHS)
    print("Training Finished!")

if __name__ == "__main__":
    main()