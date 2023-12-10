"""
Filename: tissue_labeling/scripts/main.py
Created Date: Friday, December 8th 2023
Author: Sabeen Lohawala
Description: Training code of TissueLabeling

Copyright (c) 2023, Sabeen Lohawala. MIT
"""
import argparse
import glob
import json
import os
import sys

import torch
import wandb

from TissueLabeling.config import Configuration
from TissueLabeling.data.dataset import get_data_loader
from TissueLabeling.models.metrics import Dice
from TissueLabeling.models.segformer import Segformer
from TissueLabeling.models.unet import Unet
from TissueLabeling.models.simple_unet import SimpleUnet
from TissueLabeling.parser import get_args
from TissueLabeling.training.trainer import Trainer
from TissueLabeling.utils import init_cuda, init_fabric, init_wandb, set_seed, main_timer

def select_model(config):
    """
    Selects the model based on the model name provided in the config file.
    """
    if config.model_name == "segformer":
        print("Segformer model found!")
        model = Segformer(config.nr_of_classes, pretrained=config.pretrained)
    elif config.model_name == "unet":
        print("Unet model found!")
        model = Unet(
            dim=16,
            channels=1,
            dim_mults=(2, 4, 8, 16, 32, 64),
        )
    elif config.model_name == "simple_unet":
        print('Simple Unet model found!')
        model = SimpleUnet(1)
    else:
        print(f"Invalid model name provided: {config.model_name}")
        sys.exit()

    return model


def update_config(config):
    """
    Updates the config file based on the command line arguments.
    """
    if sys.argv[1] == "train":
        config = Configuration(config)

    elif sys.argv[1] == "resume-train":
        chkpt_folder = config.logdir

        config_file = os.path.join(chkpt_folder, "config.json")
        if not os.path.exists(config_file):
            sys.exit(f"Configuration file not found at {config_file}")

        with open(config_file) as json_file:
            data = json.load(json_file)
        assert isinstance(data, dict), "Invalid Object Type"

        dice_list = sorted(glob.glob(os.path.join(chkpt_folder, "model*")))
        if not dice_list:
            sys.exit("No checkpoints exist to resume training")

        data["checkpoint"] = dice_list[-1]
        data["start_epoch"] = int(os.path.basename(dice_list[-1]).split("_")[-1])

        args = argparse.Namespace(**data)
        config = Configuration(args, "config_resume.json")

    else:
        sys.exit("Invalid Sub-command")

    return config

@main_timer
def main():
    """
    The main function that executes the entire program.
    """
    args = get_args()

    config = update_config(args)
    model = select_model(config)

    fabric = init_fabric(precision=config.precision)  # , devices=2, strategy='ddp')
    set_seed(config.seed)
    init_cuda()

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    # loss function
    loss_fn = Dice(config.nr_of_classes, fabric, config.data_dir)

    # get data loader
    train_loader, val_loader, _ = get_data_loader(config)

    # fabric setup
    train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)
    model, optimizer = fabric.setup(model, optimizer)

    # model params to track with wandb
    model_params = {
        'learning rate': config.lr,
        '# epochs': config.num_epochs,
        'batch size': config.batch_size,
        'model': config.model_name,
        'dataset': config.data_dir,
        'validation frequency': "epoch",
        'precision': config.precision
        }

    # init WandB
    if fabric.global_rank == 0 and config.wandb_on:
        init_wandb("Brain Segmentation", fabric, model_params, config.wandb_description)
        save_frequency = len(train_loader) if model_params['validation frequency'] == "epoch" else 1000
        wandb.watch(model, log_freq=save_frequency)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        fabric=fabric,
        config=config,
    )
    trainer.train(config.num_epochs)
    print("Training Finished!")


if __name__ == "__main__":
    main()
