"""
Filename: tissue_labeling/scripts/main.py
Created Date: Wednesday, March 6th, 2024
Author: Sabeen Lohawala
Description: Similar to training code of TissueLabeling to test SoftmaxFocalLoss implementation

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
from TissueLabeling.metrics.metrics import Dice
from TissueLabeling.models.segformer import Segformer
from TissueLabeling.models.original_unet import OriginalUnet
from TissueLabeling.models.attention_unet import AttentionUnet
from TissueLabeling.parser import get_args
from TissueLabeling.training.trainer import Trainer
from TissueLabeling.utils import (
    init_cuda,
    init_fabric,
    init_wandb,
    set_seed,
    main_timer,
)
from TissueLabeling.metrics.losses import SoftmaxFocalLoss


def select_model(config):
    """
    Selects the model based on the model name provided in the config file.
    """
    if config.model_name == "segformer":
        model = Segformer(config.nr_of_classes, pretrained=config.pretrained)
    elif config.model_name == "original_unet":
        model = OriginalUnet(image_channels=1, nr_of_classes=config.nr_of_classes)
    elif config.model_name == "attention_unet":
        model = AttentionUnet(
            dim=16,
            channels=1,
            dim_mults=(2, 4, 8, 16, 32, 64),
        )
    elif config.model_name == "nobrainer_unet":
        model = OriginalUnet(image_channels=1, nr_of_classes=config.nr_of_classes)
    else:
        print(f"Invalid model name provided: {config.model_name}")
        sys.exit()

    print(f"{config.model_name} found")
    if config.checkpoint:
        print(f"Loading from checkpoint...")
        model.load_state_dict(torch.load(config.checkpoint)["model"])

    return model


def update_config(config):
    """
    Updates the config file based on the command line arguments.
    """
    if sys.argv[1] == "train":
        config = Configuration(config)

    elif sys.argv[1] == "resume-train":
        chkpt_folder = os.path.join("results/", config.logdir)

        config_file = os.path.join(chkpt_folder, "config.json")
        if not os.path.exists(config_file):
            sys.exit(f"Configuration file not found at {config_file}")

        with open(config_file) as json_file:
            data = json.load(json_file)
        assert isinstance(data, dict), "Invalid Object Type"

        dice_list = sorted(glob.glob(os.path.join(chkpt_folder, "checkpoint*")))
        if not dice_list:
            sys.exit("No checkpoints exist to resume training")

        data["checkpoint"] = dice_list[-1]
        data["start_epoch"] = int(
            os.path.basename(dice_list[-1]).split(".")[0].split("_")[-1]
        )

        configs = sorted(glob.glob(os.path.join(chkpt_folder, "config*.json")))
        config_file_name = f"config_resume_{len(configs):02d}.json"
        args = argparse.Namespace(**data)
        config = Configuration(args, config_file_name)

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

    fabric = init_fabric(precision=config.precision)
    set_seed(config.seed)
    init_cuda()

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    # loss function
    # loss_fn = Dice(fabric, config)
    loss_fn = SoftmaxFocalLoss(alpha=0.25)

    # get data loader
    train_loader, val_loader, _ = get_data_loader(config)

    # fabric setup
    train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)
    model, optimizer = fabric.setup(model, optimizer)

    # init WandB
    if fabric.global_rank == 0 and config.wandb_on:
        # model params to track with wandb
        model_params = {
            "learning rate": config.lr,
            "# epochs": config.num_epochs,
            "batch size": config.batch_size,
            "model": config.model_name,
            "dataset": config.data_dir,
            "validation frequency": "epoch",
            "precision": config.precision,
        }
        init_wandb("Brain Segmentation", fabric, model_params, config.wandb_description)
        save_frequency = (
            len(train_loader)
            if model_params["validation frequency"] == "epoch"
            else 1000
        )
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
    trainer.train_and_validate()
    print("Training Finished!")


if __name__ == "__main__":
    main()
