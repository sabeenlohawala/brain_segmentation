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

from TissueLabeling.config import Configuration
from TissueLabeling.data.dataset import NoBrainerDataset
from TissueLabeling.models.metrics import Dice
from TissueLabeling.models.segformer import Segformer
from TissueLabeling.models.unet import Unet
from TissueLabeling.parser import get_args
from TissueLabeling.training.trainer import Trainer
from TissueLabeling.utils import init_cuda, init_fabric, set_seed

# NR_OF_CLASSES = 51 # set to 2 for binary classification
# DATA_DIR = args.data_dir
# SAVE_EVERY = "epoch"
# PRECISION = '32-true' #"16-mixed"
# PRETRAINED = args.pretrained


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
    else:
        print(f"Invalid model name provided: {config.model_name}")
        sys.exit()

    return model


def update_config(config):
    """
    Updates the config file based on the command line arguments.
    """
    if sys.argv[1] == "train":
        config = Configuration(args)

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
    train_dataset = NoBrainerDataset(
        "/om2/user/sabeen/nobrainer_data_norm/data_prepared_segmentation_small/train/extracted_tensors"
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    # fabric setup
    train_loader = fabric.setup_dataloaders(train_loader)
    model, optimizer = fabric.setup(model, optimizer)

    trainer = Trainer(
        model=model,
        nr_of_classes=config.nr_of_classes,
        train_loader=train_loader,
        val_loader=None,
        loss_fn=loss_fn,
        optimizer=optimizer,
        fabric=fabric,
        batch_size=config.batch_size,
        wandb_on=False,
        pretrained=config.pretrained,
        logdir=config.logdir,
    )
    trainer.train(config.num_epochs)
    print("Training Finished!")


if __name__ == "__main__":
    main()
