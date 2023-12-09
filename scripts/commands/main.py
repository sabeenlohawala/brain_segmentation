"""
Filename: tissue_labeling/scripts/main.py
Created Date: Friday, December 8th 2023
Author: Sabeen Lohawala
Description: Training code of TissueLabeling

Copyright (c) 2023, Sabeen Lohawala. MIT
"""
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
# BATCH_SIZE = args.batch_size
# LEARNING_RATE = args.lr # 3e-6
# N_EPOCHS = args.num_epochs
# DATA_DIR = args.data_dir
# MODEL_NAME = args.model_name
# SEED = args.seed
# SAVE_EVERY = "epoch"
# PRECISION = '32-true' #"16-mixed"
# PRETRAINED = args.pretrained
# LOGDIR = args.logdir


def main():
    """
    The main function that executes the entire program.

    Parameters:
    - None

    Return:
    - None
    """
    args = get_args()
    config = Configuration(args)

    # you can move this as well to config.py
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
        print(model)
    else:
        print(f"Invalid model name provided: {config.model_name}")
        sys.exit()

    # TODO: loading model from checkpoint

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
        num_workers=4,
        pin_memory=True,
    )
    train_loader = fabric.setup_dataloaders(train_loader)

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
        pretrained=config.pretrained,
        logdir=config.logdir,
    )
    trainer.train(config.num_epochs)
    print("Training Finished!")


if __name__ == "__main__":
    main()
