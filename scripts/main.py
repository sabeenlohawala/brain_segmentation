"""
Filename: tissue_labeling/scripts/main.py
Created Date: Friday, December 8th 2023
Author: Sabeen Lohawala
Description: Training code of TissueLabeling

Copyright (c) 2023, Sabeen Lohawala. MIT
"""
import os

import torch

from TissueLabeling.config import Configuration
from TissueLabeling.data.dataset import get_data_loader
from TissueLabeling.models.metrics import Dice
from TissueLabeling.models.segformer import Segformer
from TissueLabeling.models.unet import Unet
from TissueLabeling.parser import get_args
from TissueLabeling.training.trainer import Trainer
from TissueLabeling.utils import init_cuda, init_fabric, init_wandb, set_seed

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
    args = get_args()
    config = Configuration(args)
    print("LOGDIR", config.logdir)

    # TODO: do this check in config.py
    if not os.path.exists(config.data_dir):
        raise Exception("Dataset not found")

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
        print(config.model_name)
        raise Exception("Invalid model name provided")

    # TODO: loading model from checkpoint

    fabric = init_fabric(precision=config.precision)  # , devices=2, strategy='ddp')
    set_seed(config.seed)
    init_cuda()

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    # loss function
    loss_fn = Dice(config.nr_of_classes, fabric, config.data_dir)

    # get data loader
    # train_loader, val_loader, _ = get_data_loader(f'/om2/user/sabeen/nobrainer_data_norm/data_prepared_segmentation_{DATASET}', batch_size=BATCH_SIZE, pretrained=config.pretrained)
    train_loader, val_loader, _ = get_data_loader(
        config.data_dir, batch_size=config.batch_size, pretrained=config.pretrained
    )

    # fabric setup
    train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)
    model, optimizer = fabric.setup(model, optimizer)

    # model params to track with wandb
    model_params = {
        "learning rate": config.lr,
        "# epochs": config.num_epochs,
        "batch size": config.batch_size,
        "model": config.model_name,
        "data_dir": config.data_dir,
        "validation frequency": config.save_every,
        "precision": config.precision,
    }

    # init WandB
    if fabric.global_rank == 0:
        init_wandb(
            config.wandb_on,
            config.wandb_run_title,
            fabric,
            model_params,
            config.wandb_description,
        )
        save_frequency = len(train_loader) if config.save_every == "epoch" else 1000
        if config.wandb_on:
            wandb.watch(model, log_freq=save_frequency)

    trainer = Trainer(
        model=model,
        nr_of_classes=config.nr_of_classes,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        fabric=fabric,
        batch_size=config.batch_size,
        wandb_on=config.wandb_on,
        pretrained=config.pretrained,
        logdir=config.logdir,
    )
    trainer.train(config.num_epochs)
    print("Training Finished!")


if __name__ == "__main__":
    main()
