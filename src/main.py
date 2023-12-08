
import torch
import argparse
import wandb
import os

from data.dataset import get_data_loader
from utils import set_seed, init_cuda, init_fabric, init_wandb
from models.metrics import Dice
from models.segformer import Segformer
from models.unet import Unet
from training.trainer import Trainer
from tissue_labeling_config import Configuration

parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
parser.add_argument('--logdir', help="Tensorboard directory", type=str,required=False, default=os.getcwd())
parser.add_argument('--wandb_description', help="Description add to the wandb run", type=str, required=False)

parser.add_argument('--model_name', help="Name of model to use for segmentation", type=str, default='segformer')
parser.add_argument('--num_epochs', help="Number of epochs to train", type=int, required=False, default=20)
parser.add_argument('--batch_size', help="Batch size for training", type=int, required=False, default=64)
parser.add_argument('--lr', help="Learning for training", type=float, required=False, default=6e-5)

parser.add_argument('--data_dir', help="Directory of which dataset to train on", type=str, default='/om2/user/sabeen/nobrainer_data_norm/new_small_no_aug_51')
parser.add_argument('--pretrained', help="Whether to use pretrained model", type=bool, required=False, default=True)
parser.add_argument('--seed', help="Random seed value", type=int, required=False, default=42)


args = parser.parse_args()

# WANDB_ON = args.wandb_description is not None
# WANDB_RUN_DESCRIPTION = args.wandb_description
# WANDB_RUN_TITLE = "Brain Segmentation"

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

    config = Configuration(args)
    print('LOGDIR', config.logdir)

    if not os.path.exists(config.data_dir):
        raise Exception('Dataset not found')

    # model
    if config.model_name == 'segformer':
        print('Segformer model found!')
        model = Segformer(config.nr_of_classes, pretrained=config.pretrained)
    elif config.model_name == 'unet':
        print('Unet model found!')
        model = Unet(
            dim=16,
            channels=1,
            dim_mults=(2, 4, 8, 16, 32, 64),
        )
        print(model)
    else:
        print(config.model_name)
        raise Exception('Invalid model name provided')

    # # TODO: loading model from checkpoint
    
    fabric = init_fabric(precision=config.precision)#, devices=2, strategy='ddp')
    set_seed(config.seed)
    init_cuda()

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    # loss function
    loss_fn = Dice(config.nr_of_classes, fabric, config.data_dir)

    # get data loader
    # train_loader, val_loader, _ = get_data_loader(f'/om2/user/sabeen/nobrainer_data_norm/data_prepared_segmentation_{DATASET}', batch_size=BATCH_SIZE, pretrained=config.pretrained)
    train_loader, val_loader, _ = get_data_loader(config.data_dir, batch_size=config.batch_size, pretrained=config.pretrained)

    # fabric setup
    train_loader, val_loader = fabric.setup_dataloaders(train_loader,val_loader)
    model, optimizer = fabric.setup(model, optimizer)

    # model params to track with wandb
    model_params = {
        'learning rate': config.lr,
        '# epochs': config.num_epochs,
        'batch size': config.batch_size,
        'model': config.model_name,
        'data_dir': config.data_dir,
        'validation frequency': config.save_every,
        'precision': config.precision
        }

    # init WandB
    if fabric.global_rank == 0:
        init_wandb(config.wandb_on, config.wandb_run_title, fabric, model_params, config.wandb_description)
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
         logdir = config.logdir
    )
    trainer.train(config.num_epochs)
    print("Training Finished!")

if __name__ == "__main__":
    main()