
import torch
import argparse
import wandb

from data.dataset import get_data_loader
from utils import set_seed, init_cuda, init_fabric, init_wandb
from models.metrics import Dice
from models.segformer import Segformer
from training.trainer import Trainer
from training.trainer import auto_train

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
parser.add_argument('--dataset', help="Which dataset to train on", type=str, default='small')
parser.add_argument('--pretrained', help="Whether to use pretrained model", type=bool, required=False, default=True)

args = parser.parse_args()

WANDB_ON = args.wandb_description is not None
WANDB_RUN_DESCRIPTION = args.wandb_description
WANDB_RUN_TITLE = "Brain Segmentation"

NR_OF_CLASSES = 107 # set to 2 for binary classification
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate # 3e-6
N_EPOCHS = args.num_epochs
DATASET = args.dataset
MODEL_NAME = args.model_name
SEED = args.seed
SAVE_EVERY = "epoch"
PRECISION = '32-true' #"16-mixed"
PRETRAINED = args.pretrained

@auto_train
def my_train(model, nr_of_classes, train_loader, val_loader, loss_fn, optimizer, fabric, batch_size, wandb_on, pretrained, n_epochs):
    trainer = Trainer(
        model=model,
        nr_of_classes=nr_of_classes,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        fabric=fabric,
        batch_size=batch_size,
        wandb_on=wandb_on,
        pretrained=pretrained
)
    trainer.train(n_epochs)

def main():

    if DATASET not in ['small', 'medium']:
        raise Exception('Invalid dataset provided')
    else:
        print('Dataset found!')

    # model
    if MODEL_NAME == 'segformer':
        print('Model found!')
        model = Segformer(NR_OF_CLASSES, pretrained=PRETRAINED)
    else:
        raise Exception('Invalid model name provided')

    # TODO: loading model from checkpoint
    
    fabric = init_fabric(precision=PRECISION, devices=2, strategy='ddp')
    set_seed(SEED) # TODO: replace with seed_everything(SEED)?
    init_cuda()

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # loss function
    loss_fn = Dice(NR_OF_CLASSES, fabric)

    # get data loader
    train_loader, val_loader, _ = get_data_loader(f'/om2/user/sabeen/nobrainer_data_norm/data_prepared_segmentation_{DATASET}', batch_size=BATCH_SIZE, pretrained=PRETRAINED)

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
        init_wandb(WANDB_ON, WANDB_RUN_TITLE, fabric, model_params, WANDB_RUN_DESCRIPTION)
        save_frequency = len(train_loader) if SAVE_EVERY == "epoch" else 1000
        if WANDB_ON:
            wandb.watch(model, log_freq=save_frequency)

    my_train(
         model=model,
         nr_of_classes=NR_OF_CLASSES,
         train_loader=train_loader,
         val_loader=val_loader,
         loss_fn=loss_fn,
         optimizer=optimizer,
         fabric=fabric,
         batch_size=BATCH_SIZE,
         wandb_on=WANDB_ON,
         pretrained=PRETRAINED,
         n_epochs=N_EPOCHS
    )
    print("Training Finished!")

if __name__ == "__main__":
    main()