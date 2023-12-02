import torch
import os
from data.dataset import get_data_loader
from models.metrics import Dice
from models.segformer import Segformer
from training.trainer import Trainer
from lightning.fabric import Fabric, seed_everything
from torch.utils.data import Dataset, DataLoader
import webdataset as wds
import numpy as np
import glob

torch.set_float32_matmul_precision("high")

NR_OF_CLASSES = 107  # set to 2 for binary classification
BATCH_SIZE = 10
LEARNING_RATE = 1.0
N_EPOCHS = 1
DATASET = "small"
MODEL_NAME = "segformer"
SEED = 42
SAVE_EVERY = "epoch"
PRECISION = "32-true"  # "16-mixed"

class NewCustomDataset(Dataset):
    def __init__(self,file_dir):
        # save_dir = '/om2/user/sabeen/nobrainer_data_norm/data_prepared_segmentation_small/train/extracted_tensors'
        self.images = glob.glob(f'{file_dir}/brain*.npy')
        self.masks = glob.glob((f'{file_dir}/mask*.npy'))

        self.images = self.images[:100]
        self.masks = self.masks[:100]
        # self.keys = np.load(f'{file_dir}/keys.npy')
    
    def __getitem__(self,idx):
        # returns (image, mask)
        return torch.from_numpy(np.load(self.images[idx])), torch.from_numpy(np.load(self.masks[idx]))
    
    def __len__(self):
        return len(self.images)

def main():
    fabric = Fabric(devices=2, accelerator="gpu", precision=PRECISION)
    fabric.launch()
    seed_everything(SEED)

    # make environment infos available
    os.environ["RANK"] = str(fabric.global_rank)
    # local world size
    os.environ["WORLD_SIZE"] = str(torch.cuda.device_count())

    model = Segformer(NR_OF_CLASSES)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # loss function
    loss_fn = Dice(NR_OF_CLASSES, fabric) # TODO: replace with NLL

    # get data loader
    train_dataset = NewCustomDataset('/om2/user/sabeen/nobrainer_data_norm/data_prepared_segmentation_small/train/extracted_tensors')
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
    )
    train_loader = fabric.setup_dataloaders(train_loader)

    # train_loader, val_loader, _ = get_data_loader(DATASET, batch_size=BATCH_SIZE)
    # train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)
    

    model, optimizer = fabric.setup(model, optimizer)
    for epoch in range(N_EPOCHS):
        print(f"{epoch} of {N_EPOCHS}")
        print(len(list(enumerate(train_loader))))
        model.train()
        for i, (image, mask) in enumerate(train_loader):
            # image = fabric.to_device(image)
            # mask = fabric.to_device(mask)
            # mask[mask != 0] = 1 # uncomment for binary classification check
            print(i, fabric.global_rank)
            optimizer.zero_grad()
            probs = model(image)
            loss, _, _ = loss_fn(mask.long(), probs)
            print(f'loss {fabric.global_rank}:', loss.item())

            fabric.backward(loss)
            optimizer.step()
    # fabric.barrier()
    # if fabric.global_rank == 0:
    PATH = "/om2/user/sabeen/brain_segmentation/checkpoint.ckpt"
    state = {
        'epoch': epoch,
        # 'batch_idx': batch_idx,
        'model': model,
        'optimizer': optimizer,
        }
    # torch.save(model,PATH)
    fabric.save(PATH,state)
    # fabric.barrier()
    print('Training finished!')

if __name__ == "__main__":
    main()
