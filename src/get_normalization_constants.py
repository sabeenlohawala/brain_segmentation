import os
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset
import torch
import webdataset as wds
from typing import Tuple
import matplotlib.pyplot as plt
import tarfile
import csv
import lightning as L

from data.dataset import get_data_loader
from models.segformer import Segformer
from training.trainer import Trainer
from utils import load_brains, set_seed, crop, init_cuda, init_fabric, init_wandb

DATASET = dataset = 'medium'
BATCH_SIZE = 64
NR_OF_CLASSES = 6

train_path = ''
save_path = '/om2/user/sabeen/nobrainer_data_norm/matth406_medium_6_classes/'


if dataset == "medium":
    train_idxs = '000{000..236}.tar'
    val_idxs = '0000{00..39}.tar'

train_loader, val_loader, _ = get_data_loader(DATASET, batch_size=BATCH_SIZE)

dataset_mean, dataset_std = 0, 0
pixel_counts = {i: 0 for i in range(NR_OF_CLASSES)}
idx = 0
for image, mask, _ in train_loader:
    # print(image[0,:,:,:].squeeze().shape)
    # print(image[0].squeeze().shape)
    # break
    # print(idx)
    for i in range(BATCH_SIZE):
        brain_slice = image[i].squeeze()
        mask_slice = mask[i].squeeze()

        image_mean, image_std = torch.mean(brain_slice), torch.std(brain_slice)
        dataset_mean += image_mean
        dataset_std += image_std

        # pixel distribution per image
        unique, counts = np.unique(mask_slice, return_counts=True)
        for i,j in zip(unique, counts):
            pixel_counts[i] += j
        
        idx += 1

dataset_mean /= idx
dataset_std /= idx
np.save(f"{save_path}/normalization_constants.npy", np.array([dataset_mean, dataset_std]))
print("Dataset mean: ", dataset_mean, "Dataset std: ", dataset_std, 'Idx:', idx)
np.save(f"{save_path}/pixel_counts.npy", np.array(list(pixel_counts.values())))


# trying to load model from checkpoint
# PRECISION = '32-true' #"16-mixed"
# NR_OF_CLASSES = 107 # set to 2 for binary classification
# LEARNING_RATE = 3e-6

# checkpoint_path = "/home/sabeen/brain_segmentation/models/checkpoint.ckpt"
# fabric = init_fabric(precision=PRECISION)

# model = Segformer(NR_OF_CLASSES)
# optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# full_checkpoint = fabric.load(checkpoint_path)
# model.load_state_dict(full_checkpoint["model"])
# optimizer.load_state_dict(full_checkpoint["optimizer"])

# print(full_checkpoint.keys())