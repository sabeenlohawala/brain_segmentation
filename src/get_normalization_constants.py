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
from models.metrics import Dice
from models.segformer import Segformer
from training.trainer import Trainer
from utils import load_brains, set_seed, crop, init_cuda, init_fabric, init_wandb

DATASET = dataset = 'medium'
BATCH_SIZE = 64
NR_OF_CLASSES = 6

NR_OF_CLASSES = 107 # set to 2 for binary classification
LEARNING_RATE = 3e-6
N_EPOCHS = 1
MODEL_NAME = "segformer"
SEED = 700
SAVE_EVERY = "epoch"
PRECISION = '32-true' #"16-mixed"

train_path = ''
save_path = '/om2/user/sabeen/nobrainer_data_norm/matth406_medium_6_classes/'

set_seed(SEED)
    
fabric = init_fabric(precision=PRECISION)#,devices=2,num_nodes=1,strategy='ddp') # accelerator="gpu", devices=2, num_nodes=1
init_cuda()

# model
model = Segformer(NR_OF_CLASSES)
# model = fabric.to_device(model)

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# loss function
loss_fn = Dice(NR_OF_CLASSES, fabric)

# get data loader
train_loader, val_loader, _ = get_data_loader(DATASET, batch_size=BATCH_SIZE)
model, optimizer = fabric.setup(model, optimizer)

# get normalization constants
# if dataset == "medium":
#     train_idxs = '000{000..236}.tar'
#     val_idxs = '0000{00..39}.tar'

# train_loader, val_loader, _ = get_data_loader(DATASET, batch_size=BATCH_SIZE)

# dataset_mean, dataset_std = 0, 0
# pixel_counts = {i: 0 for i in range(NR_OF_CLASSES)}
# idx = 0
# for image, mask, _ in train_loader:
#     # print(image[0,:,:,:].squeeze().shape)
#     # print(image[0].squeeze().shape)
#     # break
#     # print(idx)
#     for i in range(BATCH_SIZE):
#         brain_slice = image[i].squeeze()
#         mask_slice = mask[i].squeeze()

#         image_mean, image_std = torch.mean(brain_slice), torch.std(brain_slice)
#         dataset_mean += image_mean
#         dataset_std += image_std

#         # pixel distribution per image
#         unique, counts = np.unique(mask_slice, return_counts=True)
#         for i,j in zip(unique, counts):
#             pixel_counts[i] += j
        
#         idx += 1

# dataset_mean /= idx
# dataset_std /= idx
# np.save(f"{save_path}/normalization_constants.npy", np.array([dataset_mean, dataset_std]))
# print("Dataset mean: ", dataset_mean, "Dataset std: ", dataset_std, 'Idx:', idx)
# np.save(f"{save_path}/pixel_counts.npy", np.array(list(pixel_counts.values())))

# saving images of brain and mask slices
# for image, mask, _ in train_loader:
#     # print(image[0,:,:,:].squeeze().shape)
#     # print(image[0].squeeze().shape)
#     # break
#     # print(idx)
#     for i in range(BATCH_SIZE):
#         brain_slice = image[i].squeeze()
#         mask_slice = mask[i].squeeze()
#         colors = plt.cm.hsv(np.linspace(0, 1, 107))
#         # new plt cmap
#         cmap = ListedColormap(colors)
#         bounds=np.arange(0,107+1)
#         norm = BoundaryNorm(bounds, cmap.N)

#         fig, ax = plt.subplots()
#         ax.imshow(mask_slice, cmap=cmap, norm=norm)
#         ax.axis('off')
#         fig.canvas.draw()
#         fig.savefig('mask_slice_3.png')
#         image = Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
#         # image = wandb.Image(image, caption=caption)
#         plt.close()

#         fig, ax = plt.subplots()
#         ax.imshow(brain_slice, cmap='gray', norm=None)
#         ax.axis('off')
#         fig.canvas.draw()
#         fig.savefig('brain_slice_3.png')
#         image = Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
#         # image = wandb.Image(image, caption=caption)
#         plt.close()

#         for angle in [30,60,90]:
#             brain_slice_rot = torch.from_numpy(rotate(brain_slice,angle,reshape=False)).to(torch.float32)
#             mask_slice_rot = torch.from_numpy(rotate(mask_slice,angle,reshape=False,order=0)).to(torch.float32)
#             fig, ax = plt.subplots()
#             ax.imshow(mask_slice_rot, cmap=cmap, norm=norm)
#             ax.axis('off')
#             fig.canvas.draw()
#             fig.savefig(f'mask_slice_{angle}_3.png')
#             image = Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
#             # image = wandb.Image(image, caption=caption)
#             plt.close()

#             fig, ax = plt.subplots()
#             ax.imshow(brain_slice_rot, cmap='gray', norm=None)
#             ax.axis('off')
#             fig.canvas.draw()
#             fig.savefig(f'brain_slice_{angle}_3.png')
#             image = Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
#             # image = wandb.Image(image, caption=caption)
#             plt.close()
        
#         # save_image(brain_slice, 'brain_slice_3.png')
#         # save_image(mask_slice, 'mask_slice_3.png')

#         # image_mean, image_std = torch.mean(brain_slice), torch.std(brain_slice)
#         # dataset_mean += image_mean
#         # dataset_std += image_std

#         # # pixel distribution per image
#         # unique, counts = np.unique(mask_slice, return_counts=True)
#         # for i,j in zip(unique, counts):
#         #     pixel_counts[i] += j
        
#         # idx += 1
#         break
#     break


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