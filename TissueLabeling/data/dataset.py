#!/.venv/bin/python
# -*- coding: utf-8 -*-
"""
@Author: Matthias Steiner
@Contact: matth406@mit.edu
@File: dataset.py
@Date: 2023/04/03
"""
import glob
import os

import random
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import affine_transform

from TissueLabeling.brain_utils import mapping


class NoBrainerDataset(Dataset):
    def __init__(self, mode: str, config) -> None:
        """
        Initializes the object with the given `file_dir` and `pretrained` parameters.

        Args:
            mode: The subdirectory where the files are located.
            pretrained: Whether or not to use pretrained models.

        Returns:
            None
        """
        if mode == 'val':
            mode = 'validation'

        if mode not in ['train','test','validation']:
            raise Exception(f"{mode} is not a valid data mode. Choose from 'train', 'test', or 'validation'.")
        self.mode = mode

        # Set the model name
        self.model_name = config.model_name

        self.nr_of_classes = config.nr_of_classes

        # Set the pretrained attribute
        self.pretrained = config.pretrained

        # Get a list of all the brain image files in the specified directory
        self.images = sorted(glob.glob(f"{config.data_dir}/{mode}/brain*.npy"))

        # Get a list of all the mask files in the specified directory
        self.masks = sorted(glob.glob(f"{config.data_dir}/{mode}/mask*.npy"))

        # Get a list of all the affine matrices for rigid transformations
        self.rotate_affines = sorted(glob.glob(f"{config.aug_rotate_dir}/{mode}/affine*.npy"))
        self.zoom_affines = sorted(glob.glob(f"{config.aug_zoom_dir}/{mode}/affine*.npy"))

        # only augment the train, not validation or test
        if self.mode == 'train':
            self.augment = config.augment
            self.aug_rotate = config.aug_rotate
            self.aug_zoom = config.aug_zoom
            self.aug_null = config.aug_null
            self.aug_flip = config.aug_flip
        else:
            self.augment = 0
            self.aug_rotate = 0
            self.aug_zoom = 0
            self.aug_null = 0
            self.aug_flip = 0

        if self.augment:
            print(f'augmenting data!')
            self.images = self.images + self.images
            self.masks = self.masks + self.masks
            self.rotate_affines = self.rotate_affines + self.rotate_affines
            self.zoom_affines = self.zoom_affines + self.zoom_affines

        # Limit the number of images and masks to the first 100 during debugging
        if config.debug:
            print('debug mode')
            self.images = self.images[:100]
            self.masks = self.masks[:100]
            self.rotate_affines = self.rotate_affines[:100]
            self.zoom_affines = self.zoom_affines[:100]

        # Load the normalization constants from the file directory
        self.normalization_constants = np.load(
            os.path.join(f'{config.data_dir}/{mode}','..','normalization_constants.npy')
        )

        if os.path.exists(f"{config.data_dir}/{mode}/keys.npy"):
            self.keys = np.load(f"{config.data_dir}/{mode}/keys.npy")

    def __getitem__(self, idx):
        # returns (image, mask)
        image = torch.from_numpy(np.load(self.images[idx]))
        mask = torch.from_numpy(np.load(self.masks[idx]))

        # randomly rotate
        rotate_coin_toss = random.randint(0,1)
        if self.aug_rotate and rotate_coin_toss == 1:
            affine = torch.from_numpy(np.load(self.rotate_affines[idx]))
            image = affine_transform(image.squeeze(),affine,mode="constant")
            mask = affine_transform(mask.squeeze(),affine,mode="constant",order=0)
            image = torch.from_numpy(image).unsqueeze(dim=0)
            mask = torch.from_numpy(mask).unsqueeze(dim=0)

        # randomly zoom
        zoom_coin_toss = random.randint(0,1)
        if self.aug_zoom and zoom_coin_toss == 1:
            affine = torch.from_numpy(np.load(self.zoom_affines[idx]))
            image = affine_transform(image.squeeze(),affine,mode="constant")
            mask = affine_transform(mask.squeeze(),affine,mode="constant",order=0)
            image = torch.from_numpy(image).unsqueeze(dim=0)
            mask = torch.from_numpy(mask).unsqueeze(dim=0)
        
        # random null
        null_coin_toss = random.randint(0,1)
        null_type = random.randint(1,2) if self.aug_null == 3 else self.aug_null
        if null_type == 1 and null_coin_toss == 1:
            # left/right null
            mid = image.shape[1] // 2
            if random.randint(0,1) == 1:
                image[:,:,:mid] = 0
                mask[:,:,:mid] = 0
            else:
                image[:,:,mid:] = 0
                mask[:,:,mid:] = 0
        elif null_type == 2 and null_coin_toss == 1:
            # up/down null
            mid = image.shape[2] // 2
            if random.randint(0,1) == 1:
                image[:,:mid,:] = 0
                mask[:,:mid,:] = 0
            else:
                image[mid:,:] = 0
                mask[mid:,:] = 0
        elif null_type == 4 and null_coin_toss == 1:
            # random pixel null
            random_mask = torch.rand(image.size()) < 0.5
            image[random_mask] = 0
            mask[random_mask] = 0
        
        # random flip
        flip_coin_toss = random.randint(0,1)
        flip_type = random.randint(1,2) if self.aug_flip == 3 else self.aug_flip # choose left/right or up/down
        if flip_type != 0 and flip_coin_toss == 1:
            image = torch.flip(image,dims=(flip_type,))

        if self.model_name == 'simple_unet':
            image = image[:,1:161,1:193]
            mask = mask[:,1:161,1:193]

        if self.nr_of_classes == 2:
            mask[mask != 0] = 1
        elif self.nr_of_classes == 7:
            mask = mapping(mask,self.nr_of_classes,original=False)
            
        # normalize image
        image = (
            image - self.normalization_constants[0]
        ) / self.normalization_constants[1]

        if self.pretrained:
            return image.repeat((3, 1, 1)), mask
        return image, mask

    def __len__(self):
        return len(self.images)

    def normalize(self, sample):
        image = sample[0]
        image = (
            image - self.normalization_constants[0]
        ) / self.normalization_constants[1]
        sample = (image, sample[1], sample[2])
        return sample


def get_data_loader(
    # data_dir: str,
    config,
    num_workers: int = 4 * torch.cuda.device_count(),
):
    # train_dataset = NoBrainerDataset(f"{config.data_dir}/train", config)
    # val_dataset = NoBrainerDataset(f"{config.data_dir}/validation", config)
    # test_dataset = NoBrainerDataset(f"{config.data_dir}/test", config)
    train_dataset = NoBrainerDataset("train", config)
    val_dataset = NoBrainerDataset("validation", config)
    test_dataset = NoBrainerDataset("test", config)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size,shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size)

    return (train_loader, val_loader, test_loader)
