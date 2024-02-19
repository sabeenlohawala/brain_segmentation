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
from TissueLabeling.data.cutout import Cutout
from TissueLabeling.data.mask import Mask


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
        self.affines = sorted(glob.glob(f"{config.aug_dir}/{mode}/affine*.npy"))

        # only augment the train, not validation or test
        if self.mode == 'train':
            self.augment = config.augment
            self.aug_mask = config.aug_mask
            self.aug_cutout = config.aug_cutout
            self.aug_mask = config.aug_mask
            self.cutout_obj = Cutout(config.cutout_n_holes, config.cutout_length)
            self.mask_obj = Mask(config.mask_n_holes, config.mask_length)
        else:
            self.augment = 0
            self.aug_mask = 0
            self.aug_cutout = 0
            self.aug_mask = 0
            self.cutout_obj = None
            self.mask_obj = None

        if self.augment:
            print(f'augmenting data!')
            self.images = self.images[:] + self.images[:]
            self.masks = self.masks[:] + self.masks[:]
            self.affines = self.affines[:] + self.affines[:]

        # Limit the number of images and masks to the first 100 during debugging
        if config.debug:
            print('debug mode')
            self.images = self.images[:100]
            self.masks = self.masks[:100]
            self.affines = self.affines[:100]

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

        # randomly augment
        augment_coin_toss = random.randint(0,1)
        if self.augment and augment_coin_toss == 1:
            # apply affine
            affine = torch.from_numpy(np.load(self.affines[idx]))
            image = affine_transform(image.squeeze(),affine,mode="constant")
            mask = affine_transform(mask.squeeze(),affine,mode="constant",order=0)

            image = torch.from_numpy(image)
            mask = torch.from_numpy(mask)
        
            # random left/right flip
            flip_coin_toss = random.randint(0,1)
            if flip_coin_toss == 1:
                image = torch.flip(image,dims=(1,))
                mask = torch.flip(mask,dims=(1,))
            
            # apply cutout
            if self.aug_cutout == 1:
                image = self.cutout_obj(image)
                
            # apply mask
            if self.aug_mask == 1: # TODO: if or elif?
                image, mask = self.mask_obj(image,mask)
            
            # resize image to [1,h,w] again
            image = image.unsqueeze(dim=0)
            mask = mask.unsqueeze(dim=0)

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
