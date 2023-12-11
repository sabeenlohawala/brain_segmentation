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

import numpy as np
import torch
from torch.utils.data import Dataset


class NoBrainerDataset(Dataset):
    def __init__(self, file_dir: str, config) -> None:
        """
        Initializes the object with the given `file_dir` and `pretrained` parameters.

        Args:
            file_dir: The directory where the files are located.
            pretrained: Whether or not to use pretrained models.

        Returns:
            None
        """
        # Set the model name
        self.model_name = config.model_name

        # Set the pretrained attribute
        self.pretrained = config.pretrained

        # Get a list of all the brain image files in the specified directory
        self.images = glob.glob(f"{file_dir}/brain*.npy")

        # Get a list of all the mask files in the specified directory
        self.masks = glob.glob(f"{file_dir}/mask*.npy")

        # Limit the number of images and masks to the first 100 during debugging
        if config.debug:
            print('debug mode')
            self.images = self.images[:100]
            self.masks = self.masks[:100]

        # Load the normalization constants from the file directory
        self.normalization_constants = np.load(
            os.path.join(file_dir,'..','normalization_constants.npy')
        )

        if os.path.exists(f"{file_dir}/keys.npy"):
            self.keys = np.load(f"{file_dir}/keys.npy")

    def __getitem__(self, idx):
        # returns (image, mask)
        image = torch.from_numpy(np.load(self.images[idx]))
        mask = torch.from_numpy(np.load(self.masks[idx]))
        if self.model_name == 'simple_unet':
            image = image[:,1:161,1:193]
            mask = mask[:,1:161,1:193]

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
    train_dataset = NoBrainerDataset(f"{config.data_dir}/train", config)
    val_dataset = NoBrainerDataset(f"{config.data_dir}/validation", config)
    test_dataset = NoBrainerDataset(f"{config.data_dir}/test", config)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size)

    return (train_loader, val_loader, test_loader)
