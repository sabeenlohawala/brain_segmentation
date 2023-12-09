#!/.venv/bin/python
# -*- coding: utf-8 -*-
"""
@Author: Matthias Steiner
@Contact: matth406@mit.edu
@File: dataset.py
@Date: 2023/04/03
"""

import glob

import numpy as np
import torch
from torch.utils.data import Dataset


class NoBrainerDataset(Dataset):
    def __init__(self, file_dir: str, pretrained: bool = True) -> None:
        """
        Initializes the object with the given `file_dir` and `pretrained` parameters.

        Args:
            file_dir: The directory where the files are located.
            pretrained: Whether or not to use pretrained models.

        Returns:
            None
        """
        # Set the pretrained attribute
        self.pretrained = pretrained

        # Get a list of all the brain image files in the specified directory
        self.images = glob.glob(f"{file_dir}/brain*.npy")

        # Get a list of all the mask files in the specified directory
        self.masks = glob.glob(f"{file_dir}/mask*.npy")

        # Limit the number of images and masks to the first 100
        self.images = self.images[:100]
        self.masks = self.masks[:100]

        # Load the normalization constants from the file directory
        self.normalization_constants = np.load(
            f"{file_dir}/../normalization_constants.npy"
        )

        # Uncomment the following line if the 'keys.npy' file is present in the file directory
        # self.keys = np.load(f"{file_dir}/keys.npy")

    def __getitem__(self, idx):
        # returns (image, mask)
        image = torch.from_numpy(np.load(self.images[idx]))
        mask = torch.from_numpy(np.load(self.masks[idx]))

        # normalize image
        image = (
            image - self.normalization_constants[0]
        ) / self.normalization_constants[1]

        if self.pretrained:
            return image.repeat((3, 1, 1)), mask
        return image, mask

    def __len__(self):
        return len(self.images)

    # def normalize(self,sample):
    #     image = sample[0]
    #     image = (image - self.normalization_constants[0]) / self.normalization_constants[1]
    #     sample = (image, sample[1], sample[2])
    #     return sample
