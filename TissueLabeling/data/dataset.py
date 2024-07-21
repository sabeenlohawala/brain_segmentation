#!/.venv/bin/python
# -*- coding: utf-8 -*-
"""
File: dataset.py
Author: Sabeen Lohawala
Date: 2024-05-11
Description: This file contains classes for two methods of reading in the KWYK dataset slices and a function
to return the specified PyTorch Dataloaders for the train, validation, and test split.
"""
import glob
import os
import sys

import h5py as h5
import random
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import affine_transform
from torchvision import transforms
import albumentations as A
from sklearn.model_selection import train_test_split
import nibabel as nib

from TissueLabeling.data.cutout import Cutout
from TissueLabeling.data.mask import Mask
from TissueLabeling.utils import center_pad_tensor
from TissueLabeling.brain_utils import (
    mapping,
    create_affine_transformation_matrix,
    draw_value_from_distribution,
    null_half,
    apply_background,
    draw_random_shapes_background,
    draw_random_grid_background,
    draw_random_noise_background,
    null_cerebellum_brain_stem
)

class HDF5Dataset(Dataset):
    """
    A class representing the KWYK dataset read/written to HDF5 files.
    """

    def __init__(self, mode:str, config):
        """
        Initializes a new HDF5Dataset for the specified mode and config.

        Args:
            mode (str): Either 'train', 'validation', or 'test' to specify which dataset.
            config (TissueLabeling.config.Configuration): contains the parameters specified at the start of this run.
        """
        if mode == 'val':
            mode = 'validation'
        if mode not in ['train','validation','test']:
            raise Exception("invalid mode, choose from train, validation, or test")
        self.mode = mode

        random.seed(42)

        h5_dir = '/om/scratch/tmp/sabeen/kwyk_chunk/' # where the hdf5 files are stored
        h5_file_paths = sorted(glob.glob(os.path.join(h5_dir, '*.h5')))
        self.h5_pointers = [h5.File(h5_path,'r') for h5_path in h5_file_paths]

        if config.background_percent_cutoff > 0:
            slice_nonbrain_dir = '/om/scratch/Fri/sabeen/kwyk_h5_nonbrains'
            slice_nonbrain_file_paths = sorted(glob.glob(os.path.join(slice_nonbrain_dir, '*nonbrain*.npy')))
            all_slice_nonbrain = [np.load(slice_nonbrain_path) for slice_nonbrain_path in slice_nonbrain_file_paths]
            all_slice_nonbrain[-1] = np.pad(all_slice_nonbrain[-1], ((0,0),(0,all_slice_nonbrain[0].shape[1] - all_slice_nonbrain[-1].shape[1]),(0,0),(0,0)),mode='constant', constant_values=65535)
            self.slice_nonbrain = np.vstack(all_slice_nonbrain) # shape = [10,1150,3,256]

            # keep track of which slices have fewer percentage of background pixels than config.background_percent_cutoff
            filter_value = config.background_percent_cutoff * 256*256 
            keep_indices = torch.nonzero(torch.from_numpy(self.slice_nonbrain.astype(np.int32)) < filter_value) # [num_slices, 4] - (shard_idx, shard_vol_idx, axis, slice_idx)
        else: # only filters out slices that have no tissue
            slice_nonbrain_dir = '/om/scratch/Fri/sabeen/kwyk_h5_matthias'
            slice_nonbrain_file_paths = sorted(glob.glob(os.path.join(slice_nonbrain_dir, '*matthias*.npy')))
            all_slice_nonbrain = [np.load(slice_nonbrain_path) for slice_nonbrain_path in slice_nonbrain_file_paths]
            all_slice_nonbrain[-1] = np.pad(all_slice_nonbrain[-1], ((0,0),(0,all_slice_nonbrain[0].shape[1] - all_slice_nonbrain[-1].shape[1]),(0,0),(0,0)),mode='constant', constant_values=0)
            self.slice_nonbrain = np.vstack(all_slice_nonbrain) # shape = [10,1150,3,256]
            keep_indices = torch.nonzero(torch.from_numpy(self.slice_nonbrain.astype(np.uint8)) != 0) # [num_slices, 4] - (shard_idx, shard_vol_idx, axis, slice_idx)

        # train-val-test-split
        if 'shard' in config.data_size:
            # train-val-test split a single shard of data
            if config.data_size != 'shard': # config.data_size = 'shard-#' where # in [0,10)
                _, shard_num = config.data_size.split('-')
                shard_num = int(shard_num)
                train_shard = shard_num % 10
                val_shard = (train_shard + 1) % 10
                test_shard = (val_shard + 1) % 10
                train_indices = list(range(train_shard * 1150,min((train_shard+1) * 1150,11480)))
                val_indices = list(range(val_shard*1150, min((val_shard + 1)*1150,11480)))
                test_indices = list(range(test_shard * 1150, min((test_shard+1) * 1150,11480)))
                random.shuffle(train_indices)
                random.shuffle(val_indices)
                random.shuffle(test_indices)
                train_indices = train_indices[:80]
                val_indices = val_indices[:10]
                test_indices = test_indices[:10]
            else: # train, val, and test split come from a different shard of data each
                train_indices = list(range(1150))
                val_indices = list(range(1150,1150*2))
                test_indices = list(range(1150*2,1150*3))
        else:
            train_indices, rem_indices = train_test_split(np.arange(0,11479),test_size = 0.2, random_state = 42)
            val_indices, test_indices = train_test_split(rem_indices,test_size = 0.5, random_state = 42)

            # keep only a subset of indices if not using the full dataset
            if config.data_size in ['small', 'med','medium']:
                end_idx = 10 if config.data_size == 'small' else 1150
                train_indices = train_indices[:int(end_idx * 0.8)]
                val_indices = val_indices[:int(end_idx * 0.1)]
                test_indices = test_indices[:int(end_idx * 0.1)]

        mode_indices = train_indices if mode == 'train' else val_indices if mode == 'validation' else test_indices

        keep_vol_indices = keep_indices[:,0] * 1150 + keep_indices[:,1] # convert to flattened indices
        self.filtered_matrix = keep_indices[np.isin(keep_vol_indices,mode_indices)]

        if config.debug:
            print("debug mode")
            self.filtered_matrix = self.filtered_matrix[:100]
        
        # store relevant config parameters
        self.nr_of_classes = config.nr_of_classes
        self.pretrained = config.pretrained
        if self.mode == "train": # only apply augmentations to training data
            self.augment = config.augment
            self.intensity_scale = config.intensity_scale
            self.aug_elastic = config.aug_elastic
            self.aug_piecewise_affine = config.aug_piecewise_affine
            self.aug_percent = config.aug_percent
            self.aug_null_half = config.aug_null_half
            self.aug_null_cerebellum_brain_stem = config.aug_null_cerebellum_brain_stem
            self.aug_background_manipulation = config.aug_background_manipulation
            self.aug_shapes_background = config.aug_shapes_background
            self.aug_grid_background = config.aug_grid_background
            self.aug_noise_background = config.aug_noise_background
            self.possible_backgrounds = []
            if self.aug_shapes_background:
                self.possible_backgrounds.append(1)
            if self.aug_grid_background:
                self.possible_backgrounds.append(2)
            if self.aug_noise_background:
                self.possible_backgrounds.append(3)
        else:
            self.augment = 0
            self.intensity_scale = 0
            self.aug_elastic = 0
            self.aug_piecewise_affine = 0
            self.aug_percent = 0
            self.aug_null_half = 0
            self.aug_null_cerebellum_brain_stem = 0
            self.aug_background_manipulation = 0
            self.aug_shapes_background = 0
            self.aug_grid_background = 0
            self.aug_noise_background = 0
            self.possible_backgrounds = set()

        self.class_mapping = None # stores the mapping from original freesurfer labels to ids to labels for nr_of_classes
        self.right_classes = None # stores the mapping for regions prefixed with 'right' or 'rh'
        self.left_classes = None # stores the mapping for regions prefixed with 'left' or 'lh'
        self.null_classes = None # stores the mapping for regions belonging in cerebellum or brain stem

        # list of albumentations augmentations that will be applied based on config
        transform_list = [
            A.Affine(rotate=(-15,15),scale=(1-0.2,1+0.2),shear=(-0.69,0.69),interpolation=2,mask_interpolation=0,always_apply=True),
            A.HorizontalFlip(p=0.5),
        ]
        if self.intensity_scale:
            transform_list.append(A.RandomBrightnessContrast(always_apply=True))
        if self.aug_elastic:
            transform_list.append(A.ElasticTransform(always_apply=True))
        if self.aug_piecewise_affine:
            transform_list.append(A.PiecewiseAffine(always_apply=True))
            
        if not config.aug_null_half and config.aug_mask:
            # masking augmentation: null out a region of both the feature image and the same region in the corresponding label mask
            transform_list.append(A.CoarseDropout(max_holes=config.mask_n_holes,
                                                  max_height=config.mask_length,
                                                  max_width=config.mask_length,
                                                  min_holes=config.mask_n_holes,
                                                  min_height=config.mask_length,
                                                  min_width=config.mask_length,
                                                  mask_fill_value=0,
                                                  always_apply=True))
        elif not config.aug_null_half and config.aug_cutout:
            # cut-out augmentation: null out a region of ONLY the feature image and NOT the corresponding label mask
            transform_list.append(A.CoarseDropout(max_holes=config.cutout_n_holes,
                                                  max_height=config.cutout_length,
                                                  max_width=config.cutout_length,
                                                  min_holes=config.cutout_n_holes,
                                                  min_height=config.cutout_length,
                                                  min_width=config.cutout_length,
                                                  mask_fill_value=None, # mask will not be affected
                                                  always_apply=True))
        self.transform = A.Compose(transform_list)

    def __getitem__(self, index):
        """
        Gets the slice at the corresponding index.

        Args:
            index (int): index of slice to get
        
        Returns:
            feature_slice (torch.tensor): the MRI slices of size [1,h,w]
            label_slice (torch.tensor): the corresponding label slice of size [1,h,w] where freesurfer labels 
                                        have been mapped to the config.nr_of_classes
        """
        shard_idx, shard_vol_idx, axis, slice_idx = self.filtered_matrix[index]
        indices = [shard_vol_idx,slice(None),slice(None)]
        indices.insert(axis+1,slice_idx)
        feature_slice = (self.h5_pointers[shard_idx][f'features_axis{axis}'][tuple(indices)]).astype(np.float32) # (256, 256)
        label_slice = (self.h5_pointers[shard_idx][f'labels_axis{axis}'][tuple(indices)]).astype(np.int16) # (256, 256)
        feature_slice[label_slice == 0] = 0 # skull stripping
        feature_slice = feature_slice / 255.0 # make intensities 0 to 1 instead of 0 to 255

        # add augmentations
        augment_coin_toss = 1 if random.random() < self.aug_percent else 0
        if self.augment and augment_coin_toss == 1:
            transformed = self.transform(image = feature_slice, mask = label_slice)
            feature_slice = transformed['image']
            label_slice = transformed['mask']
            feature_slice[label_slice == 0] = 0

            # null half of the brain and possibly cerebellum and brain stem
            null_coin_toss = 1 if random.random() < 0.5 else 0
            if self.aug_null_half and null_coin_toss:
                feature_slice, label_slice, right_classes, left_classes = null_half(image=feature_slice, mask=label_slice, keep_left=random.randint(0, 1) == 1,right_classes=self.right_classes,left_classes=self.left_classes)
                self.right_classes = right_classes
                self.left_classes = left_classes

                null_cerebellum_brain_stem_coin_toss = 1 if self.aug_null_cerebellum_brain_stem and random.random() < 0.5 else 0
                if null_cerebellum_brain_stem_coin_toss:
                    feature_slice, label_slice, null_classes = null_cerebellum_brain_stem(image=feature_slice, mask=label_slice, null_classes=self.null_classes)
                    self.null_classes = null_classes
            
            # background manipulation augmentations
            if self.aug_background_manipulation:
                apply_background_coin_toss = random.random() < 0.5
                if apply_background_coin_toss:
                    background_type = random.choice(self.possible_backgrounds)
                    if background_type == 1:
                        background = draw_random_shapes_background(feature_slice.shape)
                    elif background_type == 2:
                        background = draw_random_grid_background(label_slice.shape)
                    elif background_type == 3:
                        background = draw_random_noise_background(label_slice.shape)
                        
                    feature_slice = apply_background(feature_slice,label_slice,background)

        label_slice, class_mapping = mapping(np.array(label_slice), nr_of_classes=self.nr_of_classes, reference_col='original', class_mapping=self.class_mapping)
        self.class_mapping = class_mapping

        feature_slice = torch.from_numpy(feature_slice)
        label_slice = torch.from_numpy(label_slice)
        
        # resize image from [h,w] to [1,h,w] again
        feature_slice = feature_slice.unsqueeze(dim=0)
        label_slice = label_slice.unsqueeze(dim=0)

        if self.pretrained:
            feature_slice = feature_slice.repeat((3, 1, 1)) # pretrained segformer takes 3-channel images as input

        return (feature_slice, label_slice)

    def __len__(self):
        """
        Gets the length of the dataset.

        Returns:
            int: the number of slices in the dataset
        """
        return self.filtered_matrix.shape[0]

class NoBrainerDataset(Dataset):
    """
    A class reprsenting KWYK dataset slices stored as .npy files.
    """

    def __init__(self, mode: str, config) -> None:
        """
        Initializes a new dataset for the specified mode and config.

        Args:
            mode (str): Either 'train', 'validation', or 'test' to specify which dataset.
            config (TissueLabeling.config.Configuration): contains the parameters specified at the start of this run.
        """
        random.seed(42)

        if mode == "val":
            mode = "validation"

        if mode not in ["train", "test", "validation"]:
            raise Exception(
                f"{mode} is not a valid data mode. Choose from 'train', 'test', or 'validation'."
            )
        self.mode = mode

        # store relevant config parameters
        self.model_name = config.model_name
        self.nr_of_classes = config.nr_of_classes
        self.data_size = config.data_size
        self.pretrained = config.pretrained
        self.pad_old_data = config.pad_old_data
        self.use_norm_consts = config.use_norm_consts

        self.new_kwyk_data = config.new_kwyk_data
        if self.new_kwyk_data:
        #     background_percent_cutoff = config.background_percent_cutoff # 0.99
        #     valid_feature_filename = f"{config.data_dir}/{mode}/valid_feature_files_{int(background_percent_cutoff*100)}.json"
        #     valid_label_filename = f"{config.data_dir}/{mode}/valid_label_files_{int(background_percent_cutoff*100)}.json"
        #     if os.path.exists(valid_feature_filename) and os.path.exists(
        #         valid_label_filename
        #     ):
        #         with open(valid_feature_filename) as f:
        #             images = json.load(f)
        #         with open(valid_label_filename) as f:
        #             masks = json.load(f)
        #     else:
        #         with open(
        #             os.path.join(config.data_dir, "percent_backgrounds.json")
        #         ) as f:
        #             percent_backgrounds = json.load(f)
        #         # keep only files from current mode with percent_background < cutoff
        #         images = sorted(
        #             [
        #                 file
        #                 for file, percent_background in percent_backgrounds.items()
        #                 if percent_background < background_percent_cutoff
        #                 and mode in file
        #                 and "features" in file
        #             ]
        #         )
        #         masks = sorted(
        #             [
        #                 file
        #                 for file, percent_background in percent_backgrounds.items()
        #                 if percent_background < background_percent_cutoff
        #                 and mode in file
        #                 and "labels" in file
        #             ]
        #         )
        #         with open(valid_feature_filename, "w") as f:
        #             json.dump(images, f)
        #         with open(valid_label_filename, "w") as f:
        #             json.dump(masks, f)

        #     combined_data = list(zip(images, masks))

        #     # Shuffle the combined list using a specific seed
        #     random.seed(42)
        #     random.shuffle(combined_data)
        #     shuffled_images, shuffled_masks = zip(*combined_data)

        #     if config.data_size == "small":
        #         num_files = int(len(shuffled_images) * 0.001)
        #     elif config.data_size == "med" or config.data_size == "medium":
        #         num_files = int(len(shuffled_images) * 0.1)
        #     else:
        #         num_files = len(shuffled_images)

        #     self.images = shuffled_images[:num_files]
        #     self.masks = shuffled_masks[:num_files]

            self.images = sorted(glob.glob(f"{config.data_dir}/{mode}/features/*orig*"))
            self.masks = sorted(glob.glob(f"{config.data_dir}/{mode}/labels/*aseg*"))

            # correspond to exact same dataset size as Matthias's
            if config.data_size == "small":
                num_files = 3238 if mode == 'train' else 417 if mode == 'validation' else 388
            elif config.data_size == "med" or config.data_size == 'medium':
                num_files = 314160 if mode == 'train' else 39020 if mode == 'validation' else 38965
            else:
                num_files = len(self.images)
            self.images = self.images[:num_files]
            self.masks = self.masks[:num_files]
        else:
            # Get a list of all the brain image files in the specified directory
            self.images = sorted(glob.glob(f"{config.data_dir}/{mode}/brain*.npy"))

            # Get a list of all the mask files in the specified directory
            self.masks = sorted(glob.glob(f"{config.data_dir}/{mode}/mask*.npy"))

        # only augment the train, not validation or test
        if self.mode == "train":
            self.augment = config.augment
            self.aug_percent = config.aug_percent
            self.aug_mask = config.aug_mask
            self.aug_cutout = config.aug_cutout
            self.aug_mask = config.aug_mask
            self.aug_null_half = config.aug_null_half
            self.cutout_obj = Cutout(config.cutout_n_holes, config.cutout_length)
            self.mask_obj = Mask(config.mask_n_holes, config.mask_length)
            self.intensity_scale = (
                transforms.ColorJitter(brightness=0.2, contrast=0.2)
                if config.intensity_scale
                else None
            )
            self.aug_background_manipulation = config.aug_background_manipulation
            self.aug_shapes_background = config.aug_shapes_background
            self.aug_grid_background = config.aug_grid_background
        else:
            self.augment = 0
            self.aug_percent = 0
            self.aug_mask = 0
            self.aug_cutout = 0
            self.aug_mask = 0
            self.aug_null_half = 0
            self.cutout_obj = None
            self.mask_obj = None
            self.intensity_scale = None
            self.aug_background_manipulation = 0
            self.aug_shapes_background = 0
            self.aug_grid_background = 0

        # Limit the number of images and masks to the first 100 during debugging
        if config.debug:
            print("debug mode")
            self.images = self.images[:100]
            self.masks = self.masks[:100]

        # Load the normalization constants from the file directory
        if not self.new_kwyk_data:
            self.normalization_constants = np.load(
                os.path.join(
                    f"{config.data_dir}/{mode}", "..", "normalization_constants.npy"
                )
            )

        if os.path.exists(f"{config.data_dir}/{mode}/keys.npy"):
            self.keys = np.load(f"{config.data_dir}/{mode}/keys.npy")
        
        self.class_mapping = None # stores the mapping from original freesurfer labels to ids to labels for nr_of_classes
        self.right_classes = None # stores the mapping for regions prefixed with 'right' or 'rh'
        self.left_classes = None # stores the mapping for regions prefixed with 'left' or 'lh'

        # list of albumentations augmentations to apply
        transform_list = [
            A.Affine(rotate=(-15,15),scale=(1-0.2,1+0.2),shear=(-0.69,0.69),interpolation=2,mask_interpolation=0,always_apply=True),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(always_apply=True),
            A.ElasticTransform(always_apply=True),
        ]
        if not config.aug_null_half and config.aug_mask:
            # masking augmentation: null out region of feature image AND corresponding label mask
            transform_list.append(A.CoarseDropout(max_holes=config.mask_n_holes,
                                                  max_height=config.mask_length,
                                                  max_width=config.mask_length,
                                                  min_holes=config.mask_n_holes,
                                                  min_height=config.mask_length,
                                                  min_width=config.mask_length,
                                                  mask_fill_value=0,
                                                  always_apply=True))
        elif not config.aug_null_half and config.aug_cutout:
            # cutout augmentation: null out region of feature image ONLY but NOT corresponding label mask
            transform_list.append(A.CoarseDropout(max_holes=config.cutout_n_holes,
                                                  max_height=config.cutout_length,
                                                  max_width=config.cutout_length,
                                                  min_holes=config.cutout_n_holes,
                                                  min_height=config.cutout_length,
                                                  min_width=config.cutout_length,
                                                  mask_fill_value=None, # mask will not be affected
                                                  always_apply=True))
        self.transform = A.Compose(transform_list)

        if self.pad_old_data:
            print('will pad')
        else:
            print('will NOT pad')


    def __getitem__(self, idx):
        """
        Gets the slice at the corresponding index.

        Args:
            index (int): index of slice to get
        
        Returns:
            image (torch.tensor): the MRI slice at index of size [1,h,w]
            mask (torch.tensor): the corresponding label slice of size [1,h,w] where freesurfer labels 
                                        have been mapped to the config.nr_of_classes
        """
        image = torch.from_numpy(np.load(self.images[idx]).astype(np.float32))
        mask = torch.from_numpy(np.load(self.masks[idx]).astype(np.int16))

        if not self.new_kwyk_data:
            if not self.use_norm_consts:
                image = image / 255.0
            else:
                # normalize image
                image = (
                    image - self.normalization_constants[0]
                ) / self.normalization_constants[1]

        # randomly augment
        augment_coin_toss = 1 if random.random() < self.aug_percent else 0
        if self.augment and augment_coin_toss == 1:

            # resize from [1,h,w] to [h,w]
            image = np.array(image.squeeze(0))
            mask = np.array(mask.squeeze(0))

            transformed = self.transform(image = image.astype(np.float32), mask = mask)
            image = transformed['image']
            mask = transformed['mask']
            image[mask == 0] = 0
            
            if self.aug_background_manipulation:
                apply_background_coin_toss = random.random() < 0.5
                if apply_background_coin_toss:
                    shapes_background_coin_toss = random.random() < 0.5 if self.aug_grid_background == self.aug_shapes_background else self.aug_shapes_background

                    if shapes_background_coin_toss:
                        background = draw_random_shapes_background(image.shape)
                    else:
                        background = draw_random_grid_background(image.shape)
                        
                    image = apply_background(image,mask,background)

            # null half of the brain
            null_coin_toss = 1 if random.random() < 0.5 else 0
            if self.aug_null_half and null_coin_toss:
                image, mask, right_classes, left_classes = null_half(image=image, mask=mask, keep_left=random.randint(0, 1) == 1,right_classes=self.right_classes,left_classes=self.left_classes)
                self.right_classes = right_classes
                self.left_classes = left_classes

            # resize image from [h,w] to [1,h,w]
            image = torch.from_numpy(image)
            mask = torch.from_numpy(mask)
            image = image.unsqueeze(dim=0)
            mask = mask.unsqueeze(dim=0)

        # make image dimensions compatible for unet models if not using new 256x256 dataset
        if not self.new_kwyk_data and "unet" in self.model_name:
            image = image[:, 1:161, 1:193]
            mask = mask[:, 1:161, 1:193]

        if not self.new_kwyk_data: # these may work anymore
            if self.nr_of_classes == 2:
                mask[mask != 0] = 1
            elif self.nr_of_classes == 6 or self.nr_of_classes == 16:
                mask, class_mapping = mapping(mask, nr_of_classes=self.nr_of_classes, reference_col="50-class", class_mapping=self.class_mapping)
                self.class_mapping = class_mapping
            elif self.nr_of_classes == 50:
                mask[mask > 49] = 0

            if self.pad_old_data:
                # center pad old data (which was cropped to 162x194) to be 256x256
                image = center_pad_tensor(image, 256, 256)
                mask = center_pad_tensor(mask, 256, 256)

        if self.new_kwyk_data:
            image = image.to(torch.float32)
            mask, class_mapping = mapping(np.array(mask), nr_of_classes=self.nr_of_classes, reference_col='original', class_mapping=self.class_mapping)
            mask = torch.from_numpy(mask)
            self.class_mapping = class_mapping

        if self.pretrained:
            image = image.repeat((3, 1, 1)) # pretrained segformer requires 3-channel image data
        return image, mask

    def __len__(self):
        """
        Gets the length of the dataset.

        Returns:
            int: the number of slices in the dataset
        """
        return len(self.images)
    

def get_data_loader(
    # data_dir: str,
    config,
    num_workers: int = 4 * torch.cuda.device_count(),
):
    """
    Returns the PyTorch dataloaders for the datasets created based on the parameters specified in config.

    Args:
        config (TissueLabeling.config.Configuration): contains the parameters specified at the start of this run.
    
    Returns:
        train_loader (torch.utils.data.DataLoader): the PyTorch Dataloader for the training split of data
        val_loader (torch.utils.data.DataLoader): the PyTorch Dataloader for the validation split of data
        test_loader (torch.utils.data.DataLoader): the PyTorch Dataloader for the test split of data
        tuple (int, int): the height and width of the images in the datasets
    """

    # whether to use the new dataset (256x256 slices) or old dataset created by Matthias (162x194 slices)
    if config.new_kwyk_data != 0:
        train_dataset = HDF5Dataset(mode='train',config=config)
        val_dataset = HDF5Dataset(mode='validation',config=config)
        test_dataset = HDF5Dataset(mode='test',config=config)
    else:
        train_dataset = NoBrainerDataset("train", config)
        val_dataset = NoBrainerDataset("validation", config)
        test_dataset = NoBrainerDataset("test", config)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.batch_size
    )

    return train_loader, val_loader, test_loader, tuple(train_dataset[0][0].shape[1:])
