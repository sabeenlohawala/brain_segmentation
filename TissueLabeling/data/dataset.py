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
import sys

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
from TissueLabeling.brain_utils import (
    mapping,
    create_affine_transformation_matrix,
    draw_value_from_distribution,
    null_half,
    apply_background,
    draw_random_shapes_background,
    draw_random_grid_background,
)

class KWYKVolumeDataset(torch.utils.data.Dataset):
    def __init__(self, mode, config, volume_data_dir, slice_info_file, bg_percent=0.99):
        self.mode = mode
        self.matrix = np.load(slice_info_file, allow_pickle=True)
        self.matrix = torch.Tensor(self.matrix)

        self.feature_label_files = list(zip(
            sorted(glob.glob(os.path.join(volume_data_dir, "*orig*.nii.gz")))[:self.matrix.shape[0]],
            sorted(glob.glob(os.path.join(volume_data_dir, "*aseg*.nii.gz")))[:self.matrix.shape[0]]
        ))

        # perform train-val-test split and apply to self.matrix and self.feature_label_files
        indices = np.arange(self.matrix.shape[0])
        train_indices, remaining_indices = train_test_split(indices, test_size=0.2, random_state=42)
        validation_indices, test_indices = train_test_split(remaining_indices, test_size=0.5, random_state=42)
        if mode == 'train':
            keep_indices = train_indices
        elif mode == 'val' or mode == 'validation':
            keep_indices = validation_indices
        elif mode == 'test':
            keep_indices = test_indices
        else:
            raise Exception(
                f"{mode} is not a valid data mode. Choose from 'train', 'test', or 'validation'."
            )
        self.matrix = self.matrix[keep_indices]
        self.feature_label_files = [feature_label_pair for i, feature_label_pair in enumerate(self.feature_label_files) if i in keep_indices]

        self.filtered_matrix = self.matrix < bg_percent
        self.nonzero_indices = torch.nonzero(
            self.filtered_matrix
        )  # [num_slices, 3] - (file_idx, direction_idx, slice_idx)

        assert self.nonzero_indices.shape[0] <= torch.numel(
            self.matrix
        ), "select bg slice count cannot be more than total slice count"

        # Select dataset size
        if config.data_size == "small":
            num_files = int(self.nonzero_indices.shape[0] * 0.001)
        elif config.data_size == "med" or config.data_size == "medium":
            num_files = int(self.nonzero_indices.shape[0] * 0.1)
        else:
            num_files = self.nonzero_indices.shape[0]
        
        # Limit the number of images and masks to the first 100 during debugging
        if config.debug:
            print("debug mode")
            num_files = min(num_files,40)
        self.nonzero_indices = self.nonzero_indices[:num_files,:]

        self.nr_of_classes = config.nr_of_classes
        self.pretrained = config.pretrained
        if self.mode == "train":
            self.augment = config.augment
            self.aug_percent = config.aug_percent
            # self.aug_mask = config.aug_mask
            # self.aug_cutout = config.aug_cutout
            self.aug_null_half = config.aug_null_half
            # self.cutout_obj = Cutout(config.cutout_n_holes, config.cutout_length)
            # self.mask_obj = Mask(config.mask_n_holes, config.mask_length)
            # self.intensity_scale = (
            #     transforms.ColorJitter(brightness=0.2, contrast=0.2)
            #     if config.intensity_scale
            #     else None
            # )
            self.aug_background_manipulation = config.aug_background_manipulation
            self.aug_shapes_background = config.aug_shapes_background
            self.aug_grid_background = config.aug_grid_background
        else:
            self.augment = 0
            self.aug_percent = 0
            # self.aug_mask = 0
            # self.aug_cutout = 0
            self.aug_null_half = 0
            # self.cutout_obj = None
            # self.mask_obj = None
            # self.intensity_scale = None
            self.aug_background_manipulation = 0
            self.aug_shapes_background = 0
            self.aug_grid_background = 0

        self.class_mapping = None
        transform_list = [
            A.Affine(rotate=(-15,15),scale=(1-0.2,1+0.2),shear=(-0.69,0.69),interpolation=2,mask_interpolation=0,always_apply=True),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(always_apply=True),
            A.ElasticTransform(always_apply=True),
        ]
        if not config.aug_null_half and config.aug_mask:
            print('adding mask!')
            transform_list.append(A.CoarseDropout(max_holes=config.mask_n_holes,
                                                  max_height=config.mask_length,
                                                  max_width=config.mask_length,
                                                  min_holes=config.mask_n_holes,
                                                  min_height=config.mask_length,
                                                  min_width=config.mask_length,
                                                  mask_fill_value=0,
                                                  always_apply=True))
        elif not config.aug_null_half and config.aug_cutout:
            print('adding cutout!')
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
        file_idx, direction_idx, slice_idx = self.nonzero_indices[index]

        feature_file, label_file = self.feature_label_files[file_idx]

        feature_vol = torch.from_numpy(nib.load(feature_file).get_fdata().astype(np.float32))
        label_vol = torch.from_numpy(nib.load(label_file).get_fdata().astype(np.int16))

        # do you want a cropped slice (sure do it here, of course you still need that fixed number)

        # all the above 3 if conditions can be written in a single line:
        feature_slice = torch.index_select(
            feature_vol, direction_idx, torch.Tensor(slice_idx)
        ).squeeze()
        label_slice = torch.index_select(
            label_vol, direction_idx, torch.Tensor(slice_idx)
        ).squeeze()

        feature_slice = feature_slice / 255.0

        # TODO:
        # i now wonder if rotating as well can be dont at get time...probably not advisable unless you have torch functions to do the same
        # (if you care) see nitorch (written by yael) or neurite (written by adrian)

        augment_coin_toss = 1 if random.random() < self.aug_percent else 0
        if self.augment and augment_coin_toss == 1:
            print('augmenting!')
            feature_slice = np.array(feature_slice)
            label_slice = np.array(label_slice)

            transformed = self.transform(image = feature_slice, mask = label_slice)
            feature_slice = transformed['image']
            label_slice = transformed['mask']

            # null half
            null_coin_toss = 1 if random.random() < 0.5 else 0
            if self.aug_null_half and null_coin_toss:
                print('nulling half!')
                feature_slice, label_slice = null_half(feature_slice, label_slice, random.randint(0, 1) == 1)
            
            if self.aug_background_manipulation:
                apply_background_coin_toss = random.random() < 0.5
                if apply_background_coin_toss:
                    shapes_background_coin_toss = random.random() < 0.5 if self.aug_grid_background == self.aug_shapes_background else self.aug_shapes_background
                    if shapes_background_coin_toss:
                        print('shapes background')
                        background = draw_random_shapes_background(feature_slice.shape)
                    else:
                        print('grid background')
                        background = draw_random_grid_background(label_slice.shape)
                        
                    feature_slice = apply_background(feature_slice,label_slice,background)

            feature_slice = torch.from_numpy(feature_slice)
            label_slice = torch.from_numpy(label_slice)
        
        # resize image to [1,h,w] again
        feature_slice = feature_slice.unsqueeze(dim=0)
        label_slice = label_slice.unsqueeze(dim=0)

        label_slice, class_mapping = mapping(np.array(label_slice), nr_of_classes=self.nr_of_classes, reference_col='original', class_mapping=self.class_mapping)
        self.class_mapping = class_mapping
        if self.pretrained:
            feature_slice = feature_slice.repeat((3, 1, 1))

        return (feature_slice, torch.from_numpy(label_slice))

    def __len__(self):
        return self.nonzero_indices.shape[0]

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
        random.seed(42)

        if mode == "val":
            mode = "validation"

        if mode not in ["train", "test", "validation"]:
            raise Exception(
                f"{mode} is not a valid data mode. Choose from 'train', 'test', or 'validation'."
            )
        self.mode = mode

        # Set the model name
        self.model_name = config.model_name

        self.nr_of_classes = config.nr_of_classes
        self.data_size = config.data_size

        # Set the pretrained attribute
        self.pretrained = config.pretrained

        self.new_kwyk_data = config.new_kwyk_data
        if self.new_kwyk_data:
            background_percent_cutoff = config.background_percent_cutoff # 0.99
            valid_feature_filename = f"{config.data_dir}/{mode}/valid_feature_files_{int(background_percent_cutoff*100)}.json"
            valid_label_filename = f"{config.data_dir}/{mode}/valid_label_files_{int(background_percent_cutoff*100)}.json"
            if os.path.exists(valid_feature_filename) and os.path.exists(
                valid_label_filename
            ):
                with open(valid_feature_filename) as f:
                    images = json.load(f)
                with open(valid_label_filename) as f:
                    masks = json.load(f)
            else:
                with open(
                    os.path.join(config.data_dir, "percent_backgrounds.json")
                ) as f:
                    percent_backgrounds = json.load(f)
                # keep only files from current mode with percent_background < cutoff
                images = sorted(
                    [
                        file
                        for file, percent_background in percent_backgrounds.items()
                        if percent_background < background_percent_cutoff
                        and mode in file
                        and "features" in file
                    ]
                )
                masks = sorted(
                    [
                        file
                        for file, percent_background in percent_backgrounds.items()
                        if percent_background < background_percent_cutoff
                        and mode in file
                        and "labels" in file
                    ]
                )
                with open(valid_feature_filename, "w") as f:
                    json.dump(images, f)
                with open(valid_label_filename, "w") as f:
                    json.dump(masks, f)

            combined_data = list(zip(images, masks))

            # Shuffle the combined list using a specific seed
            random.seed(42)
            random.shuffle(combined_data)
            shuffled_images, shuffled_masks = zip(*combined_data)

            if config.data_size == "small":
                num_files = int(len(shuffled_images) * 0.001)
            elif config.data_size == "med" or config.data_size == "medium":
                num_files = int(len(shuffled_images) * 0.1)
            else:
                num_files = len(shuffled_images)

            self.images = shuffled_images[:num_files]
            self.masks = shuffled_masks[:num_files]
            self.affines = []
        else:
            # Get a list of all the brain image files in the specified directory
            self.images = sorted(glob.glob(f"{config.data_dir}/{mode}/brain*.npy"))

            # Get a list of all the mask files in the specified directory
            self.masks = sorted(glob.glob(f"{config.data_dir}/{mode}/mask*.npy"))

            # Get a list of all the affine matrices for rigid transformations
            self.affines = sorted(glob.glob(f"{config.aug_dir}/{mode}/affine*.npy"))

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
            self.affines = self.affines[:100]

        # Load the normalization constants from the file directory
        if not self.new_kwyk_data:
            self.normalization_constants = np.load(
                os.path.join(
                    f"{config.data_dir}/{mode}", "..", "normalization_constants.npy"
                )
            )

        if os.path.exists(f"{config.data_dir}/{mode}/keys.npy"):
            self.keys = np.load(f"{config.data_dir}/{mode}/keys.npy")
        
        self.class_mapping = None
        transform_list = [
            A.Affine(rotate=(-15,15),scale=(1-0.2,1+0.2),shear=(-0.69,0.69),interpolation=2,mask_interpolation=0,always_apply=True),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(always_apply=True),
            A.ElasticTransform(always_apply=True),
        ]
        if not config.aug_null_half and config.aug_mask:
            transform_list.append(A.CoarseDropout(max_holes=config.mask_n_holes,
                                                  max_height=config.mask_length,
                                                  max_width=config.mask_length,
                                                  min_holes=config.mask_n_holes,
                                                  min_height=config.mask_length,
                                                  min_width=config.mask_length,
                                                  mask_fill_value=0,
                                                  always_apply=True))
        elif not config.aug_null_half and config.aug_cutout:
            transform_list.append(A.CoarseDropout(max_holes=config.cutout_n_holes,
                                                  max_height=config.cutout_length,
                                                  max_width=config.cutout_length,
                                                  min_holes=config.cutout_n_holes,
                                                  min_height=config.cutout_length,
                                                  min_width=config.cutout_length,
                                                  mask_fill_value=None, # mask will not be affected
                                                  always_apply=True))
        self.transform = A.Compose(transform_list)


    def __getitem__(self, idx):
        # returns (image, mask)
        image = torch.from_numpy(np.load(self.images[idx]).astype(np.float32))
        mask = torch.from_numpy(np.load(self.masks[idx]).astype(np.int16))

        # randomly augment
        augment_coin_toss = 1 if random.random() < self.aug_percent else 0
        if self.augment and augment_coin_toss == 1:
            # apply affine
            # if self.new_kwyk_data:
            #     affine = torch.from_numpy(self._get_affine_matrix(image))
            # else:
            #     affine = torch.from_numpy(np.load(self.affines[idx]))
            # image = affine_transform(image.squeeze(), affine, mode="constant")
            # mask = affine_transform(mask.squeeze(), affine, mode="constant", order=0)

            # image = torch.from_numpy(image)
            # mask = torch.from_numpy(mask)

            # # random left/right flip
            # flip_coin_toss = random.randint(0, 1)
            # if flip_coin_toss == 1:
            #     image = torch.flip(image, dims=(1,))
            #     mask = torch.flip(mask, dims=(1,))

            # # apply cutout
            # cutout_coin_toss = random.randint(0, 1)
            # if self.aug_cutout == 1 and cutout_coin_toss:
            #     image = self.cutout_obj(image)

            # # apply mask
            # mask_coin_toss = random.randint(0,1)
            # if self.aug_mask == 1 and mask_coin_toss:  # TODO: if or elif?
            #     image, mask = self.mask_obj(image, mask)

            # if self.intensity_scale:
            #     if not self.new_kwyk_data:
            #         image = self.intensity_scale(image)
            #     else:
            #         image = self.intensity_scale(image / 255.0) * 255

            # Albumentations
            image = np.array(image.squeeze(0))
            mask = np.array(mask.squeeze(0))
            if not self.new_kwyk_data:
                image = image / 255.0

            transformed = self.transform(image = image.astype(np.float32), mask = mask)
            image = transformed['image']
            mask = transformed['mask']
            if not self.new_kwyk_data:
                image = image * 255.0
            
            if self.aug_background_manipulation:
                apply_background_coin_toss = random.random() < 0.5
                if apply_background_coin_toss:
                    shapes_background_coin_toss = random.random() < 0.5 if self.aug_grid_background == self.aug_shapes_background else self.aug_shapes_background

                    if shapes_background_coin_toss:
                        background = draw_random_shapes_background(image.shape)
                    else:
                        background = draw_random_grid_background(image.shape)
                        
                    image = apply_background(image,mask,background)

            # null half
            null_coin_toss = 1 if random.random() < 0.5 else 0
            if self.aug_null_half and null_coin_toss:
                image, mask = null_half(image, mask, random.randint(0, 1) == 1)

            # resize image to [1,h,w] again
            image = torch.from_numpy(image)
            mask = torch.from_numpy(mask)
            image = image.unsqueeze(dim=0)
            mask = mask.unsqueeze(dim=0)

        if not self.new_kwyk_data and "unet" in self.model_name:
            image = image[:, 1:161, 1:193]
            mask = mask[:, 1:161, 1:193]

        if not self.new_kwyk_data: # these prob don't work anymore
            if self.nr_of_classes == 2:
                mask[mask != 0] = 1
            elif self.nr_of_classes == 6 or self.nr_of_classes == 16:
                # mask = mapping(mask, self.nr_of_classes, original=False, map_class_num=51) # mapping mod
                mask, class_mapping = mapping(mask, nr_of_classes=self.nr_of_classes, reference_col="50-class", class_mapping=self.class_mapping)
                self.class_mapping = class_mapping
            elif self.nr_of_classes == 50:
                mask[mask > 49] = 0

            # normalize image
            image = (
                image - self.normalization_constants[0]
            ) / self.normalization_constants[1]

        if self.new_kwyk_data:
            image = image.to(torch.float32)
            # mask = torch.tensor(
            #     mapping(np.array(mask), self.nr_of_classes, original=True)
            # ) # mapping mod
            mask, class_mapping = mapping(np.array(mask), nr_of_classes=self.nr_of_classes, reference_col='original', class_mapping=self.class_mapping)
            mask = torch.from_numpy(mask)
            self.class_mapping = class_mapping

        if self.pretrained:
            image = image.repeat((3, 1, 1))
        return image, mask

    def __len__(self):
        return len(self.images)

    def _get_affine_matrix(self, image):
        # which augmentations to perform (based on SynthSeg)
        scaling_bounds = 0.2  # the scaling coefficients will be sampled from U(1-scaling_bounds; 1+scaling_bounds)
        rotation_bounds = 15  # the rotation angles will be sampled from U(-rotation_bounds; rotation_bounds)
        shearing_bounds = 0.012  # the shearing coefficients will be sampled from U(-shearing_bounds; shearing_bounds)
        translation_bounds = False  # no translation is performed, as this is already modelled by the random cropping
        enable_90_rotations = False

        # randomize augmentations
        batchsize = 1
        n_dims = 2
        center = np.array([image.shape[0] // 2, image.shape[1] // 2])

        scaling = draw_value_from_distribution(
            scaling_bounds,
            size=n_dims,
            centre=1,
            default_range=0.15,
            return_as_tensor=False,
            batchsize=batchsize,
        )

        rotation = draw_value_from_distribution(
            rotation_bounds,
            size=1,
            default_range=15.0,
            return_as_tensor=False,
            batchsize=batchsize,
        )

        shearing = draw_value_from_distribution(
            shearing_bounds,
            size=n_dims**2 - n_dims,
            default_range=0.01,
            return_as_tensor=False,
            batchsize=batchsize,
        )
        affine_matrix = create_affine_transformation_matrix(
            n_dims=2,
            scaling=scaling,
            rotation=rotation,
            shearing=shearing,
            translation=None,
        )

        # Translate the center back to the origin
        translation_matrix1 = np.array(
            [[1, 0, -center[0]], [0, 1, -center[1]], [0, 0, 1]]
        )

        # Translate the center to the original position
        translation_matrix2 = np.array(
            [[1, 0, center[0]], [0, 1, center[1]], [0, 0, 1]]
        )

        final_matrix = translation_matrix2 @ affine_matrix @ translation_matrix1
        return final_matrix


def get_data_loader(
    # data_dir: str,
    config,
    num_workers: int = 4 * torch.cuda.device_count(),
):
    if config.new_kwyk_data:
        train_dataset = KWYKVolumeDataset(mode="train", config=config, volume_data_dir='/om2/scratch/Mon/sabeen/kwyk-volumes/rawdata/',slice_info_file='/om2/user/sabeen/kwyk_data/new_kwyk_full.npy',bg_percent=0.8)
        val_dataset = KWYKVolumeDataset(mode="validation", config=config, volume_data_dir='/om2/scratch/Mon/sabeen/kwyk-volumes/rawdata/',slice_info_file='/om2/user/sabeen/kwyk_data/new_kwyk_full.npy',bg_percent=0.8)
        test_dataset = KWYKVolumeDataset(mode="test", config=config, volume_data_dir='/om2/scratch/Mon/sabeen/kwyk-volumes/rawdata/',slice_info_file='/om2/user/sabeen/kwyk_data/new_kwyk_full.npy',bg_percent=0.8)
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
