import argparse
import os
import pickle
import random
import json
import glob

import numpy as np
import torch
# import webdataset as wds
from scipy.ndimage import rotate
from sklearn.model_selection import train_test_split

from TissueLabeling.brain_utils import brain_coord, load_brains, mapping

# import nobrainer


# parser = argparse.ArgumentParser()
# parser.add_argument(
#     "type", help="which dataset to build: 'small', 'medium' or 'large'", type=str
# )
# # parser.add_argument("shardsize", help="Number of samples each shard should contain",
# #                     type=int)
# parser.add_argument(
#     "dataset_name",
#     help="name of dataset, gives name of folder where data is saved",
#     type=str,
# )
# args = parser.parse_args()

SEED = 42
TYPE = 'small' #args.type
# SHARD_SIZE = args.shardsize
HEIGHT = 162
WIDTH = 194
NR_OF_CLASSES = 51
AUG_ANGLES = list(range(15, 180 + 15, 15))
# DATASET_NAME = args.dataset_name
DATASET_NAME = 'new_med_no_aug_51' if TYPE == 'medium' else 'new_small_no_aug_51'
# POSSIBLE_AUGMENTATIONS = ['rotation'] # ['rotation','null','zoom']

file_path = "/nese/mit/group/sig/users/matth406/nobrainer_data/data/SharedData/segmentation/freesurfer_asegs"
matthias_idx_path = "/nese/mit/group/sig/users/matth406/nobrainer_data/data/SharedData/segmentation/idx.dat"
save_path_basic = f"/om2/user/sabeen/nobrainer_data_norm/{DATASET_NAME}"

kwyk_slice_path = '/om/scratch/Fri/sabeen/kwyk_slice_split_250'
kwyk_idx_path = f"{kwyk_slice_path}/idx.dat"

def matthias_files_to_volume_dir_slice():
    # create and store train/val/test split of ALL files once and save it
    print("Loading existing file names...")
    with open(matthias_idx_path, "rb") as f:
        (
            train_images,
            train_masks,
            validation_images,
            validation_masks,
            test_images,
            test_masks,
        ) = pickle.load(f)

    # take only subset of data
    end_idx = 10 if TYPE == "small" else 1000 if TYPE == "medium" else -1
    train_images = train_images[: int(end_idx * 0.8)]
    train_masks = train_masks[: int(end_idx * 0.8)]
    validation_images = validation_images[: int(end_idx * 0.1)]
    validation_masks = validation_masks[: int(end_idx * 0.1)]
    test_images = test_images[: int(end_idx * 0.1)]
    test_masks = test_masks[: int(end_idx * 0.1)]

    images_files_splitted = (train_images, validation_images, test_images)
    mask_files_splitted = (train_masks, validation_masks, test_masks)

    # 1. find smallest possible crop over train, val and test set if not provided
    # if HEIGHT == None and WIDTH == None:

    height, width = 162, 194
    too_small = 0

    path_map = {}

    # 2. create image shards
    for mode_idx, mode in enumerate(["train", "validation", "test"]):
        path_map[mode] = {}

        print("Mode: ", mode)

        image_files = images_files_splitted[mode_idx]
        mask_files = mask_files_splitted[mode_idx]

        save_path_mode = f"{save_path_basic}/{mode}"

        idx = 0

        for image_file, mask_file in zip(image_files, mask_files):
            brain, brain_mask, image_nr = load_brains(image_file, mask_file, file_path)
            # brain_mask = mapping(brain_mask, nr_of_classes=NR_OF_CLASSES, original=True)

            # slice the MRI volume in 3 directions
            for d in range(3):
                for i in range(brain.shape[d]):
                    # get the slice
                    if d == 0:
                        brain_slice = brain[i, :, :]
                        mask_slice = brain_mask[i, :, :]
                    elif d == 1:
                        brain_slice = brain[:, i, :]
                        mask_slice = brain_mask[:, i, :]
                    elif d == 2:
                        brain_slice = brain[:, :, i]
                        mask_slice = brain_mask[:, :, i]

                    # skip slices with no or little brain (20% cutoff)
                    if np.sum(brain_slice) < 52428:
                        continue

                    # find the crop
                    for j in range(256):
                        if (brain_slice[j] != 0).any():
                            cut_top_temp = j
                            break
                    for j in range(256):
                        if (brain_slice[255 - j] != 0).any():
                            cut_bottom_temp = 255 - j
                            break
                    for j in range(256):
                        if (brain_slice[:, j] != 0).any():
                            cut_left_temp = j
                            break
                    for j in range(256):
                        if (brain_slice[:, 255 - j] != 0).any():
                            cut_right_temp = 255 - j
                            break

                    height_temp = cut_bottom_temp - cut_top_temp + 1
                    width_temp = cut_right_temp - cut_left_temp + 1

                    assert height_temp <= height, "Crop height is too big"
                    assert width_temp <= width, "Crop width is too big"
                    print("Patch height: ", height_temp)
                    print("Patch width: ", width_temp)

                    # crop image-optimal patch:
                    brain_slice = brain_slice[
                        cut_top_temp : cut_bottom_temp + 1,
                        cut_left_temp : cut_right_temp + 1,
                    ]
                    mask_slice = mask_slice[
                        cut_top_temp : cut_bottom_temp + 1,
                        cut_left_temp : cut_right_temp + 1,
                    ]

                    if brain_slice.shape[0] < 50 or brain_slice.shape[1] < 50:
                        too_small += 1
                        continue

                    assert (brain_slice > 0).any(), "Crop is empty"

                    brain_filename = f"{save_path_mode}/brain_{idx}.npy"
                    mask_filename = f"{save_path_mode}/mask_{idx}.npy"
                    path_map[mode][f'{brain_filename}\n{mask_filename}'] = [image_file,mask_file,d,i]
                    # path_map[mode][mask_filename] = [mask_file,d,i]

                    idx += 1

    print("Number of patches too small: ", too_small)

    with open(f'{save_path_basic}/matthias_path_map.json', 'w') as f:
        json.dump(path_map,f)
    
    return path_map
    
def path_map_to_kwyk_slices(path_map):
    matthias_path_to_kwyk_path = {}
    with open(kwyk_idx_path, "rb") as f:
        (
            kwyk_train_images,
            kwyk_train_masks,
            kwyk_validation_images,
            kwyk_validation_masks,
            kwyk_test_images,
            kwyk_test_masks,
        ) = pickle.load(f)
    kwyk_train_saved_features = sorted(glob.glob(f'{kwyk_slice_path}/train/features/*orig*'))
    kwyk_validation_saved_features = sorted(glob.glob(f'{kwyk_slice_path}/validation/features/*orig*'))
    kwyk_test_saved_features = sorted(glob.glob(f'{kwyk_slice_path}/test/features/*orig*'))

    kwyk_saved_files = {
        'train': kwyk_train_saved_features,
        'validation': kwyk_validation_saved_features,
        'test': kwyk_test_saved_features
    }

    not_found = {}
    
    for mode in ['train','validation','test']:
        mode_path_map = path_map[mode]
        matthias_path_to_kwyk_path[mode] = {}
        not_found[mode] = []
        for key,val in mode_path_map.items():
            image_vol_file, mask_vol_file, d, i = val
            matthias_image_file, matthias_mask_file = key.split('\n')
            if f'/om2/scratch/Mon/sabeen/kwyk-volumes/rawdata/{image_vol_file}' in kwyk_test_images:
                kwyk_mode_dir = f'{kwyk_slice_path}/test'
            elif f'/om2/scratch/Mon/sabeen/kwyk-volumes/rawdata/{image_vol_file}' in kwyk_validation_images:
                kwyk_mode_dir = f'{kwyk_slice_path}/validation'
            else:
                kwyk_mode_dir = f'{kwyk_slice_path}/train'
            image_vol_base = image_vol_file.split('.')[0]
            mask_vol_base = mask_vol_file.split('.')[0]
            kwyk_feature_slice_path = f'{kwyk_mode_dir}/features/{image_vol_base}_{d*256+i}.npy'
            kwyk_label_slice_path = f'{kwyk_mode_dir}/labels/{mask_vol_base}_{d*256+i}.npy'
            if kwyk_feature_slice_path not in kwyk_saved_files[kwyk_mode_dir.split('/')[-1]]:
                print(f'Not found: {kwyk_feature_slice_path}')
                not_found[mode].append(kwyk_feature_slice_path)
            matthias_path_to_kwyk_path[mode][matthias_image_file] = kwyk_feature_slice_path
            matthias_path_to_kwyk_path[mode][matthias_mask_file] = kwyk_label_slice_path
    
    matthias_path_to_kwyk_path['not found'] = not_found
    with open(f'{save_path_basic}/matthias_to_kwyk.json', 'w') as f:
        json.dump(matthias_path_to_kwyk_path,f)
            


def main():
    if os.path.exists(f'{save_path_basic}/matthias_path_map.json'):
        print('path map exists')
        with open(f'{save_path_basic}/matthias_path_map.json') as f:
            path_map = json.load(f)
    else:
        print('creating path map')
        path_map = matthias_files_to_volume_dir_slice()

    path_map_to_kwyk_slices(path_map)


if __name__ == "__main__":
    main()