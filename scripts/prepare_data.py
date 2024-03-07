import argparse
import os
import pickle
import random

import numpy as np
import torch
import webdataset as wds
from scipy.ndimage import rotate
from sklearn.model_selection import train_test_split

from TissueLabeling.brain_utils import brain_coord, load_brains, mapping

# import nobrainer


parser = argparse.ArgumentParser()
parser.add_argument(
    "type", help="which dataset to build: 'small', 'medium' or 'large'", type=str
)
# parser.add_argument("shardsize", help="Number of samples each shard should contain",
#                     type=int)
parser.add_argument(
    "dataset_name",
    help="name of dataset, gives name of folder where data is saved",
    type=str,
)
args = parser.parse_args()

SEED = 42
TYPE = args.type
# SHARD_SIZE = args.shardsize
HEIGHT = 162
WIDTH = 194
NR_OF_CLASSES = 51
AUG_ANGLES = list(range(15, 180 + 15, 15))
DATASET_NAME = args.dataset_name
# POSSIBLE_AUGMENTATIONS = ['rotation'] # ['rotation','null','zoom']


def main():
    # create and store train/val/test split of ALL files once and save it
    file_path = "/nese/mit/group/sig/users/matth406/nobrainer_data/data/SharedData/segmentation/freesurfer_asegs"
    idx_path = "/nese/mit/group/sig/users/matth406/nobrainer_data/data/SharedData/segmentation/idx.dat"
    # save_path_basic = f'/om2/user/sabeen/nobrainer_data_norm/data_prepared_segmentation_{TYPE}_3'
    save_path_basic = f"/om2/user/sabeen/nobrainer_data_norm/{DATASET_NAME}"
    # for mode in ['train','validation','test']:
    #     save_path_mode = f'{save_path_basic}/{mode}'

    #     if not os.path.exists(save_path_mode):
    #         os.makedirs(save_path_mode)

    if os.path.exists(idx_path):
        # load existing file names
        print("Loading existing file names...")
        with open(idx_path, "rb") as f:
            (
                train_images,
                train_masks,
                validation_images,
                validation_masks,
                test_images,
                test_masks,
            ) = pickle.load(f)
    else:
        # get file names
        files = sorted(os.listdir(file_path))
        image_files = [file for file in files if "_orig" in file]
        mask_files = [file for file in files if "_aseg" in file]
        image_files = sorted(image_files)
        mask_files = sorted(mask_files)
        # train-validation-test split
        train_images, test_images, train_masks, test_masks = train_test_split(
            image_files, mask_files, test_size=0.2, random_state=SEED
        )
        validation_images, test_images, validation_masks, test_masks = train_test_split(
            test_images, test_masks, test_size=0.5, random_state=SEED
        )
        data = [
            train_images,
            train_masks,
            validation_images,
            validation_masks,
            test_images,
            test_masks,
        ]
        with open(idx_path, "wb") as f:
            pickle.dump(data, f)

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

    # 2. create image shards
    for mode_idx, mode in enumerate(["train", "validation", "test"]):
        steps = 10

        print("Mode: ", mode)

        image_files = images_files_splitted[mode_idx]
        mask_files = mask_files_splitted[mode_idx]

        save_path_mode = f"{save_path_basic}/{mode}"

        # if not os.path.exists(save_path_mode):
        #     os.makedirs(save_path_mode)

        # sink = wds.ShardWriter(f"{save_path_basic}/{mode}/{mode}-%06d.tar", maxcount = SHARD_SIZE)

        idx = 0
        dataset_mean, dataset_std = 0, 0
        pixel_counts = {i: 0 for i in range(NR_OF_CLASSES)}

        for image_file, mask_file in zip(image_files, mask_files):
            brain, brain_mask, image_nr = load_brains(image_file, mask_file, file_path)
            brain_mask = mapping(brain_mask, nr_of_classes=NR_OF_CLASSES, original=True)

            # TODO Add Random Rotation here
            # rotate_flag = random.choice([True,False])
            # if rotate_flag:
            #     possible_rot_angles = [0,15,30,45]
            #     angles = [random.choice(possible_rot_angles),random.choice(possible_rot_angles),random.choice(possible_rot_angles)]
            #     while angles == [0,0,0]:
            #         angles = [random.choice(possible_rot_angles),random.choice(possible_rot_angles),random.choice(possible_rot_angles)]

            #     assert brain.shape == brain_mask.shape

            #     affine = nobrainer.transform.get_affine(brain.shape,rotation=angles)
            #     brain = nobrainer.transform.warp(brain,affine,order=0)
            #     brain_mask = nobrainer.transform.warp(brain_mask,labels_affine,order=0)

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
                    if (
                        cut_bottom_temp - cut_top_temp + 1 > height
                        or cut_right_temp - cut_left_temp + 1 > width
                    ):
                        print("cut_top_temp: ", cut_top_temp)
                        print("cut_bottom_temp: ", cut_bottom_temp)
                        print("cut_left_temp: ", cut_left_temp)
                        print("cut_right_temp: ", cut_right_temp)
                        if d == 0:
                            brain_slice_temp = brain[i, :, :]
                        elif d == 1:
                            brain_slice_temp = brain[:, i, :]
                        elif d == 2:
                            brain_slice_temp = brain[:, :, i]
                        import matplotlib.pyplot as plt

                        print("Sum: ", np.sum(brain_slice_temp))
                        plt.imshow(brain_slice_temp, cmap="gray")
                        plt.savefig("images/brain_slice.png")

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

                    # adjust the crop to largest rectangle
                    if height_temp < height:
                        diff = height - height_temp
                        # even difference
                        if (diff % 2) == 0:
                            # cut_bottom_temp += diff//2
                            # cut_top_temp -= diff//2
                            brain_slice = np.pad(
                                brain_slice, ((diff // 2, diff // 2), (0, 0))
                            )
                            mask_slice = np.pad(
                                mask_slice, ((diff // 2, diff // 2), (0, 0))
                            )
                        # odd difference
                        else:
                            # cut_bottom_temp += diff//2
                            # cut_top_temp -= diff//2 + 1
                            brain_slice = np.pad(
                                brain_slice, ((diff // 2, diff // 2 + 1), (0, 0))
                            )
                            mask_slice = np.pad(
                                mask_slice, ((diff // 2, diff // 2 + 1), (0, 0))
                            )
                    if width_temp < width:
                        diff = width - width_temp
                        # even difference
                        if (diff % 2) == 0:
                            # cut_right_temp += diff//2
                            # cut_left_temp -= diff//2
                            brain_slice = np.pad(
                                brain_slice, ((0, 0), (diff // 2, diff // 2))
                            )
                            mask_slice = np.pad(
                                mask_slice, ((0, 0), (diff // 2, diff // 2))
                            )
                        # odd difference
                        else:
                            # cut_right_temp += diff//2
                            # cut_left_temp -= diff//2 + 1
                            brain_slice = np.pad(
                                brain_slice, ((0, 0), (diff // 2, diff // 2 + 1))
                            )
                            mask_slice = np.pad(
                                mask_slice, ((0, 0), (diff // 2, diff // 2 + 1))
                            )

                    print(brain_slice.shape)
                    print(brain_slice.shape[0] == height)
                    assert brain_slice.shape[0] == height, "Crop height is not correct"
                    assert brain_slice.shape[1] == width, "Crop width is not correct"

                    if mode == "train":
                        # mean and std per image
                        image_mean, image_std = np.mean(brain_slice), np.std(
                            brain_slice
                        )
                        dataset_mean += image_mean
                        dataset_std += image_std

                        # pixel distribution per image
                        unique, counts = np.unique(mask_slice, return_counts=True)
                        for i, j in zip(unique, counts):
                            pixel_counts[i] += j

                    # brain_slice = (brain_slice - image_mean) / image_std
                    # standardize = torch.tensor([image_mean, image_std])

                    # to torch tensor
                    brain_slice = torch.from_numpy(brain_slice).to(torch.float32)
                    mask_slice = torch.from_numpy(mask_slice).to(torch.float32)

                    # get final brain location coordinates
                    (
                        cut_top_temp,
                        cut_bottom_temp,
                        cut_left_temp,
                        cut_right_temp,
                    ) = brain_coord(brain_slice)

                    # sink.write({
                    #     "__key__": str(idx), # key used to identify the object
                    #     'image_nr.id': image_nr, # image number stored as id --> integer
                    #     'slice_nr.id': i, # slice number stored as id --> integer
                    #     'slice_direction.id': d, # slice direction stored as id --> integer
                    #     'slice_augmentation_1.txt': 'None',
                    #     'slice_augmentation_2.txt': 'None',
                    #     'slice_augmentation_3.txt': 'None',
                    #     'slice_rot.id': 0, # angle of rotation for augmentation
                    #     'slice_null.txt': 'None',
                    #     "brain.pth": brain_slice.unsqueeze(0), # brain slice stored as pth --> tensor
                    #     "mask.pth": mask_slice.unsqueeze(0), # mask slice stored as pth --> tensor
                    #     # "standardize.pth": standardize, # image-wise mean and std stored as pth --> tensor
                    #     "batch_idx_bottom.id": cut_bottom_temp, # bottom index of the batch --> integer
                    #     "batch_idx_top.id": cut_top_temp, # top index of the batch --> integer
                    #     "batch_idx_left.id": cut_left_temp, # left index of the batch --> integer
                    #     "batch_idx_right.id": cut_right_temp, # right index of the batch --> integer
                    # })

                    brain_filename = f"{save_path_mode}/brain_{idx}.npy"
                    mask_filename = f"{save_path_mode}/mask_{idx}.npy"
                    np.save(brain_filename, brain_slice.unsqueeze(0))
                    np.save(mask_filename, mask_slice.unsqueeze(0))

                    idx += 1
                    # TODO: add augmentations
                    # if mode == "train":
                    #     # num_augmentations = random.randint(1,len(POSSIBLE_AUGMENTATIONS))
                    #     # augmentations_to_apply = random.sample(POSSIBLE_AUGMENTATIONS,num_augmentations)
                    #     # angle = 0
                    #     # null_side = 'None'
                    #     # for augmentation in augmentations_to_apply:
                    #     #     if augmentation == 'rotate':
                    #     #         angle = random.choice(AUG_ANGLES)
                    #     #         brain_slice = torch.from_numpy(rotate(brain_slice,angle,reshape=False)).to(torch.float32)
                    #     #         mask_slice = torch.from_numpy(rotate(mask_slice,angle,reshape=False,order=0)).to(torch.float32)
                    #     #     elif augmentation == 'null':
                    #     #         null_side = ['left','right','top','down']
                    #     #         if null_side == 'left':
                    #     #             mid = brain_slice.shape[0] // 2
                    #     #             brain_slice[:mid,:] = 0
                    #     #             mask_slice[:mid,:] = 0
                    #     #         elif null_side == 'right':
                    #     #             mid = brain_slice.shape[0] // 2
                    #     #             brain_slice[mid:,:] = 0
                    #     #             mask_slice[mid:,:] = 0
                    #     #         elif null_side == 'top':
                    #     #             mid = brain_slice.shape[1] // 2
                    #     #             brain_slice[:,:mid] = 0
                    #     #             mask_slice[:,:mid] = 0
                    #     #         elif null_side == 'down':
                    #     #             mid = brain_slice.shape[1] // 2
                    #     #             brain_slice[:,mid:] = 0
                    #     #             mask_slice[:,mid] = 0
                    #     #         else:
                    #     #             raise Exception(f'{null_side} is not a valid option for null_side')
                    #     #     elif augmentation == 'zoom':
                    #     #         pass

                    #     # rotation augmentation
                    #     augmentations_to_apply = ['rotation']
                    #     null_side = 'None'
                    #     angle = random.choice(AUG_ANGLES)
                    #     rotated_brain = torch.from_numpy(rotate(brain_slice,angle,reshape=False)).to(torch.float32)
                    #     rotated_slice = torch.from_numpy(rotate(mask_slice,angle,reshape=False,order=0)).to(torch.float32)

                    #     # (cut_top_temp, cut_bottom_temp, cut_left_temp, cut_right_temp) = brain_coord(rotated_brain)
                    #     while len(augmentations_to_apply) < 3:
                    #         augmentations_to_apply.append('None')

                    #     sink.write({
                    #         "__key__": str(idx), # key used to identify the object
                    #         'image_nr.id': image_nr, # image number stored as id --> integer
                    #         'slice_nr.id': i, # slice number stored as id --> integer
                    #         'slice_direction.id': d, # slice direction stored as id --> integer
                    #         'slice_augmentation_1.txt': augmentations_to_apply[0], # 1st in list of augmentations --> string?
                    #         'slice_augmentation_2.txt': augmentations_to_apply[1], # 2nd in list of augmentations --> string?
                    #         'slice_augmentation_3.txt': augmentations_to_apply[2], # 3rd in list of augmentations --> string?
                    #         'slice_rot.id': angle, # angle of rotation for augmentation --> integer
                    #         'slice_null.txt': null_side, # which side of the image was null (or None if no null) --> string
                    #         "brain.pth": rotated_brain.unsqueeze(0), # brain slice stored as pth --> tensor
                    #         "mask.pth": rotated_slice.unsqueeze(0), # mask slice stored as pth --> tensor
                    #         # "standardize.pth": standardize, # image-wise mean and std stored as pth --> tensor
                    #         "batch_idx_bottom.id": cut_bottom_temp, # bottom index of the batch --> integer
                    #         "batch_idx_top.id": cut_top_temp, # top index of the batch --> integer
                    #         "batch_idx_left.id": cut_left_temp, # left index of the batch --> integer
                    #         "batch_idx_right.id": cut_right_temp, # right index of the batch --> integer
                    #     })

                    #     brain_filename = f'{save_path_mode}/brain_{idx}.npy'
                    #     mask_filename = f'{save_path_mode}/mask_{idx}.npy'
                    #     np.save(brain_filename,rotated_brain.unsqueeze(0))
                    #     np.save(mask_filename,rotated_slice.unsqueeze(0))

                    #     idx += 1

        if mode == "train":
            dataset_mean /= idx
            dataset_std /= idx
            np.save(
                f"{save_path_basic}/normalization_constants.npy",
                np.array([dataset_mean, dataset_std]),
            )
            print("Dataset mean: ", dataset_mean, "Dataset std: ", dataset_std)
            np.save(
                f"{save_path_basic}/pixel_counts.npy",
                np.array(list(pixel_counts.values())),
            )

        # sink.close()

    print("Number of patches too small: ", too_small)


if __name__ == "__main__":
    main()
