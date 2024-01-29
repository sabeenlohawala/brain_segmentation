import csv
import os
from typing import Tuple

import nibabel as nib
import numpy as np
import torch

def load_brains(image_file: str, mask_file: str, file_path: str):
    # ensure that mask and image numbers match
    image_nr = image_file.split("_")[1]
    mask_nr = mask_file.split("_")[1]
    assert image_nr == mask_nr, "image and mask numbers do not match"

    image_path = os.path.join(file_path, image_file)
    mask_path = os.path.join(file_path, mask_file)

    brain = nib.load(image_path)
    brain_mask = nib.load(mask_path)

    brain = brain.get_fdata()
    brain_mask = brain_mask.get_fdata()
    brain_mask = brain_mask.astype(int)
    # apply skull stripping
    brain[brain_mask == 0] = 0

    return brain, brain_mask, image_nr

def crop(image: np.array, height: int, width: int) -> np.array:
    # find image-optimal crop
    for j in range(256):
        if (image[j] != 0).any():
            cut_top_temp = j
            break
    for j in range(256):
        if (image[255 - j] != 0).any():
            cut_bottom_temp = 255 - j
            break
    for j in range(256):
        if (image[:, j] != 0).any():
            cut_left_temp = j
            break
    for j in range(256):
        if (image[:, 255 - j] != 0).any():
            cut_right_temp = 255 - j
            break

    # image-optimal size:
    height_temp = cut_bottom_temp - cut_top_temp + 1
    width_temp = cut_right_temp - cut_left_temp + 1
    assert height_temp <= height, "Crop height is too big"
    assert width_temp <= width, "Crop width is too big"

    # crop image-optimal patch:
    image = image[
        cut_top_temp : cut_bottom_temp + 1, cut_left_temp : cut_right_temp + 1
    ]

    if image.shape[0] < 50 or image.shape[1] < 50:
        pass

    assert (image > 0).any(), "Crop is empty"

    # adjust the crop to largest rectangle
    if height_temp < height:
        diff = height - height_temp
        # even difference
        if (diff % 2) == 0:
            image = np.pad(image, ((diff // 2, diff // 2), (0, 0)))
        # odd difference
        else:
            image = np.pad(image, ((diff // 2, diff // 2 + 1), (0, 0)))
    if width_temp < width:
        diff = width - width_temp
        # even difference
        if (diff % 2) == 0:
            image = np.pad(image, ((0, 0), (diff // 2, diff // 2)))
        # odd difference
        else:
            image = np.pad(image, ((0, 0), (diff // 2, diff // 2 + 1)))

    return image

def brain_coord(slice: torch.tensor) -> Tuple[int, int, int, int]:
    """
    Computes the coordnates of a rectangle
    that contains the brain area of a slice

    Args:
        slice (torch.tensor): brain slice of shape [H, W]

    Returns:
        Tuple(int, int, int, int): Coordinates of the rectangle
    """

    H, W = slice.shape[-2:]
    H, W = H - 1, W - 1
    # majority vote of the 4 corners
    bg_value = torch.stack(
        (slice[0, 0], slice[-1, -1], slice[0, -1], slice[-1, 0])
    ).mode()[0]

    for j in range(H):
        if (slice[j] != bg_value).any():
            cut_top_temp = j
            break
    for j in range(H):
        if (slice[H - j] != bg_value).any():
            cut_bottom_temp = H - j
            break
    for j in range(W):
        if (slice[:, j] != bg_value).any():
            cut_left_temp = j
            break
    for j in range(W):
        if (slice[:, W - j] != bg_value).any():
            cut_right_temp = W - j
            break

    return (cut_top_temp, cut_bottom_temp, cut_left_temp, cut_right_temp)


def brain_area(slice: torch.tensor) -> torch.tensor:
    """
    Computes the brain area of a slice

    Args:
        slice (torch.tensor): brain slice

    Returns:
        torch.tensor: brain area
    """

    cut_top_temp, cut_bottom_temp, cut_left_temp, cut_right_temp = brain_coord(slice)

    return slice[cut_top_temp : cut_bottom_temp + 1, cut_left_temp : cut_right_temp + 1]


# def mapping(mask: np.array):
#     class_mapping = {}
#     # labels = []
#     with open(
#         "/home/matth406/unsupervised_brain/data/class-mapping.csv", newline=""
#     ) as csvfile:
#         spamreader = csv.reader(csvfile, delimiter=",", quotechar="|")
#         # skip header
#         next(spamreader, None)
#         for row in spamreader:
#             class_mapping[int(row[1])] = int(row[4])
#     #         labels.append(int(row[1]))
#     # labels = np.array(labels)

#     # labels = []
#     # with open('/home/matth406/unsupervised_brain/data/class-mapping.csv', newline='') as csvfile:
#     #     spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
#     #     # skip header
#     #     next(spamreader, None)
#     #     for row in spamreader:
#     #         labels.append(int(row[1]))
#     # labels = np.array(labels)
#     # # labels = torch.tensor([   0, 1024,    2,    3,    4,    5, 1025,    7,    8, 1026,   10,   11,
#     # #       12,   13,   14,   15,   16,   17,   18, 1034, 1035,   24,   26,   28,
#     # #       30,   31,   41,   42,   43,   44,   46,   47, 1027,   49,   50,   51,
#     # #       52,   53,   54, 1028,   58, 1029,   60,   62,   63, 1030, 1031,   72,
#     # #     1032,   77, 1033,   80,   85,  251,  252,  253,  254,  255, 1009, 1010,
#     # #     1011, 2034, 1012, 1013, 1014, 2035, 1015, 1007, 2033, 2000, 2001, 2002,
#     # #     2003, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015,
#     # #     2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 1000, 2025, 1002, 1003,
#     # #     2024, 1005, 1006, 2031, 2032, 1008, 1001, 2026, 2027, 2028, 2029, 2030,
#     # #     1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023])

#     # class_mapping = {value.item(): index for index, value in enumerate(labels)}
#     u, inv = np.unique(mask, return_inverse=True)
#     num_classes = 50  # len(class_mapping)
#     for x in u:
#         if x not in class_mapping:
#             class_mapping[x] = num_classes

#     # # we collect all classes not in the mapping table as an additional "other" class
#     # mask = np.array([class_mapping[int(x)] if x in labels else len(labels) for x in u])[inv].reshape(mask.shape)
#     for old, new in class_mapping.items():
#         mask[mask == old] = -1 * new
#     mask = mask * -1

#     return mask

def mapping(mask: np.array, nr_of_classes=51, original=True):

    # if original == True, map from original --> num-class column
    # if original == False, map from index --> num-class column

    # TODO: handle binary case!!

    class_mapping = {}
    # labels = []
    with open(
        "/om2/user/sabeen/nobrainer_data_norm/class_mapping.csv", newline=""
    ) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=",", quotechar="|")
        # skip header
        header = next(spamreader, None)
        original_index = header.index('original') # original numbered segments
        new_index = header.index('index') # 107-class numbered segments
        map_index = original_index if original else new_index

        col_index = new_index if nr_of_classes == 107 else (header.index('2-class') if nr_of_classes == 2 else header.index(f'{nr_of_classes-1}-class'))

        for row in spamreader:
            class_mapping[int(row[map_index])] = int(row[col_index])

    # class_mapping = {value.item(): index for index, value in enumerate(labels)}
    u, inv = np.unique(mask, return_inverse=True)
    # num_classes = 50  # len(class_mapping)
    for x in u:
        if x not in class_mapping:
            class_mapping[x] = nr_of_classes - 1

    # we collect all classes not in the mapping table as an additional "other" class
    # mask = np.array([class_mapping[int(x)] if x in labels else len(labels) for x in u])[inv].reshape(mask.shape)
    for old, new in class_mapping.items():
        mask[mask == old] = -1 * new
    mask = mask * -1

    return mask