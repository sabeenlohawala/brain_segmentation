import glob
import os
from multiprocessing import Pool
import multiprocessing

import sys
import numpy as np
from typing import Optional, Tuple
from collections import Counter, ChainMap
from functools import reduce
import random
import pickle
import json
import argparse
from sklearn.model_selection import train_test_split
import nobrainer

from ext.lab2im import utils, edit_volumes
from datetime import datetime

from TissueLabeling.brain_utils import mapping, load_brains_v2

parser = argparse.ArgumentParser()
parser.add_argument(
    "transform_dir", help="Where to save the transformed volumes", type=str
)
parser.add_argument(
    "slice_dest_dir",
    help="Directory where the slices extracted from the transformed volumes are saved",
    type=str,
)
parser.add_argument(
    "--rotate_vol",
    help="Flag for whether to rotate the brain volume",
    type=int,
    required=False,
    default=0,
)
args = parser.parse_args()

gettrace = getattr(sys, "gettrace", None)
DEBUG = True if gettrace() else False

SOURCE_DIR_00 = "/om2/scratch/tmp/sabeen/kwyk/rawdata/"

TRANSFORM_DIR = args.transform_dir  # "/om2/user/sabeen/kwyk_tranform"
FEATURE_TRANFORM_DIR = f"{TRANSFORM_DIR}/features" if TRANSFORM_DIR != SOURCE_DIR_00 else SOURCE_DIR_00
LABEL_TRANFORM_DIR = f"{TRANSFORM_DIR}/labels" if TRANSFORM_DIR != SOURCE_DIR_00 else SOURCE_DIR_00\

SLICE_DEST_DIR = args.slice_dest_dir  # "/om2/user/sabeen/kwyk_final"
IDX_PATH = f"{SLICE_DEST_DIR}/idx.dat"
os.makedirs(SLICE_DEST_DIR,exist_ok=True)

ROTATE_VOL = args.rotate_vol

SEED = 42

def main_timer(func):
    """Decorator to time any function"""

    def function_wrapper():
        start_time = datetime.now()
        # print(f'Start Time: {start_time.strftime("%A %m/%d/%Y %H:%M:%S")}')

        func()

        end_time = datetime.now()
        # print(f'End Time: {end_time.strftime("%A %m/%d/%Y %H:%M:%S")}')
        print(
            f"Function: {func.__name__} Total runtime: {end_time - start_time} (HH:MM:SS)"
        )

    return function_wrapper


# Crop label volume
def cropLabelVol(V, margin=10, threshold=0):
    # Make sure it's 3D
    margin = np.array(margin)
    if len(margin.shape) < 2:
        margin = [margin, margin, margin]

    if len(V.shape) < 2:
        V = V[..., np.newaxis]
    if len(V.shape) < 3:
        V = V[..., np.newaxis]

    # Now
    idx = np.where(V > threshold)
    i1 = np.max([0, np.min(idx[0]) - margin[0]]).astype("int")
    j1 = np.max([0, np.min(idx[1]) - margin[1]]).astype("int")
    k1 = np.max([0, np.min(idx[2]) - margin[2]]).astype("int")
    i2 = np.min([V.shape[0], np.max(idx[0]) + margin[0] + 1]).astype("int")
    j2 = np.min([V.shape[1], np.max(idx[1]) + margin[1] + 1]).astype("int")
    k2 = np.min([V.shape[2], np.max(idx[2]) + margin[2] + 1]).astype("int")

    cropping = [i1, j1, k1, i2, j2, k2]
    cropped = V[i1:i2, j1:j2, k1:k2]

    return cropped, cropping


def applyCropping(V, cropping):
    i1 = cropping[0]
    j1 = cropping[1]
    k1 = cropping[2]
    i2 = cropping[3]
    j2 = cropping[4]
    k2 = cropping[5]

    if len(V.shape) > 2:
        Vcropped = V[i1:i2, j1:j2, k1:k2, ...]
    else:
        Vcropped = V[i1:i2, j1:j2]

    return Vcropped


def transform_feature_label_pair(feature: str, label: str) -> Tuple[int, int, int]:
    """
    Apply a series of transformations to a feature and label volume.

    Args:
        feature (str): Path to the feature volume.
        label (str): Path to the label volume.

    Returns:
        Tuple[Tuple[int, int, int]]: A tuple containing the shape of the transformed feature volume.
    """
    print(f"Processing subject: {os.path.basename(feature)}")

    feature_dest = os.path.join(FEATURE_TRANFORM_DIR, os.path.basename(feature))
    label_dest = os.path.join(LABEL_TRANFORM_DIR, os.path.basename(label))

    label_vol, label_aff, label_hdr = utils.load_volume(label, im_only=False)
    feature_vol, feature_aff, feature_hdr = utils.load_volume(feature, im_only=False)

    label_vol = label_vol.astype("uint16")
    
    # apply skull stripping (from Matthias's brain_utils.load_brains function)
    feature_vol[label_vol == 0] = 0

    mask, cropping = cropLabelVol(label_vol)
    feature_vol = applyCropping(feature_vol, cropping)
    feature_vol = feature_vol * (mask > 0)
    feature_vol = feature_vol / 255.0

    utils.save_volume(feature_vol, feature_aff, feature_hdr, feature_dest)
    utils.save_volume(mask, label_aff, label_hdr, label_dest)

    return feature_vol.shape


def transform_kwyk_dataset() -> np.ndarray:
    """
    Apply a series of transformations to the entire kwyk dataset.

    Args:
        None

    Returns:
        max_dims (nd.array): 3-D numpy array containing the shape of the transformed volume.
    """

    feature_label_pairs = get_feature_label_pairs()

    file_count = len(feature_label_pairs)

    if (
        os.path.exists(FEATURE_TRANFORM_DIR)
        and os.path.exists(LABEL_TRANFORM_DIR)
        and len(os.listdir(FEATURE_TRANFORM_DIR)) == file_count
        and len(os.listdir(LABEL_TRANFORM_DIR)) == file_count
    ):
        max_dims = np.load(os.path.join(TRANSFORM_DIR, "max_dims.npy"))
    else:
        os.makedirs(FEATURE_TRANFORM_DIR, exist_ok=True)
        os.makedirs(LABEL_TRANFORM_DIR, exist_ok=True)

        input_ids = np.random.choice(range(file_count), file_count, replace=False)

        n_procs = 1 if DEBUG else multiprocessing.cpu_count()

        with Pool(processes=n_procs) as pool:
            shapes = pool.starmap(
                transform_feature_label_pair,
                [feature_label_pairs[idx] for idx in input_ids],
            )

        # determine slice dimensions based on max dataset dims
        max_dims = np.max(np.vstack(shapes), axis=0)
        print("max dims:", max_dims)

        np.save(os.path.join(TRANSFORM_DIR, "max_dims.npy"), max_dims)

    return max_dims


def get_feature_label_pairs(features_dir=SOURCE_DIR_00, labels_dir=SOURCE_DIR_00):
    """
    Get pairs of feature and label filenames.
    """
    features = sorted(glob.glob(os.path.join(features_dir, "*orig*")))
    labels = sorted(glob.glob(os.path.join(labels_dir, "*aseg*")))

    return list(zip(features, labels))


def extract_feature_label_slices(
    feature: str,
    label: str,
    max_shape: Tuple[int, int],
    slice_dir: str,
    get_pixel_counts: bool = False,
) -> Counter:
    """
    Extract all of the slices with > 20% brain in the pair of feature-label volumes that
    are stored at the filepaths specified by feature and label.

    Args:
        feature (str): Path to the feature volume
        label (str): Path to the label volume
        max_shape (Tuple[int,int]): A tuple containing the shape to pad each slice to.
        slice_dir (str): Main directory where feature and label slice .npy files are saved.

    Returns:
        pixel_counts (Counter): A Counter where the keys are the class labels in the label
        and the values are the number of pixels in the slice equal to each key.
    """
    feature_slice_dest_dir = os.path.join(slice_dir, "features")
    label_slice_dest_dir = os.path.join(slice_dir, "labels")

    os.makedirs(feature_slice_dest_dir, exist_ok=True)
    os.makedirs(label_slice_dest_dir, exist_ok=True)

    label_vol = (utils.load_volume(label, im_only=True)).astype("uint16")
    feature_vol = utils.load_volume(feature, im_only=True)
    if TRANSFORM_DIR == SOURCE_DIR_00:
        feature_vol[label_vol == 0] = 0 # skull stripping?
        feature_vol = feature_vol / 255.0

    feature_base_filename = os.path.basename(feature).split(".")[0]
    label_base_filename = os.path.basename(label).split(".")[0]

    # Add random rotation to entire volume
    if ROTATE_VOL:
        # randomly choose an angle between -20 to 20 for all axes
        angles = np.random.uniform(-20, 20, size=3)
        # assert feature_vol.shape == label_vol.shape

        affine = nobrainer.transform.get_affine(feature_vol.shape, rotation=angles)
        feature_vol = np.array(nobrainer.transform.warp(feature_vol, affine, order=1))
        label_vol = np.array(
            nobrainer.transform.warp(label_vol, affine, order=0)
        ).astype("uint16")

    slice_idx = 0
    pixel_counts = Counter()
    all_percent_backgrounds = {}

    for d in range(3):
        # V1: not parallelized: for looping over all slices (currently faster than V2)
        # for i in range(label_vol.shape[d]):
        #     # get the slice
        #     if d == 0:
        #         feature_slice = feature_vol[i, :, :]
        #         label_slice = label_vol[i, :, :]
        #     elif d == 1:
        #         feature_slice = feature_vol[:, i, :]
        #         label_slice = label_vol[:, i, :]
        #     elif d == 2:
        #         feature_slice = feature_vol[:, :, i]
        #         label_slice = label_vol[:, :, i]

        #     idx_and_counts = process_slice(
        #         feature_slice,
        #         label_slice,
        #         slice_idx,
        #         max_shape,
        #         get_pixel_counts,
        #         feature_base_filename,
        #         label_base_filename,
        #         feature_slice_dest_dir,
        #         label_slice_dest_dir,
        #     )
        #     pixel_counts += idx_and_counts
        #     # increase slice_idx
        #     slice_idx += 1

        # V2: trying to paralellize using map (currently slower than V1)
        feature_base_filename = os.path.basename(feature).split('.')[0]
        label_base_filename = os.path.basename(label).split('.')[0]
        if d == 0:
            slice_counts_and_percent_backgrounds = map(lambda i: process_slice(feature_vol[i,:,:],label_vol[i,:,:],slice_idx = slice_idx + i, max_shape = max_shape, get_pixel_counts = get_pixel_counts, feature_base_filename = feature_base_filename, label_base_filename = label_base_filename, feature_slice_dest_dir = feature_slice_dest_dir, label_slice_dest_dir = label_slice_dest_dir), range(feature_vol.shape[d]))
        elif d == 1:
            slice_counts_and_percent_backgrounds = map(lambda i: process_slice(feature_vol[:,i,:],label_vol[:,i,:],slice_idx = slice_idx + i, max_shape = max_shape, get_pixel_counts = get_pixel_counts, feature_base_filename = feature_base_filename, label_base_filename = label_base_filename, feature_slice_dest_dir = feature_slice_dest_dir, label_slice_dest_dir = label_slice_dest_dir), range(feature_vol.shape[d]))
        else:
            slice_counts_and_percent_backgrounds = map(lambda i: process_slice(feature_vol[:,:,i],label_vol[:,:,i],slice_idx = slice_idx + i, max_shape = max_shape, get_pixel_counts = get_pixel_counts, feature_base_filename = feature_base_filename, label_base_filename = label_base_filename, feature_slice_dest_dir = feature_slice_dest_dir, label_slice_dest_dir = label_slice_dest_dir), range(feature_vol.shape[d]))
        slice_counts, percent_backgrounds = zip(*slice_counts_and_percent_backgrounds)
        slice_idx += feature_vol.shape[d]
        pixel_counts += sum(slice_counts, Counter())
        all_percent_backgrounds.update(dict(ChainMap(*percent_backgrounds)))

    return pixel_counts, all_percent_backgrounds


def process_slice(
    feature_slice,
    label_slice,
    slice_idx,
    max_shape,
    get_pixel_counts,
    feature_base_filename,
    label_base_filename,
    feature_slice_dest_dir,
    label_slice_dest_dir,
):
    """
    Processes the feature and label slice by padding them to max_shape and saving them as .npy files.

    Args:
        feature_slice
        label_slice
        slice_idx
        max_shape
        get_pixel_counts
        feature_base_filename
        label_base_filename
        feature_slice_dest_dir
        label_slice_dest_dir

    Returns:
        slice_idx (int): the slice index
        slice_pixel_counts (Counter): Counter where the keys are the labels in label_slice
        and the value is the number of pixel in label_slice equal to that key if get_pixel_counts
        is True, else empty Counter.
    """
    slice_pixel_counts = Counter()
    # discard slices with < 20% brain (> 80% background)
    count_background = np.sum(label_slice == 0)
    percent_background = count_background / (label_slice.shape[0] * label_slice.shape[1])
    # if count_background > 0.8 * (label_slice.shape[0] * label_slice.shape[1]):
    #     return slice_pixel_counts

    # pad slices
    pad_rows = max(0, max_shape[0] - label_slice.shape[0])
    pad_cols = max(0, max_shape[1] - label_slice.shape[1])

    # padding for each side
    pad_top = pad_rows // 2
    pad_bottom = pad_rows - pad_top
    pad_left = pad_cols // 2
    pad_right = pad_cols - pad_left

    padded_feature_slice = np.pad(
        feature_slice,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=0,
    )
    padded_label_slice = np.pad(
        label_slice,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=0,
    )

    # save .npy files
    feature_slice_filename = f"{feature_base_filename}_{slice_idx:03d}.npy"
    label_slice_filename = f"{label_base_filename}_{slice_idx:03d}.npy"
    np.save(
        os.path.join(feature_slice_dest_dir, feature_slice_filename),
        padded_feature_slice[np.newaxis, :],
    )
    np.save(
        os.path.join(label_slice_dest_dir, label_slice_filename),
        padded_label_slice[np.newaxis, :],
    )

    if get_pixel_counts:
        unique, counts = np.unique(padded_label_slice, return_counts=True)
        slice_pixel_counts.update(
            {label: count for label, count in zip(unique, counts)}
        )
    return slice_pixel_counts, {os.path.join(feature_slice_dest_dir, feature_slice_filename): percent_background, os.path.join(label_slice_dest_dir, label_slice_filename): percent_background}


def get_train_val_test_split():
    """
    Obtains the 80/10/10 train/val/test split for the transformed kwyk volumes. Loads this form
    IDX_PATH if it exists, otherwise generates split and saves to IDX_PATH.

    Args:
        None

    Returns:
        train_features: List of filenames for the feature volumes in the training split.
        train_labels: List of filenames for the label voluems in the training split.
        validation_features: List of filenames for the feature volumes in the validation split.
        validation_labels:  List of filenames for the label volumes in the validation split.
        test_features: List of filenames for the feature volumes in the test split.
        test_labels: List of filenames for the label volumes in the test split.
    """
    if os.path.exists(IDX_PATH):
        # load existing file names
        print("Loading existing file names...")
        with open(IDX_PATH, "rb") as f:
            (
                train_features,
                train_labels,
                validation_features,
                validation_labels,
                test_features,
                test_labels,
            ) = pickle.load(f)
    else:
        # get file names
        feature_label_pairs = get_feature_label_pairs(
            features_dir=FEATURE_TRANFORM_DIR, labels_dir=LABEL_TRANFORM_DIR
        )
        feature_files = [feature for feature, _ in feature_label_pairs]
        label_files = [label for _, label in feature_label_pairs]
        feature_files = sorted(feature_files)
        label_files = sorted(label_files)

        # train-validation-test split
        train_features, test_features, train_labels, test_labels = train_test_split(
            feature_files, label_files, test_size=0.2, random_state=SEED
        )

        (
            validation_features,
            test_features,
            validation_labels,
            test_labels,
        ) = train_test_split(
            test_features, test_labels, test_size=0.5, random_state=SEED
        )

        data = [
            train_features,
            train_labels,
            validation_features,
            validation_labels,
            test_features,
            test_labels,
        ]
        with open(IDX_PATH, "wb") as f:
            pickle.dump(data, f)
    return (
        train_features,
        train_labels,
        validation_features,
        validation_labels,
        test_features,
        test_labels,
    )


def extract_kwyk_slices(max_shape):
    """
    Extracts all the slices with > 20% brain in each of the 3 directions for the entire kwyk dataset, and saves
    them in directories separated by the train/validation/test split.
    """
    (
        train_features,
        train_labels,
        validation_features,
        validation_labels,
        test_features,
        test_labels,
    ) = get_train_val_test_split()

    feature_label_pairs_by_mode = {
        "train": zip(train_features, train_labels),
        "validation": zip(validation_features, validation_labels),
        "test": zip(test_features, test_labels),
    }

    train_pixel_counts = Counter()
    all_percent_backgrounds = dict()
    for mode, zipped in feature_label_pairs_by_mode.items():
        print(f"Extracting {mode} slices...")

        n_procs = 1 if DEBUG else multiprocessing.cpu_count()
        with Pool(processes=n_procs) as pool:
            pixel_counts_and_percent_backgrounds = pool.starmap(
                extract_feature_label_slices,
                [
                    (
                        feature,
                        label,
                        max_shape,
                        os.path.join(SLICE_DEST_DIR, mode),
                        mode == "train",
                    )
                    for feature, label in zipped
                ],
            )

        pixel_counts, percent_backgrounds = zip(*pixel_counts_and_percent_backgrounds)
        all_percent_backgrounds.update(dict(ChainMap(*percent_backgrounds)))
        if mode == "train":
            train_pixel_counts += sum(pixel_counts, Counter())

    # Aggregate and save pixel counts
    with open(
        os.path.join(SLICE_DEST_DIR, "train_pixel_counts.pkl"), "wb"
    ) as pickle_file:
        pickle.dump(dict(train_pixel_counts), pickle_file)
    with open(os.path.join(SLICE_DEST_DIR,"percent_backgrounds.json"), "w") as f:
        json.dump(all_percent_backgrounds,f)


def get_max_slice_dims(max_vol_dims):
    """
    Returns the max height and width of 2D slices extracted from a volume with volume.shape = max_vol_dims.

    slice[i,:,:] -> shape is dim[1] x dim[2]
    slice[:,i,:] -> shape is dim[0] x dim[2]
    slice[:,:,i] -> shape is dim[0] x dim[1]
    Therefore, in order for slices extracted from all three directions to be the same shape, slice shape
    should be (max(dim[0], dim[1]), max(dim[1], dim[2]))
    """

    max_rows = max(max_vol_dims[0], max_vol_dims[1])
    max_cols = max(max_vol_dims[1], max_vol_dims[2])
    return max_rows, max_cols


@main_timer
def main():
    # Obtain shapes (and pixel_counts?) after cropping full kwyk dataset
    if TRANSFORM_DIR == SOURCE_DIR_00:
        max_vol_dims = (256,256,256)
    else:
        max_vol_dims = transform_kwyk_dataset()

    # Extract slices, pad them to be of shape (max_rows, max_cols) and save them as .npy files
    extract_kwyk_slices(get_max_slice_dims(max_vol_dims))


if __name__ == "__main__":
    main()
