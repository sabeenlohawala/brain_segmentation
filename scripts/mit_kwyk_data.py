import glob
import os
from multiprocessing import Pool
import multiprocessing

import sys
import numpy as np
from typing import Optional, Tuple
from collections import Counter
import random
import pickle
import json
import argparse
from sklearn.model_selection import train_test_split
import nobrainer

from ext.lab2im import utils, edit_volumes
from datetime import datetime

from TissueLabeling.brain_utils import mapping

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
    default=0
)
args = parser.parse_args()

gettrace = getattr(sys, "gettrace", None)
DEBUG = True if gettrace() else False

SOURCE_DIR_00 = "/nese/mit/group/sig/data/kwyk/rawdata" # TODO: command line arg?

TRANSFORM_DIR = args.transform_dir # "/om2/user/sabeen/kwyk_tranform" # Done: command line arg
FEATURE_TRANFORM_DIR = f"{TRANSFORM_DIR}/features"
LABEL_TRANFORM_DIR = f"{TRANSFORM_DIR}/labels"
IDX_PATH = f"{TRANSFORM_DIR}/idx.dat"

SLICE_DEST_DIR = args.slice_dest_dir # "/om2/user/sabeen/kwyk_final" # Done: command line arg

ROTATE_VOL = args.rotate_vol # True # Done: make this command line arg

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

def transform_feature_label_pair(
    feature: str, label: str
) -> Tuple[Tuple[int, int, int], Optional[np.ndarray]]:
    """
    Apply a series of transformations to a feature and label volume.

    Args:
        feature (str): Path to the feature volume.
        label (str): Path to the label volume.

    Returns:
        Tuple[Tuple[int, int, int], Optional[np.ndarray]]: A tuple containing 
        the shape of the transformed feature volume and the pixel counts (None if not computed).
    """
    print(f"Processing subject: {os.path.basename(feature)}")

    feature_dest = os.path.join(FEATURE_TRANFORM_DIR, os.path.basename(feature))
    label_dest = os.path.join(LABEL_TRANFORM_DIR, os.path.basename(label))

    label_vol, label_aff, label_hdr = utils.load_volume(label, im_only=False)
    feature_vol, feature_aff, feature_hdr = utils.load_volume(feature, im_only=False)

    label_vol = label_vol.astype('int32')

    mask, cropping = cropLabelVol(label_vol)
    feature_vol = applyCropping(feature_vol, cropping)
    feature_vol = feature_vol * (mask > 0)
    feature_vol = feature_vol / 255.0

    utils.save_volume(feature_vol, feature_aff, feature_hdr, feature_dest)
    utils.save_volume(mask, label_aff, label_hdr, label_dest)

    # TODO: is pixel counts needed at vol level?
    unique,counts = np.unique(mask,return_counts = True)
    pixel_counts = {label:count for label,count in zip(unique,counts)}

    return feature_vol.shape , pixel_counts

def transform_kwyk_dataset():
    """
    Apply a series of transformations to the entire kwyk dataset.
    """
    
    feature_label_pairs = get_feature_label_pair()

    file_count = len(feature_label_pairs)

    if os.path.exists(FEATURE_TRANFORM_DIR) and os.path.exists(LABEL_TRANFORM_DIR) and len(os.listdir(FEATURE_TRANFORM_DIR)) == file_count and len(os.listdir(LABEL_TRANFORM_DIR)) == file_count:
        all_pixel_counts = None
        max_dims = np.load(os.path.join(TRANSFORM_DIR,'max_dims.npy'))
    else:
        os.makedirs(FEATURE_TRANFORM_DIR, exist_ok=True)
        os.makedirs(LABEL_TRANFORM_DIR, exist_ok=True)

        input_ids = np.random.choice(range(file_count), file_count, replace=False)

        n_procs = 1 if DEBUG else multiprocessing.cpu_count()

        with Pool(processes=n_procs) as pool:
            shapes_and_pixel_counts = pool.starmap(
                transform_feature_label_pair,
                [feature_label_pairs[idx] for idx in input_ids],
            )

        shapes = [shape for shape, _ in shapes_and_pixel_counts]
        pixel_counts = [pixel_counts for _, pixel_counts in shapes_and_pixel_counts]

        all_keys = {key for d in pixel_counts for key in d.keys()}
        all_pixel_counts = {int(label):sum(p.get(label,0) for p in pixel_counts) for label in all_keys} # save in kwyk_tranform? -> this might be slightly different after padding (more zeros)

        # TODO: Matthias finds pixel_counts over training data only -> is this needed anymore?
        # with open(os.path.join(TRANSFORM_DIR,'all_pixel_counts.pkl'), 'wb') as pickle_file:
        #     pickle.dump(all_pixel_counts, pickle_file)
        
        # Step 2: determine slice dimensions based on max dataset dims
        max_dims = np.max(np.vstack(shapes), axis=0)
        print('max dims:', max_dims)

        np.save(os.path.join(TRANSFORM_DIR,'max_dims.npy'), max_dims)

    return max_dims #, all_pixel_counts


def get_feature_label_pair(features_dir=SOURCE_DIR_00, labels_dir = SOURCE_DIR_00):
    """
    Get pairs of feature and label filenames.
    """
    features = sorted(glob.glob(os.path.join(features_dir, "*orig*")))[:1000]
    labels = sorted(glob.glob(os.path.join(labels_dir, "*aseg*")))[:1000]

    return list(zip(features, labels))

def extract_feature_label_slices(
    feature: str, label: str, max_shape: Tuple[int,int], slice_dir: str, get_pixel_counts: bool=False):
    """
    Extract all of the slices with > 20% brain in the pair of feature-label volumes that
    are stored at the filepaths specified by feature and label.

    Args:
        feature (str): Path to the feature volume
        label (str): Path to the label volume
        max_shape (Tuple[int,int]): A tuple containing the shape to pad each slice to.
        slice_dir (str): Main directory where feature and label slice .npy files are saved.
    
    Returns:
        None
    """
    feature_slice_dest_dir = os.path.join(slice_dir, "features")
    label_slice_dest_dir = os.path.join(slice_dir, "labels")

    os.makedirs(feature_slice_dest_dir, exist_ok=True)
    os.makedirs(label_slice_dest_dir, exist_ok=True)

    label_vol = (utils.load_volume(label, im_only=True)).astype('int32')
    feature_vol = utils.load_volume(feature, im_only=True)

    # Add random rotation to entire volume
    if ROTATE_VOL:
        # randomly choose an angle between 0 to 20 for all axes
        angles = np.random.uniform(0,20,size=3)
        assert feature_vol.shape == label_vol.shape

        affine = nobrainer.transform.get_affine(feature_vol.shape,rotation=angles)
        feature_vol = np.array(nobrainer.transform.warp(feature_vol,affine,order=1))
        label_vol = np.array(nobrainer.transform.warp(label_vol,affine,order=0)).astype('int32') # Done: check whether this is necessary or can I leave as ints?

    slice_idx = 0
    if get_pixel_counts:
        pixel_counts = Counter()
    else:
        pixel_counts = None

    for d in range(3):
        for i in range(label_vol.shape[d]):
            # get the slice
            if d == 0:
                feature_slice = feature_vol[i, :, :]
                label_slice = label_vol[i, :, :]
            elif d == 1:
                feature_slice = feature_vol[:, i, :]
                label_slice = label_vol[:, i, :]
            elif d == 2:
                feature_slice = feature_vol[:, :, i]
                label_slice = label_vol[:, :, i]

            # discard slices with < 20% brain (> 80% background)
            count_background = np.sum(label_slice == 0)
            if count_background > 0.8 * (label_slice.shape[0] * label_slice.shape[1]):
                continue

            # pad slices
            pad_rows = max(0,max_shape[0] - label_slice.shape[0])
            pad_cols = max(0,max_shape[1] - label_slice.shape[1])

            # padding for each side
            pad_top = pad_rows // 2
            pad_bottom = pad_rows - pad_top
            pad_left = pad_cols // 2
            pad_right = pad_cols - pad_left

            padded_feature_slice = np.pad(feature_slice, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
            padded_label_slice = np.pad(label_slice, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)

            # save .npy files
            feature_slice_filename = f"{os.path.basename(feature).split('.')[0]}_{slice_idx:03d}.npy"
            label_slice_filename = f"{os.path.basename(label).split('.')[0]}_{slice_idx:03d}.npy"
            np.save(os.path.join(feature_slice_dest_dir,feature_slice_filename), padded_feature_slice[np.newaxis,:])
            np.save(os.path.join(label_slice_dest_dir,label_slice_filename), padded_label_slice[np.newaxis,:])


            # Done: get pixel_counts
            if get_pixel_counts:
                unique,counts = np.unique(padded_label_slice,return_counts = True)
                pixel_counts.update({label:count for label,count in zip(unique,counts)})

            # increase slice_idx
            slice_idx += 1

    # TODO: generate synth affines? -> Or apply in dataset.py
    return pixel_counts

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
        feature_label_pairs = get_feature_label_pair(features_dir=FEATURE_TRANFORM_DIR,labels_dir=LABEL_TRANFORM_DIR)
        feature_files = [feature for feature, _ in feature_label_pairs]
        label_files = [label for _, label in feature_label_pairs]
        feature_files = sorted(feature_files)
        label_files = sorted(label_files)

        # train-validation-test split
        train_features, test_features, train_labels, test_labels = train_test_split(
            feature_files, label_files, test_size=0.2, random_state=SEED
        )
        validation_features, test_features, validation_labels, test_labels = train_test_split(
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
    return train_features, train_labels, validation_features, validation_labels, test_features, test_labels

def extract_kwyk_slices(max_shape):
    """
    Extracts all the slices with > 20% brain in each of the 3 directions for the entire kwyk dataset, and saves
    them in directories separated by the train/validation/test split.
    """
    train_features, train_labels, validation_features, validation_labels, test_features, test_labels = get_train_val_test_split()

    feature_label_pairs_by_mode = {'train': zip(train_features,train_labels),
                                   'validation': zip(validation_features,validation_labels),
                                   'test': zip(test_features, test_labels)}

    train_pixel_counts = Counter()
    for mode, zipped in feature_label_pairs_by_mode.items():
        print(f"Extracting {mode} slices...")
        for feature, label in zipped:
            pixel_counts = extract_feature_label_slices(feature,label,max_shape,os.path.join(SLICE_DEST_DIR, mode),get_pixel_counts=(mode=='train'))
            if pixel_counts is not None:
                train_pixel_counts += pixel_counts
    
    # Done: aggregate and save pixel counts
    with open(os.path.join(SLICE_DEST_DIR,'train_pixel_counts.pkl'), 'wb') as pickle_file:
        pickle.dump(dict(train_pixel_counts), pickle_file)

@main_timer
def main():
    # Obtain shapes (and pixel_counts?) after cropping full kwyk dataset
    max_dims = transform_kwyk_dataset()

    max_rows = max(max_dims[0], max_dims[1])
    max_cols = max(max_dims[1], max_dims[2])

    # Extract slices, pad them to be of shape (max_rows, max_cols) and save them as .npy files
    extract_kwyk_slices((max_rows,max_cols))


if __name__ == "__main__":
    main()
