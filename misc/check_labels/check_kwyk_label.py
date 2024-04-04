import glob
import os
from multiprocessing import Pool
import multiprocessing

import sys
import numpy as np
from typing import Tuple
import json

from ext.lab2im import utils
from datetime import datetime

gettrace = getattr(sys, "gettrace", None)
DEBUG = True if gettrace() else False

SOURCE_DIR_00 = "/om2/scratch/tmp/sabeen/kwyk_data/kwyk/rawdata/"

FEATURE_TRANFORM_DIR = f"{SOURCE_DIR_00}/features"
LABEL_TRANFORM_DIR = f"{SOURCE_DIR_00}/labels"

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

def get_feature_label_class_count(feature: str, label: str, seg_class: int) -> Tuple[int, str, str]:
    """
    Apply a series of transformations to a feature and label volume.

    Args:
        feature (str): Path to the feature volume.
        label (str): Path to the label volume.

    Returns:
        Tuple[Tuple[int, int, int]]: A tuple containing the shape of the transformed feature volume.
    """
    print(f"Processing subject: {os.path.basename(feature)}")
    label_vol, label_aff, label_hdr = utils.load_volume(label, im_only=False)
    label_vol = label_vol.astype("int16")

    return (int(np.sum(label_vol == seg_class)), feature, label)


def get_kwyk_feature_label_class_count(seg_class: int) -> np.ndarray:
    """
    Apply a series of transformations to the entire kwyk dataset.

    Args:
        None

    Returns:
        max_dims (nd.array): 3-D numpy array containing the shape of the transformed volume.
    """

    feature_label_pairs = get_feature_label_pairs()

    file_count = len(feature_label_pairs)

    input_ids = np.random.choice(range(file_count), file_count, replace=False)

    n_procs = 1 if DEBUG else multiprocessing.cpu_count()

    with Pool(processes=n_procs) as pool:
        shapes = pool.starmap(
            get_feature_label_class_count,
            [
                (
                    feature_label_pairs[idx][0],
                    feature_label_pairs[idx][1], 
                    seg_class
                ) 
                for idx in input_ids
            ],
        )
    shapes = [item for item in shapes if item[0] > 0]
    with open(f'/om2/user/sabeen/tissue_labeling/misc/contains_{seg_class}.json', 'w') as jsonfile:
        json.dump(shapes, jsonfile)

    return shapes


def get_feature_label_pairs(features_dir=SOURCE_DIR_00, labels_dir=SOURCE_DIR_00):
    """
    Get pairs of feature and label filenames.
    """
    features = sorted(glob.glob(os.path.join(features_dir, "*orig*")))
    labels = sorted(glob.glob(os.path.join(labels_dir, "*aseg*")))

    return list(zip(features, labels))

@main_timer
def main():
    # Obtain shapes (and pixel_counts?) after cropping full kwyk dataset
    counts = get_kwyk_feature_label_class_count(seg_class=3)

    # Extract slices, pad them to be of shape (max_rows, max_cols) and save them as .npy files
    print(len(counts))


if __name__ == "__main__":
    main()
