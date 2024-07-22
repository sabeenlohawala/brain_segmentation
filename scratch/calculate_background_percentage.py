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
SOURCE_DIR_00 = ''
gettrace = getattr(sys, "gettrace", None)
DEBUG = True if gettrace() else False

def get_feature_label_pairs(features_dir=SOURCE_DIR_00, labels_dir=SOURCE_DIR_00):
    """
    Get pairs of feature and label filenames.
    """
    features = sorted(glob.glob(os.path.join(features_dir, "*orig*")))
    labels = sorted(glob.glob(os.path.join(labels_dir, "*aseg*")))
    # features = sorted(glob.glob(os.path.join(features_dir, "*brain*")))
    # labels = sorted(glob.glob(os.path.join(labels_dir, "*mask*")))

    return list(zip(features, labels))

def count_background(feature, label):
    label_slice = np.load(label).astype(np.int16)
    if len(label_slice.shape) > 2:
        label_slice = label_slice.squeeze(0)
    count_background = np.sum(label_slice == 0)
    percent_background = count_background / (
        label_slice.shape[0] * label_slice.shape[1]
    )
    return {feature: percent_background, label: percent_background}

data_dir = '/om2/user/sabeen/kwyk_final_1000/'
percent_background_by_mode = {}
all_percent_background = Counter()
for mode in ['train', 'validation', 'test']:
    subdir = f'{data_dir}/{mode}'
    # feature_label_pairs = get_feature_label_pairs(features_dir = subdir, labels_dir = subdir)

    features_dir = f'{subdir}/features'
    labels_dir = f'{subdir}/labels'
    feature_label_pairs = get_feature_label_pairs(features_dir = features_dir, labels_dir = labels_dir)

    n_procs = 1 if DEBUG else multiprocessing.cpu_count()
    with Pool(processes=n_procs) as pool:
            percent_backgrounds = pool.starmap(
                count_background,
                [
                    (feature, label)
                    for feature, label in feature_label_pairs
                ],
            )
    percent_background_by_mode[mode] = percent_backgrounds
    all_percent_background.update(dict(ChainMap(*percent_backgrounds)))

with open(os.path.join(data_dir, "percent_backgrounds.json"), "w") as f:
    json.dump(percent_background_by_mode, f)

