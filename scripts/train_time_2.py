import glob
import json
import os
import random
import sys
from argparse import Namespace
from datetime import datetime
from multiprocessing import Pool
import h5py as h5

import nibabel as nib
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from TissueLabeling.utils import main_timer
from torch.utils.data import Dataset

gettrace = getattr(sys, "gettrace", None)
DEBUG = True if gettrace() else False
OUT_DIR = "/om2/user/sabeen/nifti_to_numpy/"
os.makedirs(OUT_DIR, exist_ok=True)

SAVE_NAME = "/om2/user/sabeen/kwyk_data/satra.h5"
def write_kwyk_hdf5():
    N_VOLS = 10
    feature_files = sorted(glob.glob("/om2/scratch/Mon/sabeen/kwyk-volumes/rawdata/*orig*"))[:N_VOLS]
    label_files = sorted(glob.glob("/om2/scratch/Mon/sabeen/kwyk-volumes/rawdata/*aseg*"))[:N_VOLS]
    feature_label_files = zip(feature_files, label_files)

    f = h5.File(SAVE_NAME, "w")
    features = f.create_dataset(
        "kwyk_features",
        (N_VOLS, 256, 256, 256),
        dtype=np.uint8,
        chunks=True,
        compression="gzip",
        compression_opts=9,
    )
    labels = f.create_dataset(
        "kwyk_labels",
        (N_VOLS, 256, 256, 256),
        dtype=np.uint16,
        chunks=True,
        compression="gzip",
        compression_opts=9,
    )

    # TODO: parallelize
    # def write_volume(feature_file, label_file):
    #     features[idx, :, :, :] = nib.load(feature_file).dataobj
    #     labels[idx, :, :, :] = nib.load(label_file).dataobj

    # with Pool(processes=len(os.sched_getaffinity(0))) as pool:
    #     pool.map(write_volume, zip(feature_files, label_files))

    # check scale factors are all nan
    nib_files = [nib.load(file) for file in feature_files]
    scl_slopes = np.array([file.header['scl_slope'] for file in nib_files])
    scl_inters = np.array([file.header['scl_inter'] for file in nib_files])
    assert np.isnan(scl_slopes).all() and np.isnan(scl_inters).all()
    print('Assertion passed!')

    for idx, (feature_file, label_file) in enumerate(feature_label_files):
        features[idx, :, :, :] = nib.load(feature_file).dataobj
        labels[idx, :, :, :] = nib.load(label_file).dataobj


@main_timer
def read_kwyk_hdf5():
    kwyk = h5.File(SAVE_NAME, "r")
    features = kwyk["kwyk_features"]
    labels = kwyk["kwyk_labels"]

    for feature, label in zip(features, labels):
        _, _ = feature.shape, label.shape

    print("success")

class KWYKVolumeDataset(torch.utils.data.Dataset):
    def __init__(self, mode, config, volume_data_dir, slice_info_file):
        self.mode = mode
        self.matrix = torch.from_numpy(np.load(slice_info_file, allow_pickle=True))

        self.feature_label_files = list(
            zip(
                sorted(glob.glob(os.path.join(volume_data_dir, "*orig*.nii.gz")))[
                    : self.matrix.shape[0]
                ],
                sorted(glob.glob(os.path.join(volume_data_dir, "*aseg*.nii.gz")))[
                    : self.matrix.shape[0]
                ],
            )
        )

        self.nonzero_indices = torch.nonzero(
            self.matrix < config.background_percent_cutoff
        )  # [num_slices, 3] - (file_idx, direction_idx, slice_idx)

    def __getitem__(self, index):
        file_idx, direction_idx, slice_idx = self.nonzero_indices[index]
        feature_file, label_file = self.feature_label_files[file_idx]

        feature_vol = torch.from_numpy(nib.load(feature_file).get_fdata())
        label_vol = torch.from_numpy(nib.load(label_file).get_fdata())

        if direction_idx == 0:
            feature_slice = feature_vol[slice_idx, :, :]
            label_slice = label_vol[slice_idx, :, :]

        if direction_idx == 1:
            feature_slice = feature_vol[:, slice_idx, :]
            label_slice = label_vol[:, slice_idx, :]

        if direction_idx == 2:
            feature_slice = feature_vol[:, :, slice_idx]
            label_slice = label_vol[:, :, slice_idx]

        return (feature_slice, label_slice)

    def __len__(self):
        return self.nonzero_indices.shape[0]

class H5Dataset(torch.utils.data.Dataset):
    def __init__(self, mode, config, volume_data_dir, slice_info_file):
        self.mode = mode
        self.matrix = torch.from_numpy(np.load(slice_info_file, allow_pickle=True))

        kwyk = h5.File("/om2/user/sabeen/kwyk_data/satra.h5", "r")
        self.kwyk_features = kwyk["kwyk_features"]
        self.kwyk_labels = kwyk["kwyk_labels"]

        self.nonzero_indices = torch.nonzero(
            self.matrix < config.background_percent_cutoff
        )  # [num_slices, 3] - (file_idx, direction_idx, slice_idx)

    def __getitem__(self, index):
        file_idx, direction_idx, slice_idx = self.nonzero_indices[index]
        
        if direction_idx == 0:
            feature_slice = torch.from_numpy(self.kwyk_features[file_idx,slice_idx,:,:].astype(np.float32)).squeeze()
            label_slice = torch.from_numpy(self.kwyk_labels[file_idx,slice_idx,:,:].astype(np.int16)).squeeze()
        elif direction_idx == 1:
            feature_slice = torch.from_numpy(self.kwyk_features[file_idx,:,slice_idx,:].astype(np.float32)).squeeze()
            label_slice = torch.from_numpy(self.kwyk_labels[file_idx,:,slice_idx,:].astype(np.int16)).squeeze()
        else:
            feature_slice = torch.from_numpy(self.kwyk_features[file_idx,:,:,slice_idx].astype(np.float32)).squeeze()
            label_slice = torch.from_numpy(self.kwyk_labels[file_idx,:,:,slice_idx].astype(np.int16)).squeeze()

        return (feature_slice, label_slice)

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
        if mode not in ["train", "test", "validation"]:
            raise Exception(
                f"{mode} is not a valid data mode. Choose from 'train', 'test', or 'validation'."
            )
        self.mode = mode

        background_percent_cutoff = config.background_percent_cutoff  # 0.99
        valid_feature_filename = f"{config.data_dir}/{mode}/valid_feature_files_{int(background_percent_cutoff*100)}.json"
        valid_label_filename = f"{config.data_dir}/{mode}/valid_label_files_{int(background_percent_cutoff*100)}.json"

        with open(valid_feature_filename) as f:
            self.images = json.load(f)

        with open(valid_label_filename) as f:
            self.masks = json.load(f)

    def __getitem__(self, idx):
        # returns (image, mask)
        image = np.load(self.images[idx])
        mask = np.load(self.masks[idx])

        return image, mask

    def __len__(self):
        return len(self.images)

@main_timer
def loop_over_dataloder(item):
    for batch_idx, (image, mask) in enumerate(item):
        if batch_idx == 0:
            break

def main():
    config = {
        "batch_size": 288,  # CHANGE
        "background_percent_cutoff": 0.8,
        "data_dir": "/om2/scratch/Mon/sabeen/kwyk_slice_split_250/",
    }
    config = Namespace(**config)

    # train_dataset = KWYKVolumeDataset(
    #     mode="test",
    #     config=config,
    #     volume_data_dir="/om2/scratch/Mon/sabeen/kwyk-volumes/rawdata/",
    #     slice_info_file="/om2/user/sabeen/kwyk_data/new_kwyk_full.npy",
    # )

    # train_dataset = H5Dataset(
    #     mode="test",
    #     config=config,
    #     volume_data_dir="/om2/scratch/Mon/sabeen/kwyk-volumes/rawdata/",
    #     slice_info_file="/om2/user/sabeen/kwyk_data/new_kwyk_full.npy",
    # )

    # slice dataset
    train_dataset = NoBrainerDataset("train", config)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        # num_workers=1,
    )

    loop_over_dataloder(train_loader)


if __name__ == "__main__":
    main()
