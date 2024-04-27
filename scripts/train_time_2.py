import glob
import json
import os
import sys
from argparse import Namespace

import h5py as h5
import nibabel as nib
import numpy as np
import torch
from TissueLabeling.utils import main_timer
from torch.utils.data import Dataset

gettrace = getattr(sys, "gettrace", None)
DEBUG = True if gettrace() else False
OUT_DIR = "/om2/user/sabeen/nifti_to_numpy/"
os.makedirs(OUT_DIR, exist_ok=True)

SLICE_HDF5 = "/om2/scratch/Fri/hgazula/kwyk_slices.h5"
VOL_HDF5 = "/om2/scratch/Fri/hgazula/kwyk_vols.h5"

NIFTI_DIR = "/om2/scratch/Mon/sabeen/kwyk-volumes/rawdata/"
SLICE_INFO_FILE = "/om2/user/sabeen/kwyk_data/new_kwyk_full.npy"


def write_kwyk_vols_to_hdf5(save_path=None):
    N_VOLS = 10
    feature_files = sorted(glob.glob(os.path.join(NIFTI_DIR, "*orig*")))[:N_VOLS]
    label_files = sorted(glob.glob(os.path.join(NIFTI_DIR, "*aseg*")))[:N_VOLS]
    feature_label_files = zip(feature_files, label_files)

    f = h5.File(save_path, "w")
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
    scl_slopes = np.array([file.header["scl_slope"] for file in nib_files])
    scl_inters = np.array([file.header["scl_inter"] for file in nib_files])
    assert np.isnan(scl_slopes).all() and np.isnan(scl_inters).all()
    print("Assertion passed!")

    for idx, (feature_file, label_file) in enumerate(feature_label_files):
        features[idx, :, :, :] = nib.load(feature_file).dataobj
        labels[idx, :, :, :] = nib.load(label_file).dataobj

    f.close()


def write_kwyk_slices_to_hdf5(save_path=None):
    N_VOLS = 10
    feature_files = sorted(glob.glob(os.path.join(NIFTI_DIR, "*orig*")))[:N_VOLS]
    label_files = sorted(glob.glob(os.path.join(NIFTI_DIR, "*aseg*")))[:N_VOLS]
    feature_label_files = zip(feature_files, label_files)

    f = h5.File(save_path, "w")
    features_dir1 = f.create_dataset(
        "kwyk_features_dir1",
        (N_VOLS * 256, 256, 256),
        dtype=np.uint8,
        chunks=True,
        compression="gzip",
        compression_opts=9,
    )
    features_dir2 = f.create_dataset(
        "kwyk_features_dir2",
        (N_VOLS * 256, 256, 256),
        dtype=np.uint8,
        chunks=True,
        compression="gzip",
        compression_opts=9,
    )
    features_dir3 = f.create_dataset(
        "kwyk_features_dir3",
        (N_VOLS * 256, 256, 256),
        dtype=np.uint8,
        chunks=True,
        compression="gzip",
        compression_opts=9,
    )

    labels_dir1 = f.create_dataset(
        "kwyk_labels_dir1",
        (N_VOLS * 256, 256, 256),
        dtype=np.uint16,
        chunks=True,
        compression="gzip",
        compression_opts=9,
    )

    labels_dir2 = f.create_dataset(
        "kwyk_labels_dir2",
        (N_VOLS * 256, 256, 256),
        dtype=np.uint16,
        chunks=True,
        compression="gzip",
        compression_opts=9,
    )

    labels_dir3 = f.create_dataset(
        "kwyk_labels_dir3",
        (N_VOLS * 256, 256, 256),
        dtype=np.uint16,
        chunks=True,
        compression="gzip",
        compression_opts=9,
    )

    # # check scale factors are all nan
    # nib_files = [nib.load(file) for file in feature_files]
    # scl_slopes = np.array([file.header["scl_slope"] for file in nib_files])
    # scl_inters = np.array([file.header["scl_inter"] for file in nib_files])
    # assert np.isnan(scl_slopes).all() and np.isnan(scl_inters).all()
    # print("Assertion passed!")

    for idx, (feature_file, label_file) in enumerate(feature_label_files):
        print(f"writing file {idx}")
        features_dir1[idx * 256 : (idx + 1) * 256, :, :] = nib.load(
            feature_file
        ).dataobj[0:256, :, :]
        features_dir2[idx * 256 : (idx + 1) * 256, :, :] = nib.load(
            feature_file
        ).dataobj[:, 0:256, :]
        features_dir3[idx * 256 : (idx + 1) * 256, :, :] = nib.load(
            feature_file
        ).dataobj[:, :, 0:256]

        labels_dir1[idx * 256 : (idx + 1) * 256, :, :] = nib.load(label_file).dataobj[
            0:256, :, :
        ]
        labels_dir2[idx * 256 : (idx + 1) * 256, :, :] = nib.load(label_file).dataobj[
            :, 0:256, :
        ]
        labels_dir3[idx * 256 : (idx + 1) * 256, :, :] = nib.load(label_file).dataobj[
            :, :, 0:256
        ]

    f.close()


@main_timer
def read_kwyk_vol_hdf5(read_path):
    kwyk = h5.File(read_path, "r")
    features = kwyk["kwyk_features"]
    labels = kwyk["kwyk_labels"]
    for feature, label in zip(features, labels):
        _, _ = feature.shape, label.shape
    print("success")


def read_kwyk_slice_hdf5(read_path):
    kwyk = h5.File(read_path, "r")
    features_dir1 = kwyk["kwyk_features_dir1"]
    labels_dir1 = kwyk["kwyk_labels_dir1"]

    for feature, label in zip(features_dir1, labels_dir1):
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


class H5SliceDataset(torch.utils.data.Dataset):
    def __init__(self, mode, config, volume_data_dir, slice_info_file):
        self.mode = mode
        self.matrix = torch.from_numpy(np.load(slice_info_file, allow_pickle=True))

        kwyk = h5.File("/om2/scratch/Fri/hgazula/kwyk_slices.h5", "r")
        self.kwyk_features_dir1 = kwyk["kwyk_features_dir1"]
        self.kwyk_features_dir2 = kwyk["kwyk_features_dir2"]
        self.kwyk_features_dir3 = kwyk["kwyk_features_dir3"]

        self.kwyk_labels_dir1 = kwyk["kwyk_labels_dir1"]
        self.kwyk_labels_dir2 = kwyk["kwyk_labels_dir2"]
        self.kwyk_labels_dir3 = kwyk["kwyk_labels_dir3"]

        self.nonzero_indices = torch.nonzero(
            self.matrix < config.background_percent_cutoff
        )  # [num_slices, 3] - (file_idx, direction_idx, slice_idx)

    def __getitem__(self, index):
        file_idx, direction_idx, slice_idx = self.nonzero_indices[index]

        new_idx = file_idx * 256 + slice_idx

        if direction_idx == 0:
            feature_slice = torch.from_numpy(
                self.kwyk_features_dir1[new_idx, :, :].astype(np.float32)
            ).squeeze()
            label_slice = torch.from_numpy(
                self.kwyk_labels_dir1[new_idx, :, :].astype(np.int16)
            ).squeeze()
        elif direction_idx == 1:
            feature_slice = torch.from_numpy(
                self.kwyk_features_dir2[new_idx, :, :].astype(np.float32)
            ).squeeze()
            label_slice = torch.from_numpy(
                self.kwyk_labels_dir2[new_idx, :, :].astype(np.int16)
            ).squeeze()
        else:
            feature_slice = torch.from_numpy(
                self.kwyk_features_dir3[new_idx, :, :].astype(np.float32)
            ).squeeze()
            label_slice = torch.from_numpy(
                self.kwyk_labels_dir3[new_idx, :, :].astype(np.int16)
            ).squeeze()

        return (feature_slice, label_slice)

    def __len__(self):
        return self.nonzero_indices.shape[0]


class H5VolDataset(torch.utils.data.Dataset):
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
            feature_slice = torch.from_numpy(
                self.kwyk_features[file_idx, slice_idx, :, :].astype(np.float32)
            ).squeeze()
            label_slice = torch.from_numpy(
                self.kwyk_labels[file_idx, slice_idx, :, :].astype(np.int16)
            ).squeeze()
        elif direction_idx == 1:
            feature_slice = torch.from_numpy(
                self.kwyk_features[file_idx, :, slice_idx, :].astype(np.float32)
            ).squeeze()
            label_slice = torch.from_numpy(
                self.kwyk_labels[file_idx, :, slice_idx, :].astype(np.int16)
            ).squeeze()
        else:
            feature_slice = torch.from_numpy(
                self.kwyk_features[file_idx, :, :, slice_idx].astype(np.float32)
            ).squeeze()
            label_slice = torch.from_numpy(
                self.kwyk_labels[file_idx, :, :, slice_idx].astype(np.int16)
            ).squeeze()

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
def loop_over_dataloader(config, item):
    train_loader = torch.utils.data.DataLoader(
        item,
        batch_size=config.batch_size,
        shuffle=False,
    )

    for batch_idx, (image, mask) in enumerate(train_loader):
        if batch_idx == 0:
            break


def time_dataloaders():
    config = {
        "batch_size": 64,  # CHANGE
        "background_percent_cutoff": 0.8,
        "data_dir": "/om2/scratch/Mon/sabeen/kwyk_slice_split_250/",
    }
    config = Namespace(**config)

    print("time for nifti volumes")
    kwyk_dataset = KWYKVolumeDataset(
        mode="test",
        config=config,
        volume_data_dir=NIFTI_DIR,
        slice_info_file=SLICE_INFO_FILE,
    )
    loop_over_dataloader(config, kwyk_dataset)

    print("time for h5 vols")
    h5vol_dataset = H5VolDataset(
        mode="test",
        config=config,
        volume_data_dir=NIFTI_DIR,
        slice_info_file=SLICE_INFO_FILE,
    )
    loop_over_dataloader(config, h5vol_dataset)

    print("time for h5 slices")
    h5slice_dataset = H5VolDataset(
        mode="test",
        config=config,
        volume_data_dir=NIFTI_DIR,
        slice_info_file=SLICE_INFO_FILE,
    )
    loop_over_dataloader(config, h5slice_dataset)

    print("time for slices")
    train_dataset = NoBrainerDataset("train", config)
    loop_over_dataloader(config, train_dataset)


if __name__ == "__main__":
    # write_kwyk_slices_to_hdf5(save_path=SLICE_HDF5)
    # read_kwyk_slice_hdf5(read_path=SLICE_HDF5)

    # write_kwyk_vols_to_hdf5(save_path=VOL_HDF5)
    # read_kwyk_vol_hdf5(read_path=VOL_HDF5)

    time_dataloaders()
