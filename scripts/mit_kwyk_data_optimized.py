import glob
import os
from datetime import datetime
from multiprocessing import Pool

import nibabel as nib
import numpy as np
import torch
from sklearn.model_selection import train_test_split

DATA_DIR = "/om2/scratch/Mon/sabeen/kwyk-volumes/rawdata/"
SAVE_DIR = os.getcwd()  # "/om2/user/sabeen/kwyk_data/"
SAVE_NAME = "new_kwyk_full.npy"
N_VOLS = 100  # number of volumes to load (this is only for testing)


def main_timer(func):
    """Decorator to time any function"""

    def function_wrapper(*args, **kwargs):
        start_time = datetime.now()
        # print(f'Start Time: {start_time.strftime("%A %m/%d/%Y %H:%M:%S")}')
        result = func(*args, **kwargs)
        end_time = datetime.now()
        # print(f'End Time: {end_time.strftime("%A %m/%d/%Y %H:%M:%S")}')
        print(
            f"Function: {func.__name__} Total runtime: {end_time - start_time} (HH:MM:SS)"
        )
        return result

    return function_wrapper


def calculate_bg(label_file):
    print(os.path.basename(label_file))
    label_vol = nib.load(label_file).get_fdata().astype(np.int16)

    bg = label_vol == 0
    bgcount_ax0 = np.sum(bg, axis=(1, 2)) / 256**2
    bgcount_ax1 = np.sum(bg, axis=(0, 2)) / 256**2
    bgcount_ax2 = np.sum(bg, axis=(0, 1)) / 256**2

    bg_count = np.stack((bgcount_ax0, bgcount_ax1, bgcount_ax2), axis=0)
    return bg_count


@main_timer
def main():
    # feature_files = sorted(glob.glob(os.path.join(DATA_DIR, "*orig*.nii.gz")))[:N_VOLS]
    label_files = sorted(glob.glob(os.path.join(DATA_DIR, "*aseg*.nii.gz")))[:N_VOLS]

    nprocs = len(os.sched_getaffinity(0))
    with Pool(processes=nprocs) as p:
        output = p.map(calculate_bg, label_files)

    final_output = np.dstack(output)  # [num_slices, num_directions, num_files]
    final_output = np.moveaxis(
        final_output, -1, 0
    )  # [num_files, num_directions, num_slices]
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    np.save(os.path.join(SAVE_DIR, SAVE_NAME), final_output)


class SampleDataset(torch.utils.data.Dataset):
    def __init__(self, mode, bg_percent=0.99):
        self.matrix = torch.from_numpy(np.load(SAVE_NAME, allow_pickle=True))

        self.feature_label_files = list(
            zip(
                sorted(glob.glob(os.path.join(DATA_DIR, "*orig*.nii.gz")))[
                    : self.matrix.shape[0]
                ],
                sorted(glob.glob(os.path.join(DATA_DIR, "*aseg*.nii.gz")))[
                    : self.matrix.shape[0]
                ],
            )
        )

        train_matrix, rem_matrix, train_files, rem_files = train_test_split(
            self.matrix, self.feature_label_files, test_size=0.2, random_state=42
        )
        val_matrix, test_matrix, val_files, test_files = train_test_split(
            rem_matrix, rem_files, test_size=0.5, random_state=42
        )

        temp_dict = {
            "train": [train_matrix, train_files],
            "val": [val_matrix, val_files],
            "test": [test_matrix, test_files],
        }

        self.matrix, self.feature_label_files = temp_dict.get(mode, [None, None])
        assert self.matrix is not None, "mode must be in 'train', 'val', or 'test'"

        self.nonzero_indices = torch.nonzero(
            self.matrix < bg_percent
        )  # [num_slices, 3] - (file_idx, direction_idx, slice_idx)

        assert self.nonzero_indices.shape[0] <= torch.numel(
            self.matrix
        ), "select bg slice count cannot be more than total slice count"

    def __getitem__(self, index):
        file_idx, direction_idx, slice_idx = self.nonzero_indices[index]

        feature_file, label_file = self.feature_label_files[file_idx]

        feature_vol = torch.from_numpy(
            nib.load(feature_file).get_fdata().astype(np.float32)
        )
        label_vol = torch.from_numpy(nib.load(label_file).get_fdata().astype(np.int16))

        feature_slice = torch.index_select(feature_vol, direction_idx, slice_idx)
        label_slice = torch.index_select(label_vol, direction_idx, slice_idx)

        return (
            feature_slice.squeeze().unsqueeze(0),
            label_slice.squeeze().unsqueeze(0),
        )

    def __len__(self):
        return self.nonzero_indices.shape[0]


if __name__ == "__main__":
    main()
    dataset = SampleDataset(mode="train", bg_percent=0.8)
    a, b = dataset[0]
    print(a.shape, b.shape)
    a, b = dataset[4688]
    print(a.shape, b.shape)
