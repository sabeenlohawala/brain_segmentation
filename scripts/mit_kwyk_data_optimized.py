import glob
import os
from datetime import datetime
from multiprocessing import Pool

import nibabel as nib
import numpy as np
import torch
from sklearn.model_selection import train_test_split

DATA_DIR = "/nese/mit/group/sig/data/kwyk/rawdata"
SAVE_DIR = "/om2/user/sabeen/kwyk_data"
SAVE_NAME = "output_10.npy"
N_VOLS = 10  # number of volumes to load (this is only for testing)


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
    label_vol = (nib.load(label_file).get_fdata().astype(np.int16))

    bgcount_ax0 = np.sum(label_vol == 0, axis=(1,2)) / 256**2
    bgcount_ax1 = np.sum(label_vol == 0, axis=(0,2)) / 256**2
    bgcount_ax2 = np.sum(label_vol == 0, axis=(0,1)) / 256**2

    bg_count = np.stack((bgcount_ax0, bgcount_ax1, bgcount_ax2), axis=0)
    return bg_count


@main_timer
def main():
    # feature_files = sorted(glob.glob(os.path.join(DATA_DIR, "*orig*.nii.gz")))[:N_VOLS]
    label_files = sorted(glob.glob(os.path.join(DATA_DIR, "*aseg*.nii.gz")))[:N_VOLS]

    nprocs = len(os.sched_getaffinity(0))
    with Pool(processes=nprocs) as p:
        output = p.map(
            calculate_bg, label_files
        ) 

    final_output = np.dstack(output)  # [num_slices, num_directions, num_files]
    final_output = np.moveaxis(
        final_output, -1, 0
    )  # [num_files, num_directions, num_slices]
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    np.save(os.path.join(SAVE_DIR, SAVE_NAME), final_output)


class SampleDataset(torch.utils.data.Dataset):
    def __init__(self, mode, volume_data_dir, slice_info_file, bg_percent=0.99):
        self.matrix = np.load(data_file, allow_pickle=True)
        self.matrix = torch.Tensor(self.matrix)

        self.feature_label_files = list(zip(
            sorted(glob.glob(os.path.join(DATA_DIR, "*orig*.nii.gz")))[:self.matrix.shape[0]],
            sorted(glob.glob(os.path.join(DATA_DIR, "*aseg*.nii.gz")))[:self.matrix.shape[0]]
        ))

        # perform train-val-test split and apply to self.matrix and self.feature_label_files
        indices = np.arange(self.matrix.shape[0])
        train_indices, remaining_indices = train_test_split(indices, test_size=0.2, random_state=42)
        validation_indices, test_indices = train_test_split(remaining_indices, test_size=0.5, random_state=42)
        if mode == 'train':
            keep_indices = train_indices
        elif mode == 'val' or mode == 'validation':
            keep_indices = validation_indices
        elif mode == 'test':
            keep_indices = test_indices
        else:
            raise Exception(
                f"{mode} is not a valid data mode. Choose from 'train', 'test', or 'validation'."
            )
        self.matrix = self.matrix[keep_indices]
        self.feature_label_files = [feature_label_pair for i, feature_label_pair in enumerate(self.feature_label_files) if i in keep_indices]

        self.filtered_matrix = self.matrix < bg_percent
        self.nonzero_indices = torch.nonzero(
            self.filtered_matrix
        )  # [num_slices, 3] - (file_idx, direction_idx, slice_idx)

        assert self.nonzero_indices.shape[0] <= torch.numel(
            self.matrix
        ), "select bg slice count cannot be more than total slice count"

    def __getitem__(self, index):
        file_idx, direction_idx, slice_idx = self.nonzero_indices[index]

        feature_file, label_file = self.feature_label_files[file_idx]

        feature_vol = torch.from_numpy(nib.load(feature_file).get_fdata().astype(np.float32))
        label_vol = torch.from_numpy(nib.load(label_file).get_fdata().astype(np.int16))

        # do you want a cropped slice (sure do it here, of course you still need that fixed number)

        # all the above 3 if conditions can be written in a single line:
        feature_slice = torch.index_select(
            feature_vol, direction_idx, torch.Tensor(slice_idx)
        )
        label_slice = torch.index_select(
            label_vol, direction_idx, torch.Tensor(slice_idx)
        )

        # TODO:
        # i now wonder if rotating as well can be dont at get time...probably not advisable unless you have torch functions to do the same
        # (if you care) see nitorch (written by yael) or neurite (written by adrian)

        return (feature_slice, label_slice)

    def __len__(self):
        return self.nonzero_indices.shape[0]


if __name__ == "__main__":
    # main()
    dataset = SampleDataset(mode='validation',data_dir=os.path.join(SAVE_DIR,SAVE_NAME), bg_percent=0.8)
    a, b = dataset[0]
    print(a.shape, b.shape)
