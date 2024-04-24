import glob
import os
from datetime import datetime
from multiprocessing import Pool

import nibabel as nib
import numpy as np
import torch

DATA_DIR = "/nese/mit/group/sig/data/kwyk/rawdata"
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


def calculate_bg(feature_file):
    print(os.path.basename(feature_file))
    feature_vol = (
        nib.load(feature_file).get_fdata().astype(np.uint)
    )  # btw shouldn't this be done on label maps?

    bgcount_ax1 = [
        np.count_nonzero(feature_vol[i, :, :] == 0) for i in range(256)
    ]  # TODO: bg percentages
    bgcount_ax2 = [
        np.count_nonzero(feature_vol[:, i, :] == 0) for i in range(256)
    ]  # TODO: optimizing this if you can and want
    bgcount_ax3 = [np.count_nonzero(feature_vol[:, :, i] == 0) for i in range(256)]

    # is the above the same as (Please check for me): if they match then for loop is not needed
    # bgcount_ax1 = np.sum(np.count_nonzero(feature_vol == 0, axis=1), axis=2)  # for axis 0
    # bgcount_ax2 = np.sum(np.count_nonzero(feature_vol == 0, axis=0), axis=2)  # for axis 1
    # bgcount_ax3 = np.sum(np.count_nonzero(feature_vol == 0, axis=0), axis=1)  # for axis 2

    bg_count = np.stack((bgcount_ax1, bgcount_ax2, bgcount_ax3), axis=0)
    return bg_count


@main_timer
def main():
    feature_files = sorted(glob.glob(os.path.join(DATA_DIR, "*orig*.nii.gz")))[:N_VOLS]
    label_files = sorted(glob.glob(os.path.join(DATA_DIR, "*aseg*.nii.gz")))[:N_VOLS]

    nprocs = len(os.sched_getaffinity(0))
    with Pool(processes=nprocs) as p:
        output = p.map(
            calculate_bg, feature_files
        )  # passing in labels is redundant... you know how to change this

    final_output = np.dstack(output)  # [num_slices, num_directions, num_files]
    final_output = np.moveaxis(
        final_output, -1, 0
    )  # [num_files, num_directions, num_slices]
    np.save("output.npy", final_output)


class SampleDataset(torch.utils.data.Dataset):
    def __init__(self, bg_count=30000):  # change this to percent
        self.matrix = np.load("output.npy", allow_pickle=True)
        self.matrix = torch.Tensor(self.matrix)

        self.filtered_matrix = self.matrix > bg_count
        self.feature_vols = sorted(glob.glob(os.path.join(DATA_DIR, "*orig*.nii.gz")))[
            :N_VOLS
        ]
        self.label_vols = sorted(glob.glob(os.path.join(DATA_DIR, "*aseg*.nii.gz")))[
            :N_VOLS
        ]
        # the above 2 lines can be a tuple if you prefer

        self.nz_indices = torch.nonzero(
            self.filtered_matrix
        )  # [num_slices, 3] - (file_idx, direction_idx, slice_idx)

        assert self.nz_indices.shape[0] <= torch.numel(
            self.matrix
        ), "select bg slice count cannot be more than total slice count"

    def __getitem__(self, index):
        file_idx, direction_idx, slice_idx = self.nz_indices[index]

        feature_vol = self.feature_vols[file_idx]
        label_vol = self.label_vols[file_idx]

        feature_vol = torch.from_numpy(nib.load(feature_vol).get_fdata())
        label_vol = torch.from_numpy(nib.load(label_vol).get_fdata())

        # do you want a cropped slice (sure do it here, of course you still need that fixed number)

        # if direction_idx == 0:
        #     feature_slice = feature_vol[slice_idx, :, :]
        #     label_slice = label_vol[slice_idx, :, :]

        # if direction_idx == 1:
        #     feature_slice = feature_vol[:, slice_idx, :]
        #     label_slice = label_vol[:, slice_idx, :]

        # if direction_idx == 2:
        #     feature_slice = feature_vol[:, :, slice_idx]
        #     label_slice = label_vol[:, :, slice_idx]

        # all the above 3 if conditions can be written in a single line:
        feature_slice = torch.index_select(
            feature_vol, direction_idx, torch.Tensor(slice_idx)
        )
        label_slice = torch.index_select(
            label_vol, direction_idx, torch.Tensor(slice_idx)
        )

        # TODO:
        # where and how to do the train/val/test split: on the npy matrix that you loaded. agree?
        # i now wonder if rotating as well can be dont at get time...probably not advisable unless you have torch functions to do the same
        # (if you care) see nitorch (written by yael) or neurite (written by adrian)

        return (feature_slice, label_slice)

    def __len__(self):
        return self.nz_indices.shape[0]


if __name__ == "__main__":
    main()
    # dataset = SampleDataset()
    # a, b = dataset[0]
    # print(a.shape, b.shape)
