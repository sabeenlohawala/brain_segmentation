import os
import glob
import h5py as h5
import sys
import argparse

import numpy as np
from multiprocessing import Pool

from TissueLabeling.utils import main_timer

parser = argparse.ArgumentParser()
parser.add_argument(
    "h5_dir", 
    help="Where the hdf5 chunks are saved", 
    type=str
)
parser.add_argument(
    "shard_idx",
    help="Which hdf5 shard to do this for",
    type=int,
)
parser.add_argument(
    "slice_nonbrain_dir",
    help="Where the slice_nonbrain.npys are saved",
    type=str,
)
args = parser.parse_args()

# H5_PATHS = '/om2/scratch/Sat/satra/'
H5_DIR = args.h5_dir
SHARD_IDX = args.shard_idx
SAVE_DIR = args.slice_nonbrain_dir

gettrace = getattr(sys, "gettrace", None)
DEBUG = True if gettrace() else False

h5_file_paths = glob.glob(os.path.join(H5_DIR, '*.h5'))
h5_pointers = [h5.File(h5_path,'r') for h5_path in h5_file_paths]

def get_vol_nonzero(shard_idx,vol_shape,shard_vol_idx):
    print(f'Processing shard {shard_idx} volume {shard_vol_idx}')
    f = h5_pointers[shard_idx]
    vol_shape = f['labels_axis0'].shape
    slices_nonbrain = np.ones((3,256), dtype=np.uint16)
    for axis in range(3):
        for slice_idx in range(vol_shape[axis+1]):
            indices = [shard_vol_idx, slice(None), slice(None)]
            indices.insert(axis + 1, slice_idx)
            img = f[f'labels_axis{axis}'][tuple(indices)]
            slices_nonbrain[axis,slice_idx] = min(np.sum(img==0), 65535)
    return slices_nonbrain

@main_timer
def get_shard_nonzero(shard_idx):
    # slices_nonbrain = np.ones((1150,3,256), dtype=np.uint16)
    vol_shape = h5_pointers[shard_idx]['labels_axis0'].shape
    # for shard_vol_idx in range(vol_shape[0]):
    n_procs = 1 if DEBUG else len(os.sched_getaffinity(0))
    print(f'N PROC {n_procs}')
    with Pool(processes=n_procs) as pool:
        slices_nonbrain = pool.starmap(
            get_vol_nonzero,
            [
                (
                    shard_idx,
                    vol_shape,
                    shard_vol_idx
                )
                for shard_vol_idx in range(vol_shape[0])
            ],
        )
    return np.stack(slices_nonbrain)

def main():
    h5_file_paths = glob.glob(os.path.join(H5_DIR, '*.h5'))
    h5_pointers = []
    for h5_path in h5_file_paths:
        h5_pointers.append(h5.File(h5_path,'r'))

    # slices_nonbrain = np.ones((len(h5_file_paths),3,h5_pointers[0][f'labels_axis0'].shape[0],256), dtype=np.uint16)
    # slices_nonbrain = np.ones((len(h5_file_paths),h5_pointers[0][f'labels_axis0'].shape[0],3,256),dtype=np.uint16) # [num_shards, chunk_size, num_axes, num_slices]
    slices_nonbrain = []

    # for shard_idx, f in enumerate(h5_pointers[:3]):
    slices_nonbrain.append(get_shard_nonzero(SHARD_IDX))

    # save np.stack(...)
    np.save(f'{SAVE_DIR}/slice_nonbrain_{SHARD_IDX:02d}',np.stack(slices_nonbrain))

if __name__ == "__main__":
    main()