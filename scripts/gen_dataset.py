import glob
import os
from pathlib import Path

import h5py as h5
import nibabel as nib
import numpy as np
import pydra
from pydra import mark
from pydra.engine.specs import File


def write_kwyk_data(feature_files: list[File],
                    label_files: list[File],
                    save_path: str,
                    comp: int=2) -> File:
    """Write an HDF5 dataset with slices chunked for read efficiency"""

    N_VOLS = len(feature_files)
    feature_opts = {'dtype': np.uint8,
                    'shape': (N_VOLS, 256, 256, 256),
                    'compression': comp}
    label_opts = feature_opts.copy()
    label_opts.update({'dtype': np.uint16})

    f = h5.File(save_path, "w")
    feature_ds = []
    label_ds = []
    for idx, chunks in enumerate([(1, 1, 256, 256),
                                  (1, 256, 1, 256),
                                  (1, 256, 256, 1)]):
        feature_ds.append(f.create_dataset(f"features_axis{idx}", chunks=chunks, **feature_opts))
        label_ds.append(f.create_dataset(f"labels_axis{idx}", chunks=chunks, **label_opts))
    for idx, fname in enumerate(feature_files):
        feature = nib.load(fname).dataobj
        label = nib.load(label_files[idx]).dataobj
        # add vol rotation here
        for ds_idx in range(len(feature_ds)):
            feature_ds[ds_idx][idx] = feature
            label_ds[ds_idx][idx] = label

    f.close()
    return save_path

if __name__ == "__main__":
    OUT_DIR = f"/om2/scratch/Sat/{os.environ['USER']}"  # CHECK THIS
    os.makedirs(OUT_DIR, exist_ok=True)

    CHUNK_HDF5_template = os.path.join(OUT_DIR, "kwyk_chunk_{shardidx:02d}.h5")
    NIFTI_DIR = "/om2/scratch/Sat/satra/rawdata/"  # DO NOT CHANGE
    
    feature_files = sorted(glob.glob(os.path.join(NIFTI_DIR, "*orig*")))
    label_files = sorted(glob.glob(os.path.join(NIFTI_DIR, "*aseg*")))

    N = len(feature_files)
    DEBUG = False
    if DEBUG:
        N_VOLS_PER_SHARD = 10
        N_SHARDS = 5
    else:
        N_VOLS_PER_SHARD = 1150
        N_SHARDS = int(np.ceil(N / N_VOLS_PER_SHARD))
    features = []
    labels = []
    paths = []

    print(f"Processing {N} files with {N_VOLS_PER_SHARD} per shard into {N_SHARDS} shards")
    for idx in range(N_SHARDS):
        paths.append(CHUNK_HDF5_template.format(shardidx=idx))
        start = idx * N_VOLS_PER_SHARD
        end = min((idx + 1) * N_VOLS_PER_SHARD, N)
        features.append(feature_files[start:end])
        labels.append(label_files[start:end])
        #write_kwyk_data(features[-1], labels[-1], paths[-1])
        
    write_task_pdt = mark.task(write_kwyk_data)
    cache_dir = (Path(os.getcwd()) / 'wf_cache').absolute()
    write_task = write_task_pdt() # not using cache for the moment
    write_task.split(splitter=('feature_files', 'label_files', 'save_path'),
                     feature_files=features,
                     label_files=labels,
                     save_path=paths)
    with pydra.Submitter(plugin='cf', n_procs=N_SHARDS) as sub:
        sub(runnable=write_task)
