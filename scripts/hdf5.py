import glob

import h5py as h5
import nibabel as nib
import numpy as np
from TissueLabeling.utils import main_timer

@main_timer
def write_kwyk_hdf5():
    N_VOLS = 10
    feature_files = sorted(
        glob.glob("/om2/scratch/Mon/sabeen/kwyk-volumes/rawdata/*orig*")
    )[:N_VOLS]
    label_files = sorted(
        glob.glob("/om2/scratch/Mon/sabeen/kwyk-volumes/rawdata/*aseg*")
    )[:N_VOLS]

    f = h5.File("kwyk.h5", "w")
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

    for idx, feature_file, label_file in enumerate(zip(feature_files, label_files)):
        features[idx, :, :, :] = nib.load(feature_file).dataobj
        labels[idx, :, :, :] = nib.load(label_file).dataobj

@main_timer
def read_kwyk_hdf5():
    kwyk = h5.File("kwyk.h5", "r")
    features = kwyk["kwyk_features"]
    labels = kwyk["kwyk_labels"]

    for feature, label in zip(features, labels):
        _, _ = feature.shape, label_shape

    print("success")

if __name__ == "__main__":
    write_kwyk_hdf5()
    read_kwyk_hdf5()
    # put assertion for same scale factor for all files