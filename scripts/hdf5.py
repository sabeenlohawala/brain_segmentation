import glob

import h5py as h5
import nibabel as nib
import numpy as np
from TissueLabeling.utils import main_timer
from multiprocessing import Pool

SAVE_NAME = "/om2/user/sabeen/kwyk_data/satra.h5"
@main_timer
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


if __name__ == "__main__":
    write_kwyk_hdf5()
    read_kwyk_hdf5()
    # put assertion for same scale factor for all files