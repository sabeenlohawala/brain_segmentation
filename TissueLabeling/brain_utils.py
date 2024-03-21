import csv
import os
from typing import Tuple

import nibabel as nib
import numpy as np
import torch


def load_brains(image_file: str, mask_file: str, file_path: str):
    # ensure that mask and image numbers match
    image_nr = image_file.split("_")[1]
    mask_nr = mask_file.split("_")[1]
    assert image_nr == mask_nr, "image and mask numbers do not match"

    image_path = os.path.join(file_path, image_file)
    mask_path = os.path.join(file_path, mask_file)

    brain = nib.load(image_path)
    brain_mask = nib.load(mask_path)

    brain = brain.get_fdata()
    brain_mask = brain_mask.get_fdata()
    brain_mask = brain_mask.astype(int)
    # apply skull stripping
    brain[brain_mask == 0] = 0

    return brain, brain_mask, image_nr


def crop(image: np.array, height: int, width: int) -> np.array:
    # find image-optimal crop
    for j in range(256):
        if (image[j] != 0).any():
            cut_top_temp = j
            break
    for j in range(256):
        if (image[255 - j] != 0).any():
            cut_bottom_temp = 255 - j
            break
    for j in range(256):
        if (image[:, j] != 0).any():
            cut_left_temp = j
            break
    for j in range(256):
        if (image[:, 255 - j] != 0).any():
            cut_right_temp = 255 - j
            break

    # image-optimal size:
    height_temp = cut_bottom_temp - cut_top_temp + 1
    width_temp = cut_right_temp - cut_left_temp + 1
    assert height_temp <= height, "Crop height is too big"
    assert width_temp <= width, "Crop width is too big"

    # crop image-optimal patch:
    image = image[
        cut_top_temp : cut_bottom_temp + 1, cut_left_temp : cut_right_temp + 1
    ]

    if image.shape[0] < 50 or image.shape[1] < 50:
        pass

    assert (image > 0).any(), "Crop is empty"

    # adjust the crop to largest rectangle
    if height_temp < height:
        diff = height - height_temp
        # even difference
        if (diff % 2) == 0:
            image = np.pad(image, ((diff // 2, diff // 2), (0, 0)))
        # odd difference
        else:
            image = np.pad(image, ((diff // 2, diff // 2 + 1), (0, 0)))
    if width_temp < width:
        diff = width - width_temp
        # even difference
        if (diff % 2) == 0:
            image = np.pad(image, ((0, 0), (diff // 2, diff // 2)))
        # odd difference
        else:
            image = np.pad(image, ((0, 0), (diff // 2, diff // 2 + 1)))

    return image


def brain_coord(slice: torch.tensor) -> Tuple[int, int, int, int]:
    """
    Computes the coordnates of a rectangle
    that contains the brain area of a slice

    Args:
        slice (torch.tensor): brain slice of shape [H, W]

    Returns:
        Tuple(int, int, int, int): Coordinates of the rectangle
    """

    H, W = slice.shape[-2:]
    H, W = H - 1, W - 1
    # majority vote of the 4 corners
    bg_value = torch.stack(
        (slice[0, 0], slice[-1, -1], slice[0, -1], slice[-1, 0])
    ).mode()[0]

    for j in range(H):
        if (slice[j] != bg_value).any():
            cut_top_temp = j
            break
    for j in range(H):
        if (slice[H - j] != bg_value).any():
            cut_bottom_temp = H - j
            break
    for j in range(W):
        if (slice[:, j] != bg_value).any():
            cut_left_temp = j
            break
    for j in range(W):
        if (slice[:, W - j] != bg_value).any():
            cut_right_temp = W - j
            break

    return (cut_top_temp, cut_bottom_temp, cut_left_temp, cut_right_temp)


def brain_area(slice: torch.tensor) -> torch.tensor:
    """
    Computes the brain area of a slice

    Args:
        slice (torch.tensor): brain slice

    Returns:
        torch.tensor: brain area
    """

    cut_top_temp, cut_bottom_temp, cut_left_temp, cut_right_temp = brain_coord(slice)

    return slice[cut_top_temp : cut_bottom_temp + 1, cut_left_temp : cut_right_temp + 1]


def mapping(mask: np.array, nr_of_classes=51, original=True):
    # if original == True, map from original --> num-class column
    # if original == False, map from index --> num-class column

    # TODO: handle binary case!!

    class_mapping = {}
    # labels = []
    with open(
        "/om2/user/sabeen/nobrainer_data_norm/class_mapping.csv", newline=""
    ) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=",", quotechar="|")
        # skip header
        header = next(spamreader, None)
        original_index = header.index("original")  # original numbered segments
        new_index = header.index("index")  # 107-class numbered segments
        map_index = original_index if original else new_index

        col_index = (
            new_index
            if nr_of_classes == 107
            else (
                header.index("2-class")
                if nr_of_classes == 2
                else header.index(f"{nr_of_classes-1}-class")
            )
        )

        for row in spamreader:
            class_mapping[int(row[map_index])] = int(row[col_index])

    # class_mapping = {value.item(): index for index, value in enumerate(labels)}
    u, inv = np.unique(mask, return_inverse=True)
    # num_classes = 50  # len(class_mapping)
    for x in u:
        if x not in class_mapping:
            class_mapping[x] = nr_of_classes - 1

    # we collect all classes not in the mapping table as an additional "other" class
    # mask = np.array([class_mapping[int(x)] if x in labels else len(labels) for x in u])[inv].reshape(mask.shape)
    for old, new in class_mapping.items():
        mask[mask == old] = -1 * new
    mask = mask * -1

    return mask


def create_affine_transformation_matrix(
    n_dims, scaling=None, rotation=None, shearing=None, translation=None
):
    """
    From https://github.com/MGH-LEMoN/photo-reconstruction/blob/main/scripts/hcp_replicate_photos.py#L85C40-L85C40.
    Create a 4x4 affine transformation matrix from specified values
    :param n_dims: integer
    :param scaling: list of 3 scaling values
    :param rotation: list of 3 angles (degrees) for rotations around 1st, 2nd, 3rd axis
    :param shearing: list of 6 shearing values
    :param translation: list of 3 values
    :return: 4x4 numpy matrix
    """

    T_scaling = np.eye(n_dims + 1)
    T_shearing = np.eye(n_dims + 1)
    T_translation = np.eye(n_dims + 1)

    if scaling is not None:
        T_scaling[np.arange(n_dims + 1), np.arange(n_dims + 1)] = np.append(scaling, 1)

    if shearing is not None:
        shearing_index = np.ones((n_dims + 1, n_dims + 1), dtype="bool")
        shearing_index[np.eye(n_dims + 1, dtype="bool")] = False
        shearing_index[-1, :] = np.zeros((n_dims + 1))
        shearing_index[:, -1] = np.zeros((n_dims + 1))
        T_shearing[shearing_index] = shearing

    if translation is not None:
        T_translation[
            np.arange(n_dims), n_dims * np.ones(n_dims, dtype="int")
        ] = translation

    if n_dims == 2:
        if rotation is None:
            rotation = np.zeros(1)
        else:
            rotation = np.asarray(rotation) * (np.pi / 180)
        T_rot = np.eye(n_dims + 1)
        T_rot[np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1])] = [
            np.cos(rotation[0]),
            np.sin(rotation[0]),
            np.sin(rotation[0]) * -1,
            np.cos(rotation[0]),
        ]
        return T_translation @ T_rot @ T_shearing @ T_scaling

    else:
        if rotation is None:
            rotation = np.zeros(n_dims)
        else:
            rotation = np.asarray(rotation) * (np.pi / 180)
        T_rot1 = np.eye(n_dims + 1)
        T_rot1[np.array([1, 2, 1, 2]), np.array([1, 1, 2, 2])] = [
            np.cos(rotation[0]),
            np.sin(rotation[0]),
            np.sin(rotation[0]) * -1,
            np.cos(rotation[0]),
        ]
        T_rot2 = np.eye(n_dims + 1)
        T_rot2[np.array([0, 2, 0, 2]), np.array([0, 0, 2, 2])] = [
            np.cos(rotation[1]),
            np.sin(rotation[1]) * -1,
            np.sin(rotation[1]),
            np.cos(rotation[1]),
        ]
        T_rot3 = np.eye(n_dims + 1)
        T_rot3[np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1])] = [
            np.cos(rotation[2]),
            np.sin(rotation[2]),
            np.sin(rotation[2]) * -1,
            np.cos(rotation[2]),
        ]
        return T_translation @ T_rot3 @ T_rot2 @ T_rot1 @ T_shearing @ T_scaling


def draw_value_from_distribution(
    hyperparameter,
    size=1,
    distribution="uniform",
    centre=0.0,
    default_range=10.0,
    positive_only=False,
    return_as_tensor=False,
    batchsize=None,
):
    """Sample values from a uniform, or normal distribution of given hyperparameters.
    These hyperparameters are to the number of 2 in both uniform and normal cases.
    :param hyperparameter: values of the hyperparameters. Can either be:
    1) None, in each case the two hyperparameters are given by [center-default_range, center+default_range],
    2) a number, where the two hyperparameters are given by [centre-hyperparameter, centre+hyperparameter],
    3) a sequence of length 2, directly defining the two hyperparameters: [min, max] if the distribution is uniform,
    [mean, std] if the distribution is normal.
    4) a numpy array, with size (2, m). In this case, the function returns a 1d array of size m, where each value has
    been sampled independently with the specified hyperparameters. If the distribution is uniform, rows correspond to
    its lower and upper bounds, and if the distribution is normal, rows correspond to its mean and std deviation.
    5) a numpy array of size (2*n, m). Same as 4) but we first randomly select a block of two rows among the
    n possibilities.
    6) the path to a numpy array corresponding to case 4 or 5.
    7) False, in which case this function returns None.
    :param size: (optional) number of values to sample. All values are sampled independently.
    Used only if hyperparameter is not a numpy array.
    :param distribution: (optional) the distribution type. Can be 'uniform' or 'normal'. Default is 'uniform'.
    :param centre: (optional) default centre to use if hyperparameter is None or a number.
    :param default_range: (optional) default range to use if hyperparameter is None.
    :param positive_only: (optional) whether to reset all negative values to zero.
    :param return_as_tensor: (optional) whether to return the result as a tensorflow tensor
    :param batchsize: (optional) if return_as_tensor is true, then you can sample a tensor of a given batchsize. Give
    this batchsize as a tensorflow tensor here.
    :return: a float, or a numpy 1d array if size > 1, or hyperparameter is itself a numpy array.
    Returns None if hyperparameter is False.
    """

    # return False is hyperparameter is False
    if hyperparameter is False:
        return None

    # reformat parameter_range
    # hyperparameter = load_array_if_path(hyperparameter, load_as_numpy=True)
    hyperparameter = None
    if not isinstance(hyperparameter, np.ndarray):
        if hyperparameter is None:
            hyperparameter = np.array(
                [[centre - default_range] * size, [centre + default_range] * size]
            )
        elif isinstance(hyperparameter, (int, float)):
            hyperparameter = np.array(
                [[centre - hyperparameter] * size, [centre + hyperparameter] * size]
            )
        elif isinstance(hyperparameter, (list, tuple)):
            assert (
                len(hyperparameter) == 2
            ), "if list, parameter_range should be of length 2."
            hyperparameter = np.transpose(np.tile(np.array(hyperparameter), (size, 1)))
        else:
            raise ValueError(
                "parameter_range should either be None, a number, a sequence, or a numpy array."
            )
    elif isinstance(hyperparameter, np.ndarray):
        assert (
            hyperparameter.shape[0] % 2 == 0
        ), "number of rows of parameter_range should be divisible by 2"
        n_modalities = int(hyperparameter.shape[0] / 2)
        modality_idx = 2 * np.random.randint(n_modalities)
        hyperparameter = hyperparameter[modality_idx : modality_idx + 2, :]

    # draw values as tensor
    if return_as_tensor:
        print("dont return as tensor?")
        parameter_value = None
        # shape = KL.Lambda(lambda x: tf.convert_to_tensor(hyperparameter.shape[1], 'int32'))([])
        # if batchsize is not None:
        #     shape = KL.Lambda(lambda x: tf.concat([x[0], tf.expand_dims(x[1], axis=0)], axis=0))([batchsize, shape])
        # if distribution == 'uniform':
        #     parameter_value = KL.Lambda(lambda x: tf.random.uniform(shape=x,
        #                                                             minval=hyperparameter[0, :],
        #                                                             maxval=hyperparameter[1, :]))(shape)
        # elif distribution == 'normal':
        #     parameter_value = KL.Lambda(lambda x: tf.random.normal(shape=x,
        #                                                            mean=hyperparameter[0, :],
        #                                                            stddev=hyperparameter[1, :]))(shape)
        # else:
        #     raise ValueError("Distribution not supported, should be 'uniform' or 'normal'.")

        # if positive_only:
        #     parameter_value = KL.Lambda(lambda x: K.clip(x, 0, None))(parameter_value)

    # draw values as numpy array
    else:
        if distribution == "uniform":
            parameter_value = np.random.uniform(
                low=hyperparameter[0, :], high=hyperparameter[1, :]
            )
        elif distribution == "normal":
            parameter_value = np.random.normal(
                loc=hyperparameter[0, :], scale=hyperparameter[1, :]
            )
        else:
            raise ValueError(
                "Distribution not supported, should be 'uniform' or 'normal'."
            )

        if positive_only:
            parameter_value[parameter_value < 0] = 0

    return parameter_value
