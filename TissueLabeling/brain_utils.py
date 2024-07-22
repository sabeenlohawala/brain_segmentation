"""
File: brain_utils.py
Author: Sabeen Lohawala
Date: 2024-05-08
Description: This file contains helpful functions for operating on brain images.
"""

import csv
import os
from typing import Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import cv2

def load_brains(image_file: str, mask_file: str, file_path: str):
    """
    Loads the feature and label volumes corresponding to the specified file path and names.
    
    Requires both files to be located in the directory specified by file_path and for the file nameing
    convention to be something like brain_04.nii.gz and mask_04.nii.gz where the ending number of the brain
    and mask files is used to determine which files belong together.

    Args:
        image_file (str): filename of the feature volume
        mask_file (str): filename of the label volume
        file_path (str): directory where both volumes are located
    
    Returns:
        brain (torch.Tensor): the loaded skull-stripped feature volume
        brain_mask (torch.Tensor): the loaded label volume containing the freesurfer labels
        image_nr (int): the number used to identify the feature and label pair in the filename
    """
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
    """
    Determines whether the image can be cropped to heigh and width without resulting in 
    any loss of tissue pixels.

    Args:
        image (np.array): an array of size (256,256) where all background pixels contain the value 0
                          and all non-background pixels contain a value other than 0
        height (int): the height to which the image should be cropped
        width (int): the width to which the image should be cropped

    Returns:
        image (np.array): the input array cropped to size (height, width)

    Throws:
        AssertionError if the image cannot be cropped to size (height, width) without
        resulting in loss of non-background pixels or the crop results in an all-background image.
    """
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


def mapping(mask: np.array, nr_of_classes=50, reference_col="original", class_mapping=None):
    """
    Maps the labels in the input mask to the labels corresponding to the specified nr_of_classes
    based on a mapping table if a mapping is not passed in.

    Args:
        mask (np.array): a 2D array where each cell contains an integer label that is to be
                         mapped to a new label according to the mapping table
        nr_of_classes (int): the number of classes to segment, which determines the mapping table
        reference_col (str): (optional) the name of the column that corresponds to the labels 
                             in the input mask
        class_mapping (dict | None): (optional) a dictionary containing the mapping table from
                                     reference_col to the nr_of_classes
    
    Returns:
        mask (np.array): The same mask with all pixel values now mapped to new pixel values specified
                         in the mapping table.
        class_mapping (dict): A dictionary containing the mapping, where keys are the old pixel values
                              and the values are the new values to map to.
    """
    if class_mapping is None:
        new_col = "index" if nr_of_classes==106 else f"{nr_of_classes}-class"
        class_mapping = {}
        df = pd.read_csv("/om2/user/sabeen/nobrainer_data_norm/class_mapping.csv")
        for value in df[reference_col].unique():
            filtered_rows = df[df[reference_col] == value]
            numbers = list(set(filtered_rows[new_col].tolist()))
            assert len(numbers) <= 1, "unique mapping does not exist"
            class_mapping[value] = numbers[0]

    # class_mapping = {value.item(): index for index, value in enumerate(labels)}
    u, inv = np.unique(mask, return_inverse=True)
    # num_classes = 50  # len(class_mapping)
    # for x in u:
    #     if x not in class_mapping:
    #         # class_mapping[x] = nr_of_classes - 1
    #         class_mapping[x] = 0

    # we collect all classes not in the mapping table as an additional "other" class
    # mask = np.array([class_mapping[int(x)] if x in labels else len(labels) for x in u])[inv].reshape(mask.shape)
    for old, new in class_mapping.items():
        mask[mask == old] = -1 * new
    mask[mask > 0] = 0 # all unknown classes are mapped to 0
    mask = mask * -1

    return mask, class_mapping

def null_cerebellum_brain_stem(image: np.array, mask: np.array, null_classes = None):
    """
    This function is used to set any pixels belonging to the cerebellum or brain stem to 0 in the image
    and its corresponding mask if this operation does not result in an image containing all 0s.

    Requires that the mask contains the original freesurfer labels.

    Args:
        image (np.array): the feature slice in which to null the cerebellum and brain stem
        mask (np.array): the label slice in which to null the cerebellum and brain stem
        null_classes (list | None): a list containing which labels correspond to the cerebellum and brain stem
    
    Returns:
        null_image (np.array): the feature slice after the cerebellum and brain stem have been nulled
        null_mask (np.array): the label slice after the cerebellum and brain stem have been nulled
        null_classes (list | None): a list containing which labels correspond to the cerebellum and brain stem

    """
    null_image = image.copy()
    null_mask = mask.copy()
    if not null_classes:
        df = pd.read_csv("/om2/user/sabeen/nobrainer_data_norm/class_mapping.csv")
        null_classes = list({df['original'][i] for i in df['index'] if 'cerebellum' in df['label'][i].lower() or 'brain-stem' in df['label'][i].lower()})
    
    null_elts = np.isin(mask,null_classes)
    if np.sum(null_elts) > 0:
        null_image[null_elts] = 0.0
        null_mask[null_elts] = 0

    # prevent all-background samples
    if (null_mask == 0).all():
        null_image = image.copy()
        null_mask = mask.copy()

    return null_image, null_mask, null_classes

def null_half(image: np.array, mask: np.array, keep_left=True, right_classes = None, left_classes = None):
    """
    This function is used to set pixels belonging to one hemisphere of the brain to 0 in the image
    and its corresponding mask if this operation does not result in an image containing all 0s.

    Requires that the mask contains the original freesurfer labels.

    Args:
        image (np.array): the feature slice in which to null half of the brain
        mask (np.array): the label slice in which to null half of the brain
        keep_left (bool): a flag to indicate whether to null the left hemisphere
        right_classes (list | None): a list containing which labels correspond to right hemisphere labels
        left_classes (list | None): a list containing which labels correspond to left hemisphere labels
    
    Returns:
        null_image (np.array): the feature slice after half of the brain has been nulled
        null_mask (np.array): the label slice after half of the brain has been nulled
        right_classes (list | None): a list containing which labels correspond to right hemisphere labels
        left_classes (list | None): a list containing which labels correspond to left hemisphere labels

    """
    null_image = image.copy()
    null_mask = mask.copy()
    if not right_classes or not left_classes:
        df = pd.read_csv("/om2/user/sabeen/nobrainer_data_norm/class_mapping.csv")
        right_classes = list({df['original'][i] for i in df['index'] if 'Right' in df['label'][i] or '-rh-' in df['label'][i]})
        left_classes = list({df['original'][i] for i in df['index'] if 'Left' in df['label'][i] or '-lh-' in df['label'][i]})

    # only null half for slices that have labels from both halves
    mask_set = np.unique(mask)
    if np.isin(mask_set, right_classes).any() and np.isin(mask_set, left_classes).any():
        null_classes = right_classes if keep_left else left_classes
        null_elts = np.isin(mask, null_classes)
        null_image[null_elts] = 0.0
        null_mask[null_elts] = 0
    
    # prevent all-background samples
    if (null_mask == 0).all():
        null_image = image.copy()
        null_mask = mask.copy()
    
    return null_image, null_mask, right_classes, left_classes

def apply_background(image,mask,background):
    """
    This function is used to apply new background behind the tissue on the feature image ONLY.
    Requires that image, mask, and background are all the same shape.

    Args:
        image (np.array): the feature slice on which the new background will be applied
        mask (np.array): the label slice, which will remain unmodified
        background (np.array): the new background for the feature slice

    Returns:
        combined (np.array): the feature slice with the new background
    
    Throws:
        AssertionError if the image and background are not of the same shape
    """
    assert image.shape == background.shape
    combined = np.where(mask == 0, background, image)
    return combined

def draw_random_shapes_background(shape=(256, 256), num_shapes=5):
    """
    This function is used to draw random lines, rectangles, ellipses, and polygons
    on a canvas specified by shape.

    Args:
        shape (tuple): a tuple containing 2 ints specifying the shape of the canvas
        num_shapes (int): the number of random shapes to draw on the canvas
    
    Returns:
        canvas (np.array): the canvas containing the random shapes with all intensity
                           values as floats normalized between 0 to 1.0 instead of 0 to 255
    """
    canvas = np.zeros(shape, dtype=np.uint8)  # Create a blank canvas
    
    for _ in range(num_shapes):
        shape_type = np.random.choice(['line', 'rectangle', 'circle', 'ellipse', 'polygon'])
        color = np.random.randint(0,255)  # Random intensity value between 0 and 1
        
        if shape_type == 'line':
            pt1 = (np.random.randint(0, shape[1]), np.random.randint(0, shape[0]))
            pt2 = (np.random.randint(0, shape[1]), np.random.randint(0, shape[0]))
            cv2.line(canvas, pt1, pt2, color, thickness=np.random.randint(1, 5))
        elif shape_type == 'rectangle':
            pt1 = (np.random.randint(0, shape[1]), np.random.randint(0, shape[0]))
            pt2 = (np.random.randint(pt1[0], shape[1]), np.random.randint(pt1[1], shape[0]))
            cv2.rectangle(canvas, pt1, pt2, color, thickness=-1)#np.random.randint(1, 5))
        elif shape_type == 'circle':
            center = (np.random.randint(0, shape[1]), np.random.randint(0, shape[0]))
            radius = np.random.randint(10, min(shape) // 4)
            cv2.circle(canvas, center, radius, color, thickness=-1)#np.random.randint(1, 5))
        elif shape_type == 'ellipse':
            center = (np.random.randint(0, shape[1]), np.random.randint(0, shape[0]))
            axes = (np.random.randint(10, min(shape) // 4), np.random.randint(10, min(shape) // 4))
            angle = np.random.randint(0, 180)
            cv2.ellipse(canvas, center, axes, angle, 0, 360, color, thickness=-1)#np.random.randint(1, 5))
        elif shape_type == 'polygon':
            num_vertices = np.random.randint(3, 10)
            vertices = np.random.randint(0, shape[1], size=(num_vertices, 2))
            vertices = vertices.reshape((-1, 1, 2))
            cv2.polylines(canvas, [vertices], isClosed=True, color=color, thickness=np.random.randint(1, 5))

    return canvas.astype(np.float32) / 255.0  # Normalize to range [0, 1]

def draw_random_grid_background(shape=(256, 256), intensity_range=(0,1), thickness_range=(1, 5), spacing_range=(5, 20)):
    """
    This function is used to draw a grid background with randomized cell and line intensities, thickness, and spacing.

    Args:
        shape (tuple): a tuple containing 2 ints specifying the shape of the canvas
        intensity_range (tuple): contains two float values between 0 to 1 that specify the range from which the
                                 intensities for the grid cells and lines will be drawn uniformly at random
        thickness_range (tuple): contains two numerical values that specify the range from which the thickness
                                 of the lines of the grid will be drawn uniformly at random
        spacing_range (tuple): contains two numerical values that specify the range from which the spacing between
                               the grid lines (which corresponds to the size of the grid cells) will be drawn
                               uniformly at random
    
    Returns:
        canvas (np.array): the canvas containing the random grid background
    """
    background_intensity = np.random.uniform(*intensity_range)  # Random background intensity
    line_intensity = np.random.uniform(*intensity_range)  # Random line intensity
    line_thickness = np.random.randint(*thickness_range)  # Random line thickness
    grid_spacing = np.random.randint(*spacing_range)  # Random grid spacing
    
    canvas = np.ones(shape, dtype=np.float32) * background_intensity  # Fill the canvas with the background intensity
    
    # Create horizontal lines
    canvas[::grid_spacing, :] = line_intensity
    
    # Create vertical lines
    canvas[:, ::grid_spacing] = line_intensity
    
    return canvas

def draw_random_noise_background(shape=(256,256)):
    """
    This function is used to fill a canvas with random values between 0 and 1.

    Args:
        shape (tuple): a tuple containing 2 ints specifying the shape of the canvas

    """
    return np.random.rand(*shape)

def create_affine_transformation_matrix(
    n_dims, scaling=None, rotation=None, shearing=None, translation=None
):
    """
    From https://github.com/MGH-LEMoN/photo-reconstruction/blob/e788e21c38193bdc0043466b1ee8dc66deb6f54b/ext/lab2im/utils.py#L803.
    Create a 4x4 affine transformation matrix from specified values
    :param n_dims: integer
    :param scaling: list of 3 scaling values
    :param rotation: list of 3 angles (degrees) for rotations around 1st, 2nd, 3rd axis
    :param shearing: list of 6 shearing values
    :param translation: list of 3 values
    :return: 4x4 numpy affine
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
    """
    From https://github.com/MGH-LEMoN/photo-reconstruction/blob/e788e21c38193bdc0043466b1ee8dc66deb6f54b/ext/lab2im/utils.py#L1317.
    Sample values from a uniform, or normal distribution of given hyperparameters.
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


### NOBRAINER FUNCTIONS: CONVERTED FROM TF TO NP ###
# def _get_coordinates(volume_shape):
#     Nx, Ny, Nz = volume_shape
#     x = np.linspace(0, Nx - 1, Nx)
#     y = np.linspace(0, Ny - 1, Ny)
#     z = np.linspace(0, Nz - 1, Nz)
#     xx, yy, zz = np.meshgrid(x,y,z, indexing='ij')
#     coords = np.stack((xx,yy,zz),axis=3)
#     return np.reshape(coords,(-1,3))

# def _warp_coords(affine, volume_shape):
#     coords = _get_coordinates(volume_shape=volume_shape)
#     coor_homog = np.concatenate([coords,np.ones((coords.shape[0],1)).astype(coords.dtype)], axis=1)
#     return (coor_homog @ np.transpose(affine))[..., :3]

# def _get_voxels(volume, coords):
#     """Get voxels from volume at points. These voxels are in a flat tensor."""
#     x = volume.astype(np.float32)
#     coords = coords.astype(np.float32)

#     if len(x.shape) < 3:
#         raise ValueError("`volume` must be at least rank 3")
#     if len(coords.shape) != 2 or coords.shape[1] != 3:
#         raise ValueError("`coords` must have shape `(N, 3)`.")

#     rows, cols, depth, *n_channels = x.shape

#     # Points in flattened array representation.
#     fcoords = coords[:, 0] * cols * depth + coords[:, 1] * depth + coords[:, 2]

#     # Some computed finds are out of range of the image's flattened size.
#     # Zero those so we don't get errors. These points in the volume are filled later.
#     fcoords_size = np.size(fcoords) * 1.0
#     fcoords = np.clip(fcoords, 0, fcoords_size - 1)
#     xflat = np.squeeze(np.reshape(x, [np.prod(x.shape[:3]), -1]))

#     # Reorder image data to transformed space.
#     xflat = np.take(xflat, indices=fcoords.astype(np.int32))

#     # Zero image data that was out of frame.
#     outofframe = (
#         np.any(coords < 0, -1)
#         | (coords[:, 0] > rows)
#         | (coords[:, 1] > cols)
#         | (coords[:, 2] > depth)
#     )

#     if n_channels:
#         outofframe = np.stack([outofframe for _ in range(n_channels[0])], axis=-1)

#     xflat = xflat * np.logical_not(outofframe).astype(xflat.dtype)

#     return xflat

# def _trilinear_interpolation(volume, coords): # MATCHES NOBRAINER RESULTS
#     """Trilinear interpolation.

#     Implemented according to
#     https://en.wikipedia.org/wiki/Trilinear_interpolation#Method
#     https://github.com/Ryo-Ito/spatial_transformer_network/blob/2555e846b328e648a456f92d4c80fce2b111599e/warp.py#L137-L222
#     """
#     volume = volume.astype(np.float32)
#     coords = coords.astype(np.float32)
#     coords_floor = np.floor(coords)

#     shape = volume.shape
#     xlen = shape[0]
#     ylen = shape[1]
#     zlen = shape[2]

#     # Get lattice points. x0 is point below x, and x1 is point above x. Same for y and
#     # z.
#     x0 = coords_floor[:, 0].astype(np.int32)
#     x1 = x0 + 1
#     y0 = coords_floor[:, 1].astype(np.int32)
#     y1 = y0 + 1
#     z0 = coords_floor[:, 2].astype(np.int32)
#     z1 = z0 + 1

#     # Clip values to the size of the volume array.
#     x0 = np.clip(x0, 0, xlen - 1)
#     x1 = np.clip(x1, 0, xlen - 1)
#     y0 = np.clip(y0, 0, ylen - 1)
#     y1 = np.clip(y1, 0, ylen - 1)
#     z0 = np.clip(z0, 0, zlen - 1)
#     z1 = np.clip(z1, 0, zlen - 1)

#     i000 = x0 * ylen * zlen + y0 * zlen + z0
#     i001 = x0 * ylen * zlen + y0 * zlen + z1
#     i010 = x0 * ylen * zlen + y1 * zlen + z0
#     i011 = x0 * ylen * zlen + y1 * zlen + z1
#     i100 = x1 * ylen * zlen + y0 * zlen + z0
#     i101 = x1 * ylen * zlen + y0 * zlen + z1
#     i110 = x1 * ylen * zlen + y1 * zlen + z0
#     i111 = x1 * ylen * zlen + y1 * zlen + z1

#     if len(volume.shape) == 3:
#         volume_flat = np.reshape(volume, [-1])
#     else:
#         volume_flat = np.reshape(volume, [-1, volume.shape[-1]])

#     c000 = np.take(volume_flat, i000)
#     c001 = np.take(volume_flat, i001)
#     c010 = np.take(volume_flat, i010)
#     c011 = np.take(volume_flat, i011)
#     c100 = np.take(volume_flat, i100)
#     c101 = np.take(volume_flat, i101)
#     c110 = np.take(volume_flat, i110)
#     c111 = np.take(volume_flat, i111)

#     xd = coords[:, 0] - x0.astype(np.float32)
#     yd = coords[:, 1] - y0.astype(np.float32)
#     zd = coords[:, 2] - z0.astype(np.float32)

#     # Interpolate along x-axis.
#     c00 = c000 * (1 - xd) + c100 * xd
#     c01 = c001 * (1 - xd) + c101 * xd
#     c10 = c010 * (1 - xd) + c110 * xd
#     c11 = c011 * (1 - xd) + c111 * xd

#     # Interpolate along y-axis.
#     c0 = c00 * (1 - yd) + c10 * yd
#     c1 = c01 * (1 - yd) + c11 * yd

#     c = c0 * (1 - zd) + c1 * zd

#     return np.reshape(c, volume.shape)

# def _nearest_neighbor_interpolation(volume, coords): #MATCHES NOBRAINER VERSION
#     """Three-dimensional nearest neighbors interpolation."""
#     volume_f = _get_voxels(volume=volume, coords=np.round(coords))
#     return np.reshape(volume_f, volume.shape)

# def warp_features_labels(features, labels, affine, scalar_label=False):
#     """Warp features and labels tensors according to affine matrix.

#     Trilinear interpolation is used for features, and nearest neighbor
#     interpolation is used for labels.

#     Parameters
#     ----------
#     features: Rank 3 tensor, volumetric feature data.
#     labels: Rank 3 tensor or N
#     affine: Tensor with shape `(4, 4)`, affine affine.

#     Returns
#     -------
#     Tuple of warped features, warped labels.
#     """
#     # features = tf.convert_to_tensor(features)
#     # labels = tf.convert_to_tensor(labels)

#     warped_coords = _warp_coords(affine=affine, volume_shape=features.shape)
#     features = _trilinear_interpolation(volume=features, coords=warped_coords)
#     if not scalar_label:
#         labels = _nearest_neighbor_interpolation(volume=labels, coords=warped_coords)
#     return (features, labels)

def get_affine(volume_shape, rotation=[0, 0, 0], translation=[0, 0, 0]):
    """Return 4x4 affine, which encodes rotation and translation of 3D tensors.

    Parameters
    ----------
    rotation: iterable of three numbers, the yaw, pitch, and roll,
        respectively, in radians.
    translation: iterable of three numbers, the number of voxels to translate
        in the x, y, and z directions.

    Returns
    -------
    Tensor with shape `(4, 4)` and dtype float32.
    """
    volume_shape = np.array(volume_shape).astype(np.float32)
    rotation = np.array(rotation).astype(np.float32)
    translation = np.array(translation).astype(np.float32)
    if volume_shape.shape[0] < 3:
        raise ValueError("`volume_shape` must have at least three values")
    if rotation.shape[0] != 3:
        raise ValueError("`rotation` must have three values")
    if translation.shape[0] != 3:
        raise ValueError("`translation` must have three values")

    # ROTATION
    # yaw
    rx = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(rotation[0]), -np.sin(rotation[0]), 0],
            [0, np.sin(rotation[0]), np.cos(rotation[0]), 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32
    )

    # pitch
    ry = np.array(
        [
            [np.cos(rotation[1]), 0, np.sin(rotation[1]), 0],
            [0, 1, 0, 0],
            [-np.sin(rotation[1]), 0, np.cos(rotation[1]), 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32
    )

    # roll
    rz = np.array(
        [
            [np.cos(rotation[2]), -np.sin(rotation[2]), 0, 0],
            [np.sin(rotation[2]), np.cos(rotation[2]), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32
    )

    # Rotation around origin.
    transform = rz @ ry @ rx

    center = (volume_shape[:3] / 2 - 0.5).astype(np.float32)
    neg_center = -1 * center
    center_to_origin = np.array(
        [
            [1, 0, 0, neg_center[0]],
            [0, 1, 0, neg_center[1]],
            [0, 0, 1, neg_center[2]],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )

    origin_to_center = np.array(
        [
            [1, 0, 0, center[0]],
            [0, 1, 0, center[1]],
            [0, 0, 1, center[2]],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )

    # Rotation around center of volume.
    transform = origin_to_center @ transform @ center_to_origin

    # TRANSLATION
    translation = np.array(
        [
            [1, 0, 0, translation[0]],
            [0, 1, 0, translation[1]],
            [0, 0, 1, translation[2]],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )

    transform = translation @ transform

    # REFLECTION
    #
    # TODO.
    # See http://web.iitd.ac.in/~hegde/cad/lecture/L6_3dtrans.pdf#page=7
    # and https://en.wikipedia.org/wiki/Transformation_matrix#Reflection_2

    return transform
