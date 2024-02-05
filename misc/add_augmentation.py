import os
import glob

import numpy as np
import torch

from skimage.transform import resize
from TissueLabeling.brain_utils import create_affine_transformation_matrix

# def create_affine_transformation_matrix(
#     n_dims, scaling=None, rotation=None, shearing=None, translation=None
# ):
#     """Create a 4x4 affine transformation matrix from specified values
#     :param n_dims: integer
#     :param scaling: list of 3 scaling values
#     :param rotation: list of 3 angles (degrees) for rotations around 1st, 2nd, 3rd axis
#     :param shearing: list of 6 shearing values
#     :param translation: list of 3 values
#     :return: 4x4 numpy matrix
#     """

#     T_scaling = np.eye(n_dims + 1)
#     T_shearing = np.eye(n_dims + 1)
#     T_translation = np.eye(n_dims + 1)

#     if scaling is not None:
#         T_scaling[np.arange(n_dims + 1), np.arange(n_dims + 1)] = np.append(
#             scaling, 1
#         )

#     if shearing is not None:
#         shearing_index = np.ones((n_dims + 1, n_dims + 1), dtype="bool")
#         shearing_index[np.eye(n_dims + 1, dtype="bool")] = False
#         shearing_index[-1, :] = np.zeros((n_dims + 1))
#         shearing_index[:, -1] = np.zeros((n_dims + 1))
#         T_shearing[shearing_index] = shearing

#     if translation is not None:
#         T_translation[
#             np.arange(n_dims), n_dims * np.ones(n_dims, dtype="int")
#         ] = translation

#     if n_dims == 2:
#         if rotation is None:
#             rotation = np.zeros(1)
#         else:
#             rotation = np.asarray(rotation) * (np.pi / 180)
#         T_rot = np.eye(n_dims + 1)
#         T_rot[np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1])] = [
#             np.cos(rotation[0]),
#             np.sin(rotation[0]),
#             np.sin(rotation[0]) * -1,
#             np.cos(rotation[0]),
#         ]
#         return T_translation @ T_rot @ T_shearing @ T_scaling

#     else:

#         if rotation is None:
#             rotation = np.zeros(n_dims)
#         else:
#             rotation = np.asarray(rotation) * (np.pi / 180)
#         T_rot1 = np.eye(n_dims + 1)
#         T_rot1[np.array([1, 2, 1, 2]), np.array([1, 1, 2, 2])] = [
#             np.cos(rotation[0]),
#             np.sin(rotation[0]),
#             np.sin(rotation[0]) * -1,
#             np.cos(rotation[0]),
#         ]
#         T_rot2 = np.eye(n_dims + 1)
#         T_rot2[np.array([0, 2, 0, 2]), np.array([0, 0, 2, 2])] = [
#             np.cos(rotation[1]),
#             np.sin(rotation[1]) * -1,
#             np.sin(rotation[1]),
#             np.cos(rotation[1]),
#         ]
#         T_rot3 = np.eye(n_dims + 1)
#         T_rot3[np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1])] = [
#             np.cos(rotation[2]),
#             np.sin(rotation[2]),
#             np.sin(rotation[2]) * -1,
#             np.cos(rotation[2]),
#         ]
#         return T_translation @ T_rot3 @ T_rot2 @ T_rot1 @ T_shearing @ T_scaling
    
# directory where images and masks are stored
data_dir = '/om2/user/sabeen/nobrainer_data_norm/new_small_no_aug_51'

# directory where affine matrices are stored
aug_dir = '/om2/user/sabeen/nobrainer_data_norm/20240204_test'

modes = ['test',]#'validation', 'train']

for mode in modes:
    # mode = 'test'
    file_dir = f'{data_dir}/{mode}'
    images = sorted(glob.glob(f"{file_dir}/brain*.npy"))
    image = np.load(images[0]).squeeze()
    center = np.array([image.shape[0] // 2, image.shape[1] // 2])

    for image_file in images:
        file_suffix = image_file[len(file_dir)+1:].split('_')[-1] # to get the #.npy from '...brain_#.npy'
        save_path = f'{aug_dir}/{mode}/affine_{file_suffix}'

        rotation = np.random.randint(-30,31,1)
        rotation_matrix = create_affine_transformation_matrix(n_dims = 2,
                                                    scaling = None,
                                                    rotation = rotation,
                                                    shearing = None,
                                                    translation = None)

        # Translate the center back to the origin
        translation_matrix1 = np.array([[1, 0, -center[0]],
                                        [0, 1, -center[1]],
                                        [0, 0, 1]])

        # Translate the center to the original position
        translation_matrix2 = np.array([[1, 0, center[0]],
                                        [0, 1, center[1]],
                                        [0, 0, 1]])

        final_matrix = translation_matrix2 @ rotation_matrix @ translation_matrix1
    
        np.save(save_path,final_matrix)