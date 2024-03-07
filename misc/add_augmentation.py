import os
import glob
import json

import numpy as np

from skimage.transform import resize
from TissueLabeling.brain_utils import create_affine_transformation_matrix, draw_value_from_distribution

# set seed
np.random.seed(42)

# which augmentations to perform
flipping = True  # enable right/left flipping
scaling_bounds = 0.2  # the scaling coefficients will be sampled from U(1-scaling_bounds; 1+scaling_bounds)
rotation_bounds = 15  # the rotation angles will be sampled from U(-rotation_bounds; rotation_bounds)
shearing_bounds = 0.012  # the shearing coefficients will be sampled from U(-shearing_bounds; shearing_bounds)
translation_bounds = False  # no translation is performed, as this is already modelled by the random cropping
enable_90_rotations = False
# nonlin_std = 4.  # this controls the maximum elastic deformation (higher = more deformation)
# bias_field_std = 0.7  # this controls the maximum bias field corruption (higher = more bias)

batchsize = 1
n_dims = 2

# directory where images and masks are stored
data_dir = '/om2/user/sabeen/nobrainer_data_norm/new_med_no_aug_51'

# directory where affine matrices are stored
aug_dir = '/om2/user/sabeen/nobrainer_data_norm/20240217_med_synth_aug'

modes = ['test', 'validation', 'train']
augmentation_dict = {}

for mode in modes:
    print(f'Mode: {mode}')
    # mode = 'test'
    file_dir = f'{data_dir}/{mode}'
    if not os.path.exists(f'{aug_dir}/{mode}'):
        print("Making aug dirs...")
        os.makedirs(f'{aug_dir}/{mode}')
    
    images = sorted(glob.glob(f"{file_dir}/brain*.npy"))
    image = np.load(images[0]).squeeze()
    center = np.array([image.shape[0] // 2, image.shape[1] // 2])

    augmentation_dict[mode] = {}
    augmentation_dict_path = f'{aug_dir}/augmentation_dict.json'

    for image_file in images:
        file_suffix = image_file[len(file_dir)+1:].split('_')[-1] # to get the #.npy from '...brain_#.npy'
        affine_file = f'affine_{file_suffix}'
        save_path = f'{aug_dir}/{mode}/{affine_file}'

        # randomize augmentations
        
        scaling = draw_value_from_distribution(scaling_bounds,
                                               size=n_dims,
                                               centre=1,
                                               default_range=.15,
                                               return_as_tensor=False,
                                               batchsize=batchsize)

        rotation = draw_value_from_distribution(rotation_bounds,
                                                size=1,
                                                default_range=15.0,
                                                return_as_tensor=False,
                                                batchsize=batchsize)

        shearing = draw_value_from_distribution(shearing_bounds,
                                                size=n_dims ** 2 - n_dims,
                                                default_range=.01,
                                                return_as_tensor=False,
                                                batchsize=batchsize)

        augmentation_dict[mode][affine_file] = {}
        augmentation_dict[mode][affine_file]['rotation'] = rotation.tolist()
        augmentation_dict[mode][affine_file]['scaling'] = scaling.tolist()
        augmentation_dict[mode][affine_file]['shearing'] = shearing.tolist()
        augmentation_dict[mode][affine_file]['translation'] = None

        affine_matrix = create_affine_transformation_matrix(n_dims = 2,
                                                    scaling = scaling,
                                                    rotation = rotation,
                                                    shearing = shearing,
                                                    translation = None)

        # Translate the center back to the origin
        translation_matrix1 = np.array([[1, 0, -center[0]],
                                        [0, 1, -center[1]],
                                        [0, 0, 1]])

        # Translate the center to the original position
        translation_matrix2 = np.array([[1, 0, center[0]],
                                        [0, 1, center[1]],
                                        [0, 0, 1]])

        final_matrix = translation_matrix2 @ affine_matrix @ translation_matrix1
    
        np.save(save_path,final_matrix)

    with open(augmentation_dict_path, "w") as outfile: 
        json.dump(augmentation_dict, outfile)
print('Finished!')