import os
import glob
import json

import numpy as np

from skimage.transform import resize
from TissueLabeling.brain_utils import create_affine_transformation_matrix

# set seed
np.random.seed(42)

# which augmentations to perform
rotate_flag = 0
zoom_flag = 1
scale_range = 0.2
    
# directory where images and masks are stored
data_dir = '/om2/user/sabeen/nobrainer_data_norm/new_small_no_aug_51'

# directory where affine matrices are stored
aug_dir = '/om2/user/sabeen/nobrainer_data_norm/20240208_zoom_20_small_aug_51'

modes = ['test', 'validation', 'train']
augmentation_dict = {}

for mode in modes:
    print(f'Mode: {mode}')
    # mode = 'test'
    file_dir = f'{data_dir}/{mode}'
    images = sorted(glob.glob(f"{file_dir}/brain*.npy"))
    image = np.load(images[0]).squeeze()
    center = np.array([image.shape[0] // 2, image.shape[1] // 2])

    augmentation_dict[mode] = {}
    augmentation_dict_path = f'{aug_dir}/augmentation_dict.json'

    for image_file in images:
        file_suffix = image_file[len(file_dir)+1:].split('_')[-1] # to get the #.npy from '...brain_#.npy'
        affine_file = f'affine_{file_suffix}'
        save_path = f'{aug_dir}/{mode}/{affine_file}'
        
        # coin toss: whether to rotate or zoom (or both)
        if rotate_flag == 1 and zoom_flag == 1:
            print('randomizing rotate/zoom combo')
            rot_flip = np.random.choice([0,1])
            scale_flip = np.random.choice([0,1])
        else:
            rot_flip = rotate_flag
            scale_flip = zoom_flag

        # randomize augmentations
        rotation = np.random.randint(-30,31,1) if rot_flip == 1 else np.array([0])
        scaling_factor = np.random.uniform(1 - scale_range, 1 + scale_range) if scale_flip == 1 else 1
        augmentation_dict[mode][affine_file] = {}
        augmentation_dict[mode][affine_file]['rotation_angle'] = int(rotation[0])
        augmentation_dict[mode][affine_file]['scaling_factor'] = scaling_factor

        affine_matrix = create_affine_transformation_matrix(n_dims = 2,
                                                    scaling = (scaling_factor,scaling_factor),
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

        final_matrix = translation_matrix2 @ affine_matrix @ translation_matrix1
    
        np.save(save_path,final_matrix)

    with open(augmentation_dict_path, "w") as outfile: 
        json.dump(augmentation_dict, outfile)
print('Finished!')