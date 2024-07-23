# Recognizing Brain Regions in 2D Images from Brain Tissue

This repository contains the code developed for my master's thesis project titled *[Recognizing Brain Regions in 2D Images from Brain Tissue](https://sabeenlohawala.cargo.site/thesis)*. The project aims to develop a deep learning method to automatically segment 50 different anatomical regions 2D images of human brain tissue, primarily focusing on sMRI scans.

# Environment Setup

Ensure python is installed. The code in this project was tested and run using Python 3.10. Then, install the necessary libraries in the order described below  to ensure the dependencies are met.

1. Create a virtual environment for the project. This can be done using the following command:

    ```
    conda create -n tissue_labeling python==3.10 pip
    ```

2. Install PyTorch 2.2 and TensorFlow 2.14. You **MUST** use pip to install both to ensure that both libraries can detect the CUDA drivers, like in the following commands*:

    ```
    pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118

    pip install tensorflow==2.14
    ```
    *Make sure the CUDA drivers you install are compatible with your OS and the PyTorch version you install is compatible with that CUDA driver. See this link to see the install command for your system: https://pytorch.org/get-started/previous-versions/.

3. Install the remaining libraries

    ```
    pip install wandb

    pip install scipy

    pip install albumentations

    pip install nibabel

    pip install pytorch-lightning

    pip install lightning

    pip install pandas

    pip install torchinfo

    pip install transformers

    pip install matplotlib

    pip install einops
    ```

# Dataset

This models in this project were trained on slices from the [KWYK dataset](https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2019.00067/full). The KWYK dataset consists of 11480 MRI scans (called feature volumes) and their corresponding anatomical segmentation volumes (called label volumes). The feature volumes consist of intensity values ranging from 0 to 255, which are then normalized to be floats between 0 to 1 in [TissueLabeling/data/dataset.py](https://github.com/sabeenlohawala/tissue_labeling/blob/bb9a5a56f0449ddc742c384e6e011204d22cee41/TissueLabeling/data/dataset.py#L219). The label volumes consist of integer values corresponding to their freesurfer labels. The [class_mapping.csv](https://github.com/sabeenlohawala/tissue_labeling/blob/bb9a5a56f0449ddc742c384e6e011204d22cee41/misc/class_mapping.csv#L1) shows which freesurfer labels correspond to which regions.

The goal in this project is to segment 2D images of the brain, not 3D volumes. So, individual slices were extracted from the volumes to build the dataset. Continue reading below to learn how to generate the slice dataset from the volumes.

## Image Slice Dataset Storage

### HDF5 Dataset (Recommended)
Storing the dataset as HDF5 files is recommended, as it is more efficient and scalable, especially for large datasets. The HDF5Dataset also contains the most updated implementation of the various augmentations used in this project.

When using the HDF5 dataset structure, make sure to include the  command line argument `--new_kwyk_data 1` when the `main.py` script is run.

The .npy dataset can be generated using the script at [`scripts/gen_dataset.py`](https://github.com/sabeenlohawala/tissue_labeling/blob/bb9a5a56f0449ddc742c384e6e011204d22cee41/scripts/gen_dataset.py#L1). Additionally, run the script at [`scripts/gen_dataset_nonbrain.py`](https://github.com/sabeenlohawala/tissue_labeling/blob/bb9a5a56f0449ddc742c384e6e011204d22cee41/scripts/gen_dataset_nonbrain.py#L1) for .hf5 shard saved to store information about the percentage of pixels that are background (i.e. equal to 0) for each slice in the dataset; this information can be used for filtering the dataset during experiments. See [`submit_gen_dataset_nonbrain.sh`](https://github.com/sabeenlohawala/tissue_labeling/blob/bb9a5a56f0449ddc742c384e6e011204d22cee41/submit_gen_dataset_nonbrain.sh#L1) for an example of how to run this script.

**!!! IMPORTANT: Change the paths in the following locations to point to your dataset**
1. Set the `OUTDIR` ([scripts/gen_dataset.py, line 46](https://github.com/sabeenlohawala/tissue_labeling/blob/bb9a5a56f0449ddc742c384e6e011204d22cee41/scripts/gen_dataset.py#L46)) to point to the directory where you would like to write the .hf5 shards.
2. Set `NIFTIDIR` ([scripts/gen_dataset.py, line 50](https://github.com/sabeenlohawala/tissue_labeling/blob/bb9a5a56f0449ddc742c384e6e011204d22cee41/scripts/gen_dataset.py#L50)) to point to the directory where the .nii.gz volumes are saved. Note: the feature volumes and label volumes must be located in the same directory.
3. Set `self.data_dir` ([TissueLabeling/config.py line 163](https://github.com/sabeenlohawala/tissue_labeling/blob/bb9a5a56f0449ddc742c384e6e011204d22cee41/TissueLabeling/config.py#L163)) to point to directory where `.hf5` shards are stored.
4. Set `h5_dir` ([TissueLabeling/data/dataset.py, line 62](https://github.com/sabeenlohawala/tissue_labeling/blob/bb9a5a56f0449ddc742c384e6e011204d22cee41/TissueLabeling/data/dataset.py#L62)) to point to directory where `.hf5` shards are stored.
5. Set `slice_nonbrain_dir` ([TissueLabeling/data/dataset.py, line 67](https://github.com/sabeenlohawala/tissue_labeling/blob/bb9a5a56f0449ddc742c384e6e011204d22cee41/TissueLabeling/data/dataset.py#L67)) to point to the directory where the output from `scripts/gen_dataset_nonbrain.py` are stored.

### .npy Dataset
In the earliest version, each MRI feature slice and its corresponding label slice was stored as an individual .npy file. This version of the dataset may still be helpful for initial explorations of this repo or new datasets, but it is less efficient and scalable, and not all of the functions or augmentations are kept as up-to-date as those in the HDF5 dataset.

When using a .npy dataset structure, make sure to include the command line argument `--new_kwyk_data 0` when the `main.py` script is run.

These .npy datasets can be generated using the script at [`scripts/mit_kwyk_data_optimized.py`](https://github.com/sabeenlohawala/tissue_labeling/blob/e49946030e41d4117ee800fb7d7d4c8d4be72425/scripts/mit_kwyk_data_optimized.py#L1). See [`job_prepare_new_data.sh`](https://github.com/sabeenlohawala/tissue_labeling/blob/e49946030e41d4117ee800fb7d7d4c8d4be72425/job_prepare_new_kwyk_data.sh#L1) for an example of how to run this script.

The overall directory structure for the .npy files is shown below. The various .npy datasets (e.g. `new_med_no_aug_51`, `new_small_aug_107`) are all contained within the `data_root_dir` (e.g. `nobrainer_data_norm`).

**!!! IMPORTANT: Change the paths in the following locations to point to your dataset**
1. Set the `data_root_dir` in [TissueLabeling/config.py, line 25](https://github.com/sabeenlohawala/tissue_labeling/blob/e49946030e41d4117ee800fb7d7d4c8d4be72425/TissueLabeling/config.py#L25) to the path to your `data_root_dir`.

#### .npy Dataset Structure

```
nobrainer_data_norm
│
├── new_med_no_aug_51
│   ├── normalization_counts.npy
│   ├── pixel_counts.npy
│   ├── train
│   │   ├── brain_0.npy
│   │   ├── mask_0.npy
│   │   ├── brain_1.npy
│   │   ├── mask_1.npy
|   |   └── ...
│   ├── validation
│   │   ├── brain_0.npy
│   │   ├── mask_0.npy
│   │   ├── brain_1.npy
│   │   ├── mask_1.npy
|   |   └── ...
│   └── test
│   │   ├── brain_0.npy
│   │   ├── mask_0.npy
│   │   ├── brain_1.npy
│   │   ├── mask_1.npy
|   |   └── ...
└── new_small_aug_107
    └── ...
```

### Setting Other Hardcoded Paths
1. [TissueLabeling/training/logging.py, line 312](https://github.com/sabeenlohawala/tissue_labeling/blob/e49946030e41d4117ee800fb7d7d4c8d4be72425/TissueLabeling/training/logging.py#L312): modify the path to point to the location of misc/FreeSurferColorLUT.txt
2. [TissueLabeling/training/logging.py, line 315](https://github.com/sabeenlohawala/tissue_labeling/blob/e49946030e41d4117ee800fb7d7d4c8d4be72425/TissueLabeling/training/logging.py#L315): modify the path to point to the location of misc/readme
3. [TissueLabeling/brain_utils.py, line 207](https://github.com/sabeenlohawala/tissue_labeling/blob/e49946030e41d4117ee800fb7d7d4c8d4be72425/TissueLabeling/brain_utils.py#L207): modify the path to point to the location of misc/class_mapping.csv
4. [TissueLabeling/brain_utils.py, line 252](https://github.com/sabeenlohawala/tissue_labeling/blob/e49946030e41d4117ee800fb7d7d4c8d4be72425/TissueLabeling/brain_utils.py#L252): modify the path to point to the location of misc/class_mapping.csv
5. [TissueLabeling/brain_utils.py, line 291](https://github.com/sabeenlohawala/tissue_labeling/blob/e49946030e41d4117ee800fb7d7d4c8d4be72425/TissueLabeling/brain_utils.py#L291): modify the path to point to the location of misc/class_mapping.csv
6. [TissueLabeling/training/logging.py, line 60](https://github.com/sabeenlohawala/tissue_labeling/blob/e49946030e41d4117ee800fb7d7d4c8d4be72425/TissueLabeling/training/logging.py#L60): set `file_path` to point to the directory where the .nii.gz volumes are saved. Note: the feature volumes and label volumes must be located in the same directory.
7. [TissueLabeling/training/logging.py, line 58-59](https://github.com/sabeenlohawala/tissue_labeling/blob/e49946030e41d4117ee800fb7d7d4c8d4be72425/TissueLabeling/training/logging.py#L58C9-L59C41): set `image_file` and `mask_file` to the feature volume and its corresponding label volume that you will log the model predictions for.

## How to Run

The model training is run using the script in [scripts/commands/main.py](https://github.com/sabeenlohawala/tissue_labeling/blob/e49946030e41d4117ee800fb7d7d4c8d4be72425/scripts/commands/main.py#L1). See [submit_requeue.sh](https://github.com/sabeenlohawala/tissue_labeling/blob/e49946030e41d4117ee800fb7d7d4c8d4be72425/submit_requeue.sh#L1) for an example of how to submit an experiment as a batch job on the SLURM. There are two main options when running the script using the command line. The basic commands that use default experiment parameters, as well as a table listing the possible command line arguments, their description, and their possible values are included below:
1. Train a new model.
    
    `python scripts/commands/main.py train`

2. Resume training the model from a checkpoint.

    `python scripts/commands/main.py resume-train --logdir directory_where_checkpoints_are_saved`

To obtain the loss and metrics for the held-out test script, you can use a similar type of command for with the scripts/commands/test.py file:
    `python scripts/commands/test.py test --logdir directory_where_checkpoints_are_saved`

### `train` Command-line Arguments
| Argument Name    | Possible Values   | Description
| -------- | ------- |------- |
| --logdir  | any str    | Path to the directory where config.json, all Tensorboard logs, and model checkpoints are saved for the run. |
| --model_name | 'segformer', 'original_unet', 'attention_unet'    | Name of the ML model that will be trained for the segmentation task. Modify [`select_model()`](https://github.com/sabeenlohawala/tissue_labeling/blob/e49946030e41d4117ee800fb7d7d4c8d4be72425/scripts/commands/main.py#L36C1-L68C17) function in scripts/commands/main.py if you add new models. |
| --num_epochs    | any int    | The number of epochs to train during the run. |
| --batch_size  | any int    | The batch size for a single GPU. If training across multiple GPUs (which the code is already set-up to do with PyTorch Lightning Fabric), the effective batch size for the run becomes batch_size * num_gpus.|
| --lr  | any float    | This is the learning rate for training. Experiments we ran for this project were most successful with lr = 1e-3. |
| --pretrained  | 0 or 1    | Flag for whether to use the pretrained SegFormer model, which is trained on 3-channel images instead of 1-channel. Used to select initialize model as pre-trained SegFormer and configure datasets to output 3-channel images by repeating color channel 3 times. |
| --nr_of_classes  | 2, 6, 16, 50    | The number of anatomical classes to train the model to segment. If you want to segment a different number of classes, you must create your own mapping in the mapping.csv (TODO), specify the region names in voxelmorph readme (TODO). |
| --seed  | any int    | Random seed value. |
| --debug  | 0 or 1    | Flag for whether run is for debugging code rather than actual experiment. This reduces the size of the train, val, and test datasets to the first 100 items to run faster. |
| --wandb_description  | any str    | Description for the run that's logged to wandb. |
| --save_checkpoint  | 0 or 1    | Flag for whether to save model checkpoints to directory specified by logdir. |
| --log_images  | 0 or 1    | Flag for whether to log model segmentation outputs for a few test slices to tensorboard. |
| --checkpoint_freq  | any int    | Frequency at which to save checkpoints to logdir. |
| --image_log_freq  | any int    | Frequency at which to log images to tensorboard. |
| --data_size  | 'small', 'medium', 'large', 'shard', 'shard-<#>'    | Indicates how large a subset of the data to train on. 'small' corresponds to a dataset made up of slices from 10 volumes, 'medium' from 1150 volumes, and 'large' from all volumes. If using the HDF5 dataset, specifying 'shard' indicates that the train, validation, and test datasets should be the first three shards, respectively; specifying 'shard-<#>' (e.g. 'shard-5') indicates that (TODO: is this implemented correctly?)| 
| --loss_fn  | 'dice', 'focal'    | Indicates which loss function to use during training. | 
| --metric  | 'dice'    | Indicates which metric to calculate and log during the run. | 
| --class_specific_scores  | 0 or 1    | Flag for whether to log the dice score for each class that is being segmented, rather than only the overall dice scores. (NOTE: even though this is implemented in the metric, not all parts of the training code may support this, so double check TissueLabeling/training/trainer.py.) | 
| --new_kwyk_data  | 0 or 1    | Indicates whether to use the HDF5Dataset or the .npy Dataset. | 
| --background_percent_cutoff  | float between 0.0 to 1.0    | The maximum fraction of a feature slice that can be background (i.e. equal to 0). Any slices with a greater proportion of background pixels are filtered out and excluded from the dataset. | 
| --pad_old_data  | 0 or 1    | TODO | 
| --use_norm_consts  | 0 or 1    | TODO | 
| --augment  | 0 or 1    | Flag for whether to apply augmentations to some of the data during training. Only images in the training dataset, not validation or test, are augmented. | 
| --aug_percent  | float between 0.0 to 1.0    | What fraction of the training data should be augmented. | 
| --aug_mask  | 0 or 1    | Flag for whether to include masking augmentation. | 
| --aug_cutout  | 0 or 1    | Flag for whether to include cutout augmentation. | 
| --cutout_n_holes  | any int    | Number of cutout regions to make when augmentation is applied. | 
| --cutout_length  | any int    | Side length of the cutout that's made when augmentation is applied. | 
| --mask_n_holes  | any int    | Number of mask regions to make when augmentation is applied. |
| --mask_length  | any int   | Side length of the mask that's made when augmentation is applied. | 
| --intensity_scale  | 0 or 1    | Flag for whether to include random brightness and contrast jittering augmentation. | 
| --aug_elastic  | 0 or 1    | Flag for whether to include elastic transformation augmentation. | 
| --aug_piecewise_affine  | 0 or 1    | Flag for whether to include piecewise affine augmentation. | 
| --aug_null_half  | 0 or 1    | Flag for whether to include nulling one hemisphere of the brain as an augmentation. | 
| --aug_null_cerebellum_brain_stem  | 0 or 1    | Flag for whether to include nulling the cerebellum and brain stem when half of the brain is nulled. | 
| --aug_background_manipulation  | 0 or 1    | Flag for whether to include background manipulation. | 
| --aug_shapes_background  | 0 or 1    | Flag for whether to include random shapes as a possible background manipulation. | 
| --aug_grid_background  | 0 or 1    | Flag for whether to include a random grid as a possible background manipulation. | 
| --aug_noise_background  | 0 or 1    | Flag for whether to include random noise as a possible background manipulation. | 
<!-- | --data_dir  |     | | -->

### `resume-train` Command-line Arguments
| Argument Name    | Possible Values   | Description
| -------- | ------- |------- |
| --logdir  | any str    | Path to the directory where config.json, all Tensorboard logs, and model checkpoints are saved for the run. |
| --debug  | 0 or 1    | Flag for whether run is for debugging code rather than actual experiment. This reduces the size of the train, val, and test datasets to the first 100 items to run faster. |

### `test` Command-line Arguments
| Argument Name    | Possible Values   | Description
| -------- | ------- |------- |
| --logdir  | any str    | Path to the directory where config.json, all Tensorboard logs, and model checkpoints are saved for the run. |
| --debug  | 0 or 1    | Flag for whether run is for debugging code rather than actual experiment. This reduces the size of the train, val, and test datasets to the first 100 items to run faster. |

## Weights & Biases

If you plan on using Weights & Biases to log runs, you will need to change the default paths in the [`init_wandb()`](https://github.com/sabeenlohawala/tissue_labeling/blob/e49946030e41d4117ee800fb7d7d4c8d4be72425/TissueLabeling/utils.py#L135C5-L145C6) function in TissueLabeling/utils.py. Change the values passed into the `entity` and `dir` parameters in the call to `wandb.init()` to match those of your W&B account and your local machine, respectively.

# Augmentations
The argument `--augment 1` must be included in the command line in order for any augmentations to be applied during training.
If augmentations are enabled, some portion of the images in the training dataset are randomly selected to be augmented. It is important to note that augmentations are applied to training data ONLY, not to validation or test data.

## Traditional Augmentations
Including the command line argument `--augment 1` automatically selects random affine transformation and random horizontal flipping augmentations. For any augmented sample, the angle of rotation is chosen uniformly at random from -15º to +15º, the random scaling factor is chosen uniformly at random from 0.8 to 1.2, and the shearing factor is chosen uniformly at random from -0.012 to +0.012. Additionally, the augmented slice was horizontally flipped with a probability of 50%.

The affine transformation and random flipping augmentations are applied to both the feature slice and the corresponding segmentation mask slices in order to align the pixel-level labels to the perturbed slice. In the case of the affine transformation, trilinear interpolation is used to transform the feature slice, which contains continuous intensity values, and nearest neighbor interpolation is used to transform the corresponding mask slice, which contains discrete brain region labels.

## Intensity Scaling

To include intensity scaling augmentations, add the flag `--intensity_scale 1` to the command line.

Because the goal is to develop a model that is robust enough to segment any 2D photograph, regardless of its brightness or contrast, like a neuroanatomist would be able to, it is important that the model is trained to be robust to these types of variations in the intensity values of the images. A random brightness and contrast augmentation is applied to jitter the intensity values of the pixels in the feature slice, while the ground truth label mask remained unmodified. For this project, the brightness and contrast were both jittered by a random factor between -0.2 and +0.2, a standard default range in intensity scaling augmentations.

## Cut-outs and Masking

To include the cut-out augmentation, add the flag `--aug_cutout 1` to the command line. When the cut-out augmentation is applied to a sample, a random square region of the feature slice is nulled out and set to 0 while its corresponding label slice is left unmodified. You can specify the number of square cut-out regions and the size of these regions using the arguments `--cutout_n_holes` and `--cutout_length`, respectively. The cut-out augmentation when applied to the brain MRI data can emulate occluded parts of the pictured brain tissue, like those caused by glares. By nulling out only a portion of the image but not the corresponding mask, the cut-out augmentation tests the model’s ability to learn the segmentation of slices well-enough to “fill in” the missing information.

A related augmentation, which nulls out the corresponding part of the label map as well, is called "masking." To include the masking augmentation, add the flag `--aug_mask 1` to the command line. You can specify the number of square mask regions and the size of these regions using the arguments `--mask_n_holes` and `--mask_length`, respectively. Masking can emulate places of missing data, like lesions or tears. Such an augmentation will test whether the model is able to segment the remainder of the feature slice without the additional contextual information in a different way than the cut-outs, thereby resulting in a more robust segmentation model.

When the cut-out or masking augmentation is included in the training pipeline, it is applied with 100% probability to any sample that is selected to be augmented. Because the location of the hole is randomly chosen, applications of the augmentation to different samples result in different amounts and different locations of the sample being nulled, with some augmented slices appearing not to have any nulled region at all if the hole falls entirely in the background, and other slices having a large portion of the tissue being nulled, with the majority of cases having fallen somewhere in between.

## Nulling Half of the Brain

To include the augmentation to randomly null one hemisphere of the image, add the flag `--aug_null_half 1` to the command line. Because the slice dataset in this project was built from MRI scans, it contained whole-brain slices only, but many datasets containing slices from other imaging modalities have only hemi-slices. In order to train the model to be able to segment hemi-slices as well as whole-brain slices, the hemi-slices can be simulated using this augmentation. When this augmentation is applied, a randomly selected half of the labels -- either those prefixed with 'left' or those prefixed with 'right' -- are set to 0 in the feature and label slices. The nulling half of the brain augmentation is applied with 50% probability to slices selected for augmentation when this augmentation is included in the training pipeline.

Additionally, in many cases the cerebellum and brain stem are dissected away first before the remainder of the brain is sliced and photographed, resulting in hemi slices without cerebellum or brain stem regions. This can be simulated using an augmentation by adding the flag `--aug_null_cerebellum_brain_stem 1` to the command line in conjunction with the `--aug_null_half 1` flag. For a sample selected to be have the cerebellum and brain stem nulled, all pixels of the feature slice and label slice that are assigned cerebellum and brain stem labels were set to 0. Then, when a slice is selected to be modified with the null-half augmentation, the cerebellum and brain stem are nulled with 50% probability to incorporate more dissection-like MRI slices.

## Background Manipulation

Background manipulation was used to decouple the brain tissue and background and add more variability to the background to help the model become more robust in detecting the brain against a variety of backgrounds that could be found in photorealistic images of the brain, rather than just the all-0 background found in the MRI dataset. To include this augmentation, add the flag `--aug_background_manipulation 1` to the command line.

There are three possible background functions that have been implemented. One or more of these must be enabled in order for background manipulation to be applied:
1. A background of random shapes of random intensity values.
    Add the flag `--aug_shapes_background 1` to the command line.
2. A background of grid of random size, line thickness, and intensity.
    Add the flag `--aug_grid_background 1` to the command line.
3. A background of random noise.
    Add the flag `--aug_noise_background 1` to the command line.

If background augmentation is included in the training pipeline, it is applied with 50% probability to a sample that is selected to be augmented. If only one of the background manipulation augmentations is selected to be used, it is applied with 100% probability to any sample selected to have background augmentation. However, if it is specified to use a combination, for example, of the random grid and random shape backgrounds, the manipulation option is chosen with 50% probability for a sample that is selected to have background augmentation.

# Next Steps

## Ablation Study
Results from some preliminary experiments run with augmentation have shown that adding augmentations helps the model generalize better to unseen data, segment photorealistic images, and better recognizes brain tissue vs. background. However, a full abalation study comparing the effects of each augmentation and various combinations of augmentations has not yet been run.

## nnUNet
In this project, we compared the UNet model, which has achieved state-of-the-art results in many medical segmentation models, with the SegFormer model, a newer transformer-based model. We found that the SegFormer model significantly outperformed the original UNet model. The nnUNet has been shown to surpass the original UNet in both 2D and 3D segmentation tasks. Before any firm conclusion can be made about the superior performance of the SegFormer model, the nnUNet performance should be evaluated, as well.

# Contributions

The majority of the code in this repository was written by me ([@sabeenlohawala](https://github.com/sabeenlohawala)). I received invaluable guidance and mentorship on this project, and help with structuring and debugging this code from my advisor Drs. Satrajit Ghosh ([@satra](https://github.com/satra)) and Harsha Gazula ([@hvgazula](https://github.com/hvgazula)). The earliest version of this code was adapted from a repository by Matthias Steiner ([@batschoni](https://github.com/batschoni)).