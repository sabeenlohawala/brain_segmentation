"""
File: parser.py
Author: Sabeen Lohawala
Date: 2024-05-08
Description: This file contains the function needed to parse all command line arguments.
"""

import argparse
import os


def get_args():
    """
    This is the main function of the program.
    It parses command line arguments and executes the program logic.
    """

    # Create the argument parser
    parser = argparse.ArgumentParser(
        prog="ProgramName",
        description="What the program does",
        epilog="Text at the bottom of help",
    )

    subparsers = parser.add_subparsers(help="sub-command help")

    # create subparser for "test" command
    test = subparsers.add_parser(
        "test", help="Use this sub-command for testing"
    )
    test.add_argument(
        "--logdir",
        type=str,
        help="Folder containing previous checkpoints",
    )
    test.add_argument("--debug", action="store_true", dest="debug")

    # create subparser for "resume-train" command
    resume = subparsers.add_parser(
        "resume-train", help="Use this sub-command for resuming training"
    )
    resume.add_argument(
        "--logdir",
        type=str,
        help="Folder containing previous checkpoints",
    )
    resume.add_argument("--debug", action="store_true", dest="debug")

    # create subparser for "train" command
    train = subparsers.add_parser("train", help="Use this sub-command for training")

    # Add command line arguments
    train.add_argument(
        "--logdir",
        help="Tensorboard directory",
        type=str,
        required=False,
        default=os.getcwd(),
    )
    train.add_argument(
        "--model_name",
        help="Name of model to use for segmentation",
        type=str,
        default="segformer",
    )
    train.add_argument(
        "--num_epochs",
        help="Number of epochs to train",
        type=int,
        required=False,
        default=20,
    )
    train.add_argument(
        "--batch_size",
        help="Batch size for training",
        type=int,
        required=False,
        default=64,
    )
    train.add_argument(
        "--lr", help="Learning rate for training", type=float, required=False, default=1e-3
    )
    train.add_argument(
        "--data_dir",
        help="Directory of which dataset to train on",
        type=str,
    )
    train.add_argument(
        "--pretrained",
        help="Flag for whether to use pretrained model",
        type=int,
        required=False,
        default=0,
    )
    train.add_argument(
        "--nr_of_classes",
        help="Number of classes in the dataset",
        type=int,
        required=False,
        default=51,
    )
    train.add_argument(
        "--seed", help="Random seed value", type=int, required=False, default=42
    )
    train.add_argument(
        "--debug",
        help="Flag for whether code is being debugged",
        type=int,
        required=False,
        default=0,
    )
    train.add_argument(
        "--wandb_description",
        help="Description for wandb run",
        type=str,
        required=False,
    )
    train.add_argument(
        "--save_checkpoint",
        help="Flag for whether to save checkpoints",
        type=int,
        required=False,
        default=1,
    )
    train.add_argument(
        "--log_images",
        help="Flag for whether to log images to tensorboard",
        type=int,
        required=False,
        default=0,
    )
    train.add_argument(
        "--checkpoint_freq",
        help="Frequency at which to save checkpoints",
        type=int,
        required=False,
        default=10,
    )
    train.add_argument(
        "--image_log_freq",
        help="Frequency at which to save checkpoints",
        type=int,
        required=False,
        default=10,
    )
    train.add_argument(
        "--data_size",
        help="Whether to use the small or medium sized dataset",
        type=str,
        required=False,
        default="small",
    )
    train.add_argument(
        "--augment",
        help="Flag for whether to train on augmented data",
        type=int,
        required=False,
        default=0,
    )
    train.add_argument(
        "--aug_percent",
        help="What fraction of the data should be augmented (between 0 and 1)",
        type=float,
        required=False,
        default=0.8
    )
    train.add_argument(
        "--aug_mask",
        help="Flag for whether to augment the data by masking",
        type=int,
        required=False,
        default=0,
    )
    train.add_argument(
        "--aug_cutout",
        help="Flag for whether to augment the data by adding cutouts",
        type=int,
        required=False,
        default=0,
    )
    train.add_argument(
        "--cutout_n_holes",
        help="Number of cutouts to make during augmentation",
        type=int,
        required=False,
        default=1,
    )
    train.add_argument(
        "--cutout_length",
        help="Side length of cutout to make during augmentation",
        type=int,
        required=False,
        default=32,
    )
    train.add_argument(
        "--mask_n_holes",
        help="Number of masks to make during augmentation",
        type=int,
        required=False,
        default=1,
    )
    train.add_argument(
        "--mask_length",
        help="Side length of mask to make during augmentation",
        type=int,
        required=False,
        default=32,
    )
    train.add_argument(
        "--loss_fn",
        help="Which loss function to use: dice or focal",
        type=str,
        required=False,
        default="dice",
    )
    train.add_argument(
        "--metric",
        help="Which metric to use (currently only supports dice)",
        type=str,
        required=False,
        default="dice",
    )
    train.add_argument(
        "--class_specific_scores",
        help="Whether to log class-specific dice",
        type=int,
        required=False,
        default=0,
    )
    train.add_argument(
        "--intensity_scale",
        help="Whether to apply intensity scaling",
        type=int,
        required=False,
        default=0,
    )
    train.add_argument(
        "--aug_elastic",
        help="Whether to apply elastic transformation augmentation",
        type=int,
        required=False,
        default=0,
    )
    train.add_argument(
        "--aug_piecewise_affine",
        help="Whether to apply piecewise affine augmentation",
        type=int,
        required=False,
        default=0,
    )
    train.add_argument(
        "--new_kwyk_data",
        help="Whether to use the HDF5 dataset",
        type=int,
        required=False,
        default=0,
    )
    train.add_argument(
        "--background_percent_cutoff",
        help="Max percent of slice that can be background",
        type=float,
        required=False,
        default=0.99,
    )
    train.add_argument(
        "--aug_null_half",
        help="Whether to null out half of the brain when training",
        type=int,
        required=False,
        default=0,
    )
    train.add_argument(
        "--aug_null_cerebellum_brain_stem",
        help="Whether to null out cerebellum and brain stem when training",
        type=int,
        required=False,
        default=0,
    )
    train.add_argument(
        "--aug_background_manipulation",
        help="Whether to do background manipulation as an augmentation",
        type=int,
        required=False,
        default=0
    )
    train.add_argument(
        "--aug_shapes_background",
        help="Whether to do random shapes as a background manipulation as an augmentation",
        type=int,
        required=False,
        default=0
    )
    train.add_argument(
        "--aug_grid_background",
        help="Whether to do random grid as a background manipulation as an augmentation",
        type=int,
        required=False,
        default=0
    )
    train.add_argument(
        "--aug_noise_background",
        help="Whether to do random noise as a background manipulation as an augmentation",
        type=int,
        required=False,
        default=0
    )
    train.add_argument(
        "--pad_old_data",
        help="Whether to pad cropped dataset to 256x256",
        type=int,
        required=False,
        default=0
    )
    train.add_argument(
        "--use_norm_consts",
        help="Whether to use Matthias's normalization constants or to divide image intensities by 255.0",
        type=int,
        required=False,
        default=0
    )

    # Parse the command line arguments
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    print(get_args())
