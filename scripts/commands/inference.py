"""
Created Date: Thursday, February 29, 2024
Author: Sabeen Lohawala

Copyright (c) 2023, Sabeen Lohawala. MIT
"""
import argparse
import glob
import json
import os
import sys

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

# from matplotlib.colors import BoundaryNorm, ListedColormap
# import matplotlib.colors as mcolors
from PIL import Image
from torchvision import transforms
import cv2
from torch.utils.tensorboard import SummaryWriter

from TissueLabeling.config import Configuration
from TissueLabeling.brain_utils import crop, load_brains, mapping
from TissueLabeling.models.segformer import Segformer
from TissueLabeling.models.original_unet import OriginalUnet
from TissueLabeling.models.attention_unet import AttentionUnet
from TissueLabeling.training.logging import Log_Images
from TissueLabeling.metrics.metrics import Dice


def get_config(logdir):
    """
    Gets the config file based on the command line arguments.
    """
    chkpt_folder = os.path.join("results/", logdir)

    config_file = os.path.join(chkpt_folder, "config.json")
    if not os.path.exists(config_file):
        sys.exit(f"Configuration file not found at {config_file}")

    with open(config_file) as json_file:
        data = json.load(json_file)
    assert isinstance(data, dict), "Invalid Object Type"

    checkpoint_paths = sorted(glob.glob(os.path.join(chkpt_folder, "checkpoint*")))
    if not checkpoint_paths:
        sys.exit("No checkpoints exist to resume training")

    data["checkpoint"] = checkpoint_paths[-1]
    # data["start_epoch"] = int(os.path.basename(dice_list[0]).split('.')[0].split('_')[-1])

    args = argparse.Namespace(**data)
    config = Configuration(args, "config_resume.json")

    return config, checkpoint_paths


def get_model(config, checkpoint_path=None):
    if config.model_name == "segformer":
        model = Segformer(config.nr_of_classes, pretrained=config.pretrained)
    elif config.model_name == "original_unet":
        model = OriginalUnet(image_channels=1, nr_of_classes=config.nr_of_classes)
    elif config.model_name == "attention_unet":
        model = AttentionUnet(
            dim=16,
            channels=1,
            dim_mults=(2, 4, 8, 16, 32, 64),
        )
    else:
        print(f"Invalid model name provided: {config.model_name}")
        sys.exit()

    print(f"{config.model_name} found")
    if checkpoint_path:
        print(f"Loading from checkpoint...")
        model.load_state_dict(
            torch.load(checkpoint_path, map_location=torch.device("cpu"))["model"]
        )

    return model


def rgb_map_for_data(nr_of_classes):
    def extract_numbers_names_colors(FreeSurferColorLUT=""):
        """
        Extract lists of numbers, names, and colors representing anatomical brain
        regions from FreeSurfer's FreeSurferColorLUT.txt lookup table file.

        Parameters
        ----------
        FreeSurferColorLUT : string
            full path to FreeSurferColorLUT.txt file (else uses local Python file)

        Returns
        -------
        numbers : list of integers
            numbers representing anatomical labels from FreeSurferColorLUT.txt
        names : list of integers
            names for anatomical regions from FreeSurferColorLUT.txt
        colors : list of integers
            colors associated with anatomical labels from FreeSurferColorLUT.txt

        Examples
        --------
        >>> from mindboggle.mio.labels import extract_numbers_names_colors # doctest: +SKIP
        >>> ennc = extract_numbers_names_colors # doctest: +SKIP
        >>> en1,en2,ec = ennc('/Applications/freesurfer/FreeSurferColorLUT.txt') # doctest: +SKIP

        """
        import os
        from io import open

        # from ext.mindboggle.FreeSurferColorLUT import lut_text

        def is_number(s):
            try:
                int(s)
                return True
            except ValueError:
                return False

        # if os.environ['FREESURFER_HOME']:
        #     FreeSurferColorLUT = os.path.join(
        #              os.environ['FREESURFER_HOME'], 'FreeSurferColorLUT.txt')

        if FreeSurferColorLUT and os.path.exists(FreeSurferColorLUT):
            f = open(FreeSurferColorLUT, "r")
            lines = f.readlines()
        else:
            # lut = lut_text()
            # lines = lut.split('\n')
            lines = None

        numbers = []
        names = []
        colors = []
        for line in lines:
            strings = line.split()
            if strings and is_number(strings[0]):
                numbers.append(int(strings[0]))
                names.append(strings[1])
                colors.append([int(strings[2]), int(strings[3]), int(strings[4])])

        return numbers, names, colors

    _, fs_names, fs_colors = extract_numbers_names_colors(
        "/om2/user/sabeen/freesurfer/distribution/FreeSurferColorLUT.txt"
    )

    with open("/om2/user/sabeen/readme", "r") as f:
        voxmorph_label_index = f.read().splitlines()

    # get the last 24 lines of the readme file (format--> id: name)
    if nr_of_classes == 50:
        voxmorph_label_index = [
            item.strip().split(":")
            for item in voxmorph_label_index[200:250]
            if item != ""
        ]  # HACK
    elif nr_of_classes == 51:
        voxmorph_label_index = [
            item.strip().split(":")
            for item in voxmorph_label_index[200:251]
            if item != ""
        ]  # HACK
    elif nr_of_classes == 107:
        voxmorph_label_index = [
            item.strip().split(":")
            for item in voxmorph_label_index[91:198]
            if item != ""
        ]  # HACK
    elif nr_of_classes == 7:
        voxmorph_label_index = [
            item.strip().split(":")
            for item in voxmorph_label_index[253:260]
            if item != ""
        ]  # HACK
    elif nr_of_classes == 2:
        voxmorph_label_index = [
            item.strip().split(":")
            for item in voxmorph_label_index[262:264]
            if item != ""
        ]  # HACK
    elif nr_of_classes == 17:
        voxmorph_label_index = [
            item.strip().split(":")
            for item in voxmorph_label_index[266:283]
            if item != ""
        ]  # HACK
    else:
        raise Exception(f"coloring for nr_of_classes = {nr_of_classes} not found")

    voxmorph_label_index = [
        [int(item[0]), item[1].strip()] for item in voxmorph_label_index
    ]
    voxmorph_label_index_dict = dict(voxmorph_label_index)
    my_colors = [
        fs_colors[fs_names.index(item)] for item in voxmorph_label_index_dict.values()
    ]

    return np.array(my_colors)


def create_plot(
    image: np.array,
    caption: str,
    color_range=None,
    fig_path: str = None,
):
    if fig_path is not None and len(fig_path.split(".")) == 1:
        fig_path = fig_path + ".png"

    if color_range is not None:
        image = image.astype(np.uint8)
        channels = [cv2.LUT(image, color_range[:, i]) for i in range(3)]
        new_img = np.dstack(channels)

        if fig_path is not None:
            new_img_bgr = np.dstack([channels[2], channels[1], channels[0]])
            cv2.imwrite(fig_path, new_img_bgr)
        image = Image.fromarray(np.uint8(new_img))
    else:
        img_min = np.min(image)
        img_max = np.max(image)
        new_img = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        if fig_path is not None:
            cv2.imwrite(fig_path, new_img)
        image = Image.fromarray(np.uint8(new_img))
    return image


# logdir = '/om2/user/sabeen/tissue_labeling/results/20240227-aug-Msegformer\Smed\C51\B128\LR0.001\PT1\A1'
# logdir = '/om2/user/sabeen/tissue_labeling/results/20240218-grid-Msegformer\Smed\C51\B128\LR0.001\PT0\A0'
# logdir = '/om2/user/sabeen/tissue_labeling/results/20240120-multi-4gpu-Msegformer\\Ldice\\C2\\B670\\A0'
# logdir = '/om2/user/sabeen/tissue_labeling/results/20240131-multi-4gpu-Msegformer\Ldice\C7\B670\A0'
# logdir = "/om2/scratch/tmp/sabeen/20240314-intensity-0.2-0.2-Msegformer\Ldice\Smed\C51\B512\LR0.001\PT0\A0"
# logdir = "/om2/user/sabeen/tissue_labeling/results/20240227-cut-64-1-Msegformer\Smed\C51\B128\LR0.001\PT0\A1"
logdir = "/om2/scratch/tmp/sabeen/results/20240505-50-null-Msegformer\Ldice\Sshard\RV0\BC0\C50\B288\LR0.001\PT0\A1"
config, checkpoint_paths = get_config(logdir)

# writer = SummaryWriter(logdir)
# print("SummaryWriter created")

model = get_model(config, checkpoint_paths[-1])
# config.start_epoch = int(os.path.basename(checkpoint_paths[0]).split('.')[0].split('_')[-1])
# image_logger = Log_Images(None, config, writer)
# log = image_logger.logging(model,config.start_epoch,True)
# writer.close()

if config.nr_of_classes in [2, 7, 51, 107, 50]:  # freesurfer colors available
    colors = rgb_map_for_data(config.nr_of_classes)
else:
    colors = plt.cm.hsv(np.linspace(0, 1, config.nr_of_classes))
    colors = colors[:, :-1] * 255
color_range = np.zeros((256, 3))
color_range[: colors.shape[0], :] = colors

photo = True
mri = False

if photo:
    for remove_bg, mirror in [(False, False), (False, True), (True, False), (True, True)]:
        print(f"Remove background: {remove_bg}, Mirror: {mirror}")
        img_dir = "/om2/user/sabeen/dandi/dandi_000108"
        pred_dir = f"{logdir}/dandi_000108"
        if remove_bg and not mirror:
            img_dir = "/om2/user/sabeen/dandi/dandi_000108_photoshop"
            pred_dir = f"{logdir}/dandi_000108_photoshop"
        elif not remove_bg and mirror:
            img_dir = "/om2/user/sabeen/dandi/dandi_000108_mirror"
            pred_dir = f"{logdir}/dandi_000108_mirror"
        elif remove_bg and mirror:
            img_dir = "/om2/user/sabeen/dandi/dandi_000108_photoshop_mirror"
            pred_dir = f"{logdir}/dandi_000108_photoshop_mirror"

        os.makedirs(pred_dir, exist_ok=True)

        img_files = os.listdir(img_dir)

        img_shape = (256, 256)
        img_list = []

        for i, file in enumerate(img_files):
            file_path = os.path.join(img_dir, file)

            # Load the RGB image
            rgb_image = Image.open(file_path)

            # Resize the image to match the input size expected by your model
            resize = transforms.Resize((162, 194))
            rgb_resized = resize(rgb_image)

            # Convert the RGB image to grayscale
            if not config.pretrained:
                grayscale_image = rgb_resized.convert("L")
            else:
                grayscale_image = rgb_resized

            # Convert the grayscale image to a PyTorch tensor
            img_tensor = transforms.ToTensor()(grayscale_image)
            img_list.append(img_tensor)

        img_batch = torch.stack(img_list)
        probs = model(img_batch)
        preds = probs.argmax(1)

        for i, file in enumerate(img_files):
            pred = preds[i, :, :].numpy().astype("uint8")
            p = create_plot(
                pred,
                "",
                color_range,
                fig_path=str(os.path.join(pred_dir, f"{file.split('.')[0]}_pred.png")),
            )
            real_img = img_batch[i, :, :, :].squeeze()
            if len(real_img.shape) > 2:
                real_img = real_img.permute(1, 2, 0)
            cv2.imwrite(
                str(os.path.join(pred_dir, f"{file.split('.')[0]}_gray.png")),
                (real_img.numpy() * 255).astype(np.uint8),
            )

        print("done!")

if mri:
    for mri_type in ['T2']:
        if mri_type == 'T1':
            img_dir = "/om2/user/sabeen/dandi/sub-KC001_T1map"
            pred_dir = f"{logdir}/sub-KC001_T1map"
        else:
            img_dir = "/om2/user/sabeen/dandi/sub-KC001_T2starmap"
            pred_dir = f"{logdir}/sub-KC001_T2starmap"

        os.makedirs(pred_dir, exist_ok=True)

        img_files = os.listdir(img_dir)[:3]

        img_shape = (162, 194)
        img_list = []

        for i, file in enumerate(img_files):
            file_path = os.path.join(img_dir, file)

            # Load the RGB image
            rgb_image = Image.open(file_path)

            # Resize the image to match the input size expected by your model
            resize = transforms.Resize((162, 194))
            rgb_resized = resize(rgb_image)

            # Convert the RGB image to grayscale
            # if not config.pretrained:
            #     grayscale_image = rgb_resized.convert("L")
            # else:
            grayscale_image = rgb_image

            # Convert the grayscale image to a PyTorch tensor
            img_tensor = transforms.ToTensor()(grayscale_image)
            img_list.append(img_tensor)

        img_batch = torch.stack(img_list)
        probs = model(img_batch)
        preds = probs.argmax(1)

        for i, file in enumerate(img_files):
            pred = preds[i, :, :].numpy().astype("uint8")
            p = create_plot(
                pred,
                "",
                color_range,
                fig_path=str(os.path.join(pred_dir, f"{file.split('.')[0]}_pred.png")),
            )
            real_img = img_batch[i, :, :, :].squeeze()
            if len(real_img.shape) > 2:
                real_img = real_img.permute(1, 2, 0)
            cv2.imwrite(
                str(os.path.join(pred_dir, f"{file.split('.')[0]}_gray.png")),
                (real_img.numpy() * 255).astype(np.uint8),
            )

        print("done!")