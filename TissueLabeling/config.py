import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch

import ext.utils as ext_utils


class Configuration:
    """
    Initializes an instance of the class.

    Parameters:
        args (object): An object containing arguments.
        config_file_name (str, optional): The name of the configuration file. Defaults to None.

    Returns:
        None
    """

    data_root_dir = "/om2/user/sabeen/nobrainer_data_norm"

    def __init__(self, args=None, config_file_name=None):
        self.logdir = getattr(args, "logdir", os.getcwd())
        if not os.path.isabs(self.logdir):
            self.logdir = os.path.join(
                os.getcwd(),
                "results",
                self.logdir,
            )
        if not os.path.isdir(self.logdir):
            os.makedirs(self.logdir, exist_ok=True)

        self.model_name = getattr(args, "model_name", "segformer")
        self.pretrained = (
            getattr(args, "pretrained", 0) == 1 and self.model_name == "segformer"
        )
        self.nr_of_classes = getattr(args, "nr_of_classes", 51)
        self.num_epochs = getattr(args, "num_epochs", 20)
        self.batch_size = getattr(args, "batch_size", 64)
        self.lr = getattr(args, "lr", 1e-3)

        self.loss_fn = getattr(args, "loss_fn", "dice")
        self.loss_fn = self.loss_fn.lower()
        self.metric = getattr(args, "metric", "dice")
        self.metric = self.metric.lower()
        self.class_specific_scores = (
            getattr(args, "class_specific_scores", 0) if self.metric == "dice" else 0
        )

        self.new_kwyk_data = getattr(args, "new_kwyk_data", 1)
        self.data_dir = getattr(args, "data_dir", "")
        self.data_size = getattr(args, "data_size", "small")
        self.augment = getattr(args, "augment", 0)
        self.aug_mask = getattr(args, "aug_mask", 0)
        self.aug_cutout = getattr(args, "aug_cutout", 0)
        self.cutout_n_holes = (
            getattr(args, "cutout_n_holes", 0) if self.aug_cutout == 1 else 0
        )
        self.cutout_length = (
            getattr(args, "cutout_length", 0) if self.aug_cutout == 1 else 0
        )
        self.mask_n_holes = (
            getattr(args, "mask_n_holes", 0) if self.aug_mask == 1 else 0
        )
        self.mask_length = getattr(args, "mask_length", 0) if self.aug_mask == 1 else 0
        self.intensity_scale = getattr(args, "intensity_scale", 0)
        self.null_half = getattr(args,"null_half",0)

        self.debug = getattr(args, "debug", 0)

        self.seed = getattr(args, "seed", 42)
        self.precision = "32-true"  # "16-mixed"

        self.save_checkpoint = (
            True if getattr(args, "save_checkpoint", 1) == 1 else False
        )
        self.log_images = True if getattr(args, "log_images", 0) == 1 else False
        self.checkpoint_freq = getattr(args, "checkpoint_freq", 10)
        self.image_log_freq = getattr(args, "image_log_freq", 10)
        self.checkpoint = getattr(args, "checkpoint", None)
        self.start_epoch = getattr(args, "start_epoch", 0)
        self.save_every = "epoch"

        self.wandb_description = getattr(args, "wandb_description")
        self.wandb_on = self.wandb_description is not None

        self._commit_hash = ext_utils.get_git_revision_short_hash()
        self._created_on = f'{datetime.now().strftime("%A %m/%d/%Y %H:%M:%S")}'

        self._update_data_dir()
        self.write_config(config_file_name)

    def write_config(self, file_name=None):
        """Write configuration to a file
        Args:
            CONFIG (dict): configuration
        """
        file_name = file_name if file_name else "config.json"

        dictionary = self.__dict__
        json_object = json.dumps(
            dictionary,
            sort_keys=True,
            indent=4,
            separators=(", ", ": "),
            ensure_ascii=False,
            # cls=NumpyEncoder,
        )

        config_file = os.path.join(dictionary["logdir"], file_name)

        with open(config_file, "w", encoding="utf-8") as outfile:
            print("writing config file...")
            outfile.write(json_object)

    @classmethod
    def read_config(cls, file_name):
        """Read configuration from a file"""
        with open(file_name, "r", encoding="utf-8") as fh:
            config_dict = json.load(fh)

        return argparse.Namespace(**config_dict)

    def _update_data_dir(self):
        """Update the data directory based on the number of classes"""
        if self.new_kwyk_data:
            self.data_dir = "/om2/user/sabeen/kwyk_final"
            self.aug_dir = ""
        else:
            if self.data_size == "small":
                folder_map = {
                    107: "new_small_aug_107",
                    51: "new_small_no_aug_51",
                    2: "new_small_no_aug_51",
                    7: "new_small_aug_107",
                    17: "new_small_no_aug_51"
                }
            elif self.data_size == "med" or self.data_size == "medium":
                folder_map = {51: "new_med_no_aug_51", 2: "new_med_no_aug_51", 7: "new_med_no_aug_51", 17: "new_med_no_aug_51"}
            else:
                sys.exit(
                    f"{self.data_size} is not a valid dataset size. Choose from 'small' or 'med'."
                )

            if self.nr_of_classes in folder_map:
                self.data_dir = os.path.join(
                    self.data_root_dir, folder_map[self.nr_of_classes]
                )
            else:
                sys.exit(
                    f"No dataset found for {self.nr_of_classes} classes, {self.data_size} size"
                )

            if self.augment:
                if self.nr_of_classes == 51 and self.data_size == "small":
                    self.aug_dir = os.path.join(
                        self.data_root_dir, "20240217_small_synth_aug"
                    )
                if self.nr_of_classes == 51 and self.data_size == "med":
                    self.aug_dir = os.path.join(
                        self.data_root_dir, "20240217_med_synth_aug"
                    )
            else:
                self.aug_dir = ""


if __name__ == "__main__":
    config = Configuration()
