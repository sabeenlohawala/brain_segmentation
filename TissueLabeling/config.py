import argparse
import json
import os
from datetime import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

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

    def __init__(self, args=None, config_file_name=None):
        self.logdir = getattr(args, "logdir", os.getcwd())

        if not os.path.isabs(self.logdir):
            self.logdir = os.path.join(
                os.getcwd(),
                "logs",
                self.logdir,
            )

        if not os.path.isdir(self.logdir):
            os.makedirs(self.logdir)

        self.model_name = getattr(args, "model_name", "segformer")
        self.nr_of_classes = getattr(args, "nr_of_classes", 51)
        self.num_epochs = getattr(args, "num_epochs", 20)
        self.batch_size = getattr(args, "batch_size", 64)
        self.lr = getattr(args, "lr", 6e-5)

        self.data_dir = getattr(
            args, "data_dir", "/om2/user/sabeen/nobrainer_data_norm/new_small_no_aug_51"
        )
        self.seed = getattr(args, "seed", 42)
        self.save_every = "epoch"
        self.precision = "32-true"  # "16-mixed"
        self.pretrained = getattr(args, "pretrained", True)

        self._commit_hash = ext_utils.get_git_revision_short_hash()
        self._created_on = f'{datetime.now().strftime("%A %m/%d/%Y %H:%M:%S")}'

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

        print("CONFIG", config_file)

        with open(config_file, "w", encoding="utf-8") as outfile:
            print("writing config file...")
            outfile.write(json_object)

    @classmethod
    def read_config(cls, file_name):
        """Read configuration from a file"""
        with open(file_name, "r", encoding="utf-8") as fh:
            config_dict = json.load(fh)

        return argparse.Namespace(**config_dict)


if __name__ == "__main__":
    config = Configuration()
