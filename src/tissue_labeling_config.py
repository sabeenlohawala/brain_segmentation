import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

class Configuration:

    def __init__(self, args, config_file_name=None):
        self.logdir = getattr(args, "logdir", os.getcwd())

        if not os.path.isabs(self.logdir):
            self.logdir = os.path.join(
                os.getcwd(),
                "logs",
                self.logdir,
            )

        if not os.path.isdir(self.logdir):
            os.makedirs(self.logdir)

        self.wandb_description = getattr(args, 'wandb_description', None)
        self.wandb_on = self.wandb_description is not None
        self.wandb_run_title = "Brain Segmentation"

        self.model_name = getattr(args, 'model_name', 'segformer')
        self.nr_of_classes = 51 # TODO: pass as command line arg
        self.num_epochs = getattr(args, "num_epochs", 20)
        self.batch_size = getattr(args, 'batch_size', 64)
        self.lr = getattr(args, 'lr', 6e-5)
        self.data_dir = getattr(args, 'data_dir', '/om2/user/sabeen/nobrainer_data_norm/new_small_no_aug_51')
        self.seed = getattr(args,'seed',42)
        self.save_every = "epoch"
        self.precision = '32-true' #"16-mixed"
        self.pretrained = getattr(args,'pretrained',True)

        self.write_config(config_file_name)

        # TODO: edit to work with this code base
        # self.COMMIT_HASH = ext_utils.get_git_revision_short_hash()
        # self.CREATED_ON = f'{datetime.now().strftime("%A %m/%d/%Y %H:%M:%S")}'

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

        print('CONFIG', config_file)

        with open(config_file, "w") as outfile:
            print('writing config file...')
            outfile.write(json_object)
    
    @classmethod
    def read_config(self, file_name):
        """Read configuration from a file"""
        with open(file_name, "r") as fh:
            config_dict = json.load(fh)

        return argparse.Namespace(**config_dict)


if __name__ == "__main__":
    config = Configuration()