"""
Filename: tissue_labeling/scripts/main.py
Created Date: Wednesday, December 13th 2023
Author: Sabeen Lohawala
Description: Training code of TissueLabeling

Copyright (c) 2023, Sabeen Lohawala. MIT
"""
import argparse
import glob
import json
import os
import sys

import torch
from torch import Tensor
import numpy as np
import wandb
from torchmetrics import Metric

from TissueLabeling.config import Configuration
from TissueLabeling.metrics.metrics import Classification_Metrics
from TissueLabeling.data.dataset import get_data_loader, NoBrainerDataset
from TissueLabeling.metrics.metrics import Dice
from TissueLabeling.models.segformer import Segformer
from TissueLabeling.models.unet import Unet
from TissueLabeling.models.simple_unet import SimpleUnet
from TissueLabeling.parser import get_args
from TissueLabeling.training.trainer import Trainer
from TissueLabeling.utils import init_cuda, init_fabric, init_wandb, set_seed, main_timer

def update_config(config):
    """
    Updates the config file based on the command line arguments.
    """
    if sys.argv[1] == "train":
        config = Configuration(config)

    elif sys.argv[1] == "resume-train":
        chkpt_folder = os.path.join('results/', config.logdir)

        config_file = os.path.join(chkpt_folder, "config.json")
        if not os.path.exists(config_file):
            sys.exit(f"Configuration file not found at {config_file}")

        with open(config_file) as json_file:
            data = json.load(json_file)
        assert isinstance(data, dict), "Invalid Object Type"

        dice_list = sorted(glob.glob(os.path.join(chkpt_folder, "checkpoint*")))
        if not dice_list:
            sys.exit("No checkpoints exist to resume training")

        data["checkpoint"] = dice_list[0]
        data["start_epoch"] = int(os.path.basename(dice_list[0]).split('.')[0].split('_')[-1])
        
        args = argparse.Namespace(**data)
        config = Configuration(args, "config_resume.json")

    else:
        sys.exit("Invalid Sub-command")

    return config

def generate_w_multi_gpu(config,save_path):
    print('generating files...')
    # generate and test new files on multiple gpus
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    num_devices = 2
    fabric = init_fabric(precision=config.precision,devices=num_devices)

    # model
    model = Segformer(nr_of_classes=config.nr_of_classes,pretrained=False)

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    # loss function
    old_dice = Dice(fabric, config)
    new_dice = NewDice()

    # dataloaders
    train_loader, _, _ = get_data_loader(config,)

    # fabric setup
    train_loader = fabric.setup_dataloaders(train_loader)
    model, optimizer = fabric.setup(model, optimizer)

    print(f"Saving untrained model...")
    model_save_path = f"{save_path}/checkpoint_00.ckpt"
    state = {
        "model": model,
        "optimizer": optimizer,
    }
    fabric.save(model_save_path, state)

    old_losses = []
    for epoch in range(1,2):
        print('training...')
        model.train()
        for i, (image,mask) in enumerate(train_loader):
            optimizer.zero_grad()
            probs = model(image)

            # loss functions
            old_dice_loss, old_class_dice = old_dice(mask.long(), probs,debug=False)
            old_losses.append(old_dice_loss.item())
            print(f'Process {fabric.global_rank} pre-backward old dice loss: {old_dice_loss}')

            new_dice_loss,new_class_dice = new_dice(mask,probs)
            print(f'Process {fabric.global_rank} pre-backward new dice loss: {new_dice_loss}')

            # loss by hand
            # convert mask to one-hot
            y_true_oh = torch.nn.functional.one_hot(
                mask.long().squeeze(1), num_classes=config.nr_of_classes
            ).permute(0, 3, 1, 2)

            # calculate intersect and denom without weights
            class_intersect = torch.sum((y_true_oh * probs), axis=(2, 3))
            class_union = torch.sum((y_true_oh + probs), axis=(2, 3))

            fabric.backward(new_dice_loss) # TODO
            print(f'Process {fabric.global_rank} post-backward new dice loss: {new_dice_loss}')
            optimizer.step() # TODO
            # train_metrics # TODO

            fabric.barrier()
            # if fabric.global_rank == 0:
            with torch.no_grad():
                print('Gathering...')
                class_intersect_gather = fabric.all_gather(class_intersect)
                class_union_gather = fabric.all_gather(class_union)
                # train_metrics.sync(fabric)

            print('saving files...')
            # save (brain,mask,pred) from each rank
            torch.save([image.cpu(),mask.cpu(),probs.cpu()],os.path.join(save_path,f'image_mask_probs_{fabric.global_rank}.pt'))
            # save (class_intersect, class_union) from each rank
            torch.save([class_intersect.cpu(),class_union.cpu()],os.path.join(save_path,f'itersect_denom_{fabric.global_rank}.pt'))
            # save (class_intersect_gather, class_union_gather)
            torch.save([class_intersect_gather.cpu(), class_union_gather.cpu()],os.path.join(save_path,f'itersect_denom_gather_{fabric.global_rank}.pt'))
            # save old dice_loss and class_dice
            torch.save([old_dice_loss,old_class_dice],os.path.join(save_path,f'old_dice_{fabric.global_rank}.pt'))
            # save new dice_loss and class_dice
            torch.save([new_dice_loss,new_class_dice],os.path.join(save_path,f'new_dice_{fabric.global_rank}.pt'))

            break

def main():
    args = get_args()
    config = update_config(args)
    set_seed(config.seed)
    init_cuda()
    
    save_path = '/om2/user/sabeen/nobrainer_data_norm/test_dice_data/51class_test/'

    # uncomment below to generate outputs from multi-gpu case
    # generate_w_multi_gpu(config,save_path)
    
    # generate and test new files on multiple gpus
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        generate_w_multi_gpu(config,save_path)
    
    model = Segformer(nr_of_classes=config.nr_of_classes,pretrained=False)
    model.load_state_dict(torch.load(f'{save_path}/checkpoint_00.ckpt')['model'])

    # testing existing files
    # loss_fn = Dice(fabric, config)
    # new_dice = NewDice()

    # get image/mask/probs from each multi_gp
    image_0,mask_0,probs_0 = torch.load(os.path.join(save_path,'image_mask_probs_0.pt'))
    image_1,mask_1,probs_1 = torch.load(os.path.join(save_path,'image_mask_probs_1.pt'))

    # get class_intersect/class_union
    class_intersect_0, class_union_0 = torch.load(os.path.join(save_path,f'itersect_denom_0.pt'))
    class_intersect_1, class_union_1 = torch.load(os.path.join(save_path,f'itersect_denom_1.pt'))

    # get gathered
    class_intersect_gather_0, class_union_gather_0 = torch.load(os.path.join(save_path,f'itersect_denom_gather_0.pt'))
    class_intersect_gather_1, class_union_gather_1 = torch.load(os.path.join(save_path,f'itersect_denom_gather_1.pt'))

    # get dice metrics from Matthias's code
    old_dice_loss_0, old_class_dice_0 = torch.load(os.path.join(save_path,f'old_dice_0.pt'))
    old_dice_loss_1, old_class_dice_1 = torch.load(os.path.join(save_path,f'old_dice_1.pt'))

    # get dice metrics from new torch metrics custom metric
    new_dice_loss_0, new_class_dice_0 = torch.load(os.path.join(save_path,f'new_dice_0.pt'))
    new_dice_loss_1, new_class_dice_1 = torch.load(os.path.join(save_path,f'new_dice_1.pt'))

    print()

    # # probs = model(image)
    # mask_0 = fabric.to_device(mask_0)
    # probs_0 = fabric.to_device(probs_0)
    # loss, classDice = loss_fn(mask_0.long().to('cuda:0'),probs_0.to('cuda:0'),debug=False)
    # new_loss,class_dice = new_dice(mask_0,probs_0)
    # print(loss)
    # print(new_loss)
    # print()

class NewDice(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("class_intersect", default=torch.zeros((1,51)), dist_reduce_fx="sum")
        self.add_state("class_union", default=torch.zeros((1,51)), dist_reduce_fx="sum")

    def update(self, target: Tensor, preds: Tensor) -> None:
        # preds, target = self._input_format(preds, target)

        y_true_oh = torch.nn.functional.one_hot(
            target.long().squeeze(1), num_classes=preds.shape[1]
        ).permute(0, 3, 1, 2)

        self.class_intersect = torch.sum((y_true_oh * preds), axis=(0, 2, 3)).reshape((1,51))
        self.class_union = torch.sum((y_true_oh + preds), axis=(0, 2, 3)).reshape((1,51))

    def compute(self) -> Tensor:
        all_intersect = torch.sum(self.class_intersect, axis=1)
        all_union = torch.sum(self.class_union, axis=1)

        class_dice = 2.0 * self.class_intersect / self.class_union
        dice_coeff = 2.0 * all_intersect / all_union
        dice_loss = 1 - dice_coeff
        
        return dice_loss, class_dice


if __name__ == "__main__":
    main()