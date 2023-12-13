"""
Filename: tissue_labeling/scripts/main.py
Created Date: Friday, December 8th 2023
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
import numpy as np
import wandb

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

def main():
    """
    The main function that executes the entire program.
    """
    args = get_args()

    config = update_config(args)
    save_path = '/om2/user/sabeen/nobrainer_data_norm/test_dice_data/51class/'

    config.binary = False
    if config.binary:
        config.nr_of_classes = 2
        save_path = '/om2/user/sabeen/nobrainer_data_norm/test_dice_data/binary/'
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    if len(os.listdir(save_path)) == 0:
        generate_files(config, save_path) # comment if files are already generated

    # compute the dice score using a single gpu
    print('loading files...')
    # get image/mask/probs from each multi_gp
    image_0,mask_0,probs_0 = torch.load(os.path.join(save_path,'image_mask_probs_0.pt'))
    image_1,mask_1,probs_1 = torch.load(os.path.join(save_path,'image_mask_probs_1.pt'))

    # get class_intersect/class_denom
    class_intersect_0, class_denom_0 = torch.load(os.path.join(save_path,f'itersect_denom_0.pt'))
    class_intersect_1, class_denom_1 = torch.load(os.path.join(save_path,f'itersect_denom_1.pt'))

    # get gathered
    class_intersect_gather_0, class_denom_gather_0 = torch.load(os.path.join(save_path,f'itersect_denom_gather_0.pt'))
    class_intersect_gather_1, class_denom_gather_1 = torch.load(os.path.join(save_path,f'itersect_denom_gather_1.pt'))

    # find class_intersect and class_denom for batch_size = 2
    image = torch.concat((image_0,image_1),axis=0)
    mask = torch.concat((mask_0,mask_1),axis=0)
    probs = torch.concat((probs_0,probs_1),axis=0)

    # calculate intersect and union
    y_true_oh = torch.nn.functional.one_hot(
        mask.long().squeeze(1), num_classes=config.nr_of_classes
    ).permute(0, 3, 1, 2)
    class_intersect = torch.sum(
        (y_true_oh * probs), axis=(2, 3)
        )
    class_denom = torch.sum(
        (y_true_oh + probs), axis=(2, 3)
    )

    # calculate class totals
    single_intersect_sum = torch.sum(class_intersect,axis=0)
    single_denom_sum = torch.sum(class_denom,axis=0)

    multi_intersect_sum = torch.sum(torch.concat((class_intersect_0,class_intersect_1)),axis=0)
    multi_denom_sum = torch.sum(torch.concat((class_denom_0,class_denom_1)),axis=0)

    gather_intersect_sum_0 = torch.sum(class_intersect_gather_0,axis=0)
    gather_denom_sum_0 = torch.sum(class_denom_gather_0,axis=0)

    gather_intersect_sum_1 = torch.sum(class_intersect_gather_1,axis=0)
    gather_denom_sum_1 = torch.sum(class_denom_gather_1,axis=0)
    
    # comparing single_gpu and multi_gpu
    # sanity_check: gather_0 should equal gather_1
    print(f'gather_intersect_sum_0 == gather_intersect_sum_1: {torch.sum(gather_intersect_sum_0 == gather_intersect_sum_1)} / {config.nr_of_classes} are equal')
    print(f'gather_denom_sum_0 == gather_denom_sum_1: {torch.sum(gather_denom_sum_0 == gather_denom_sum_1)} / {config.nr_of_classes} are equal')
    # sanity_check: multi should equal gather_0 and gather_1
    print(f'multi_intersect_sum == gather_intersect_sum_0: {torch.sum(multi_intersect_sum == gather_intersect_sum_0)} / {config.nr_of_classes} are equal')
    print(f'multi_denom_sum == gather_denom_sum_0: {torch.sum(multi_denom_sum == gather_denom_sum_0)} / {config.nr_of_classes} are equal')
    # Q: does single = gather?
    print(f'single_intersect_sum == gather_intersect_sum_0: {torch.sum(gather_intersect_sum_0 == single_intersect_sum)} / {config.nr_of_classes} are equal')
    print(f'single_denom_sum == gather_denom_sum_0: {torch.sum(gather_denom_sum_0 == single_denom_sum)} / {config.nr_of_classes} are equal')
    # Q: does single = multi?
    print(f'single_intersect_sum == multi_intersect_sum: {torch.sum(multi_intersect_sum == single_intersect_sum)} / {config.nr_of_classes} are equal')
    print(f'single_denom_sum == multi_denom_sum: {torch.sum(multi_denom_sum == single_denom_sum)} / {config.nr_of_classes} are equal')
    
def generate_files(config,save_path):
    # model
    model = Segformer(nr_of_classes=config.nr_of_classes,pretrained=False)
    # model = SimpleUnet(image_channels=1,num_classes=1)

    fabric = init_fabric(precision=config.precision)
    set_seed(config.seed)
    init_cuda()

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    # loss function
    loss_fn = Dice(fabric, config)

    train_loader, _, _ = get_data_loader(config,)
    train_loader = fabric.setup_dataloaders(train_loader)
    model, optimizer = fabric.setup(model, optimizer)

    # get and save predicted mask
    # train_metrics = Classification_Metrics(
    #         config.nr_of_classes, prefix="Train", wandb_on=config.wandb_on
    #     )

    # compare with dice score using multi gpu (set batch_size=1)
    for epoch in range(1,2):
        print('training...')
        model.train()
        for i, (image,mask) in enumerate(train_loader):
            mask[mask!=0] = 1
            optimizer.zero_grad()
            probs = model(image)

            loss, classDice = loss_fn(mask.long(),probs)
            
            # convert mask to one-hot
            y_true_oh = torch.nn.functional.one_hot(
                mask.long().squeeze(1), num_classes=config.nr_of_classes
            ).permute(0, 3, 1, 2)

            # calculate intersect and denom without weights
            class_intersect = torch.sum((y_true_oh * probs), axis=(2, 3))
            class_denom = torch.sum((y_true_oh + probs), axis=(2, 3))

            fabric.backward(loss)
            optimizer.step()
            # train_metrics.compute(mask.long(),probs,loss.item(),classDice)

            # skip validation

            print(f'Process {fabric.global_rank} barrier reached')
            fabric.barrier()
            # if fabric.global_rank == 0:
            with torch.no_grad():
                print('Gathering...')
                class_intersect_gather = fabric.all_gather(class_intersect)
                class_denom_gather = fabric.all_gather(class_denom)
            
            print('saving files...')
            # save (brain,mask,pred) from each rank
            torch.save([image.cpu(),mask.cpu(),probs.cpu()],os.path.join(save_path,f'image_mask_probs_{fabric.global_rank}.pt'))
            # save (class_intersect, class_denom) from each rank
            torch.save([class_intersect.cpu(),class_denom.cpu()],os.path.join(save_path,f'itersect_denom_{fabric.global_rank}.pt'))
            # save (class_intersect_gather, class_denom_gather)
            torch.save([class_intersect_gather.cpu(), class_denom_gather.cpu()],os.path.join(save_path,f'itersect_denom_gather_{fabric.global_rank}.pt'))

            # if fabric.global_rank == 0:
            #    train_metrics.log(epoch,commit=False,writer=None)  # TODO: check writer
            
            # train_metrics.reset()

            break

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

if __name__ == "__main__":
    main()
