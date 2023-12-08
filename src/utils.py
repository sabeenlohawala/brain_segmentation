
import os
import nibabel as nib
import torch
import random
import numpy as np
import lightning as L
from lightning.fabric import Fabric, seed_everything
import shutil
import wandb
import datetime
from typing import Tuple
import csv

# SEED = 42

def load_brains(image_file : str, mask_file : str, file_path : str):

        # ensure that mask and image numbers match
        image_nr = image_file.split('_')[1]
        mask_nr = mask_file.split('_')[1]
        assert image_nr == mask_nr, 'image and mask numbers do not match'

        image_path = os.path.join(file_path, image_file)
        mask_path = os.path.join(file_path, mask_file)

        brain = nib.load(image_path)
        brain_mask = nib.load(mask_path)

        brain = brain.get_fdata()
        brain_mask = brain_mask.get_fdata()
        brain_mask = brain_mask.astype(int)
        # apply skull stripping
        brain[brain_mask==0]=0

        return brain, brain_mask, image_nr

def set_seed(seed : int = 0) -> None:
    '''Set the seed before GPU training

    Args:
        seed (int, optional): seed. Defaults to 0.
    '''
    seed_everything(seed)
    
    if torch.cuda.is_available():
        # determines if cuda selects only deterministic algorithms or not
        # True = Only determinstic algo --> slower but reproducible
        torch.backends.cudnn.deterministic = False
        # determines if cuda should always select the same algorithms
        # (!! use only for fixed size inputs !!)
        # False = Always same algo --> slower but reproducible
        torch.backends.cudnn.benchmark = True

def crop(image : np.array, height: int, width : int) -> np.array:

    # find image-optimal crop
    for j in range(256):
        if (image[j]!=0).any():
            cut_top_temp = j
            break
    for j in range(256):
        if (image[255-j]!=0).any():
            cut_bottom_temp = 255-j
            break
    for j in range(256):
        if (image[:,j]!=0).any():
            cut_left_temp = j
            break
    for j in range(256):
        if (image[:,255-j]!=0).any():
            cut_right_temp = 255-j
            break

    # image-optimal size:
    height_temp = cut_bottom_temp-cut_top_temp+1
    width_temp = cut_right_temp-cut_left_temp+1
    assert height_temp<=height, "Crop height is too big"
    assert width_temp<=width, "Crop width is too big"

    # crop image-optimal patch:
    image = image[cut_top_temp:cut_bottom_temp+1, cut_left_temp:cut_right_temp+1]

    if image.shape[0] < 50 or image.shape[1] < 50:
        pass

    assert (image > 0).any(), "Crop is empty"

    # adjust the crop to largest rectangle
    if height_temp<height:
        diff = height-height_temp
        # even difference
        if (diff % 2) == 0:
            image = np.pad(image, ((diff//2,diff//2),(0, 0)))
        # odd difference
        else:
            image = np.pad(image, ((diff//2,diff//2+1),(0, 0)))
    if width_temp<width:
        diff = width-width_temp
        # even difference
        if (diff % 2) == 0:
            image = np.pad(image, ((0, 0),(diff//2,diff//2)))
        # odd difference
        else:
            image = np.pad(image, ((0, 0),(diff//2,diff//2+1)))

    return image

def init_cuda() -> None:

    torch.cuda.empty_cache()

    if torch.cuda.is_available():
        # Ampere GPUs (like A100) allow to use TF32 (which is faster than FP32)
        # see https://pytorch.org/docs/stable/notes/cuda.html
        # per default, TF32 is activated for convolutions
        print("Use TF32 for convolutions: ", torch.backends.cudnn.allow_tf32)
        # we manually activate it for matmul
        if "A100" in torch.cuda.get_device_name(0):
            torch.set_float32_matmul_precision('high')
        print("Use TF32 for matmul: ", torch.backends.cuda.matmul.allow_tf32)

        # reproducability vs speed (see set_seed function)
        # https://pytorch.org/docs/stable/notes/randomness.html
        print("Only use determnisitc CUDA algorithms: ", torch.backends.cudnn.deterministic)
        print("Use the same CUDA algorithms for each forward pass: ", torch.backends.cudnn.benchmark)


def init_wandb(wandb_on : bool, project_name : str, fabric : L.fabric, model_params : dict, description : str) -> None:
    if wandb_on:
        # check if staged artifacts exist:
        if os.path.exists("/home/sabeen/.local/share/wandb"):
            shutil.rmtree("/home/sabeen/.local/share/wandb")

        wandb.init(
        name=f'{fabric.device}-{datetime.datetime.now().month}-{datetime.datetime.now().day}-{datetime.datetime.now().hour}:{datetime.datetime.now().minute}',
        group=f'test-multigpu-{datetime.datetime.now().month}-{datetime.datetime.now().day}',
        # group=f'{datetime.datetime.now().month}-{datetime.datetime.now().day}-{datetime.datetime.now().hour}:{datetime.datetime.now().minute}',
        project=project_name,
        entity="tissue-labeling-sabeen",
        notes=description,
        config={**model_params},
        reinit=True,
        dir="/om2/scratch/Fri",)
        wandb.run.log_code("./data", include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb"))
        wandb.run.log_code("./models", include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb"))
        wandb.run.log_code("./trainer", include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb"))

def init_fabric(**kwargs) -> L.fabric:

    fabric = Fabric(**kwargs)
    fabric.launch()
    
    if torch.cuda.device_count() > 1:
        # see: https://pytorch-lightning.readthedocs.io/en/1.9.0/_modules/lightning_fabric/strategies/ddp.html
        # fabric._strategy._ddp_kwargs['broadcast_buffers']=False

        # make environment infos available
        os.environ['RANK'] = str(fabric.global_rank)
        # local world size
        os.environ['WORLD_SIZE'] = str(torch.cuda.device_count())

        print(f"Initialize Process: {fabric.global_rank}")


    return fabric


def brain_coord(slice : torch.tensor) -> Tuple[int, int, int, int]:
  '''
  Computes the coordnates of a rectangle
  that contains the brain area of a slice

  Args:
      slice (torch.tensor): brain slice of shape [H, W]

  Returns:
      Tuple(int, int, int, int): Coordinates of the rectangle
  '''
  
  H, W = slice.shape[-2:]
  H, W = H-1, W-1
  # majority vote of the 4 corners
  bg_value = torch.stack((slice[0,0], slice[-1,-1], slice[0,-1], slice[-1,0])).mode()[0]
  
  for j in range(H):
      if (slice[j]!=bg_value).any():
          cut_top_temp = j
          break
  for j in range(H):
      if (slice[H-j]!=bg_value).any():
          cut_bottom_temp = H-j
          break
  for j in range(W):
      if (slice[:,j]!=bg_value).any():
          cut_left_temp = j
          break
  for j in range(W):
      if (slice[:,W-j]!=bg_value).any():
          cut_right_temp = W-j
          break
      
  return (cut_top_temp, cut_bottom_temp, cut_left_temp, cut_right_temp)

def brain_area(slice : torch.tensor) -> torch.tensor:
  '''
  Computes the brain area of a slice

  Args:
      slice (torch.tensor): brain slice

  Returns:
      torch.tensor: brain area
  '''
  
  cut_top_temp, cut_bottom_temp, cut_left_temp, cut_right_temp = brain_coord(slice)
  
  return slice[cut_top_temp:cut_bottom_temp+1, cut_left_temp:cut_right_temp+1]

def mapping(mask: np.array):
    class_mapping = {}
    # labels = []
    with open('/home/matth406/unsupervised_brain/data/class-mapping.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        # skip header
        next(spamreader, None)
        for row in spamreader:
            class_mapping[int(row[1])] = int(row[4])
    #         labels.append(int(row[1]))
    # labels = np.array(labels)

    # labels = []
    # with open('/home/matth406/unsupervised_brain/data/class-mapping.csv', newline='') as csvfile:
    #     spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    #     # skip header
    #     next(spamreader, None)
    #     for row in spamreader:
    #         labels.append(int(row[1]))
    # labels = np.array(labels)
    # # labels = torch.tensor([   0, 1024,    2,    3,    4,    5, 1025,    7,    8, 1026,   10,   11,
    # #       12,   13,   14,   15,   16,   17,   18, 1034, 1035,   24,   26,   28,
    # #       30,   31,   41,   42,   43,   44,   46,   47, 1027,   49,   50,   51,
    # #       52,   53,   54, 1028,   58, 1029,   60,   62,   63, 1030, 1031,   72,
    # #     1032,   77, 1033,   80,   85,  251,  252,  253,  254,  255, 1009, 1010,
    # #     1011, 2034, 1012, 1013, 1014, 2035, 1015, 1007, 2033, 2000, 2001, 2002,
    # #     2003, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015,
    # #     2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 1000, 2025, 1002, 1003,
    # #     2024, 1005, 1006, 2031, 2032, 1008, 1001, 2026, 2027, 2028, 2029, 2030,
    # #     1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023])

    # class_mapping = {value.item(): index for index, value in enumerate(labels)}
    u, inv = np.unique(mask, return_inverse=True)
    num_classes = 50 #len(class_mapping)
    for x in u:
        if x not in class_mapping:
            class_mapping[x] = num_classes
    
    # # we collect all classes not in the mapping table as an additional "other" class
    # mask = np.array([class_mapping[int(x)] if x in labels else len(labels) for x in u])[inv].reshape(mask.shape)
    for old,new in class_mapping.items():
        mask[mask == old] = -1*new
    mask = mask * -1
    
    return mask