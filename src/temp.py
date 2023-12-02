import torch
import os
from data.dataset import get_data_loader
from models.metrics import Dice
from models.segformer import Segformer
from training.trainer import Trainer
from lightning.fabric import Fabric, seed_everything
from torch.utils.data import Dataset, DataLoader
import webdataset as wds
import numpy as np
import glob

# 1. for idx, (brain, mask) in enumerate(item):
#     np.save(f"brain_{idx}.npy")
#     np.save(f"mask_{idx}.npy")

# uncomment below to do (1)
def create_wds(url):
    # normalization_constants_internal = np.load(
    #     f"/om2/user/sabeen/nobrainer_data_norm/data_prepared_segmentation_small/normalization_constants.npy"
    # )
    d = (
        wds.WebDataset(
            url,
            # nodesplitter=nodesplitter if torch.cuda.device_count() > 1 else wds.shardlists.single_node_only,
        )
        # .shuffle(0)#5000 if training else 0) # shuffle the dataset with a buffer
        .decode()  # decode all files in the shard
        .to_tuple("brain.pth", "mask.pth", "__key__")  # load brain and mask
        # .map(normalize_internal)  # normalize brain slice
    )
    return d

for subset in ['train']:#,'validation','test']:
    data_dir = f'/om2/user/sabeen/nobrainer_data_norm/data_prepared_segmentation_medium/{subset}'
    save_dir = f'{data_dir}/extracted_tensors'
    tar_files = glob.glob(f'{data_dir}/*.tar')
    # print(len(tar_files), tar_files[0])
    # for i in range(1):
    for tar_file in tar_files:
        d = create_wds(tar_file)
        dataset = [item for item in d]
        keys = []
        for idx, (brain, mask, key) in enumerate(dataset):
            keys.append(key)
            brain_filename = f'{save_dir}/brain_{key}.npy'
            mask_filename = f'{save_dir}/mask_{key}.npy'
            np.save(brain_filename,brain)
            np.save(mask_filename,mask)

            # brain_loaded = torch.from_numpy(np.load(brain_filename))
            # mask_loaded = torch.from_numpy(np.load(mask_filename))
    keys_filename = f'{save_dir}/keys.npy'
    np.save(keys_filename,keys)

# 2. create a custom dataset
# class custom():
#     self.images = glo.glob('brain*.npy')
#     self.mask = glo.glob('mask*.npy')

#     def __get_*():
#         retrun np.load(self.images[idx], ....)
    
#     def__len__():
#         return len(self.images)

# uncomment below to do (2)
# class NewCustomDataset(Dataset):
#     def __init__(self,file_dir):
#         # save_dir = '/om2/user/sabeen/nobrainer_data_norm/data_prepared_segmentation_small/train/extracted_tensors'
#         self.images = glob.glob(f'{file_dir}/brain*.npy')
#         self.masks = glob.glob((f'{file_dir}/mask*.npy'))
#         # self.keys = np.load(f'{file_dir}/keys.npy')
    
#     def __getitem__(self,idx):
#         # returns (image, mask)
#         return torch.from_numpy(np.load(self.images[idx])), torch.from_numpy(np.load(self.masks[idx]))
    
#     def __len__(self):
#         return len(self.images)
    
# train_data = NewCustomDataset('/om2/user/sabeen/nobrainer_data_norm/data_prepared_segmentation_small/train/extracted_tensors')
# val_data = NewCustomDataset('/om2/user/sabeen/nobrainer_data_norm/data_prepared_segmentation_small/validation/extracted_tensors')
# test_data = NewCustomDataset('/om2/user/sabeen/nobrainer_data_norm/data_prepared_segmentation_small/test/extracted_tensors')

# print(len(train_data), len(val_data), len(test_data))