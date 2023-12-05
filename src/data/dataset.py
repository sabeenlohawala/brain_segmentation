#!/.venv/bin/python
# -*- coding: utf-8 -*-
"""
@Author: Matthias Steiner
@Contact: matth406@mit.edu
@File: dataset.py
@Date: 2023/04/03
"""

import numpy as np
from torch.utils.data import Dataset
import torch
import webdataset as wds
from typing import Tuple
import glob

DATA_USER = 'sabeen' # alternatively, 'matth406'

class NoBrainerDataset(Dataset):
    def __init__(self,file_dir,pretrained):
        self.pretrained=pretrained
        self.images = glob.glob(f'{file_dir}/brain*.npy')
        self.masks = glob.glob((f'{file_dir}/mask*.npy'))
        self.images = self.images[:100]
        self.masks = self.masks[:100]
        self.normalization_constants = np.load(f"{file_dir}/../../normalization_constants.npy")
        # self.keys = np.load(f'{file_dir}/keys.npy')
    
    def __getitem__(self,idx):
        # returns (image, mask)
        image = torch.from_numpy(np.load(self.images[idx]))
        mask = torch.from_numpy(np.load(self.masks[idx]))

        # normalize image
        image = (image - self.normalization_constants[0]) / self.normalization_constants[1]

        if self.pretrained:
            return image.repeat((3,1,1)), mask
        return image, mask
    
    def __len__(self):
        return len(self.images)
    
    # def normalize(self,sample):
    #     image = sample[0]
    #     image = (image - self.normalization_constants[0]) / self.normalization_constants[1]
    #     sample = (image, sample[1], sample[2])
    #     return sample

def get_data_loader(data_dir : str, batch_size : int, pretrained: bool, num_workers : int = 4*torch.cuda.device_count()) -> Tuple[wds.WebLoader, wds.WebLoader, wds.WebLoader]:
    if data_dir[-1] == '/':
        data_dir = data_dir[:-1]
    train_dataset = NoBrainerDataset(f'{data_dir}/train/extracted_tensors', pretrained=pretrained)
    val_dataset = NoBrainerDataset(f'{data_dir}/validation/extracted_tensors', pretrained=pretrained)
    test_dataset = NoBrainerDataset(f'{data_dir}/test/extracted_tensors', pretrained=pretrained)
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size)

    return (train_loader, val_loader, test_loader)

# def sample_nr_in_wds(path : str, batch_size : int) -> int:
#     '''
#     Computes the number of samples in a list of shards located under path

#     Args:
#         path (str): path to shards
#         batch_size (int): batch size

#     Returns:
#         int: number of samples in the whole dataset
#     '''
#     files = sorted(os.listdir(path))

#     # load first shard
#     tar = tarfile.open(os.path.join(path, files[0]), "r")
#     namelist = tar.getnames()
#     # each sample consists of several files (masks, images, etc.)
#     nr_files_per_sample = sum([sample.split('.')[0] == "0" for sample in namelist])
#     samples_per_full_shard = len(namelist)/nr_files_per_sample
#     tar.close()

#     # load last shard
#     tar = tarfile.open(os.path.join(path, files[-1]), "r")
#     namelist = tar.getnames()
#     # each sample consists of several files (masks, images, etc.)
#     samples_in_last_shard = len(namelist)/nr_files_per_sample
#     tar.close()

#     # length of all shards
#     samples_in_all_shards = (len(files)-1)*samples_per_full_shard + samples_in_last_shard
#     samples_in_all_shards = samples_in_all_shards - (samples_in_all_shards % batch_size)

#     return int(samples_in_all_shards)

# def brain_data(data_path = './data/MPC/'):
#     '''Load nii.gz files, mask the brain and return numpy tensors

#     Args:
#         data_path (str, optional): Path to nii.gz files. Defaults to './data/MPC/'.

#     Returns:
#         tuple(np.array, np.aary): masked brain and dseg
#     '''
#     # path to brain data
#     brain_path = os.path.join(data_path, 'sub-100_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz')
#     brain_mask_path = os.path.join(data_path, 'sub-100_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz')
#     dseg_path = os.path.join(data_path, 'sub-100_space-MNI152NLin2009cAsym_dseg.nii.gz')
#     aseg_path = os.path.join(data_path, 'sub-100_desc-aseg_dseg.nii.gz')

#     # load brain data
#     brain = nib.load(brain_path)
#     brain_mask = nib.load(brain_mask_path)
#     dseg = nib.load(dseg_path)
#     aseg = nib.load(aseg_path)

#     # data to numpy
#     brain = brain.get_fdata()
#     brain_mask = brain_mask.get_fdata()
#     dseg = dseg.get_fdata()
#     aseg = aseg.get_fdata()

#     # mask brain
#     idx=(brain_mask==0)
#     brain[idx]=0

#     return (brain, dseg, aseg)

# class NoBrainerDatasetOLD(Dataset):

#     def __init__(
#         self,
#         mode : str = "train",
        
#     ) -> None:
        
#         path = f'/om2/user/matth406/nobrainer_data/data_prepared/{mode}/'
#         csv = 'idx.csv'
#         self.files = np.genfromtxt(os.path.join(path, csv), delimiter=',')

#     def __len__(self) -> int:
#         return len(self.files)
    
#     def __getitem__(self, idx):

#         # if batch
#         if torch.is_tensor(idx):
#             idx = idx.tolist()

#         # get file names
#         image = np.load(self.files[idx, 0])
#         mask = np.load(self.files[idx, 1])
#         if image is None:
#             raise NameError(f'Image not found at {self.files[idx, 0]}')

#         # normalize image
#         image = torch.from_numpy(image)
#         std, mean = torch.std_mean(image)
#         image_norm = (image - mean) / (std + 0.0000001)

#         return image_norm, mask, (mean, std)

# normalization_constants = np.load("/om2/user/matth406/nobrainer_data_norm/data_prepared_medium/normalization_constants.npy")
# def normalize(sample):
#     image = sample[0]
#     image = (image - normalization_constants[0]) / normalization_constants[1]
#     sample = (image, sample[1], sample[2])
#     return sample

# def NoBrainerDataset(dataset: str = "large", mode : str = "train", batchsize : int = 1) -> wds.WebDataset:

#     training = mode == "train"

#     if dataset == "small":
#         train_idxs = '0000{00..06}.tar'
#         val_idxs = '000000.tar'
#     elif dataset == "medium":
#         train_idxs = '000{000..314}.tar'
#         val_idxs = '0000{00..39}.tar'

#     if mode == "train":
#         url = f'/om2/user/{DATA_USER}/nobrainer_data_norm/data_prepared_segmentation_{dataset}/train/train-'+train_idxs
#         nr_of_samples = sample_nr_in_wds(f'/om2/user/{DATA_USER}/nobrainer_data_norm/data_prepared_segmentation_{dataset}/train/', batchsize)

#         # NR_OF_CLASSES = 6
#         # url = f'/om2/user/matth406/nobrainer_data/data_prepared_{dataset}/train/train-'+train_idxs
#         # nr_of_samples = sample_nr_in_wds(f'/om2/user/matth406/nobrainer_data/data_prepared_{dataset}/train/', batchsize)
#     elif mode == "validation":
#         url = f'/om2/user/{DATA_USER}/nobrainer_data_norm/data_prepared_segmentation_{dataset}/validation/validation-'+val_idxs
#         nr_of_samples = sample_nr_in_wds(f'/om2/user/{DATA_USER}/nobrainer_data_norm/data_prepared_segmentation_{dataset}/validation/', batchsize)
        
#         # NR_OF_CLASSES = 6
#         # url = f'/om2/user/matth406/nobrainer_data/data_prepared_{dataset}/validation/validation-'+val_idxs
#         # nr_of_samples = sample_nr_in_wds(f'/om2/user/matth406/nobrainer_data/data_prepared_{dataset}/validation/', batchsize)
#     elif mode == "test":
#         url = f'/om2/user/{DATA_USER}/nobrainer_data_norm/data_prepared_segmentation_{dataset}/test/test-'+val_idxs
#         nr_of_samples = sample_nr_in_wds(f'/om2/user/{DATA_USER}/nobrainer_data_norm/data_prepared_segmentation_{dataset}/test/', batchsize)

#         # NR_OF_CLASSES = 6
#         # url = f'/om2/user/matth406/nobrainer_data/data_prepared_{dataset}/test/test-'+val_idxs
#         # nr_of_samples = sample_nr_in_wds(f'/om2/user/matth406/nobrainer_data/data_prepared_{dataset}/test/', batchsize)
#     else:
#         raise NotImplementedError(f'No mode {mode} implemented')
    
#     if DATA_USER == 'matth406':
#         normalization_constants_internal = np.load("/om2/user/matth406/nobrainer_data_norm/data_prepared_medium/normalization_constants.npy")
#     else:
#         normalization_constants_internal = np.load(f"/om2/user/{DATA_USER}/nobrainer_data_norm/data_prepared_segmentation_{dataset}/normalization_constants.npy")
    
#     # NR_OF_CLASSES = 6
#     # normalization_constants_internal = np.load(f"/om2/user/sabeen/nobrainer_data_norm/matth406_{dataset}_6_classes/normalization_constants.npy")

#     def normalize_internal(sample):
#         image = sample[0]
#         image = (image - normalization_constants_internal[0]) / normalization_constants_internal[1]
#         sample = (image, sample[1], sample[2])
#         return sample

#     dataset = (
#         wds.WebDataset(
#             url,
#             repeat=training,
#             shardshuffle=1000 if training else False,
#             nodesplitter=nodesplitter if torch.cuda.device_count() > 1 else wds.shardlists.single_node_only
#             )
#         .shuffle(5000 if training else 0) # shuffle the dataset with a buffer
#         .decode() # decode all files in the shard
#         .to_tuple('brain.pth', 'mask.pth', "__key__") # load brain and mask
#         .map(normalize_internal) # normalize brain slice
#         .batched(batchsize=batchsize, partial=False) # batch the dataset
#     )
#     dataset.length = nr_of_samples//batchsize
#     dataset.nsamples = nr_of_samples

#     return dataset

# def nodesplitter(src):
#     #if torch.distributed.is_initialized():
#         # if group is None:
#         #     group = torch.distributed.group.WORLD
#         # rank = torch.distributed.get_rank(group=group)
#         # size = torch.distributed.get_world_size(group=group)
        
#     # world size. I.e. #Processes * #GPUsPerProcess
#     if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
#         rank = int(os.environ["RANK"])
#         world_size = int(os.environ["WORLD_SIZE"])
#     else:
#         raise ValueError(
#             "Rank and world size information not available"
#         )
#     count = 0
#     for i, item in enumerate(src):
#         if i % world_size == rank:
#             yield item
#             count += 1
#     else:
#         yield from src

# def get_data_loader(dataset : str, batch_size : int, num_workers : int = 4*torch.cuda.device_count()) -> Tuple[wds.WebLoader, wds.WebLoader, wds.WebLoader]:
    
#     train_data = NoBrainerDataset(dataset=dataset, mode='train', batchsize=batch_size)
#     val_data = NoBrainerDataset(dataset=dataset, mode='validation', batchsize=batch_size)
#     test_data = NoBrainerDataset(dataset=dataset, mode='test', batchsize=batch_size)
#     train_loader = wds.WebLoader(train_data, num_workers=num_workers, batch_size=None, pin_memory=torch.cuda.is_available())
#     val_loader = wds.WebLoader(val_data, num_workers=num_workers, batch_size=None, pin_memory=torch.cuda.is_available())
#     test_loader = wds.WebLoader(test_data, num_workers=num_workers, batch_size=None, pin_memory=torch.cuda.is_available())
#     train_loader.length = train_data.length
#     val_loader.length = val_data.length
#     test_loader.length = test_data.length
#     train_loader.nsamples = train_data.nsamples
#     val_loader.nsamples = val_data.nsamples
#     test_loader.nsamples = test_data.nsamples

#     return (train_loader, val_loader, test_loader)