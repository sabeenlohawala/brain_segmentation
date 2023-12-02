
import lightning as L
import wandb
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.colors import ListedColormap, BoundaryNorm

from utils import load_brains, crop, mapping

class Log_Images():

    def __init__(self, fabric : L.Fabric, wandb_on: bool, nr_of_classes: int = 112):
        self.wandb_on = wandb_on

        # color map to get always the same colors for classes
        colors = plt.cm.hsv(np.linspace(0, 1, nr_of_classes))
        # new plt cmap
        self.cmap = ListedColormap(colors)
        # new plt norm
        bounds=np.arange(0,nr_of_classes+1)
        self.norm = BoundaryNorm(bounds, self.cmap.N)

        # load always the same image from validation set
        image_file = 'pac_36_orig.nii.gz'
        mask_file = 'pac_36_aseg.nii.gz'
        file_path = '/om2/user/matth406/nobrainer_data/data/SharedData/segmentation/freesurfer_asegs/'
        brain, mask, _ = load_brains(image_file, mask_file, file_path)
        mask = mapping(mask)

        self.brain_slices, self.mask_slices = [], []
        
        # randomly select slices in 3 directions
        self.slice_idx = [125, 150]
        normalization_constants = np.load("/om2/user/matth406/nobrainer_data_norm/data_prepared_medium/normalization_constants.npy")
        self.brain_slices = torch.empty((len(self.slice_idx)*3, 1, 162, 194))
        self.mask_slices = torch.empty((len(self.slice_idx)*3, 1, 162, 194))
        i = 0
        self.logging_dict = {}
        for d in range(3):
            for slice_id in self.slice_idx:
                if d == 0:
                    brain_slice = crop(brain[slice_id,:,:], 162, 194)
                    mask_slice = crop(mask[slice_id,:,:], 162, 194)
                if d == 1:
                    brain_slice = crop(brain[:, slice_id, :], 162, 194)
                    mask_slice = crop(mask[:, slice_id, :], 162, 194)
                if d == 2:
                    brain_slice = crop(brain[:,:, slice_id], 162, 194)
                    mask_slice =  crop(mask[:,:, slice_id], 162, 194)
                
                # wandb.log({f"Image d{d} c{slice_id}": self.__create_plot(brain_slice, caption="Raw Image")}, step=1)
                # wandb.log({f"True Mask d{d} c{slice_id}": self.__create_plot(mask_slice, caption="True Mask", cmap=self.cmap, norm=self.norm)}, step=1)
                self.logging_dict[f"Image d{d} c{slice_id}"] = self.__create_plot(self.wandb_on, brain_slice, caption="Raw Image")
                self.logging_dict[f"True Mask d{d} c{slice_id}"] = self.__create_plot(self.wandb_on, mask_slice, caption="True Mask", cmap=self.cmap, norm=self.norm)
                brain_slice = (brain_slice - normalization_constants[0]) / normalization_constants[1]
                brain_slice = torch.from_numpy(brain_slice).to(torch.float32)
                brain_slice = brain_slice[None, None]
                self.brain_slices[i] = brain_slice

                mask_slice = torch.tensor(mask_slice)[None, None].long()
                self.mask_slices[i] = mask_slice

                i += 1

        # send all slices to device
        self.brain_slices = self.brain_slices.repeat((1,3,1,1)) # uncomment if pretrained = True
        self.brain_slices = fabric.to_device(self.brain_slices)
        self.mask_slices = fabric.to_device(self.mask_slices)

    @staticmethod
    def __create_plot(wandb_on: bool, image : np.array, caption : str, cmap : str = 'gray', norm : plt.Normalize = None, fig_path : str = None):
        '''
        Creates a pyplot and adds it to the wandb image list.

        Args:
            image (np.array): image
            caption (str): caption of the plot
            cmap (str, optional): color map applied. Defaults to 'gray'.
            norm (plt.Normalize, optional): color normalization. Defaults to None.
            fig_path (str, optional): Path if figure should be save locally. Defaults to None.
        '''
        fig, ax = plt.subplots()
        ax.imshow(image, cmap=cmap, norm=norm)
        ax.axis('off')
        fig.canvas.draw()
        if fig_path is not None:
            fig.savefig(fig_path)
        image = Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
        if wandb_on:
            image = wandb.Image(image, caption=caption) # comment to not save to wandb
        plt.close()

        return image

    @torch.no_grad()
    def logging(self, model, e : int, commit: bool):

        model.eval()
        probs = model(self.brain_slices)
        probs = probs.argmax(1)
        probs = probs.cpu()
        model.train()

        i = 0
        logging_dict = {}
        for d in range(3):
            for slice_id in self.slice_idx:
                logging_dict[f"Predicted Mask d{d} c{slice_id}"] = self.__create_plot(self.wandb_on, probs[i], caption=f"Epoch {e}", cmap=self.cmap, norm=self.norm)
                i += 1
        current_logging_dict = self.logging_dict | logging_dict
        if self.wandb_on:
            wandb.log(current_logging_dict, commit=commit) # comment to not save to wandb