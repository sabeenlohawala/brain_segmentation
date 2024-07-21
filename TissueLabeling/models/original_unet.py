"""
File: original_unet.py
Author: Sabeen Lohawala
Date: 2024-04-03
Description: This file contains the implementation of the original unet architecture 
as described in: https://arxiv.org/pdf/1505.04597.
"""

import torch
from torch import nn
from torchinfo import summary


class Block(nn.Module):
    """
    Class implementation of a convolutional block in the UNet architecture.
    """
    def __init__(self, in_ch, out_ch, up=False):
        """
        Constructor.

        Args:
            in_ch (int): the number of channels in the input
            out_ch (int): the number of channels in the output
            up (bool, optional): flag to indicate whether this should be an up-convolutional block
                                 or down-convolutional block; default = False
        """
        super().__init__()
        if up:
            # self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            # self.transform = nn.ConvTranspose2d(out_ch, out_ch, 2, 2, 1) # MOD
            self.transform = nn.ConvTranspose2d(out_ch, out_ch // 2, 2, 2, 0)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(
        self,
        x,
    ):
        """
        Implements the forward pass of the input x through the convolutional block.

        Args:
            x (torch.Tensor): the input tensor to the convolutional block
        
        Returns:
            torch.Tensor: the result of the convolutional block operations
        """
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)


class OriginalUnet(nn.Module):
    """
    The Unet architecture based on: https://arxiv.org/pdf/1505.04597.
    """

    def __init__(self, image_channels, nr_of_classes, n_base_filters=64, n_blocks=5):
        """
        Constructor.
        This has only been trained with n_base_filters = 64 and n_blocks = 5.

        Args:
            image_channels (int): the number of channels in the images that the model is trained on
            nr_of_classes (int): the number of segmentation classes
            n_base_filters (int, optional): the base number of filters; default = 64
            n_blocks (int, optional): the number of convolutional blocks in the unet; default = 5
        """
        super().__init__()
        self.image_channels = image_channels
        self.nr_of_classes = nr_of_classes

        self.n_base_filters = n_base_filters
        self.n_blocks = n_blocks
        down_channels = (self.image_channels,) + tuple([self.n_base_filters * (2**i) for i in range(self.n_blocks)])
        up_channels = tuple([self.n_base_filters * (2**i) for i in range(self.n_blocks-1, -1, -1)])
        # down_channels = [1, 64, 128, 256, 512, 1024]
        # up_channels = [1024, 512, 256, 128, 64]
        out_dim = 1

        # Downsample
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.downs = nn.ModuleList(
            [
                Block(down_channels[i], down_channels[i + 1])
                for i in range(len(down_channels) - 2)
            ]
        )

        self.shared_block = Block(down_channels[-2], down_channels[-1], up=True)

        # Upsample
        self.ups = nn.ModuleList(
            [
                Block(up_channels[i], up_channels[i + 1], up=True)
                for i in range(len(up_channels) - 2)
            ]
            + [
                Block(up_channels[-2], up_channels[-1], up=False)
            ]  # last decoder block has no upsampling (ConvTranspose2D)
        )

        self.output = nn.Conv2d(up_channels[-1], self.nr_of_classes, out_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Implements the forward pass of the input tensor x through the model.

        Args:
            x (torch.Tensor): the input to the model
        
        Returns:
            torch.Tensor: the softmax output of the UNet model.
        """
        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x)
            residual_inputs.append(x)
            x = self.max_pool(x)
        x = self.shared_block(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)
            x = up(x)
        x = self.output(x)
        return self.softmax(x)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_channels = 1
    image_size = (256,256)#(160, 192)  # (28, 28)
    batch_size = 288

    # Note: For (28, 28), remove 2 up/down channels.

    model = OriginalUnet(image_channels=image_channels, nr_of_classes=50, n_base_filters=64, n_blocks=5).to(device)
    summary(
        model,
        input_size=(batch_size, image_channels, *image_size),
        col_names=["input_size", "output_size", "num_params"],
        depth=5,
    )
