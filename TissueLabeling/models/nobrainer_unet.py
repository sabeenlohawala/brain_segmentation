import math

import torch
from torch import nn
from torchinfo import summary


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim=None, up=False):
        super().__init__()
        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 2, 2, 1) # MOD
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 2, 2, 1) # MaxPool??
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(
        self,
        x,
    ):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)

class NobrainerUnet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """

    def __init__(self, image_channels, nr_of_classes):
        super().__init__()
        self.image_channels = image_channels
        self.nr_of_classes = nr_of_classes
        
        self.n_base_filters = 16
        down_channels = tuple([self.n_base_filters * (2**i) for i in range(5)])
        up_channels = tuple([self.n_base_filters * (2**i) for i in range(4,-1,-1)])
        out_dim = 1

        # Initial projection
        self.conv0 = nn.Conv2d(self.image_channels, down_channels[0], 3, padding=1)

        # Downsample
        self.downs = nn.ModuleList(
            [
                Block(down_channels[i], down_channels[i + 1])
                for i in range(len(down_channels) - 1)
            ]
        )
        # Upsample
        self.ups = nn.ModuleList(
            [
                Block(up_channels[i], up_channels[i + 1], up=True)
                for i in range(len(up_channels) - 1)
            ]
        )

        # self.output = nn.Conv2d(up_channels[-1], self.image_channels, out_dim)
        self.output = nn.Conv2d(up_channels[-1], self.nr_of_classes, out_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Initial conv
        x = self.conv0(x)
        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x)#, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)
            x = up(x)#, t)
        x = self.output(x)
        return self.softmax(x)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_channels = 1
    image_size = (160, 192)  # (28, 28)
    batch_size = 128

    # Note: For (28, 28), remove 2 up/down channels.

    model = NobrainerUnet(image_channels=image_channels,nr_of_classes=51).to(device)
    summary(
        model,
        input_size=[(batch_size, image_channels, *image_size), (batch_size,)],
        col_names=["input_size", "output_size", "num_params"],
        depth=5,
    )