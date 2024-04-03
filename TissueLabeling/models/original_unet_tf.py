import tensorflow as tf
from tensorflow import keras
from keras.layers import (
    InputLayer,
    Conv2D,
    Conv2DTranspose,
    Identity,
    BatchNormalization,
    ReLU,
    MaxPool2D,
    Softmax,
)


class Block(keras.layers.Layer):
    def __init__(self, out_ch, up=False, name="block"):
        super().__init__(name=name)
        if up:
            self.conv1 = Conv2D(
                filters=out_ch,
                kernel_size=3,
                padding="same",
                data_format="channels_first",
            )
            self.transform = Conv2DTranspose(
                filters=out_ch // 2,
                kernel_size=2,
                strides=2,
                padding="same",
                data_format="channels_first",
            )
        else:
            self.conv1 = Conv2D(
                filters=out_ch,
                kernel_size=3,
                padding="same",
                data_format="channels_first",
            )
            self.transform = Identity()
        self.conv2 = Conv2D(
            filters=out_ch, kernel_size=3, padding="same", data_format="channels_first"
        )
        self.bnorm1 = BatchNormalization(axis=1)
        self.bnorm2 = BatchNormalization(axis=1)
        self.relu = ReLU()

    def call(self, x):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)


class OriginalUnetTF(tf.keras.Model):
    def __init__(self, image_channels, nr_of_classes):
        super().__init__(name="NobrainerUnetTF")
        self.image_channels = image_channels
        self.nr_of_classes = nr_of_classes

        down_channels = [1, 64, 128, 256, 512, 1024]
        up_channels = [1024, 512, 256, 128, 64]
        out_dim = 1  # ??

        # Downsample
        self.max_pool = MaxPool2D(pool_size=2, strides=2, data_format="channels_first")
        # Q: tf equivalent to nn.ModuleList?? nn.Sequential has forward() but ModuleList does not. Does python list work?
        self.down1 = Block(down_channels[1], name="DownBlock1")
        self.down2 = Block(down_channels[2], name="DownBlock2")
        self.down3 = Block(down_channels[3], name="DownBlock3")
        self.down4 = Block(down_channels[4], name="DownBlock4")

        self.shared_block = Block(down_channels[5], up=True, name="SharedBlock")

        # Upsample
        self.up1 = Block(up_channels[1], up=True, name="UpBlock1")
        self.up2 = Block(up_channels[2], up=True, name="UpBlock2")
        self.up3 = Block(up_channels[3], up=True, name="UpBlock3")
        self.up4 = Block(
            up_channels[4], up=False, name="UpBlock4"
        )  # last decoder block has no upsampling (ConvTranspose2D)

        self.final_conv = Conv2D(
            filters=self.nr_of_classes,
            kernel_size=out_dim,
            data_format="channels_first",
            name="FinalConv",
        )
        self.softmax = Softmax(axis=1)
        super().__init__()

    def call(self, x):
        residual_inputs = []

        x = self.down1(x)
        residual_inputs.append(x)
        x = self.max_pool(x)

        x = self.down2(x)
        residual_inputs.append(x)
        x = self.max_pool(x)

        x = self.down3(x)
        residual_inputs.append(x)
        x = self.max_pool(x)

        x = self.down4(x)
        residual_inputs.append(x)
        x = self.max_pool(x)

        x = self.shared_block(x)

        residual_x = residual_inputs.pop()
        x = tf.concat((x, residual_x), axis=1)
        x = self.up1(x)

        residual_x = residual_inputs.pop()
        x = tf.concat((x, residual_x), axis=1)
        x = self.up2(x)

        residual_x = residual_inputs.pop()
        x = tf.concat((x, residual_x), axis=1)
        x = self.up3(x)

        residual_x = residual_inputs.pop()
        x = tf.concat((x, residual_x), axis=1)
        x = self.up4(x)

        x = self.final_conv(x)
        return self.softmax(x)

    def make(self, image_size=(160, 192), batch_size=128):
        """
        This method makes the command "model.summary()" work.
        input_shape: (C,H,W), do not specify batch B
        """
        x = tf.keras.layers.Input(
            shape=(self.image_channels, *image_size), batch_size=batch_size
        )
        model = tf.keras.Model(
            inputs=[x], outputs=self.call(x), name="NobrainerUnetTF Summary"
        )
        print(model.summary())
        return model


if __name__ == "__main__":
    device = "/gpu:0" if tf.config.list_physical_devices("GPU") else "/cpu:0"
    image_channels = 1
    image_size = (160, 192)  # (28, 28)
    batch_size = 128

    # Note: For (28, 28), remove 2 up/down channels.

    model = OriginalUnetTF(
        image_channels=image_channels, nr_of_classes=50
    )  # .to(device)

    # model.build(input_shape=(batch_size,image_channels,*image_size))
    # print(model.summary())
    model.make(image_size=image_size, batch_size=batch_size)
