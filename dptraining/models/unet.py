from functools import partial
from objax import nn, Module
from objax.functional import average_pool_2d, pad
from jax import numpy as jn

# import sys
# from pathlib import Path

# sys.path.insert(0, str(Path.cwd()))

from dptraining.models.complex import (
    Cardioid,
    ComplexGroupNorm2DWhitening,
    ComplexConv2D,
    SeparablePool2D,
    ComplexWSConv2DNoWhitenTranspose,
)


class Unet(Module):
    """
    PyTorch implementation of a U-Net model.

    O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks
    for biomedical image segmentation. In International Conference on Medical
    image computing and computer-assisted intervention, pages 234–241.
    Springer, 2015.
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        chans: int = 32,
        num_pool_layers: int = 4,
        actv=Cardioid,
    ):
        """
        Args:
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            chans: Number of output channels of the first convolution layer.
            num_pool_layers: Number of down-sampling and up-sampling layers.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers

        self.down_sample_layers = nn.Sequential([ConvBlock(in_chans, chans, actv)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock(ch, ch * 2, actv))
            ch *= 2
        self.conv = ConvBlock(ch, ch * 2, actv)

        self.up_conv = nn.Sequential()
        self.up_transpose_conv = nn.Sequential()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch, actv))
            self.up_conv.append(ConvBlock(ch * 2, ch, actv))
            ch //= 2

        self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch, actv))
        self.up_conv.append(
            nn.Sequential(
                [
                    ConvBlock(ch * 2, ch, actv),
                    ComplexConv2D(ch, self.out_chans, k=1, strides=1, padding=0),
                ],
            )
        )
        self.pool = SeparablePool2D(
            size=2, strides=2, padding=0, pool_func=average_pool_2d
        )

    def __call__(self, image, **kwargs):
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        stack = []
        output = image

        # apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = self.pool(output)

        output = self.conv(output)

        # apply up-sampling layers
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output)

            # reflect pad on the right/botton if needed to handle odd input dimensions
            padding = [[0, 0], [0, 0], [0, 0], [0, 0]]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[-1] = [1, 1]  # padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[-2] = [1, 1]  # padding bottom
            if sum([sum(p) for p in padding]) != 0:
                output = pad(output, pad_width=padding, mode="reflect")

            output = jn.concatenate([output, downsample_layer], axis=1)
            output = conv(output)

        return jn.abs(output)


class ConvBlock(Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        actv: Module,
    ):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            [
                ComplexConv2D(
                    in_chans,
                    out_chans,
                    k=3,
                    padding=1,
                ),
                ComplexGroupNorm2DWhitening(nin=out_chans, groups=16),
                actv(),
                ComplexConv2D(out_chans, out_chans, k=3, padding=1),
                ComplexGroupNorm2DWhitening(nin=out_chans, groups=out_chans),
                actv(),
            ]
        )

    def __call__(self, image):
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        return self.layers(image)


class TransposeConvBlock(Module):
    """
    A Transpose Convolutional Block that consists of one convolution transpose
    layers followed by instance normalization and LeakyReLU activation.
    """

    def __init__(self, in_chans: int, out_chans: int, actv: Module):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            [
                ComplexWSConv2DNoWhitenTranspose(
                    in_chans,
                    out_chans,
                    k=2,
                    strides=2,
                    padding=0,
                    # output_padding=0,
                ),
                ComplexGroupNorm2DWhitening(groups=16, nin=out_chans),
                actv(),
            ]
        )

    def __call__(self, image):
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H*2, W*2)`.
        """
        return self.layers(image)


# if __name__ == "__main__":
#     import numpy as np

#     unet = Unet(2, 3, 4)
#     in_data = jn.array(np.random.normal(size=(1, 2, 320, 320)))
#     out_data = unet(in_data)
#     print(out_data.shape)
