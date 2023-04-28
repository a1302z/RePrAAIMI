# pylint: skip-file
from objax import nn, Module
from objax.functional import average_pool_2d, pad
from jax import numpy as jn
from math import floor, ceil

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
        in_channels: int,
        out_channels: int,
        channels: int = 32,
        num_pool_layers: int = 4,
        actv=Cardioid,
        conv_layer: Module = ComplexConv2D,
        upconv_layer: Module = ComplexWSConv2DNoWhitenTranspose,
        norm_layer: Module = ComplexGroupNorm2DWhitening,
        pool_fn=SeparablePool2D,
        dim_mode: int = 2,  # how many dimension will the input have
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

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.num_pool_layers = num_pool_layers
        self.dim_mode = dim_mode
        assert self.dim_mode in [2, 3], "Unet only supports 2D or 3D input"

        self.down_sample_layers = nn.Sequential(
            [
                ConvBlock(
                    in_channels,
                    channels,
                    actv,
                    conv_layer=conv_layer,
                    norm_layer=norm_layer,
                )
            ]
        )
        ch = channels
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(
                ConvBlock(
                    ch, ch * 2, actv, conv_layer=conv_layer, norm_layer=norm_layer
                )
            )
            ch *= 2
        self.conv = ConvBlock(
            ch, ch * 2, actv, conv_layer=conv_layer, norm_layer=norm_layer
        )

        self.up_conv = nn.Sequential()
        self.up_transpose_conv = nn.Sequential()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(
                TransposeConvBlock(
                    ch * 2, ch, actv, conv_layer=upconv_layer, norm_layer=norm_layer
                )
            )
            self.up_conv.append(
                ConvBlock(
                    ch * 2, ch, actv, conv_layer=conv_layer, norm_layer=norm_layer
                )
            )
            ch //= 2

        self.up_transpose_conv.append(
            TransposeConvBlock(
                ch * 2, ch, actv, conv_layer=upconv_layer, norm_layer=norm_layer
            )
        )
        self.up_conv.append(
            nn.Sequential(
                [
                    ConvBlock(
                        ch * 2, ch, actv, conv_layer=conv_layer, norm_layer=norm_layer
                    ),
                    conv_layer(ch, self.out_channels, k=1, strides=1, padding=0),
                ],
            )
        )
        self.pool = pool_fn

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
            match self.dim_mode:
                case 2:
                    padding = [[0, 0], [0, 0], [0, 0], [0, 0]]
                    if output.shape[-1] != downsample_layer.shape[-1]:
                        padding[-1] = [1, 1]  # padding right
                    if output.shape[-2] != downsample_layer.shape[-2]:
                        padding[-2] = [1, 1]  # padding bottom
                    if sum([sum(p) for p in padding]) != 0:
                        output = pad(output, pad_width=padding, mode="reflect")
                case 3:
                    padding = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
                    if output.shape[-1] != downsample_layer.shape[-1]:
                        diff = abs(output.shape[-1] - downsample_layer.shape[-1]) / 2.0
                        padding[-1] = [int(floor(diff)), int(ceil(diff))]
                    if output.shape[-2] != downsample_layer.shape[-2]:
                        diff = abs(output.shape[-2] - downsample_layer.shape[-2]) / 2.0
                        padding[-2] = [int(floor(diff)), int(ceil(diff))]
                    if output.shape[-3] != downsample_layer.shape[-3]:
                        diff = abs(output.shape[-3] - downsample_layer.shape[-3]) / 2.0
                        padding[-3] = [int(floor(diff)), int(ceil(diff))]
                    if sum([sum(p) for p in padding]) != 0:
                        output = pad(output, pad_width=padding, mode="reflect")
                case other:
                    raise ValueError(f"{other} dimensions not supported")

            output = jn.concatenate([output, downsample_layer], axis=1)
            output = conv(output)

        return output


class ConvBlock(Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(
        self,
        in_channels: int,
        out_chans: int,
        actv: Module,
        conv_layer: Module = ComplexConv2D,
        norm_layer: Module = ComplexGroupNorm2DWhitening,
    ):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            [
                conv_layer(
                    in_channels,
                    out_chans,
                    k=3,
                    padding=1,
                ),
                norm_layer(nin=out_chans, groups=16),
                actv,
                conv_layer(out_chans, out_chans, k=3, padding=1),
                norm_layer(nin=out_chans, groups=out_chans),
                actv,
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

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        actv: Module,
        conv_layer: Module = ComplexWSConv2DNoWhitenTranspose,
        norm_layer: Module = ComplexGroupNorm2DWhitening,
    ):
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
                conv_layer(
                    in_chans,
                    out_chans,
                    k=2,
                    strides=2,
                    padding=0,
                    # output_padding=0,
                ),
                norm_layer(groups=16, nin=out_chans),
                actv,
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
