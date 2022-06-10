# Based on this paper: https://arxiv.org/abs/2205.04095
from typing import Callable
from objax import nn, functional, Module
from functools import partial
from jax import numpy as jnp

from dptraining.models.layers import is_groupnorm, AdaptivePooling


# def getActivationFunction(activation_fc_str: str = "selu"):
#     """
#     This is a helper function to return all the different activation functions
#     we want to consider.
#     """
#     if activation_fc_str == "selu":
#         activation_fc = nn.SELU()
#     elif activation_fc_str == "relu":
#         activation_fc = nn.ReLU()
#     elif activation_fc_str == "leaky_relu":
#         activation_fc = nn.LeakyReLU()

#     return activation_fc


# def getPoolingFunction(pool_fc_str: str, **kwargs):
#     """
#     This is a helper function to return all the different pooling operations.

#     Args:
#         pool_fc_str: str to select the specific function

#     """
#     if pool_fc_str == "mxp":
#         # keep dimensions for CIFAR10 dimenions assuming a downsampling
#         # only through halving.
#         pool_fc = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
#     elif pool_fc_str == "avg":
#         # keep dimensions for CIFAR10 dimenions assuming a downsampling
#         # only through halving.
#         pool_fc = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
#     elif pool_fc_str == "identity":
#         pool_fc = nn.Identity()

#     return pool_fc


# def getNormFunction(norm_groups: int, num_features: int, **kwargs):
#     """
#     This is a helper function to return all the different normalizations we want to consider.

#     Args:
#         norm_groups: the number of normalization groups of GN, or to select IN, Identity, BN
#         num_features: number of channels

#     """
#     if norm_groups > 0:
#         # for num_groups = num_features => InstanceNorm
#         # for num_groups = 1 => LayerNorm
#         norm_fc = nn.GroupNorm(
#             num_groups=min(norm_groups, num_features),
#             num_channels=num_features,
#             affine=True,
#         )
#     # extra cases: InstanceNorm, Identity (no norm), BatchNorm (not DP-compatible)
#     elif norm_groups == 0:
#         norm_fc = nn.InstanceNorm2d(
#             num_features=num_features,
#         )
#     elif norm_groups == -1:
#         norm_fc = nn.Identity()
#     elif norm_groups == -2:
#         norm_fc = nn.BatchNorm2d(num_features=num_features)

#     return norm_fc


# NOTE: Specific features have been designed for CIFAR10 Input specifically
# NOTE: Difference to Stage 1 Network: Downsampling and channel adaptation
#       happens in first Conv2d.
class SmoothBlock(Module):
    """
    Inspired by Dense Blocks and Residual Blocks of ResNet and DenseNet.

    Args:
        in_channels: the number of channels (feature maps) of the incoming embedding
        out_channels: the number of channels after the first convolution
        pool_fc_str: selected pooling operation (mxp, avg, identity)
        norm_groups: number of norm groups for group norm (or selected IN, BN)
        activation_fc_str: choose activation function
        dsc: whether to use depthwise seperable convolutions or not
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_groups: int,
        conv_cls: Module = nn.Conv2D,
        act_func: Module = functional.relu,
        norm_cls: Module = nn.GroupNorm2D,
        pool_func: Callable = partial(functional.max_pool_2d, size=2),
        dsc: bool = False,
    ):
        super().__init__()

        if not dsc:
            self.conv_layers = conv_cls(
                in_channels,
                out_channels,
                k=3,
                strides=1,
                padding=1,
                use_bias=False,
            )
        else:
            self.conv_layers = nn.Sequential(
                [
                    conv_cls(
                        in_channels,
                        in_channels,
                        groups=in_channels,
                        k=3,
                        strides=1,
                        padding=1,
                        use_bias=False,
                    ),
                    conv_cls(in_channels, out_channels, k=1, use_bias=False),
                ]
            )

        # set post conv operations
        self.pooling = pool_func
        if is_groupnorm(norm_cls):
            self.norm = norm_cls(groups=norm_groups, nin=out_channels)
        else:
            self.norm = norm_cls(nin=out_channels)
        self.activation_fcs = act_func
        # self.pooling = getPoolingFunction(pool_fc_str=pool_fc_str)
        # self.norm = getNormFunction(norm_groups=norm_groups, num_features=out_channels)
        # self.activation_fcs = getActivationFunction(activation_fc_str)

    def __call__(self, inpt, training=False):
        out = inpt
        out = self.conv_layers(out)
        out = self.pooling(out)
        out = self.norm(out, training)
        out = self.activation_fcs(out)

        return out


class SmoothStack(Module):
    """
    Helper module to stack the different smooth blocks.

    Args:
        in_channels: the number of channels (feature maps) of the incoming embedding
        out_channels: the number of channels after the first convolution
        pool_fc_str: selected pooling operation (mxp, avg, identity)
        norm_groups: number of norm groups for group norm (or selected IN, BN)
        activation_fc_str: choose activation function
        num_blocks: number of smooth blocks
        dsc: whether to use depthwise seperable convolutions or not
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_groups: str,
        num_blocks: int,
        dsc: bool,
        conv_cls: Module = nn.Conv2D,
        act_func: Module = functional.relu,
        norm_cls: Module = nn.GroupNorm2D,
        pool_func: Callable = partial(functional.max_pool_2d, size=2),
    ):
        super().__init__()

        # first block to get the right number of channels (from previous block to current)
        self.smooth_stack = nn.Sequential(
            [
                SmoothBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    norm_groups=norm_groups,
                    dsc=dsc,
                    conv_cls=conv_cls,
                    act_func=act_func,
                    norm_cls=norm_cls,
                    pool_func=pool_func,
                )
            ]
        )

        # EXTEND adds array as elements of existing array, APPEND adds array as new element of array
        self.smooth_stack.extend(
            [
                SmoothBlock(
                    in_channels=out_channels * i + in_channels,
                    out_channels=out_channels,
                    norm_groups=norm_groups,
                    dsc=dsc,
                    conv_cls=conv_cls,
                    act_func=act_func,
                    norm_cls=norm_cls,
                    pool_func=pool_func,
                )
                for i in range(1, num_blocks)
            ]
        )

    def __call__(self, inpt, training=False):
        out = inpt
        for layer in self.smooth_stack:
            temp = layer(out, training)
            # concatenate at channel dimension
            out = jnp.concatenate((out, temp), 1)
        return out


class SmoothNet(Module):
    """
    The SmoothNet class. The v1 SmoothNets can be considered as:
    (1) Wide, (2) DenseNets (w/o Bottlenecks) with (3) SELU activations
    and (4) DP-compatible normalization and max pooling.

    Args:
        pool_fc_str: set pooling operation after conv (or none)
        norm_groups: the number of groups to be used in the group
        normalization (0:=IN, -1:=ID, -2:=BN)
        activation_fc_str: choose activation function
        depth: a factor multiplied with number of conv blocks per stage of base model
        width: a factor multiplied with number of channels per conv block of base model
                := num_blocks (as defined in the scaling approach)
        dsc: whether depthwise seperable convolutions are used or normal convolutions
    """

    def __init__(
        self,
        norm_groups: int = 8,
        in_channels: int = 3,
        depth: float = 1.0,
        width: float = 1.0,
        num_classes: int = 10,
        dsc: bool = False,
        conv_cls: Module = nn.Conv2D,
        act_func: Module = functional.selu,
        norm_cls: Module = nn.GroupNorm2D,
        pool_func: Callable = partial(
            functional.max_pool_2d, size=3, strides=1, padding=1
        ),
        linear_cls: Module = nn.Linear,
        out_func: Callable = lambda x: x,
    ):
        super().__init__()

        ## STAGE 0 ##
        # the stage 1 base model has 8 channels in stage 0
        # the stage 1 base model has 1 conv block per stage (in both stage 0 and 1)
        width_stage_zero = int(width * 8)
        depth_stage_zero = int(depth * 1)

        self.stage_zero = SmoothStack(
            in_channels=in_channels,
            out_channels=width_stage_zero,
            norm_groups=norm_groups,
            num_blocks=depth_stage_zero,
            dsc=dsc,
            conv_cls=conv_cls,
            act_func=act_func,
            norm_cls=norm_cls,
            pool_func=pool_func,
        )

        ## STAGE 1 ##
        # the stage 1 base model has 16 channels in stage 1
        # the stage 1 base model has 1 conv block per stage (in both stage 0 and 1)
        width_stage_one = int(width * 16)
        depth_stage_one = int(depth * 1)

        # DenseTransition #
        # recalculate the number of input features
        # depth_stage_zero (total num of blocks) + input channels (3 for CIFAR10)
        width_stage_zero = width_stage_zero * depth_stage_zero + 3
        # same as original Tranistion Layers in DenseNet
        # features are halved through 1x1 Convs and AvgPool is used to halv the dims
        self.dense_transition = nn.Sequential(
            # getAfterConvFc(after_conv_fc_str, width_stage_zero),
            [
                conv_cls(
                    width_stage_zero,
                    width_stage_zero // 2,
                    k=1,
                    strides=1,
                    use_bias=False,
                ),
                partial(functional.average_pool_2d, size=2, strides=2),
            ]
        )
        width_stage_zero = width_stage_zero // 2

        self.stage_one = SmoothStack(
            in_channels=width_stage_zero,
            out_channels=width_stage_one,
            norm_groups=norm_groups,
            num_blocks=depth_stage_one,
            dsc=dsc,
            conv_cls=conv_cls,
            act_func=act_func,
            norm_cls=norm_cls,
            pool_func=pool_func,
        )

        self.pre_final = partial(functional.average_pool_2d, size=2, strides=2)
        self.width_stage_one = width_stage_one

        ## Final FC Block ##
        # output_dim is fixed to 4 (even if 8 makes more sense for the stage 1 StageConvModel)
        output_dim = 4
        self.adaptive_pool = AdaptivePooling(functional.average_pool_2d, 4)
        # self.adaptive_pool = lambda x: x.mean((2, 3))

        self.fc1 = linear_cls(width_stage_one * output_dim**2, 256)
        self.fc2 = linear_cls(256, 128)
        self.fc3 = linear_cls(128, num_classes)
        self.relu1 = act_func
        self.relu2 = act_func
        self.out_func = out_func

    def __call__(self, x, training=False):
        batch_size = x.shape[0]
        out = self.stage_zero(x, training)
        # with dense transition layer
        # no dim or feature reduction will happen in the stages themselves
        out = self.dense_transition(out)
        out = self.stage_one(out, training)
        # as input to last FC layer only the output of the last conv_block
        # should be considered in the dense connection case
        # last pooling layer to downsampling (same as in DenseNet)
        out = self.pre_final(out)
        # only get output of last conv layer
        out = out[:, -self.width_stage_one :, :, :]
        out = self.adaptive_pool(out)
        out = out.reshape(batch_size, -1)
        out = self.relu1(self.fc1(out))
        out = self.relu2(self.fc2(out))
        out = self.fc3(out)
        out = self.out_func(out)
        return out


# the standard SmoothNet used in the original paper is a SmoothNet W80D50
def get_smoothnet(
    width: float = 8.0,
    depth: float = 5.0,
    norm_groups: int = 8,
    pool_func: Callable = partial(functional.max_pool_2d, size=3, strides=1, padding=1),
    **kwargs
):
    model = SmoothNet(
        width=width, depth=depth, norm_groups=norm_groups, pool_func=pool_func, **kwargs
    )

    return model
