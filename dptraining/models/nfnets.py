"""
Adapted implementation from https://github.com/psh01087/simple-nfnet
"""

from functools import partial
from typing import Callable

from jax import nn as jnn
from jax import numpy as jn
from jax.lax import rsqrt
from objax import TrainVar, StateVar, nn, functional, Module
from objax.random import normal, DEFAULT_GENERATOR
from objax.constants import ConvPadding

from numpy import prod


# import sys
# from pathlib import Path

# sys.path.insert(0, str(Path.cwd()))

# from dptraining.models.layers import ConvWS2D


nfnet_params = {
    "F0": {"width": [256, 512, 1536, 1536], "depth": [1, 2, 6, 3], "drop_rate": 0.2},
    "F1": {"width": [256, 512, 1536, 1536], "depth": [2, 4, 12, 6], "drop_rate": 0.3},
    "F2": {"width": [256, 512, 1536, 1536], "depth": [3, 6, 18, 9], "drop_rate": 0.4},
    "F3": {"width": [256, 512, 1536, 1536], "depth": [4, 8, 24, 12], "drop_rate": 0.4},
    "F4": {"width": [256, 512, 1536, 1536], "depth": [5, 10, 30, 15], "drop_rate": 0.5},
    "F5": {"width": [256, 512, 1536, 1536], "depth": [6, 12, 36, 18], "drop_rate": 0.5},
    "F6": {"width": [256, 512, 1536, 1536], "depth": [7, 14, 42, 21], "drop_rate": 0.5},
    "F7": {"width": [256, 512, 1536, 1536], "depth": [8, 16, 48, 24], "drop_rate": 0.5},
}

# These extra constant values ensure that the activations
# are variance preserving
class VPGELU(Module):
    def __call__(self, input, training=True):
        return jnn.gelu(input) * 1.7015043497085571


class VPReLU(Module):
    def __call__(self, input, training=True):
        return functional.relu(input) * 1.7139588594436646


activations_dict = {"gelu": VPGELU(), "relu": VPReLU()}


class NFNet(Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        variant: str = "F0",
        stochdepth_rate: float = None,
        alpha: float = 0.2,
        se_ratio: float = 0.5,
        activation: Callable = VPGELU(),
    ):
        super().__init__()

        if not variant in nfnet_params:
            raise RuntimeError(
                f"Variant {variant} does not exist and could not be loaded."
            )

        block_params = nfnet_params[variant]

        self.activation = activation
        self.drop_rate = jn.array(block_params["drop_rate"])
        self.num_classes = num_classes

        self.stem = Stem(in_channels=in_channels, activation=activation)

        num_blocks, index = sum(block_params["depth"]), 0

        blocks = []
        expected_std = 1.0
        nin = block_params["width"][0] // 2

        block_args = zip(
            block_params["width"],
            block_params["depth"],
            [0.5] * 4,  # bottleneck pattern
            [128] * 4,  # group pattern. Original groups [128] * 4
            [1, 2, 2, 2],  # stride pattern
        )

        for (block_width, stage_depth, expand_ratio, group_size, stride) in block_args:
            for block_index in range(stage_depth):
                beta = 1.0 / expected_std

                block_sd_rate = stochdepth_rate * index / num_blocks
                nout = block_width

                blocks.append(
                    NFBlock(
                        nin=nin,
                        nout=nout,
                        strides=stride if block_index == 0 else 1,
                        alpha=alpha,
                        beta=beta,
                        se_ratio=se_ratio,
                        group_size=group_size,
                        stochdepth_rate=block_sd_rate,
                        activation=activation,
                    )
                )

                nin = nout
                index += 1

                if block_index == 0:
                    expected_std = 1.0

                expected_std = (expected_std**2 + alpha**2) ** 0.5

        self.body = nn.Sequential(blocks)

        final_conv_channels = 2 * nin
        self.final_conv = ConvWS2D(nin=nout, nout=final_conv_channels, k=1)
        # self.pool = partial(functional.average_pool_2d, size=1)

        self.dropout = (
            nn.Dropout(self.drop_rate) if self.drop_rate > 0.0 else lambda x: x
        )

        self.linear = nn.Linear(
            final_conv_channels,
            self.num_classes,
            w_init=partial(
                normal,
                mean=0,
                stddev=0.01,
                generator=DEFAULT_GENERATOR,
            ),
        )

    def __call__(self, x, training=True):
        out = self.stem(x)
        out = self.body(out, training=training)
        out = self.activation(self.final_conv(out))
        pool = jn.mean(out, axis=(2, 3))

        # if training:
        #     pool = self.dropout(pool, training=training)

        return self.linear(pool)

    # def exclude_from_weight_decay(self, name: str) -> bool:
    #     # Regex to find layer names like
    #     # "stem.6.bias", "stem.6.gain", "body.0.skip_gain",
    #     # "body.0.conv0.bias", "body.0.conv0.gain"
    #     regex = re.compile("stem.*(bias|gain)|conv.*(bias|gain)|skip_gain")
    #     return len(regex.findall(name)) > 0

    # def exclude_from_clipping(self, name: str) -> bool:
    #     # Last layer should not be clipped
    #     return name.startswith("linear")


class Stem(Module):
    def __init__(self, in_channels: int, activation: Callable = VPGELU()):
        super().__init__()

        self.activation = activation
        self.conv0 = ConvWS2D(nin=in_channels, nout=16, k=3, strides=2)
        self.conv1 = ConvWS2D(nin=16, nout=32, k=3, strides=1)
        self.conv2 = ConvWS2D(nin=32, nout=64, k=3, strides=1)
        self.conv3 = ConvWS2D(nin=64, nout=128, k=3, strides=2)

    def __call__(self, x):
        out = self.conv0(x)
        out = self.activation(out)
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.activation(out)
        out = self.conv3(out)
        return out


class NFBlock(Module):
    def __init__(
        self,
        nin: int,
        nout: int,
        expansion: float = 0.5,
        se_ratio: float = 0.5,
        strides: int = 1,
        beta: float = 1.0,
        alpha: float = 0.2,
        group_size: int = 1,
        stochdepth_rate: float = None,
        activation: Callable = VPGELU(),
    ):

        super().__init__()

        self.nin = nin
        self.nout = nout
        self.expansion = expansion
        self.se_ratio = se_ratio
        self.activation = activation
        self.beta, self.alpha = StateVar(jn.array(beta)), StateVar(jn.array(alpha))
        self.group_size = group_size

        width = int(self.nout * expansion)
        self.groups = width // group_size
        self.width = group_size * self.groups
        self.strides = strides

        self.conv0 = ConvWS2D(nin=self.nin, nout=self.width, k=1)
        self.conv1 = ConvWS2D(
            nin=self.width,
            nout=self.width,
            k=3,
            strides=strides,
            padding=ConvPadding.SAME,
            groups=self.groups,
        )
        self.conv1b = ConvWS2D(
            nin=self.width,
            nout=self.width,
            k=3,
            strides=1,
            padding=ConvPadding.SAME,
            groups=self.groups,
        )
        self.conv2 = ConvWS2D(nin=self.width, nout=self.nout, k=1)

        self.use_projection = self.strides > 1 or self.nin != self.nout
        if self.use_projection:
            if strides > 1:
                self.shortcut_avg_pool = partial(
                    functional.average_pool_2d,
                    size=2,
                    strides=2,
                    padding=ConvPadding.SAME,  # if self.nin == 1536 else 1,
                )
            self.conv_shortcut = ConvWS2D(self.nin, self.nout, k=1)

        self.squeeze_excite = SqueezeExcite(
            self.nout,
            self.nout,
            se_ratio=self.se_ratio,
            activation=activation,
        )
        self.skip_gain = TrainVar(jn.zeros(()))

        self.use_stochdepth = (
            stochdepth_rate is not None
            and stochdepth_rate > 0.0
            and stochdepth_rate < 1.0
        )
        if self.use_stochdepth:
            self.stoch_depth = StochDepth(stochdepth_rate)

    def __call__(self, x, training=True):
        out = self.activation(x) * self.beta

        if self.strides > 1:
            shortcut = self.shortcut_avg_pool(out)
            shortcut = self.conv_shortcut(shortcut)
        elif self.use_projection:
            shortcut = self.conv_shortcut(out)
        else:
            shortcut = x

        out = self.activation(self.conv0(out))
        out = self.activation(self.conv1(out))
        out = self.activation(self.conv1b(out))
        out = self.conv2(out)
        out = (self.squeeze_excite(out) * 2) * out

        if self.use_stochdepth:
            out = self.stoch_depth(out, training=training)

        return out * self.alpha * self.skip_gain + shortcut


# # Implementation mostly from https://arxiv.org/abs/2101.08692
# # Implemented changes from https://arxiv.org/abs/2102.06171 and
# #  https://github.com/deepmind/deepmind-research/tree/master/nfnets
class ConvWS2D(nn.Conv2D):
    def __init__(
        self,
        nin: int,
        nout: int,
        k,
        strides=1,
        padding=0,
        dilations=1,
        groups: int = 1,
        use_bias: bool = True,
        w_init=nn.init.xavier_normal,
    ):
        super().__init__(
            nin=nin,
            nout=nout,
            k=k,
            strides=strides,
            padding=padding,
            dilations=dilations,
            groups=groups,
            use_bias=use_bias,
            w_init=w_init,
        )

        self.gain = TrainVar(jn.ones((nout,), dtype=self.w.value.dtype))
        self.eps = StateVar(jn.array(1e-4))
        self.fan_in = StateVar(jn.array(prod(self.w.shape[1:])))

    def standardized_weights(self):
        mean = jn.mean(self.w.value, axis=(0, 1, 2), keepdims=True)
        var = jn.var(self.w.value, axis=(0, 1, 2), keepdims=True)
        scale = rsqrt(jn.maximum(var * self.fan_in.value, self.eps.value))
        return (self.w - mean) * scale * self.gain

    def __call__(self, x, training=True):
        self.w.assign(self.standardized_weights())
        return super().__call__(x)
        # return F.conv2d(
        #     input=x,
        #     weight=self.standardized_weights(),
        #     bias=self.bias,
        #     stride=self.stride,
        #     padding=self.padding,
        #     dilation=self.dilation,
        #     groups=self.groups,
        # )


class SqueezeExcite(Module):
    def __init__(
        self,
        nin: int,
        nout: int,
        se_ratio: float = 0.5,
        activation: Callable = VPGELU(),
    ):
        super().__init__()

        self.nin = nin
        self.nout = nout
        self.se_ratio = se_ratio

        self.hidden_channels = max(1, int(self.nin * self.se_ratio))

        self.activation = activation
        self.linear = nn.Linear(self.nin, self.hidden_channels)
        self.linear_1 = nn.Linear(self.hidden_channels, self.nout)
        self.sigmoid = functional.sigmoid

    def __call__(self, x, training=True):
        out = jn.mean(x, axis=(2, 3))
        out = self.linear_1(self.activation(self.linear(out)))
        out = self.sigmoid(out)

        b, c, _, _ = x.shape
        return jn.broadcast_to(out.reshape(b, c, 1, 1), x.shape)


class StochDepth(Module):
    def __init__(self, stochdepth_rate: float):
        super().__init__()

        self.drop_rate = StateVar(jn.array(stochdepth_rate))

    def __call__(self, x, training=True):
        if not training:
            return x

        batch_size = x.shape[0]
        rand_tensor = normal(
            (batch_size, 1, 1, 1)
        )  # torch.rand(batch_size, 1, 1, 1).type_as(x).to(x.device)
        keep_prob = 1 - self.drop_rate
        binary_tensor = jn.floor(rand_tensor + keep_prob)

        return x * binary_tensor


if __name__ == "__main__":
    net = NFNet(10, stochdepth_rate=0.1)
    x = normal((16, 3, 28, 28))
    out = net(x, training=True)
    print(out.shape)
