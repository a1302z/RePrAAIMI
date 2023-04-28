from typing import Callable
from objax import nn, functional, Module
from objax.util import local_kwargs, class_name
from functools import partial
from objax.constants import ConvPadding

from dptraining.models.layers import AdaptivePooling, is_groupnorm, Flatten


def conv_norm_act(  # pylint:disable=too-many-arguments
    in_channels,
    out_channels,
    conv_cls=nn.Conv2D,
    act_func=functional.relu,
    pool_func=lambda x: x,
    norm_cls=nn.GroupNorm2D,
    num_groups=32,
):
    if is_groupnorm(norm_cls):
        norm_cls = partial(norm_cls, groups=min(num_groups, out_channels))
    layers = [
        conv_cls(
            in_channels, out_channels, k=3, padding=ConvPadding.SAME, use_bias=False
        ),
        norm_cls(nin=out_channels),
        act_func,
    ]
    layers.append(pool_func)
    return nn.Sequential(layers)


class ResNet9(Module):  # pylint:disable=too-many-instance-attributes
    def __init__(  # pylint:disable=too-many-arguments
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        conv_cls: Module = nn.Conv2D,
        act_func: Module = functional.relu,
        norm_cls: Module = nn.GroupNorm2D,
        pool_func: Callable = partial(functional.max_pool_2d, size=2),
        linear_cls: Module = nn.Linear,
        out_func: Callable = lambda x: x,
        scale_norm: bool = False,
        num_groups: tuple[int, ...] = (32, 32, 32, 32),
        channels: tuple[int, ...] = (64, 128, 256, 256),
    ):
        """9-layer Residual Network. Architecture:
        conv-conv-Residual(conv, conv)-conv-conv-Residual(conv-conv)-FC
        """
        super().__init__()

        assert len(num_groups) == 4, "num_groups must be a tuple with 4 members"
        groups = num_groups

        self.conv1 = conv_norm_act(
            in_channels,
            channels[0],
            conv_cls=conv_cls,
            act_func=act_func,
            norm_cls=norm_cls,
            num_groups=groups[0],
        )
        self.conv2 = conv_norm_act(
            channels[0],
            channels[1],
            conv_cls=conv_cls,
            pool_func=pool_func,
            act_func=act_func,
            norm_cls=norm_cls,
            num_groups=groups[0],
        )

        self.res1 = nn.Sequential(
            [
                conv_norm_act(
                    channels[1],
                    channels[1],
                    conv_cls=conv_cls,
                    act_func=act_func,
                    norm_cls=norm_cls,
                    num_groups=groups[1],
                ),
                conv_norm_act(
                    channels[1],
                    channels[1],
                    conv_cls=conv_cls,
                    act_func=act_func,
                    norm_cls=norm_cls,
                    num_groups=groups[1],
                ),
            ]
        )

        self.conv3 = conv_norm_act(
            channels[1],
            channels[2],
            conv_cls=conv_cls,
            pool_func=pool_func,
            act_func=act_func,
            norm_cls=norm_cls,
            num_groups=groups[2],
        )
        self.conv4 = conv_norm_act(
            channels[2],
            channels[3],
            conv_cls=conv_cls,
            pool_func=pool_func,
            act_func=act_func,
            norm_cls=norm_cls,
            num_groups=groups[2],
        )

        self.res2 = nn.Sequential(
            [
                conv_norm_act(
                    channels[3],
                    channels[3],
                    conv_cls=conv_cls,
                    act_func=act_func,
                    norm_cls=norm_cls,
                    num_groups=groups[3],
                ),
                conv_norm_act(
                    channels[3],
                    channels[3],
                    conv_cls=conv_cls,
                    act_func=act_func,
                    norm_cls=norm_cls,
                    num_groups=groups[3],
                ),
            ]
        )

        # self.pooling = lambda x: x.mean((2, 3))
        self.pooling = AdaptivePooling(functional.average_pool_2d, 2)
        # self.MP = partial(functional.average_pool_2d, ((2, 2)))
        # self.FlatFeats = lambda x: x.reshape(x.shape[0], -1)
        self.flatten = Flatten()
        self.classifier = linear_cls(4 * channels[3], num_classes)

        if scale_norm:
            self.scale_norm_1 = (
                partial(norm_cls, groups=min(num_groups[1], channels[1]))
                if is_groupnorm(norm_cls)
                else norm_cls
            )(nin=channels[1])
            self.scale_norm_2 = (
                partial(norm_cls, groups=min(num_groups[3], channels[3]))
                if is_groupnorm(norm_cls)
                else norm_cls
            )(nin=channels[3])
        else:
            self.scale_norm_1 = lambda x: x
            self.scale_norm_2 = lambda x: x

        self.out_func = out_func

    def __call__(self, xb, *args, **kwargs):
        out = self.conv1(xb, *args, **local_kwargs(kwargs, self.conv1))
        out = self.conv2(out, *args, **local_kwargs(kwargs, self.conv2))
        out = self.res1(out, *args, **local_kwargs(kwargs, self.res1)) + out
        out = self.scale_norm_1(out, *args, **local_kwargs(kwargs, self.scale_norm_1))
        out = self.conv3(out, *args, **local_kwargs(kwargs, self.conv3))
        out = self.conv4(out, *args, **local_kwargs(kwargs, self.conv4))
        out = self.res2(out, *args, **local_kwargs(kwargs, self.res2)) + out
        out = self.scale_norm_2(out, *args, **local_kwargs(kwargs, self.scale_norm_2))
        out = self.pooling(out, *args, **local_kwargs(kwargs, self.pooling))
        out = self.flatten(out)
        out = self.classifier(out, *args, **local_kwargs(kwargs, self.classifier))
        out = self.out_func(out)
        return out

    def __repr__(self) -> str:
        return_str = f"{class_name(self)}"
        return_str += f"\n\t{str(self.conv1)}"
        return_str += f"\n\t{str(self.conv2)}"
        return_str += f"\n\t{str(self.res1)}"
        return_str += f"\n\t{str(self.scale_norm_1)}"
        return_str += f"\n\t{str(self.conv3)}"
        return_str += f"\n\t{str(self.conv4)}"
        return_str += f"\n\t{str(self.res2)}"
        return_str += f"\n\t{str(self.scale_norm_2)}"
        return_str += f"\n\t{str(self.pooling)}"
        return_str += f"\n\t{str(self.flatten)}"
        return_str += f"\n\t{str(self.classifier)}"
        return_str += f"\n\t{str(self.out_func)}"
        return return_str
