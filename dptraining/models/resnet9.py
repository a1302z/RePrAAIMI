from objax import nn, functional, Module
from objax.util import local_kwargs
from functools import partial


def conv_norm_act(  # pylint:disable=too-many-arguments
    in_channels,
    out_channels,
    pool=False,
    act_func=functional.relu,
    norm_func=nn.GroupNorm2D,
    num_groups=32,
):
    if issubclass(norm_func, nn.GroupNorm2D):
        norm_func = partial(norm_func, groups=min(num_groups, out_channels))
    layers = [
        nn.Conv2D(in_channels, out_channels, k=3, padding=1, use_bias=False),
        norm_func(nin=out_channels),
        act_func,
    ]
    if pool:
        layers.append(partial(functional.max_pool_2d, size=2))
    return nn.Sequential(layers)


class ResNet9(Module):  # pylint:disable=too-many-instance-attributes
    def __init__(  # pylint:disable=too-many-arguments
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        act_func: Module = functional.relu,
        norm_func: Module = nn.GroupNorm2D,
        scale_norm: bool = False,
        num_groups: tuple[int, ...] = (32, 32, 32, 32),
        # input_dims: tuple[int, ...] = (224, 224),
    ):
        """9-layer Residual Network. Architecture:
        conv-conv-Residual(conv, conv)-conv-conv-Residual(conv-conv)-FC
        """
        super().__init__()

        conv_block = conv_norm_act

        assert (
            isinstance(num_groups, tuple) and len(num_groups) == 4
        ), "num_groups must be a tuple with 4 members"
        groups = num_groups

        self.conv1 = conv_block(
            in_channels,
            64,
            act_func=act_func,
            norm_func=norm_func,
            num_groups=groups[0],
        )
        self.conv2 = conv_block(
            64,
            128,
            pool=True,
            act_func=act_func,
            norm_func=norm_func,
            num_groups=groups[0],
        )

        self.res1 = nn.Sequential(
            [
                conv_block(
                    128,
                    128,
                    act_func=act_func,
                    norm_func=norm_func,
                    num_groups=groups[1],
                ),
                conv_block(
                    128,
                    128,
                    act_func=act_func,
                    norm_func=norm_func,
                    num_groups=groups[1],
                ),
            ]
        )

        self.conv3 = conv_block(
            128,
            256,
            pool=True,
            act_func=act_func,
            norm_func=norm_func,
            num_groups=groups[2],
        )
        self.conv4 = conv_block(
            256,
            256,
            pool=True,
            act_func=act_func,
            norm_func=norm_func,
            num_groups=groups[2],
        )

        self.res2 = nn.Sequential(
            [
                conv_block(
                    256,
                    256,
                    act_func=act_func,
                    norm_func=norm_func,
                    num_groups=groups[3],
                ),
                conv_block(
                    256,
                    256,
                    act_func=act_func,
                    norm_func=norm_func,
                    num_groups=groups[3],
                ),
            ]
        )

        self.pooling = lambda x: x.mean((2, 3))
        # self.MP = partial(functional.average_pool_2d, ((2, 2)))
        # self.FlatFeats = lambda x: x.reshape(x.shape[0], -1)
        # self.FlatFeats = nn.Flatten()
        self.classifier = nn.Linear(256, num_classes)

        if scale_norm:
            self.scale_norm_1 = (
                partial(norm_func, groups=min(num_groups[1], 128))
                if issubclass(norm_func, nn.GroupNorm2D)
                else norm_func
            )(nin=128)
            self.scale_norm_2 = (
                partial(norm_func, groups=min(num_groups[1], 256))
                if issubclass(norm_func, nn.GroupNorm2D)
                else norm_func
            )(nin=256)
        else:
            self.scale_norm_1 = lambda x: x
            self.scale_norm_2 = lambda x: x

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
        # out = self.FlatFeats(out)
        out = self.classifier(out, *args, **local_kwargs(kwargs, self.classifier))
        return out
