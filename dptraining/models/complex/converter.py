import types
from inspect import signature
from typing import Union, Callable
from functools import partial
from collections.abc import Iterable
from jaxlib.xla_extension import CompiledFunction  # pylint:disable=no-name-in-module
from objax import Module

from objax.nn import Conv2D, BatchNorm2D, GroupNorm2D, Linear, BatchNorm, GroupNorm
from objax.functional import (
    relu,
    leaky_relu,
    celu,
    elu,
    selu,
    softplus,
    tanh,
    average_pool_2d,
    max_pool_2d,
)
from warnings import warn

import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from dptraining.models.complex import (
    ComplexWSConv2D,
    ComplexMish,
    ComplexGroupNorm2DWhitening,
    ComplexLinear,
)


DEFAULT_CONV_CLASSES = (Conv2D,)
DEFAULT_NORM_CLASSES = (BatchNorm2D, GroupNorm2D)
DEFAULT_LINEAR_CLASSES = (Linear,)
DEFAULT_ACTIVATIONS = (relu, leaky_relu, celu, elu, selu, softplus, tanh)
DEFAULT_POOLING = (average_pool_2d, max_pool_2d)


def find_args(original_layer):
    if type(original_layer) in DEFAULT_CONV_CLASSES:
        return find_conv_args(original_layer)
    elif type(original_layer) in DEFAULT_NORM_CLASSES:
        return find_norm_args(original_layer)
    elif type(original_layer) in DEFAULT_LINEAR_CLASSES:
        return find_linear_args(original_layer)
    else:
        raise ValueError(f"No arg finder for {type(original_layer)}")


def find_conv_args(original_layer):
    return dict(
        nin=original_layer.w.value.shape[2],
        nout=original_layer.w.value.shape[3],
        k=original_layer.w.value.shape[:2],
        strides=original_layer.strides,
        dilations=original_layer.dilations,
        padding=original_layer.padding,
        use_bias=original_layer.b is not None,
        w_init=original_layer.w_init,
    )


def find_norm_args(original_layer):
    if isinstance(original_layer, BatchNorm):
        return dict(
            nin=original_layer.beta.value.shape[1],
            momentum=original_layer.momentum,
            eps=original_layer.eps,
        )
    elif isinstance(original_layer, GroupNorm):
        return dict(
            nin=original_layer.nin,
            rank=len(original_layer.gamma.shape),
            groups=original_layer.groups,
            eps=original_layer.eps,
        )
    else:
        raise ValueError(f"No arg finder for {type(original_layer)}")


def find_linear_args(original_layer):
    linear_shape = original_layer.w.value.shape
    return dict(
        nin=linear_shape[0],
        nout=linear_shape[1],
        use_bias=original_layer.b is not None,
        w_init=original_layer.w_init,
    )


class ComplexModelConverter:
    def __init__(
        self,
        conversion_dict: dict[list[Union[Callable, Module]], Module] = {
            **{conv_class: ComplexWSConv2D for conv_class in DEFAULT_CONV_CLASSES},
            **{
                norm_class: ComplexGroupNorm2DWhitening
                for norm_class in DEFAULT_NORM_CLASSES
            },
            **{linear_class: ComplexLinear for linear_class in DEFAULT_LINEAR_CLASSES},
            **{activation: ComplexMish for activation in DEFAULT_ACTIVATIONS},
            **{pooling: average_pool_2d for pooling in DEFAULT_POOLING},
        },
    ) -> None:
        self.conversion_dict = conversion_dict

    def convert_module_list(self, model):
        for i, layer in enumerate(model):
            if isinstance(layer, Iterable):
                if isinstance(layer, tuple):
                    model[i] = layer = list(layer)
                model[i] = self.convert_module_list(layer)
                continue
            if isinstance(layer, Module) and not any(
                (
                    isinstance(layer, replacement_class)
                    for replacement_class in self.conversion_dict.keys()
                    if not (
                        self.is_function(replacement_class)
                        or isinstance(replacement_class, partial)
                    )
                )
            ):
                for attr_name, attr_value in layer.__dict__.items():
                    new_layer = None
                    if isinstance(attr_value, Module) or callable(attr_value):
                        new_layer = self._replace_layer(attr_value)
                    elif isinstance(attr_value, Iterable):
                        if isinstance(attr_value, tuple):
                            attr_value = list(attr_value)
                        new_layer = self.convert_module_list(attr_value)
                    if new_layer is not None:
                        setattr(layer, attr_name, new_layer)
                continue
            new_layer = self._replace_layer(layer)
            if new_layer is not None:
                model[i] = new_layer
        return model

    def is_function(self, class_in_question):
        return isinstance(
            class_in_question,
            (types.FunctionType, types.BuiltinFunctionType, CompiledFunction),
        )

    def _replace_layer(self, layer):
        for old_layer, new_layer in self.conversion_dict.items():
            if (callable(layer) or self.is_function(layer)) and layer == old_layer:
                if issubclass(new_layer, Module):
                    new_layer = new_layer()
                return new_layer
            elif (
                # not self.is_function(new_layer) and
                isinstance(layer, partial)
                and layer.func == old_layer
            ):
                signature_attrs = signature(new_layer).parameters.keys()
                kwargs = {
                    k: v for k, v in layer.keywords.items() if k in signature_attrs
                }
                if len(layer.args) == 0 and len(kwargs) == 0:
                    if issubclass(new_layer, Module):
                        new_layer = new_layer()
                    return new_layer
                return (
                    new_layer(*layer.args, **kwargs)
                    if not self.is_function(new_layer) and issubclass(new_layer, Module)
                    else partial(new_layer, *layer.args, **kwargs)
                )
            elif not self.is_function(old_layer) and isinstance(layer, old_layer):
                args = find_args(layer)
                signature_attrs = signature(new_layer).parameters.keys()
                args = {k: v for k, v in args.items() if k in signature_attrs}
                return new_layer(**args)
        warn(f"No replacement for {layer}")
        return None

    def __call__(self, model) -> Module:
        return self.convert_module_list(model)


# if __name__ == "__main__":
#     import numpy as np
#     from dptraining.models import resnet_v2

#     m = resnet_v2.ResNet18(3, 2)
#     # m = vgg.VGG19(pretrained=False)
#     converter = ComplexModelConverter()
#     m2 = converter(m)
#     print(m2)
#     data = np.random.randn(10, 3, 224, 224) + 1j * np.random.randn(10, 3, 224, 224)
#     m2(data, training=False)
