import types
from inspect import signature
from typing import Tuple, Union, Callable
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
    ComplexGroupNorm2DWhitening,
    ComplexLinear,
    ConjugateMish,
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


def _is_function(class_in_question):
    return isinstance(
        class_in_question,
        (types.FunctionType, types.BuiltinFunctionType, CompiledFunction),
    )


class ComplexModelConverter:
    """This Module replaces modules and attributes of networks to new
    classes or functions. In particular we use this to convert normal
    networks to complex networks.

    Remark: this only works if all ops are attributes of the class or
    function. If an operation is hardcoded in the __call__ method
    this will not be changed. Looking at you objax.zoo.resnet_v2
    """

    def __init__(
        self,
        new_conv_class: Union[Module, Callable] = ComplexWSConv2D,
        new_norm_class: Union[Module, Callable] = ComplexGroupNorm2DWhitening,
        new_linear_class: Union[Module, Callable] = ComplexLinear,
        new_activation: Union[Module, Callable] = ConjugateMish,
        new_pooling: Union[Module, Callable] = average_pool_2d,
        conv_classes_to_replace: Tuple[Union[Module, Callable]] = DEFAULT_CONV_CLASSES,
        norm_classes_to_replace: Tuple[Union[Module, Callable]] = DEFAULT_NORM_CLASSES,
        linear_classes_to_replace: Tuple[
            Union[Module, Callable]
        ] = DEFAULT_LINEAR_CLASSES,
        activations_to_replace: Tuple[Union[Module, Callable]] = DEFAULT_ACTIVATIONS,
        poolings_to_replace: Tuple[Union[Module, Callable]] = DEFAULT_POOLING,
    ) -> None:
        self.conversion_dict: dict[list[Union[Callable, Module]], Module] = {
            **{conv_class: new_conv_class for conv_class in conv_classes_to_replace},
            **{norm_class: new_norm_class for norm_class in norm_classes_to_replace},
            **{
                linear_class: new_linear_class
                for linear_class in linear_classes_to_replace
            },
            **{activation: new_activation for activation in activations_to_replace},
            **{pooling: new_pooling for pooling in poolings_to_replace},
        }

    def convert(self, model):
        if isinstance(model, Iterable):
            if isinstance(model, tuple):
                model = list(model)
            return self._convert_module_list(model)
        if isinstance(model, Module) and not any(
            (
                isinstance(model, replacement_class)
                for replacement_class in self.conversion_dict.keys()
                if not (
                    _is_function(replacement_class)
                    or isinstance(replacement_class, partial)
                )
            )
        ):
            for attr_name, attr_value in model.__dict__.items():
                new_layer = None
                if isinstance(attr_value, Module):
                    new_layer = self.convert(attr_value)
                elif isinstance(attr_value, Iterable):
                    if isinstance(attr_value, tuple):
                        attr_value = list(attr_value)
                    new_layer = self.convert(attr_value)
                elif callable(attr_value):
                    new_layer = self._replace_layer(attr_value)
                if new_layer is not None:
                    setattr(model, attr_name, new_layer)
            return model
        new_layer = self._replace_layer(model)
        if new_layer is None:
            return model
        return new_layer

    def _convert_module_list(self, model):
        for i, layer in enumerate(model):
            new_layer = self.convert(layer)
            # new_layer = self._replace_layer(layer)
            if new_layer is not None:
                model[i] = new_layer
        return model

    def _replace_layer(self, layer):
        for old_layer, new_layer in self.conversion_dict.items():
            if (callable(layer) or _is_function(layer)) and layer == old_layer:
                if not _is_function(new_layer) and issubclass(new_layer, Module):
                    new_layer = new_layer()
                return new_layer
            elif (
                # not is_function(new_layer) and
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
                    if not _is_function(new_layer) and issubclass(new_layer, Module)
                    else partial(new_layer, *layer.args, **kwargs)
                )
            elif not _is_function(old_layer) and isinstance(layer, old_layer):
                args = find_args(layer)
                signature_attrs = signature(new_layer).parameters.keys()
                args = {
                    k: v
                    for k, v in args.items()
                    if k in signature_attrs
                    if not k in ["w_init"]
                }
                return new_layer(**args)
        warn(f"No replacement for {layer}")
        return None

    def __call__(self, model) -> Module:
        return self.convert(model)


# if __name__ == "__main__":
#     import numpy as np
#     from dptraining.models import resnet_v2
#     from objax.nn import Sequential
#     from jax import numpy as jnp

#     m = resnet_v2.ResNet18(3, 2)
#     # m = vgg.VGG19(pretrained=False)
#     converter = ComplexModelConverter()
#     m2 = converter(m)
#     print(m2)
#     data = np.random.randn(10, 3, 224, 224) + 1j * np.random.randn(10, 3, 224, 224)
#     m2(data, training=False)

#     m3 = Sequential([m2, jnp.abs])
#     out = m3(data, training=False)
#     assert jnp.all(jnp.isreal(out))
