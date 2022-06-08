import warnings
from functools import partial
from jax import numpy as jnp
from objax.zoo import resnet_v2
from objax import nn, functional
from dptraining.models.cifar10models import Cifar10ConvNet
from dptraining.models.resnet9 import ResNet9
from dptraining.models.activations import mish
from dptraining.models.complex.activations import (
    IGaussian,
    SeparableMish,
    ComplexMish,
    ConjugateMish,
    Cardioid,
)
from dptraining.models.complex.normalization import (
    ComplexGroupNorm2D,
    ComplexGroupNorm2DWhitening,
)  # pylint:disable=duplicate-code
from dptraining.models.complex.layers import (
    ComplexConv2D,
    ComplexWSConv2D,
    ComplexLinear,
)
from dptraining.models.complex.pooling import ConjugateMaxPool2D, SeparableMaxPool2D


SUPPORTED_MODELS = ("cifar10model", "resnet18", "resnet9")
SUPPORTED_NORMALIZATION = ("bn", "gn")
SUPPORTED_ACTIVATION = ("relu", "selu", "leakyrelu", "mish")

SUPPORTED_COMPLEX_MODELS = ("resnet9",)
SUPPORTED_COMPLEX_CONV = ("conv", "convws")
SUPPORTED_COMPLEX_NORMALIZATION = ("gn", "gnw")
SUPPORTED_COMPLEX_ACTIVATION = ("mish", "sepmish", "conjmish", "igaussian", "cardioid")
SUPPORTED_COMPLEX_POOLING = ("conjmaxpool", "sepmaxpool", "avgpool")


def make_normalization_from_config(config):
    match config["model"]["normalization"]:
        case "bn":
            return nn.BatchNorm2D
        case "gn":
            return nn.GroupNorm2D
        case _ as fail:
            raise ValueError(
                f"Unsupported normalization layer '{fail}'. Legal options are: {SUPPORTED_NORMALIZATION}"
            )


def make_complex_normalization_from_config(config):
    match config["model"]["normalization"]:
        case "gn":
            return ComplexGroupNorm2D
        case "gnw":
            return ComplexGroupNorm2DWhitening
        case _ as fail:
            raise ValueError(
                f"Unsupported normalization layer '{fail}'. Legal options are: {SUPPORTED_COMPLEX_NORMALIZATION}"
            )


def make_complex_conv_from_config(config):
    match config["model"]["conv"]:
        case "conv":
            return ComplexConv2D
        case "convws":
            return ComplexWSConv2D
        case _ as fail:
            raise ValueError(
                f"Unsupported convolutional layer '{fail}'. Legal options are: {SUPPORTED_COMPLEX_CONV}"
            )


def make_activation_from_config(config):
    match config["model"]["activation"]:
        case "relu":
            return functional.relu
        case "selu":
            return functional.selu
        case "leakyrelu":
            return functional.leaky_relu
        case "mish":
            return mish
        case _ as fail:
            raise ValueError(
                f"Unsupported activation layer '{fail}'. Legal options are: {SUPPORTED_ACTIVATION}"
            )


def make_complex_activation_from_config(config):
    match config["model"]["activation"]:
        case "mish":
            return ComplexMish()
        case "sepmish":
            return SeparableMish()
        case "conjmish":
            return ConjugateMish()
        case "igaussian":
            return IGaussian()
        case "cardioid":
            return Cardioid()
        case _ as fail:
            raise ValueError(
                f"Unsupported activation layer '{fail}'. Legal options are: {SUPPORTED_COMPLEX_NORMALIZATION}"
            )


def make_complex_pooling_from_config(config):
    match config["model"]["pooling"]:
        case "conjmaxpool":
            return ConjugateMaxPool2D(2)
        case "sepmaxpool":
            return SeparableMaxPool2D(2)
        case "avgpool":
            return partial(functional.average_pool_2d, size=2)
        case _ as fail:
            raise ValueError(
                f"Unsupported pooling layer '{fail}'. Legal options are: {SUPPORTED_COMPLEX_POOLING}"
            )


def make_normal_model_from_config(config):
    match config["model"]["name"]:
        case "cifar10model":
            if "activation" in config["model"]:
                warnings.warn("No choice of activations supported for cifar 10 model")
            if "normalization" in config["model"]:
                warnings.warn("No choice of normalization supported for cifar 10 model")
            return Cifar10ConvNet(nclass=config["model"]["num_classes"])
        case "resnet18":
            return resnet_v2.ResNet18(
                config["model"]["in_channels"],
                config["model"]["num_classes"],
                normalization_fn=make_normalization_from_config(config),
                activation_fn=make_activation_from_config(config),
            )
        case "resnet9":
            return ResNet9(
                config["model"]["in_channels"],
                config["model"]["num_classes"],
                norm_cls=make_normalization_from_config(config),
                act_func=make_activation_from_config(config),
                scale_norm=config["model"]["scale_norm"]
                if "scale_norm" in config["model"]
                else False,
            )
        case _ as fail:
            raise ValueError(
                f"Unsupported model '{fail}'. Legal options are: {SUPPORTED_MODELS}"
            )


def make_complex_model_from_config(config):
    match config["model"]["name"]:
        case "resnet9":
            return ResNet9(
                config["model"]["in_channels"],
                config["model"]["num_classes"],
                conv_cls=make_complex_conv_from_config(config),
                norm_cls=make_complex_normalization_from_config(config),
                act_func=make_complex_activation_from_config(config),
                pool_func=make_complex_pooling_from_config(config),
                linear_cls=ComplexLinear,
                out_func=jnp.abs,
                scale_norm=config["model"]["scale_norm"]
                if "scale_norm" in config["model"]
                else False,
            )
        case _ as fail:
            raise ValueError(
                f"Unsupported model '{fail}'. Legal options are: {SUPPORTED_COMPLEX_MODELS}"
            )


def make_model_from_config(config):
    if "complex" in config["model"] and config["model"]["complex"]:
        model = make_complex_model_from_config(config)
    else:
        model = make_normal_model_from_config(config)
    return model
