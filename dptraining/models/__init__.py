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
SUPPORTED_COMPLEX_NORMALIZATION = ("gn",)
SUPPORTED_COMPLEX_ACTIVATION = ("mish", "sepmish", "conjmish", "igaussian", "cardioid")
SUPPORTED_COMPLEX_POOLING = ("conjmaxpool", "sepmaxpool", "avgpool")


def make_normalization_from_config(config):
    if config["model"]["normalization"] not in SUPPORTED_NORMALIZATION:
        raise ValueError(
            f"{config['model']['normalization']} not supported yet. "
            f"Currently supported normalizations: {SUPPORTED_NORMALIZATION}"
        )
    if config["model"]["normalization"] == "bn":
        norm = nn.BatchNorm2D
    elif config["model"]["normalization"] == "gn":
        norm = nn.GroupNorm2D
    else:
        raise ValueError(
            f"This shouldn't happen. "
            f"{SUPPORTED_NORMALIZATION} includes not supported norm layers."
        )
    return norm


def make_complex_normalization_from_config(config):
    if config["model"]["normalization"] not in SUPPORTED_COMPLEX_NORMALIZATION:
        raise ValueError(
            f"{config['model']['normalization']} not supported yet. "
            f"Currently supported complex normalizations: {SUPPORTED_COMPLEX_NORMALIZATION}"
        )
    if config["model"]["normalization"] == "gn":
        norm = ComplexGroupNorm2D
    else:
        raise ValueError(
            f"This shouldn't happen. "
            f"{SUPPORTED_COMPLEX_NORMALIZATION} includes not supported norm layers."
        )
    return norm


def make_complex_conv_from_config(config):
    if config["model"]["conv"] not in SUPPORTED_COMPLEX_CONV:
        raise ValueError(
            f"{config['model']['conv']} not supported yet. "
            f"Currently supported convs: {SUPPORTED_COMPLEX_CONV}"
        )
    if config["model"]["conv"] == "conv":
        conv = ComplexConv2D
    elif config["model"]["conv"] == "convws":
        conv = ComplexWSConv2D
    else:
        raise ValueError(
            f"This shouldn't happen. "
            f"{SUPPORTED_COMPLEX_CONV} includes not supported conv layers."
        )
    return conv


def make_activation_from_config(config):
    if config["model"]["activation"] not in SUPPORTED_ACTIVATION:
        raise ValueError(
            f"{config['model']['activation']} not supported yet. "
            f"Currently supported activations: {SUPPORTED_ACTIVATION}"
        )
    if config["model"]["activation"] == "relu":
        act = functional.relu
    elif config["model"]["activation"] == "selu":
        act = functional.selu
    elif config["model"]["activation"] == "leakyrelu":
        act = functional.leaky_relu
    elif config["model"]["activation"] == "mish":
        act = mish
    else:
        raise ValueError(
            f"This shouldn't happen. "
            f"{SUPPORTED_ACTIVATION} includes not supported activation layers."
        )
    return act


def make_complex_activation_from_config(config):
    if config["model"]["activation"] not in SUPPORTED_COMPLEX_ACTIVATION:
        raise ValueError(
            f"{config['model']['activation']} not supported yet. "
            f"Currently supported activations: {SUPPORTED_COMPLEX_ACTIVATION}"
        )
    if config["model"]["activation"] == "mish":
        act = ComplexMish()
    elif config["model"]["activation"] == "sepmish":
        act = SeparableMish()
    elif config["model"]["activation"] == "conjmish":
        act = ConjugateMish()
    elif config["model"]["activation"] == "igaussian":
        act = IGaussian()
    elif config["model"]["activation"] == "cardioid":
        act = Cardioid()
    else:
        raise ValueError(
            f"This shouldn't happen. "
            f"{SUPPORTED_COMPLEX_ACTIVATION} includes not supported activation layers."
        )
    return act


def make_complex_pooling_from_config(config):
    if config["model"]["pooling"] not in SUPPORTED_COMPLEX_POOLING:
        raise ValueError(
            f"{config['model']['pooling']} not supported yet. "
            f"Currently supported pooling: {SUPPORTED_COMPLEX_POOLING}"
        )
    if config["model"]["pooling"] == "conjmaxpool":
        pool = ConjugateMaxPool2D(2)
    elif config["model"]["pooling"] == "sepmaxpool":
        pool = SeparableMaxPool2D(2)
    elif config["model"]["pooling"] == "avgpool":
        pool = partial(functional.average_pool_2d, size=2)
    else:
        raise ValueError(
            f"This shouldn't happen. "
            f"{SUPPORTED_COMPLEX_POOLING} includes not supported pooling layers."
        )
    return pool


def make_normal_model_from_config(config):
    if config["model"]["name"] not in SUPPORTED_MODELS:
        raise ValueError(
            f"{config['model']['name']} not supported yet. "
            f"Currently supported models: {SUPPORTED_MODELS}"
        )
    if config["model"]["name"] == "cifar10model":
        if "activation" in config["model"]:
            warnings.warn("No choice of activations supported for cifar 10 model")
        if "normalization" in config["model"]:
            warnings.warn("No choice of normalization supported for cifar 10 model")
        model = Cifar10ConvNet(nclass=config["model"]["num_classes"])
    elif config["model"]["name"] == "resnet18":
        model = resnet_v2.ResNet18(
            config["model"]["in_channels"],
            config["model"]["num_classes"],
            normalization_fn=make_normalization_from_config(config),
            activation_fn=make_activation_from_config(config),
        )
    elif config["model"]["name"] == "resnet9":
        model = ResNet9(
            config["model"]["in_channels"],
            config["model"]["num_classes"],
            norm_cls=make_normalization_from_config(config),
            act_func=make_activation_from_config(config),
            scale_norm=config["model"]["scale_norm"]
            if "scale_norm" in config["model"]
            else False,
        )
    else:
        raise ValueError(
            f"This shouldn't happen. "
            f"{SUPPORTED_MODELS} includes not supported models."
        )

    return model


def make_complex_model_from_config(config):
    if config["model"]["name"] not in SUPPORTED_COMPLEX_MODELS:
        raise ValueError(
            f"{config['model']['name']} not supported yet. "
            f"Currently supported models: {SUPPORTED_COMPLEX_MODELS}"
        )
    if config["model"]["name"] == "resnet9":
        model = ResNet9(
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
    else:
        raise ValueError(
            f"This shouldn't happen. "
            f"{SUPPORTED_COMPLEX_MODELS} includes not supported models."
        )

    return model


def make_model_from_config(config):
    if "complex" in config["model"] and config["model"]["complex"]:
        model = make_complex_model_from_config(config)
    else:
        model = make_normal_model_from_config(config)
    return model
