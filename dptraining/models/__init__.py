import inspect
from typing import Callable
import warnings
from copy import deepcopy
from functools import partial
from objax import nn, functional
from dptraining.models.cifar10models import Cifar10ConvNet
from dptraining.models.ensemble import Ensemble
from dptraining.models.resnet9 import ResNet9
from dptraining.models.smoothnet import get_smoothnet
from dptraining.models.activations import mish
from dptraining.models.layers import ConvWS2D, ConvCentering2D
from dptraining.models.complex.activations import (
    IGaussian,
    SeparableMish,
    ComplexMish,
    ConjugateMish,
    Cardioid,
)
from dptraining.models.complex.normalization import (
    ComplexGroupNorm2DWhitening,
    ComplexBatchNorm2D,
)  # pylint:disable=duplicate-code
from dptraining.models.complex.layers import (
    ComplexConv2D,
    ComplexWSConv2D,
    ComplexLinear,
    ComplexToReal,
    ComplexWSConv2DNoWhiten,
)
from dptraining.models.complex.pooling import ConjugatePool2D, SeparablePool2D
from dptraining.models import resnet_v2, wide_resnet
from dptraining.models.complex.converter import ComplexModelConverter
from dptraining.models.unet import Unet


SUPPORTED_MODELS = ("cifar10model", "resnet18", "resnet9", "smoothnet", "wide_resnet")
SUPPORTED_NORMALIZATION = ("bn", "gn")
SUPPORTED_CONV = ("conv", "convws", "convws_nw")
SUPPORTED_ACTIVATION = ("relu", "selu", "leakyrelu", "mish")
SUPPORTED_POOLING = ("maxpool", "avgpool")

SUPPORTED_COMPLEX_MODELS = ("resnet9", "smoothnet", "unet")
SUPPORTED_COMPLEX_CONV = ("conv", "convws", "convws_nw")
SUPPORTED_COMPLEX_NORMALIZATION = ("gnw", "bn")
SUPPORTED_COMPLEX_ACTIVATION = ("mish", "sepmish", "conjmish", "igaussian", "cardioid")
SUPPORTED_COMPLEX_POOLING = ("conjmaxpool", "sepmaxpool", "avgpool")


def get_kwargs(func: Callable, already_defined: list[str], original_kwargs: dict):
    signature = inspect.getfullargspec(func)
    kwargs = {
        k: v
        for k, v in original_kwargs.items()
        if k in signature[0] and k not in already_defined
    }
    ignoring = {
        k: v
        for k, v in original_kwargs.items()
        if k not in signature[0]
        and k
        not in [
            "in_channels",
            "num_classes",
            "conv",
            "activation",
            "pooling",
            "normalization",
            "name",
        ]
    }
    print(f"Additional kwargs for {func}: {kwargs}")
    if len(ignoring) > 0:
        warnings.warn(f" -> Ignoring: {ignoring}")
    return kwargs


def make_normalization_from_config(config: dict) -> Callable:
    match config["model"]["normalization"]:
        case "bn":
            return nn.BatchNorm2D
        case "gn":
            return nn.GroupNorm2D
        case _ as fail:
            raise ValueError(
                f"Unsupported normalization layer '{fail}'. "
                f"Legal options are: {SUPPORTED_NORMALIZATION}"
            )


def make_complex_normalization_from_config(config: dict) -> Callable:
    match config["model"]["normalization"]:
        case "gnw":
            return ComplexGroupNorm2DWhitening
        case "bn":
            return ComplexBatchNorm2D
        case _ as fail:
            raise ValueError(
                f"Unsupported normalization layer '{fail}'. "
                f"Legal options are: {SUPPORTED_COMPLEX_NORMALIZATION}"
            )


def make_conv_from_config(config: dict) -> Callable:
    match config["model"]["conv"]:
        case "conv":
            return nn.Conv2D
        case "convws":
            return ConvWS2D
        case "convws_nw":
            return ConvCentering2D
        case _ as fail:
            raise ValueError(
                f"Unsupported convolutional layer '{fail}'. "
                f"Legal options are: {SUPPORTED_CONV}"
            )


def make_complex_conv_from_config(config: dict) -> Callable:
    match config["model"]["conv"]:
        case "conv":
            return ComplexConv2D
        case "convws":
            return ComplexWSConv2D
        case "convws_nw":
            return ComplexWSConv2DNoWhiten
        case _ as fail:
            raise ValueError(
                f"Unsupported convolutional layer '{fail}'. "
                f"Legal options are: {SUPPORTED_COMPLEX_CONV}"
            )


def make_activation_from_config(config: dict) -> Callable:
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


def make_complex_activation_from_config(config: dict, init_layers: bool) -> Callable:
    match config["model"]["activation"]:
        case "mish":
            act = ComplexMish
        case "sepmish":
            act = SeparableMish
        case "conjmish":
            act = ConjugateMish
        case "igaussian":
            act = IGaussian
        case "cardioid":
            act = Cardioid
        case _ as fail:
            raise ValueError(
                f"Unsupported activation layer '{fail}'. "
                f"Legal options are: {SUPPORTED_COMPLEX_NORMALIZATION}"
            )
    if init_layers:
        return act()
    return act


def make_pooling_from_config(config: dict) -> Callable:
    match config["model"]["pooling"]:
        case "maxpool":
            return functional.max_pool_2d
        case "avgpool":
            return functional.average_pool_2d
        case _ as fail:
            raise ValueError(
                f"Unsupported pooling layer '{fail}'. "
                f"Legal options are: {SUPPORTED_COMPLEX_POOLING}"
            )


def make_complex_pooling_from_config(
    config: dict, init_layers: bool, **kwargs
) -> Callable:
    match config["model"]["pooling"]:
        case "conjmaxpool":
            layer = ConjugatePool2D
        case "sepmaxpool":
            layer = SeparablePool2D
        case "avgpool":
            if len(kwargs) == 0:
                return functional.average_pool_2d
            return partial(functional.average_pool_2d, **kwargs)
        case _ as fail:
            raise ValueError(
                f"Unsupported pooling layer '{fail}'. "
                f"Legal options are: {SUPPORTED_COMPLEX_POOLING}"
            )
    if init_layers:
        return layer(**kwargs)
    else:
        return layer


def make_normal_model_from_config(config: dict) -> Callable:
    match config["model"]["name"]:
        case "cifar10model":
            if "activation" in config["model"]:
                warnings.warn("No choice of activations supported for cifar 10 model")
            if "normalization" in config["model"]:
                warnings.warn("No choice of normalization supported for cifar 10 model")
            return Cifar10ConvNet(nclass=config["model"]["num_classes"])
        case "resnet18":
            kwargs = get_kwargs(
                resnet_v2.ResNet18,
                [
                    "in_channels",
                    "num_classes",
                    "conv_layer",
                    "normalization_fn",
                    "activation_fn",
                ],
                config["model"],
            )
            return resnet_v2.ResNet18(
                config["model"]["in_channels"],
                config["model"]["num_classes"],
                conv_layer=make_conv_from_config(config),
                normalization_fn=make_normalization_from_config(config),
                activation_fn=make_activation_from_config(config),
                **kwargs,
            )
        case "wide_resnet":
            kwargs = get_kwargs(
                wide_resnet.WideResNet,
                ["nin", "nclass", "conv_layer", "bn", "act"],
                config["model"],
            )
            return wide_resnet.WideResNet(
                config["model"]["in_channels"],
                config["model"]["num_classes"],
                conv_layer=make_conv_from_config(config),
                bn=make_normalization_from_config(config),
                act=make_activation_from_config(config),
                **kwargs,
            )
        case "resnet9":
            already_defined = (
                "in_channels",
                "num_classes",
                "conv_cls",
                "norm_cls",
                "act_func",
                "pool_func",
            )
            kwargs = get_kwargs(ResNet9, already_defined, config["model"])
            return ResNet9(
                config["model"]["in_channels"],
                config["model"]["num_classes"],
                conv_cls=make_conv_from_config(config),
                norm_cls=make_normalization_from_config(config),
                act_func=make_activation_from_config(config),
                pool_func=partial(make_pooling_from_config(config), size=2),
                **kwargs,
            )
        case "smoothnet":
            already_defined = (
                "in_channels",
                "num_classes",
                "conv_cls",
                "norm_cls",
                "act_func",
                "pool_func",
            )
            kwargs = get_kwargs(get_smoothnet, already_defined, config["model"])
            return get_smoothnet(
                in_channels=config["model"]["in_channels"],
                num_classes=config["model"]["num_classes"],
                conv_cls=make_conv_from_config(config),
                norm_cls=make_normalization_from_config(config),
                act_func=make_activation_from_config(config),
                pool_func=partial(
                    make_pooling_from_config(config), size=3, strides=1, padding=1
                ),
                **kwargs,
            )
        case _ as fail:
            raise ValueError(
                f"Unsupported model '{fail}'. Legal options are: {SUPPORTED_MODELS}"
            )


def make_complex_model_from_config(config: dict) -> Callable:
    match config["model"]["name"]:
        case "resnet9":
            already_defined = (
                "in_channels",
                "num_classes",
                "conv_cls",
                "norm_cls",
                "act_func",
                "pool_func",
                "linear_cls",
                "out_func",
            )
            kwargs = get_kwargs(ResNet9, already_defined, config["model"])
            return ResNet9(
                config["model"]["in_channels"],
                config["model"]["num_classes"],
                conv_cls=make_complex_conv_from_config(config),
                norm_cls=make_complex_normalization_from_config(config),
                act_func=make_complex_activation_from_config(config, init_layers=True),
                pool_func=make_complex_pooling_from_config(
                    config,
                    init_layers=True,
                    size=2,
                ),
                linear_cls=ComplexLinear,
                out_func=ComplexToReal(),
                **kwargs,
            )
        case "smoothnet":
            already_defined = (
                "in_channels",
                "num_classes",
                "conv_cls",
                "norm_cls",
                "act_func",
                "pool_func",
                "linear_cls",
                "out_func",
            )
            kwargs = get_kwargs(get_smoothnet, already_defined, config["model"])
            return get_smoothnet(
                in_channels=config["model"]["in_channels"],
                num_classes=config["model"]["num_classes"],
                conv_cls=make_complex_conv_from_config(config),
                norm_cls=make_complex_normalization_from_config(config),
                act_func=make_complex_activation_from_config(config, init_layers=True),
                pool_func=make_complex_pooling_from_config(
                    config,
                    init_layers=True,
                    size=3,
                    strides=1,
                    padding=1,
                ),
                linear_cls=ComplexLinear,
                out_func=ComplexToReal(),
                **kwargs,
            )
        case "unet":
            already_defined = ("in_channels",)
            kwargs = get_kwargs(Unet, already_defined, config["model"])
            if not all(
                (
                    expected in kwargs.keys()
                    for expected in (
                        "out_channels",
                        "channels",
                    )
                )
            ):
                warnings.warn(
                    "We recommend to explicitly set values for [out_channels, channels]"
                    "for a UNet architecture"
                )
            return Unet(
                in_channels=config["model"]["in_channels"],
                actv=make_complex_activation_from_config(config, init_layers=False),
                **kwargs,
            )
        case _ as fail:
            raise ValueError(
                f"Unsupported model '{fail}'. Legal options are: {SUPPORTED_COMPLEX_MODELS}"
            )


def make_model_from_config(config: dict) -> Callable:
    num_replicas = config["model"]["ensemble"] if "ensemble" in config["model"] else 1
    ensemble = []
    for _ in range(num_replicas):
        if "complex" in config["model"] and config["model"]["complex"]:
            if config["model"]["name"] in SUPPORTED_COMPLEX_MODELS:
                model = make_complex_model_from_config(config)
            elif config["model"]["name"] in SUPPORTED_MODELS:
                fake_config = deepcopy(config)
                fake_config["model"].update(
                    {
                        "conv": "conv",
                        "normalization": "gn",
                        "activation": "relu",
                        "pooling": "avgpool",
                    }
                )
                model = make_normal_model_from_config(fake_config)
                converter = ComplexModelConverter(
                    new_conv_class=make_complex_conv_from_config(config),
                    new_norm_class=make_complex_normalization_from_config(config),
                    new_linear_class=ComplexLinear,
                    new_activation=make_complex_activation_from_config(
                        config, init_layers=False
                    ),
                    new_pooling=make_complex_pooling_from_config(
                        config, init_layers=False
                    ),
                )
                model = nn.Sequential([converter(model), ComplexToReal()])
            else:
                raise ValueError(f"{config['model']['name']} unknown")
        else:
            model = make_normal_model_from_config(config)
        if "ensemble" in config["model"] and config["model"]["ensemble"] > 1:
            ensemble.append(model)
    if "ensemble" in config["model"] and config["model"]["ensemble"] > 1:
        model = Ensemble(ensemble)
    return model
