import inspect
import warnings
from copy import deepcopy
from functools import partial
from typing import Callable

from objax import functional, nn
from omegaconf import OmegaConf

from dptraining.config import Config, ModelConfig, get_allowed_values
from dptraining.config.model import (
    Activation,
    ComplexActivation,
    ComplexConv,
    ComplexModelName,
    ComplexNormalization,
    ComplexPooling,
    Conv,
    Normalization,
    Pooling,
    RealActivation,
    RealConv,
    RealModelName,
    RealNormalization,
    RealPooling,
)
from dptraining.models import resnet_v2, wide_resnet
from dptraining.models.activations import mish
from dptraining.models.cifar10models import Cifar10ConvNet
from dptraining.models.complex.activations import (
    Cardioid,
    ComplexMish,
    ConjugateMish,
    IGaussian,
    SeparableMish,
)
from dptraining.models.complex.converter import ComplexModelConverter
from dptraining.models.complex.layers import (
    ComplexConv2D,
    ComplexLinear,
    ComplexToReal,
    ComplexWSConv2D,
    ComplexWSConv2DNoWhiten,
)
from dptraining.models.complex.normalization import (  # pylint:disable=duplicate-code
    ComplexBatchNorm2D,
    ComplexGroupNorm2DWhitening,
)
from dptraining.models.complex.pooling import ConjugatePool2D, SeparablePool2D
from dptraining.models.ensemble import Ensemble
from dptraining.models.layers import ConvCentering2D, ConvWS2D
from dptraining.models.unet import Unet
from dptraining.models.resnet9 import ResNet9
from dptraining.models.smoothnet import get_smoothnet


def get_kwargs(func: Callable, already_defined: list[str], model_config: ModelConfig):
    model_config_dict = deepcopy(OmegaConf.to_container(model_config))
    # Pull extra args into the root level of the model config container
    extra_args = model_config_dict.pop("extra_args", None)
    model_config_dict |= extra_args if extra_args else {}
    signature = inspect.getfullargspec(func)
    kwargs = {
        k: v
        for k, v in model_config_dict.items()
        if k in signature[0] and k not in already_defined
    }
    ignoring = {
        k: v
        for k, v in model_config_dict.items()
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


def make_normalization_from_config(config: Config) -> Callable:
    match config.model.normalization.value:
        case RealNormalization.bn.value:
            return nn.BatchNorm2D
        case RealNormalization.gn.value:
            return nn.GroupNorm2D
        case None:
            raise ValueError("No normalization layer specified (required)")
        case _:
            raise ValueError(
                f"Unsupported normalization layer '{config.model.normalization}'"
            )


def make_complex_normalization_from_config(config: Config) -> Callable:
    match config.model.normalization.value:
        case ComplexNormalization.bn.value:
            return ComplexBatchNorm2D
        case ComplexNormalization.gnw.value:
            return ComplexGroupNorm2DWhitening
        case None:
            raise ValueError("No normalization layer specified (required)")
        case _:
            raise ValueError(
                f"Unsupported normalization layer '{config.model.normalization}'"
            )


def make_conv_from_config(config: Config) -> Callable:
    match config.model.conv.value:
        case RealConv.conv.value:
            return nn.Conv2D
        case RealConv.convws.value:
            return ConvWS2D
        case RealConv.convws_nw.value:
            return ConvCentering2D
        case _:
            raise ValueError(f"Unsupported convolutional layer '{config.model.conv}'")


def make_complex_conv_from_config(config: Config) -> Callable:
    match config.model.conv.value:
        case ComplexConv.conv.value:
            return ComplexConv2D
        case ComplexConv.convws.value:
            return ComplexWSConv2D
        case ComplexConv.convws_nw.value:
            return ComplexWSConv2DNoWhiten
        case _:
            raise ValueError(f"Unsupported convolutional layer '{config.model.conv}'")


def make_activation_from_config(config: Config) -> Callable:
    match config.model.activation.value:
        case RealActivation.relu.value:
            return functional.relu
        case RealActivation.selu.value:
            return functional.selu
        case RealActivation.leakyrelu.value:
            return functional.leaky_relu
        case RealActivation.mish.value:
            return mish
        case None:
            raise ValueError("No activation layer specified (required)")
        case _:
            raise ValueError(
                f"Unsupported activation layer '{config.model.activation}'"
            )


def make_complex_activation_from_config(config: Config, init_layers: bool) -> Callable:
    match config.model.activation.value:
        case ComplexActivation.mish.value:
            act = ComplexMish
        case ComplexActivation.sepmish.value:
            act = SeparableMish
        case ComplexActivation.conjmish.value:
            act = ConjugateMish
        case ComplexActivation.igaussian.value:
            act = IGaussian
        case ComplexActivation.cardioid.value:
            act = Cardioid
        case None:
            raise ValueError("No activation layer specified (required)")
        case _:
            raise ValueError(
                f"Unsupported activation layer '{config.model.activation}'"
            )
    if init_layers:
        return act()
    return act


def make_pooling_from_config(config: Config) -> Callable:
    match config.model.pooling.value:
        case RealPooling.maxpool.value:
            return functional.max_pool_2d
        case RealPooling.avgpool.value:
            return functional.average_pool_2d
        case _:
            raise ValueError(f"Unsupported pooling layer '{config.model.pooling}'")


def make_complex_pooling_from_config(
    config: Config, init_layers: bool, **kwargs
) -> Callable:
    match config.model.pooling.value:
        case ComplexPooling.conjmaxpool.value:
            layer = ConjugatePool2D
        case ComplexPooling.sepmaxpool.value:
            layer = SeparablePool2D
        case ComplexPooling.avgpool.value:
            if len(kwargs) == 0:
                return functional.average_pool_2d
            return partial(functional.average_pool_2d, **kwargs)
        case _:
            raise ValueError(f"Unsupported pooling layer '{config.model.pooling}'")
    if init_layers:
        return layer(**kwargs)
    else:
        return layer


def make_model_from_config(config: Config) -> Callable:
    num_replicas = config.model.ensemble if config.model.ensemble else 1
    if num_replicas > 1:
        ensemble = [
            make_single_model_instance_from_config(config) for _ in range(num_replicas)
        ]
        return Ensemble(ensemble)
    else:
        return make_single_model_instance_from_config(config)


def make_single_model_instance_from_config(config: Config) -> Callable:
    if config.model.complex:
        if config.model.name.value in get_allowed_values(ComplexModelName):
            model = make_complex_model_from_config(config)
            return model
        elif config.model.name.value in get_allowed_values(RealModelName):
            fake_config = deepcopy(config)
            fake_config.model.conv = Conv.conv
            fake_config.model.normalization = Normalization.gn
            fake_config.model.activation = Activation.relu
            fake_config.model.pooling = Pooling.avgpool
            model = make_normal_model_from_config(fake_config)
            converter = ComplexModelConverter(
                new_conv_class=make_complex_conv_from_config(config),
                new_norm_class=make_complex_normalization_from_config(config),
                new_linear_class=ComplexLinear,
                new_activation=make_complex_activation_from_config(
                    config, init_layers=False
                ),
                new_pooling=make_complex_pooling_from_config(config, init_layers=False),
            )
            return nn.Sequential([converter(model), ComplexToReal()])
        else:
            raise ValueError(f"{config.model.name} unknown")
    else:
        return make_normal_model_from_config(config)


def make_normal_model_from_config(config: Config) -> Callable:
    match config.model.name.value:
        case RealModelName.cifar10model.value:
            if config.model.activation is not None:
                warnings.warn("No choice of activations supported for cifar 10 model")
            if config.model.normalization is not None:
                warnings.warn("No choice of normalization supported for cifar 10 model")
            return Cifar10ConvNet(nclass=config.model.num_classes)
        case RealModelName.resnet18.value:
            kwargs = get_kwargs(
                resnet_v2.ResNet18,
                [
                    "in_channels",
                    "num_classes",
                    "conv_layer",
                    "normalization_fn",
                    "activation_fn",
                ],
                config.model,
            )
            return resnet_v2.ResNet18(
                config.model.in_channels,
                config.model.num_classes,
                conv_layer=make_conv_from_config(config),
                normalization_fn=make_normalization_from_config(config),
                activation_fn=make_activation_from_config(config),
                **kwargs,
            )
        case RealModelName.wide_resnet.value:
            kwargs = get_kwargs(
                wide_resnet.WideResNet,
                ["nin", "nclass", "conv_layer", "bn", "act"],
                config.model,
            )
            return wide_resnet.WideResNet(
                config.model.in_channels,
                config.model.num_classes,
                conv_layer=make_conv_from_config(config),
                bn=make_normalization_from_config(config),
                act=make_activation_from_config(config),
                **kwargs,
            )
        case RealModelName.resnet9.value:
            already_defined = (
                "in_channels",
                "num_classes",
                "conv_cls",
                "norm_cls",
                "act_func",
                "pool_func",
            )
            kwargs = get_kwargs(ResNet9, already_defined, config.model)
            return ResNet9(
                config.model.in_channels,
                config.model.num_classes,
                conv_cls=make_conv_from_config(config),
                norm_cls=make_normalization_from_config(config),
                act_func=make_activation_from_config(config),
                pool_func=partial(make_pooling_from_config(config), size=2),
                **kwargs,
            )
        case RealModelName.smoothnet.value:
            already_defined = (
                "in_channels",
                "num_classes",
                "conv_cls",
                "norm_cls",
                "act_func",
                "pool_func",
            )
            kwargs = get_kwargs(get_smoothnet, already_defined, config.model)
            return get_smoothnet(
                in_channels=config.model.in_channels,
                num_classes=config.model.num_classes,
                conv_cls=make_conv_from_config(config),
                norm_cls=make_normalization_from_config(config),
                act_func=make_activation_from_config(config),
                pool_func=partial(
                    make_pooling_from_config(config), size=3, strides=1, padding=1
                ),
                **kwargs,
            )
        case _:
            raise ValueError(f"Unsupported model '{config.model.name}'")


def make_complex_model_from_config(config: Config) -> Callable:
    match config.model.name.value:
        case ComplexModelName.resnet9.value:
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
            kwargs = get_kwargs(ResNet9, already_defined, config.model)
            return ResNet9(
                config.model.in_channels,
                config.model.num_classes,
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
        case ComplexModelName.smoothnet.value:
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
            kwargs = get_kwargs(get_smoothnet, already_defined, config.model)
            return get_smoothnet(
                in_channels=config.model.in_channels,
                num_classes=config.model.num_classes,
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
        case ComplexModelName.unet.value:
            already_defined = ("in_channels",)
            kwargs = get_kwargs(Unet, already_defined, config.model)
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
                in_channels=config.model.in_channels,
                actv=make_complex_activation_from_config(config, init_layers=False),
                **kwargs,
            )
        case _:
            raise ValueError(f"Unsupported model '{config.model.name}'.")
