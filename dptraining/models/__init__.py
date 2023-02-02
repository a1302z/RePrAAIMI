import inspect
import warnings
from copy import deepcopy
from functools import partial
from typing import Callable

from objax import functional, nn, VarCollection, BaseVar
from objax.util import class_name
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
    UpConv,
    ModelName,
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
    ComplexConv2DTranspose,
    ComplexWSConv2DNoWhitenTranspose,
    ComplexWSConv2DTranspose,
)
from dptraining.models.complex.normalization import (  # pylint:disable=duplicate-code
    ComplexBatchNorm2D,
    ComplexGroupNorm2DWhitening,
)
from dptraining.models.complex.pooling import ConjugatePool2D, SeparablePool2D
from dptraining.models.ensemble import Ensemble
from dptraining.models.layers import (
    ConvCentering2D,
    ConvWS2D,
    ConvWSTranspose2D,
    ConvCenteringTranspose2D,
    Conv3D,
    ConvTranspose3D,
    ConvWS3D,
    ConvCentering3D,
    ConvWSTranspose3D,
    ConvCenteringTranspose3D,
    BatchNorm3D,
    GroupNorm3D,
    max_pool_3d,
    average_pool_3d,
)
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
        not in [  # TODO find cleaner way
            "in_channels",
            "num_classes",
            "conv",
            "activation",
            "pooling",
            "normalization",
            "name",
            "ensemble",
            "complex",
            "dim3",
            "upconv",
        ]
    }
    print(f"Additional kwargs for {func}: {kwargs}")
    if len(ignoring) > 0:
        warnings.warn(f" -> Ignoring: {ignoring}")
    return kwargs


def make_real_normalization2d_from_config(config: ModelConfig) -> Callable:
    match config.normalization.value:
        case RealNormalization.bn.value:
            return nn.BatchNorm2D
        case RealNormalization.gn.value:
            return nn.GroupNorm2D
        case None:
            raise ValueError("No normalization layer specified (required)")
        case _:
            raise ValueError(
                f"Unsupported normalization layer '{config.normalization}'"
            )


def make_real_normalization3d_from_config(config: ModelConfig) -> Callable:
    match config.normalization.value:
        case RealNormalization.bn.value:
            return BatchNorm3D
        case RealNormalization.gn.value:
            return GroupNorm3D
        case None:
            raise ValueError("No normalization layer specified (required)")
        case _:
            raise ValueError(
                f"Unsupported normalization layer '{config.normalization}'"
            )


def make_real_normalization_from_config(config: ModelConfig) -> Callable:
    if config.dim3:
        return make_real_normalization3d_from_config(config)
    else:
        return make_real_normalization2d_from_config(config)


def make_complex_normalization2d_from_config(config: ModelConfig) -> Callable:
    if config.dim3:
        raise ValueError("Complex 3D normalization not yet supported")
    match config.normalization.value:
        case ComplexNormalization.bn.value:
            return ComplexBatchNorm2D
        case ComplexNormalization.gnw.value:
            return ComplexGroupNorm2DWhitening
        case None:
            raise ValueError("No normalization layer specified (required)")
        case _:
            raise ValueError(
                f"Unsupported normalization layer '{config.normalization}'"
            )


def make_real_conv2d_from_config(config: ModelConfig) -> Callable:
    match config.conv.value:
        case RealConv.conv.value:
            return nn.Conv2D
        case RealConv.convws.value:
            return ConvWS2D
        case RealConv.convws_nw.value:
            return ConvCentering2D
        case _:
            raise ValueError(f"Unsupported convolutional layer '{config.conv}'")


def make_real_3d_conv_from_config(config: ModelConfig) -> Callable:
    match config.conv.value:
        case RealConv.conv.value:
            return Conv3D
        case RealConv.convws.value:
            return ConvWS3D
        case RealConv.convws_nw.value:
            return ConvCentering3D
        case other:
            raise ValueError(f"Unsupported 3D-convolutional layer '{other}'")


def make_real_conv_from_config(config: ModelConfig) -> Callable:
    if config.dim3:
        return make_real_3d_conv_from_config(config)
    else:
        return make_real_conv2d_from_config(config)


def make_complex_conv_from_config(config: ModelConfig) -> Callable:
    if config.dim3:
        raise ValueError("Complex 3D conv not yet supported")
    match config.conv.value:
        case ComplexConv.conv.value:
            return ComplexConv2D
        case ComplexConv.convws.value:
            return ComplexWSConv2D
        case ComplexConv.convws_nw.value:
            return ComplexWSConv2DNoWhiten
        case _:
            raise ValueError(f"Unsupported convolutional layer '{config.conv}'")


def make_real_convtranspose2d_from_config(config: ModelConfig) -> Callable:
    match config.upconv:
        case UpConv.conv:
            return nn.ConvTranspose2D
        case UpConv.convws:
            return ConvWSTranspose2D
        case UpConv.convws_nw:
            return ConvCenteringTranspose2D
        case other:
            raise ValueError(f"Unsupported up-convolutional layer '{other}'")


def make_real_convtranspose3d_from_config(config: ModelConfig) -> Callable:
    match config.conv.value:
        case UpConv.conv.value:
            return ConvTranspose3D
        case UpConv.convws.value:
            return ConvWSTranspose3D
        case UpConv.convws_nw.value:
            return ConvCenteringTranspose3D
        case other:
            raise ValueError(f"Unsupported 3D-convolutional layer '{other}'")


def make_real_convtranspose_from_config(config: ModelConfig) -> Callable:
    if config.dim3:
        return make_real_convtranspose3d_from_config(config)
    else:
        return make_real_convtranspose2d_from_config(config)


def make_complex_transposeconv2d_from_config(config: ModelConfig) -> Callable:
    if config.dim3:
        raise ValueError("Complex 3D conv transpose not yet supported")
    match config.upconv:
        case UpConv.conv:
            return ComplexConv2DTranspose
        case UpConv.convws:
            return ComplexWSConv2DTranspose
        case UpConv.convws_nw:
            return ComplexWSConv2DNoWhitenTranspose
        case other:
            raise ValueError(f"Unsupported up-convolutional layer '{other}'")


def make_real_activation_from_config(config: ModelConfig) -> Callable:
    match config.activation.value:
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
            raise ValueError(f"Unsupported activation layer '{config.activation}'")


def make_complex_activation_from_config(
    config: ModelConfig, init_layers: bool
) -> Callable:
    match config.activation.value:
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
            raise ValueError(f"Unsupported activation layer '{config.activation}'")
    if init_layers:
        return act()
    return act


def make_real_pooling2d_from_config(config: ModelConfig) -> Callable:
    match config.pooling.value:
        case RealPooling.maxpool.value:
            return functional.max_pool_2d
        case RealPooling.avgpool.value:
            return functional.average_pool_2d
        case _:
            raise ValueError(f"Unsupported pooling layer '{config.pooling}'")


def make_real_pooling3d_from_config(config: ModelConfig) -> Callable:
    match config.pooling.value:
        case RealPooling.maxpool.value:
            return max_pool_3d
        case RealPooling.avgpool.value:
            return average_pool_3d
        case _:
            raise ValueError(f"Unsupported pooling layer '{config.pooling}'")


def make_real_pooling_from_config(config: ModelConfig) -> Callable:
    if config.dim3:
        return make_real_pooling3d_from_config(config)
    else:
        return make_real_pooling2d_from_config(config)


def make_complex_pooling_from_config(
    config: ModelConfig, init_layers: bool, **kwargs
) -> Callable:
    if config.dim3:
        raise ValueError("Complex 3D pooling not yet supported")
    match config.pooling.value:
        case ComplexPooling.conjmaxpool.value:
            layer = ConjugatePool2D
        case ComplexPooling.sepmaxpool.value:
            layer = SeparablePool2D
        case ComplexPooling.conjavgpool.value:
            layer = partial(ConjugatePool2D, pool_func=functional.average_pool_2d)
        case ComplexPooling.sepavgpool.value:
            layer = partial(SeparablePool2D, pool_func=functional.average_pool_2d)
        case ComplexPooling.avgpool.value:
            if len(kwargs) == 0:
                return functional.average_pool_2d
            return partial(functional.average_pool_2d, **kwargs)
        case _:
            raise ValueError(f"Unsupported pooling layer '{config.pooling}'")
    if init_layers:
        return layer(**kwargs)
    else:
        return layer


def make_model_from_config(config: ModelConfig) -> Callable:
    num_replicas = config.ensemble if config.ensemble else 1
    if num_replicas > 1:
        ensemble = [
            make_single_model_instance_from_config(config) for _ in range(num_replicas)
        ]
        return Ensemble(ensemble)
    else:
        return make_single_model_instance_from_config(config)


def make_single_model_instance_from_config(config: ModelConfig) -> Callable:
    if config.complex:
        if config.dim3:
            raise ValueError("Complex 3D models not yet supported")
        if config.name.value in get_allowed_values(ComplexModelName):
            model = make_complex_model_from_config(config)
            return model
        elif config.name.value in get_allowed_values(RealModelName):
            fake_config = deepcopy(config)
            fake_config.conv = Conv.conv
            fake_config.normalization = Normalization.gn
            fake_config.activation = Activation.relu
            fake_config.pooling = Pooling.avgpool
            model = make_real_model_from_config(fake_config)
            converter = ComplexModelConverter(
                new_conv_class=make_complex_conv_from_config(config),
                new_norm_class=make_complex_normalization2d_from_config(config),
                new_linear_class=ComplexLinear,
                new_activation=make_complex_activation_from_config(
                    config, init_layers=False
                ),
                new_pooling=make_complex_pooling_from_config(config, init_layers=False),
            )
            return nn.Sequential([converter(model), ComplexToReal()])
        else:
            raise ValueError(f"{config.name} unknown")
    else:
        return make_real_model_from_config(config)


def make_real_model_from_config(config: ModelConfig) -> Callable:
    match config.name.value:
        case RealModelName.cifar10model.value:
            if config.activation is not None:
                warnings.warn("No choice of activations supported for cifar 10 model")
            if config.normalization is not None:
                warnings.warn("No choice of normalization supported for cifar 10 model")
            return Cifar10ConvNet(nclass=config.num_classes)
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
                config,
            )
            return resnet_v2.ResNet18(
                in_channels=config.in_channels,
                num_classes=config.num_classes,
                conv_layer=make_real_conv_from_config(config),
                normalization_fn=make_real_normalization_from_config(config),
                activation_fn=make_real_activation_from_config(config),
                **kwargs,
            )
        case RealModelName.wide_resnet.value:
            kwargs = get_kwargs(
                wide_resnet.WideResNet,
                ["nin", "nclass", "conv_layer", "bn", "act"],
                config,
            )
            return wide_resnet.WideResNet(
                nin=config.in_channels,
                nclass=config.num_classes,
                conv_layer=make_real_conv_from_config(config),
                bn=make_real_normalization_from_config(config),
                act=make_real_activation_from_config(config),
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
            kwargs = get_kwargs(ResNet9, already_defined, config)
            return ResNet9(
                in_channels=config.in_channels,
                num_classes=config.num_classes,
                conv_cls=make_real_conv_from_config(config),
                norm_cls=make_real_normalization_from_config(config),
                act_func=make_real_activation_from_config(config),
                pool_func=partial(make_real_pooling_from_config(config), size=2),
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
            kwargs = get_kwargs(get_smoothnet, already_defined, config)
            return get_smoothnet(
                in_channels=config.in_channels,
                num_classes=config.num_classes,
                conv_cls=make_real_conv_from_config(config),
                norm_cls=make_real_normalization_from_config(config),
                act_func=make_real_activation_from_config(config),
                pool_func=partial(
                    make_real_pooling_from_config(config),
                    size=3,
                    strides=1,
                    padding=1,
                ),
                **kwargs,
            )
        case RealModelName.unet.value:
            already_defined = ("in_channels", "upconv")
            kwargs = get_kwargs(Unet, already_defined, config)
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
                in_channels=config.in_channels,
                actv=make_real_activation_from_config(config),
                conv_layer=make_real_conv_from_config(config),
                upconv_layer=make_real_convtranspose_from_config(config),
                pool_fn=partial(
                    make_real_pooling_from_config(config), size=2, strides=2, padding=0
                ),
                norm_layer=make_real_normalization_from_config(config),
                dim_mode=3 if config.dim3 else 2,
                **kwargs,
            )
        case _:
            raise ValueError(f"Unsupported model '{config.name}'")


def make_complex_model_from_config(config: ModelConfig) -> Callable:
    match config.name.value:
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
            kwargs = get_kwargs(ResNet9, already_defined, config)
            return ResNet9(
                config.in_channels,
                config.num_classes,
                conv_cls=make_complex_conv_from_config(config),
                norm_cls=make_complex_normalization2d_from_config(config),
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
            kwargs = get_kwargs(get_smoothnet, already_defined, config)
            return get_smoothnet(
                in_channels=config.in_channels,
                num_classes=config.num_classes,
                conv_cls=make_complex_conv_from_config(config),
                norm_cls=make_complex_normalization2d_from_config(config),
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
            kwargs = get_kwargs(Unet, already_defined, config)
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
                in_channels=config.in_channels,
                actv=make_complex_activation_from_config(config, init_layers=True),
                conv_layer=make_complex_conv_from_config(config),
                upconv_layer=make_complex_transposeconv2d_from_config(config),
                pool_fn=make_complex_pooling_from_config(
                    config, init_layers=True, size=2, strides=2, padding=0
                ),
                **kwargs,
            )
        case _:
            raise ValueError(f"Unsupported model '{config.name}'.")


def modify_architecture_from_pretrained_model(config: Config, model: Callable):
    new_model_vars: dict[str, BaseVar] = {}
    layer_config: ModelConfig = deepcopy(config.model)
    layer_config.conv = Conv.conv
    if config.model.in_channels != config.model.pretrained_model_changes.in_channels:
        if config.model.complex:
            new_layer: Callable = make_complex_conv_from_config(layer_config)
        else:
            new_layer: Callable = make_real_conv_from_config(layer_config)
        new_layer = new_layer(  # pylint:disable=not-callable
            config.model.pretrained_model_changes.in_channels,
            config.model.in_channels,
            k=1,
        )
        match config.model.name:
            case ModelName.resnet9:
                model.conv1[0] = nn.Sequential([new_layer, model.conv1[0]])
                add_name = "(ResNet9).conv1(Sequential)[0]"
            case ModelName.smoothnet:
                model.stage_zero = nn.Sequential([new_layer, model.stage_zero])
                add_name = "(SmoothNet).conv1"
            case _:  # e.g. resnet18, wide_resnet
                model[0] = nn.Sequential([new_layer, model[0]])
                add_name = f"({class_name(model).split('.')[-1]})[0]"
        new_model_vars = {
            f"{add_name}(Sequential)[0]{layer_name}": layer_param
            for layer_name, layer_param in new_layer.vars().items()
        }
    if config.model.num_classes != config.model.pretrained_model_changes.num_classes:
        if config.model.complex:
            new_layer = ComplexLinear
        else:
            new_layer = nn.Linear
        match config.model.name:
            case ModelName.resnet9:
                new_layer = new_layer(
                    model.classifier.linr.shape[0]
                    if config.model.complex
                    else model.classifier.w.shape[0],
                    config.model.pretrained_model_changes.num_classes,
                )
                model.classifier = new_layer
                add_name = "(ResNet9).classifier"
            case ModelName.smoothnet:
                new_layer = new_layer(
                    128, config.model.pretrained_model_changes.num_classes
                )
                model.fc3 = new_layer
                add_name = "(SmoothNet).fc3"
            case ModelName.wide_resnet:
                new_layer = new_layer(
                    model[-1].shape[0]
                    if config.model.complex
                    else model[-1].w.shape[0],
                    config.model.pretrained_model_changes.num_classes,
                )
                model[-1] = new_layer
                add_name = f"(WideResNet)[{len(model)-1}]"
            case other:
                raise ValueError(
                    f"Change of number of out classes not yet supported for {other}"
                )

        for layer_name, layer_param in new_layer.vars().items():
            new_model_vars[f"{add_name}{layer_name}"] = layer_param
    new_layer_vars = VarCollection(**new_model_vars)
    # total_vars =
    if config.model.pretrained_model_changes.only_finetune:
        return new_layer_vars, new_layer_vars
    else:
        return model.vars(), new_layer_vars
