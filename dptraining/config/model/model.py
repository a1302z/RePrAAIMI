from dataclasses import dataclass
from enum import Enum
from omegaconf import MISSING
from typing import Any, Optional

# pylint:disable=invalid-name


class ModelName(Enum):
    cifar10model = 1
    resnet18 = 2
    resnet9 = 3
    smoothnet = 4
    wide_resnet = 5
    unet = 6


# We can't use union types yet, unfortunately:
# https://github.com/omry/omegaconf/issues/144


class RealModelName(
    Enum
):  # TODO: split also by classification vs segmentation/recon models
    cifar10model = 1
    resnet18 = 2
    resnet9 = 3
    smoothnet = 4
    wide_resnet = 5
    unet = 6


class ComplexModelName(Enum):
    resnet9 = 3
    smoothnet = 4
    unet = 6


class Normalization(Enum):
    bn = 1
    gn = 2
    gnw = 3


class RealNormalization(Enum):
    bn = 1
    gn = 2


class ComplexNormalization(Enum):
    bn = 1
    gnw = 3


class Conv(Enum):
    conv = 1
    convws = 2
    convws_nw = 3


class RealConv(Enum):
    conv = 1
    convws = 2
    convws_nw = 3


class ComplexConv(Enum):
    conv = 1
    convws = 2
    convws_nw = 3


class UpConv(Enum):
    conv = 1
    convws = 2
    convws_nw = 3


class Activation(Enum):
    relu = 1
    selu = 2
    leakyrelu = 3
    mish = 4
    sepmish = 5
    conjmish = 6
    igaussian = 7
    cardioid = 8


class RealActivation(Enum):
    relu = 1
    selu = 2
    leakyrelu = 3
    mish = 4


class ComplexActivation(Enum):
    mish = 4
    sepmish = 5
    conjmish = 6
    igaussian = 7
    cardioid = 8


class Pooling(Enum):
    maxpool = 1
    avgpool = 2
    conjmaxpool = 3
    sepmaxpool = 4
    conjavgpool = 5
    sepavgpool = 6


class RealPooling(Enum):
    maxpool = 1
    avgpool = 2


class ComplexPooling(Enum):
    avgpool = 2
    conjmaxpool = 3
    sepmaxpool = 4
    conjavgpool = 5
    sepavgpool = 6


@dataclass
class PretrainChanges:
    in_channels: int = MISSING
    num_classes: int = MISSING
    only_finetune: bool = False


@dataclass
class ModelConfig:
    name: ModelName = MISSING
    ensemble: Optional[int] = None
    complex: bool = False
    dim3: bool = False
    in_channels: int = MISSING
    num_classes: int = MISSING
    conv: Conv = MISSING
    activation: Optional[Activation] = None
    normalization: Optional[Normalization] = None
    pooling: Pooling = MISSING
    pretrained_model_changes: Optional[PretrainChanges] = None
    extra_args: Optional[dict[str, Any]] = None
    upconv: Optional[UpConv] = None
