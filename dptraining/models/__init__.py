import warnings
from dptraining.models.cifar10models import Cifar10ConvNet
from dptraining.models.resnet9 import ResNet9
from dptraining.models.activations import mish
from objax.zoo import resnet_v2
from objax import nn, functional


SUPPORTED_MODELS = ("cifar10model", "resnet18", "resnet9")
SUPPORTED_NORMALIZATION = ("bn", "gn")
SUPPORTED_ACTIVATION = ("relu", "selu", "leakyrelu", "mish")


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


def make_model_from_config(config):
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
            norm_func=make_normalization_from_config(config),
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
