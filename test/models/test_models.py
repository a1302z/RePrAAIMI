import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import objax
import pytest
import numpy as np
import sys
from pathlib import Path
from itertools import product
from objax import nn
from itertools import product
import pytest

sys.path.insert(0, str(Path.cwd()))

from dptraining.config.model import (
    RealActivation,
    RealConv,
    RealModelName,
    RealNormalization,
    RealPooling,
)
from dptraining.config.utils import get_allowed_names
from dptraining.config.model import ModelName

from dptraining.models import (
    make_model_from_config,
    ResNet9,
    Cifar10ConvNet,
)


def test_cifar10convnet():
    model = Cifar10ConvNet()
    model_vars = model.vars()

    predict_op = objax.Jit(lambda x: objax.functional.softmax(model(x)), model_vars)

    cifar_batch = np.random.randn(64, 3, 32, 32)

    predict_op(cifar_batch)


def test_cifar10convnet_wrong_shape():
    model = Cifar10ConvNet()
    model_vars = model.vars()

    predict_op = objax.Jit(lambda x: objax.functional.softmax(model(x)), model_vars)

    cifar_batch = np.random.randn(64, 32, 32, 3)

    with pytest.raises(AssertionError):
        predict_op(cifar_batch)


def test_make_cifar10model(utils):
    config_dict = {"model": {"name": "cifar10model", "num_classes": 10}}
    config = utils.extend_base_config(config_dict)
    m = make_model_from_config(config.model)
    random_input_data = np.random.randn(10, 3, 32, 32)
    m(random_input_data, training=False)


def test_resnet18(utils):
    config_dict = {
        "model": {
            "name": "resnet18",
            "conv": "conv",
            "activation": "selu",
            "normalization": "gn",
            "in_channels": 3,
            "num_classes": 256,
        }
    }
    config = utils.extend_base_config(config_dict)
    m = make_model_from_config(config.model)
    random_input_data = np.random.randn(2, 3, 224, 224)
    m(random_input_data, training=False)


def test_resnet9():
    norm_funcs = [nn.BatchNorm2D, nn.GroupNorm2D]
    scale_norms = [True, False]
    for nf, sn in product(norm_funcs, scale_norms):
        m = ResNet9(norm_cls=nf, scale_norm=sn)
        x = np.random.randn(2, 3, 32, 32)
        m(x, training=True)
        m(x, training=False)


def test_make_resnet9(utils):
    config_dict = {
        "model": {
            "name": "resnet9",
            "conv": "conv",
            "activation": "selu",
            "normalization": "gn",
            "in_channels": 12,
            "num_classes": 256,
            "pooling": "maxpool",
            "extra_args": {
                "num_groups": [4, 8, 16, 32],
                "channels": [8, 32, 64, 128],
            },
        }
    }
    config = utils.extend_base_config(config_dict)
    m = make_model_from_config(config.model)
    random_input_data = np.random.randn(2, 12, 224, 224)
    m(random_input_data, training=False)
    assert m.conv1[1].groups == 4
    assert m.res1[0][1].groups == 8
    assert m.conv3[1].groups == 16
    assert m.res2[0][1].groups == 32


def test_all_options(utils):
    fake_data = np.random.randn(6, 3, 39, 39)
    for model, conv, act, norm, pool in product(
        (
            ModelName.resnet9,
            ModelName.resnet18,
            ModelName.smoothnet,
            ModelName.wide_resnet,
        ),
        get_allowed_names(RealConv),
        get_allowed_names(RealActivation),
        get_allowed_names(RealNormalization),
        get_allowed_names(RealPooling),
    ):
        config_dict = {
            "model": {
                "name": model,
                "complex": False,
                "conv": conv,
                "in_channels": 3,
                "num_classes": 5,
                "activation": act,
                "normalization": norm,
                "pooling": pool,
            }
        }
        config = utils.extend_base_config(config_dict)
        if model == "cifar10model":
            with pytest.warns(UserWarning):
                model = make_model_from_config(config.model)
        else:
            model = make_model_from_config(config.model)
        pred = model(fake_data, training=True)
        assert pred.shape[1] == 5


def test_ensemble(utils):
    config_dict = {
        "model": {
            "name": "resnet9",
            "conv": "conv",
            "activation": "selu",
            "normalization": "gn",
            "in_channels": 12,
            "num_classes": 256,
            "pooling": "maxpool",
            "ensemble": 2,
            "extra_args": {
                "num_groups": [4, 8, 16, 32],
                "channels": [8, 32, 64, 128],
            },
        }
    }
    config = utils.extend_base_config(config_dict)
    model = make_model_from_config(config.model)
    assert len(model.ensemble) == 2
    random_input_data = np.random.randn(2, 12, 224, 224)
    model(random_input_data, training=False)
