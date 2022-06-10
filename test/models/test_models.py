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

from dptraining.models import (
    SUPPORTED_POOLING,
    make_model_from_config,
    ResNet9,
    Cifar10ConvNet,
    SUPPORTED_ACTIVATION,
    SUPPORTED_MODELS,
    SUPPORTED_NORMALIZATION,
    SUPPORTED_COMPLEX_POOLING,
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


def test_make_cifar10model():
    m = make_model_from_config({"model": {"name": "cifar10model", "num_classes": 10}})
    random_input_data = np.random.randn(10, 3, 32, 32)
    m(random_input_data, training=False)


def test_resnet18():
    m = make_model_from_config(
        {
            "model": {
                "name": "resnet18",
                "activation": "selu",
                "normalization": "gn",
                "in_channels": 3,
                "num_classes": 256,
            }
        }
    )
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


def test_make_resnet9():
    m = make_model_from_config(
        {
            "model": {
                "name": "resnet9",
                "activation": "selu",
                "normalization": "gn",
                "in_channels": 12,
                "num_classes": 256,
                "pooling": "maxpool",
            }
        }
    )
    random_input_data = np.random.randn(2, 12, 224, 224)
    m(random_input_data, training=False)


def test_all_options():
    fake_data = np.random.randn(6, 3, 39, 39)
    for model, act, norm, pool in product(
        SUPPORTED_MODELS,
        SUPPORTED_ACTIVATION,
        SUPPORTED_NORMALIZATION,
        SUPPORTED_POOLING,
    ):
        config = {
            "model": {
                "name": model,
                "in_channels": 3,
                "num_classes": 5,
                "activation": act,
                "normalization": norm,
                "pooling": pool,
            }
        }
        if model == "cifar10model":
            with pytest.warns(UserWarning):
                model = make_model_from_config(config)
        else:
            model = make_model_from_config(config)
        pred = model(fake_data, training=True)
        assert pred.shape[1] == 5
