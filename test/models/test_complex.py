import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import sys
import numpy as np
from pathlib import Path
from itertools import product

sys.path.insert(0, str(Path.cwd()))

from dptraining.models import (
    make_complex_model_from_config,
    SUPPORTED_COMPLEX_ACTIVATION,
    SUPPORTED_COMPLEX_POOLING,
    SUPPORTED_COMPLEX_NORMALIZATION,
    SUPPORTED_COMPLEX_MODELS,
    SUPPORTED_COMPLEX_CONV,
)


def test_complex_model_options_classification():
    fake_data = np.random.randn(4, 3, 47, 47).astype(np.complex128)
    for conv, act, norm, pool, model in product(
        SUPPORTED_COMPLEX_CONV,
        SUPPORTED_COMPLEX_ACTIVATION,
        SUPPORTED_COMPLEX_NORMALIZATION,
        SUPPORTED_COMPLEX_POOLING,
        ("resnet9", "smoothnet"),
    ):
        try:
            conf = {
                "model": {
                    "name": model,
                    "in_channels": 3,  # TODO: changing that to 2 breaks it
                    "num_classes": 7,
                    "conv": conv,
                    "activation": act,
                    "normalization": norm,
                    "pooling": pool,
                }
            }
            model = make_complex_model_from_config(conf)
            pred = model(fake_data)
            assert pred.shape[1] == 7
        except Exception as e:
            raise type(e)(f"Error for conf: {conf}\n{str(e)}")


def test_complex_model_options_reconstruction():
    fake_data = np.random.randn(4, 3, 64, 64).astype(np.complex128)
    for act, model in product(
        SUPPORTED_COMPLEX_ACTIVATION,
        ("unet",),
    ):
        try:
            conf = {
                "model": {
                    "name": model,
                    "in_channels": 3,
                    "out_channels": 3,
                    "channels": 16,
                    "activation": act,
                }
            }
            model = make_complex_model_from_config(conf)
            pred = model(fake_data)
            assert all([pred.shape[i] == fake_data.shape[i] for i in range(4)])
        except Exception as e:
            raise type(e)(f"Error for conf: {conf}\n{str(e)}")
