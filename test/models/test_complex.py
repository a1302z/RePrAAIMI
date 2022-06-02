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


def test_complex_model_options():
    fake_data = np.random.randn(4, 2, 47, 47).astype(np.complex128)
    for conv, act, norm, pool, model in product(
        SUPPORTED_COMPLEX_CONV,
        SUPPORTED_COMPLEX_ACTIVATION,
        SUPPORTED_COMPLEX_NORMALIZATION,
        SUPPORTED_COMPLEX_POOLING,
        SUPPORTED_COMPLEX_MODELS,
    ):
        try:
            conf = {
                "model": {
                    "name": model,
                    "in_channels": 2,
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
