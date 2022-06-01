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
)


def test_complex_model_options():
    fake_data = np.random.randn(4, 2, 47, 47).astype(np.complex128)
    for act, norm, pool, model in product(
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
