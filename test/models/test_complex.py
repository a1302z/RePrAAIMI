import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import sys
import numpy as np
from pathlib import Path
from itertools import product

sys.path.insert(0, str(Path.cwd()))

from dptraining.models import make_complex_model_from_config
from dptraining.config.model import (
    ComplexModelName,
    ComplexConv,
    ComplexActivation,
    ComplexNormalization,
    ComplexPooling,
)
from dptraining.config.utils import get_allowed_names


def test_complex_model_options(utils):
    fake_data = np.random.randn(4, 3, 47, 47).astype(np.complex128)
    for model, conv, act, norm, pool in product(
        get_allowed_names(ComplexModelName),
        get_allowed_names(ComplexConv),
        get_allowed_names(ComplexActivation),
        get_allowed_names(ComplexNormalization),
        get_allowed_names(ComplexPooling),
    ):
        config_dict = {
            "model": {
                "complex": True,
                "name": model,
                "in_channels": 3,  # TODO: changing that to 2 breaks it
                "num_classes": 7,
                "conv": conv,
                "activation": act,
                "normalization": norm,
                "pooling": pool,
            }
        }
        config = utils.extend_base_config(config_dict)
        try:
            model = make_complex_model_from_config(config)
            pred = model(fake_data)
            assert pred.shape[1] == 7
        except Exception as e:
            raise type(e)(f"Error for conf: {config_dict}\n{str(e)}")
