import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from dptraining.models import make_activation_from_config


def put_data_in_it(f):
    data = np.random.randn(10, 3, 64, 64)
    f(data)


def test_relu():
    f = make_activation_from_config({"model": {"activation": "relu"}})
    put_data_in_it(f)


def test_selu():
    f = make_activation_from_config({"model": {"activation": "selu"}})
    put_data_in_it(f)


def test_leakyrelu():
    f = make_activation_from_config({"model": {"activation": "leakyrelu"}})
    put_data_in_it(f)


def test_mish():
    f = make_activation_from_config({"model": {"activation": "mish"}})
    put_data_in_it(f)
