import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from dptraining.models import make_real_activation_from_config


def put_data_in_it(f):
    data = np.random.randn(10, 3, 64, 64)
    f(data)


def test_relu(utils):
    config = utils.extend_base_config({"model": {"activation": "relu"}})
    f = make_real_activation_from_config(config)
    put_data_in_it(f)


def test_selu(utils):
    config = utils.extend_base_config({"model": {"activation": "selu"}})
    f = make_real_activation_from_config(config)
    put_data_in_it(f)


def test_leakyrelu(utils):
    config = utils.extend_base_config({"model": {"activation": "leakyrelu"}})
    f = make_real_activation_from_config(config)
    put_data_in_it(f)


def test_mish(utils):
    config = utils.extend_base_config({"model": {"activation": "mish"}})
    f = make_real_activation_from_config(config)
    put_data_in_it(f)
