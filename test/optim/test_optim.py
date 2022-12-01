import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np

from jax import numpy as jnp
from objax import nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from dptraining.optim import make_optim_from_config, NAdam

fake_model = nn.Linear(100, 10)
fake_grads = [jnp.ones_like(x) * np.random.randn(*x.shape) for x in fake_model.vars()]


def test_nadam():
    opt = NAdam(fake_model.vars(), 0.5, 0.8, weight_decay=1e-4, momentum_decay=1e-3)
    opt(lr=1e-3, grads=fake_grads)


def test_make_momentum(utils):
    config_dict = {
        "optim": {"name": "momentum", "args": {"momentum": 0.5, "nesterov": True}},
    }
    config = utils.extend_base_config(config_dict)
    make_optim_from_config(config, fake_model.vars())


def test_make_adam(utils):
    config_dict = {"optim": {"name": "adam", "args": {"beta2": 0.7}}}
    config = utils.extend_base_config(config_dict)
    make_optim_from_config(config, fake_model.vars())


def test_make_nadam(utils):
    config_dict = {
        "optim": {"name": "nadam", "args": {"beta2": 0.7, "momentum_decay": 0.1}},
    }
    config = utils.extend_base_config(config_dict)
    make_optim_from_config(config, fake_model.vars())
