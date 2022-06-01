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


def test_make_momentum():
    make_optim_from_config(
        {"optim": {"name": "momentum", "momentum": 0.5, "nesterov": True}},
        fake_model.vars(),
    )


def test_make_adam():
    make_optim_from_config(
        {"optim": {"name": "adam", "beta2": 0.7}},
        fake_model.vars(),
    )


def test_make_nadam():
    make_optim_from_config(
        {"optim": {"name": "nadam", "beta2": 0.7, "momentum_decay": 0.1}},
        fake_model.vars(),
    )
