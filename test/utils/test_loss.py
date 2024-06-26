import os
from dptraining.config import LossReduction
from dptraining.config.utils import get_allowed_values

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import objax
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from dptraining.utils import make_loss_from_config


class MiniModel(objax.nn.Sequential):
    def __init__(self, nclass=10):
        ops = [
            objax.nn.Linear(128, nclass, use_bias=True),
        ]
        super().__init__(ops)


def test_cselogitssparse_loss(utils):
    model = MiniModel()
    model_vars = model.vars()
    for reduction in get_allowed_values(LossReduction):
        config_dict = {
            "loss": {"type": "cse", "reduction": reduction, "binary_loss": False},
            "hyperparams": {},
        }
        config = utils.extend_base_config(config_dict)
        loss_class = make_loss_from_config(config)
        loss_fn = loss_class.create_train_loss_fn(model_vars, model)
        mini_data = np.random.randn(3, 2, 8, 8).reshape(-1, 128)
        mini_label = np.random.randint(0, 10, size=(3,))
        loss_fn(mini_data, mini_label)


def test_combinedl2regularization(utils):
    model = MiniModel()
    model_vars = model.vars()
    for reduction in get_allowed_values(LossReduction):
        config_dict = {
            "loss": {"type": "cse", "reduction": reduction, "binary_loss": False},
            "hyperparams": {"l2regularization": 0.5},
        }
        config = utils.extend_base_config(config_dict)
        loss_class = make_loss_from_config(config)
        loss_fn = loss_class.create_train_loss_fn(model_vars, model)
        mini_data = np.random.randn(3, 2, 8, 8).reshape(-1, 128)
        mini_label = np.random.randint(0, 10, size=(3,))
        loss_fn(mini_data, mini_label)
