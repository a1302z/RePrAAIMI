import objax
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from dptraining.utils.loss import CSELogitsSparse


class MiniModel(objax.nn.Sequential):
    def __init__(self, nclass=10):
        ops = [
            objax.nn.Linear(128, nclass, use_bias=True),
        ]
        super().__init__(ops)


def test_cselogitssparse_loss():
    model = MiniModel()
    model_vars = model.vars()
    loss_fn = CSELogitsSparse.create_loss_fn(model_vars, model)
    mini_data = np.random.randn(3, 2, 8, 8).reshape(-1, 128)
    mini_label = np.random.randint(0, 10, size=(3,))
    loss = loss_fn(mini_data, mini_label)
