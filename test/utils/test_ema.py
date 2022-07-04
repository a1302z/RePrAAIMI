import numpy as np
from jax import numpy as jn
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from dptraining.utils import ExponentialMovingAverage
from dptraining.models import Cifar10ConvNet

model = Cifar10ConvNet()


def test_init_ema():
    ema = ExponentialMovingAverage(model.vars(), 0.9)


def test_update_ema():
    decay = 0.9
    ema = ExponentialMovingAverage(model.vars(), decay)
    original_weight_norms = [np.linalg.norm(v) for v in model.vars().values()]
    for k, v in model.vars().items():
        v.assign(jn.array(np.zeros_like(v)))
    ema.update()
    ema.copy_to(model.vars())
    assert all(
        [
            (1.0 - decay) * own <= np.linalg.norm(v) <= own
            for v, own in zip(model.vars().values(), original_weight_norms)
        ]
    )
