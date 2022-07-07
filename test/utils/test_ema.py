import numpy as np
from jax import numpy as jn
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from dptraining.utils import ExponentialMovingAverage
from objax.nn import Linear

model = Linear(10, 1)


def test_init_ema():
    ema = ExponentialMovingAverage(model.vars(), 0.9)


def test_update_ema():
    decay = 0.9
    model_vars = model.vars()
    for key, mv in model_vars.items():
        model_vars[key].assign(jn.ones_like(mv))
    ema = ExponentialMovingAverage(model.vars(), decay, use_num_updates=False)
    for k, v in model.vars().items():
        v.assign(jn.array(np.zeros_like(v)))
    ema.update(model.vars())
    ema.copy_to(model.vars())
    assert all([jn.all(jn.isclose(v, 0.9)).item() for v in model.vars().values()])
