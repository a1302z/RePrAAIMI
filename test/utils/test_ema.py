import numpy as np
import objax
from jax import numpy as jn
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from dptraining.utils import ExponentialMovingAverage
from objax.nn import Linear

model = Linear(10, 1)


def test_init_ema():
    ExponentialMovingAverage(model.vars(), 0.9)


def test_update_ema():
    decay = 0.9
    model_vars = model.vars()
    for key, mv in model_vars.items():
        model_vars[key].assign(jn.ones_like(mv))
    ema = ExponentialMovingAverage(model.vars(), decay, use_num_updates=False)
    for v in model.vars().values():
        v.assign(jn.array(np.zeros_like(v)))
    ema.update()
    ema.copy_to()
    assert all([jn.all(jn.isclose(v, 0.9)).item() for v in model.vars().values()])


def test_jit_ema():
    decay = 0.01
    model_vars = model.vars()
    model_var_refs = {k: objax.TrainRef(v) for k, v in model_vars.items()}
    for key, mv in model_var_refs.items():
        model_var_refs[key].assign(jn.ones_like(mv))
    ema = ExponentialMovingAverage(model.vars(), decay, use_num_updates=False)

    @objax.Function.with_vars(model_vars)
    def increase_model_weights():
        for mv in model_vars.values():
            ref = objax.TrainRef(mv)
            ref.value += jn.ones_like(mv)

    increase_model_weights = objax.Jit(increase_model_weights)

    for _ in range(3):
        increase_model_weights()
        ema.update()
        ema.copy_to()
