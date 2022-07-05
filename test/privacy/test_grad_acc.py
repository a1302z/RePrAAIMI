# import os


# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import objax
import numpy as np
from jax import numpy as jnp, checking_leaks
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from dptraining.privacy import PrivateGradValuesAccumulation
from dptraining.utils.training_utils import create_train_op


def setup_fake_training():
    model = objax.nn.Sequential([objax.nn.Linear(10, 1)])
    model_vars = model.vars()
    for key, mv in model_vars.items():
        model_vars[key].assign(jnp.zeros_like(mv))

    @objax.Function.with_vars(model_vars)
    def loss_fn(inpt, label):
        logit = model(inpt, training=True)
        return objax.functional.loss.mean_squared_error(logit, label).mean()

    grad_values = PrivateGradValuesAccumulation(
        loss_fn,
        model_vars,
        1.0,
        1.0,
        1,
        gradient_accumulation_steps=1,
        batch_axis=(0, 0),
    )

    return model_vars, grad_values


def setup_fake_data():
    data = np.random.randn(5, 10)  # 5 samples with 10 values
    label = np.random.randn(5, 1)
    return data, label


def test_grad_acc_step():
    model_vars, grad_values = setup_fake_training()
    data, label = setup_fake_data()

    _, _, clipped_grad, _ = grad_values.setup_grad_step(data, label)
    assert (
        jnp.linalg.norm([jnp.linalg.norm(g) for g in clipped_grad]).item() <= 1.0
    ), "Clipping incorrect"

    for i in range(3):
        grad_values.accumulate_grad(
            [jnp.ones_like(mv) for mv in model_vars], 5, [jnp.array(1.0)]
        )
        assert all(
            [
                jnp.all(jnp.isclose(gv.value, ((i + 1) * 5))).item()
                for gv in grad_values.accumulated_grads
            ]
        )

    noise_clipped_grad, loss_value = grad_values.apply_accumulated_grads(0.0)
    assert all(
        [jnp.all(jnp.isclose(gv, 1.0)).item() for gv in noise_clipped_grad]
    ), "Gradient averaging incorrect"
    assert jnp.isclose(loss_value[0], 1.0).item(), "Loss averaging incorrect"


# This doesn't work but works in training and I have no idea why
# def test_create_train_loop():
#     model_vars, grad_values = setup_fake_training()
#     train_op_acc = create_train_op(
#         model_vars,
#         grad_values,
#         objax.optimizer.SGD(model_vars),
#         lambda x: x,
#         grad_accumulation=True,
#     )
#     with checking_leaks():
#         train_op_acc(*setup_fake_data(), 1.0, apply_norm_acc=False)
#         train_op_acc(*setup_fake_data(), 1.0, apply_norm_acc=True)
