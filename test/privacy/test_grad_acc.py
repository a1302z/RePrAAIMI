# import os


# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import objax
import numpy as np
from jax import numpy as jnp
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from dptraining.transform.transform_pipeline import TransformPipeline
from dptraining.privacy import ClipAndAccumulateGrads

# from dptraining.utils.training_utils import create_train_op
from dptraining.models import make_model_from_config


def setup_fake_training(model=objax.nn.Sequential([objax.nn.Linear(10, 1)])):
    model_vars = model.vars()
    for key, mv in model_vars.items():
        model_vars[key].assign(jnp.zeros_like(mv))

    @objax.Function.with_vars(model_vars)
    def loss_fn(inpt, label):
        logit = model(inpt, training=True)
        return objax.functional.loss.mean_squared_error(logit, label).mean()

    grad_values = ClipAndAccumulateGrads(
        loss_fn,
        model_vars,
        1.0,
        gradient_accumulation_steps=1,
        batch_axis=(0, 0),
    )

    return model_vars, grad_values


def setup_fake_data(inpt_shape=(5, 10), otpt_shape=(5, 1)):
    data = np.random.randn(*inpt_shape)  # 5 samples with 10 values
    label = np.random.randn(*otpt_shape)
    return data, label


def test_grad_acc_step():
    model_vars, grad_values = setup_fake_training()
    data, label = setup_fake_data()

    clipped_grad, _ = grad_values.calc_per_sample_grads(data, label)
    assert (
        jnp.linalg.norm([jnp.linalg.norm(g) for g in clipped_grad]).item() <= 5.0
    ), "Clipping incorrect"

    for i in range(3):
        grad_values.accumulate_grad(
            [jnp.ones_like(mv) for mv in model_vars], [jnp.array(1.0)]
        )
        print([gv.value for gv in grad_values.accumulated_grads])
        assert all(
            [
                jnp.all(jnp.isclose(gv.value, i + 1)).item()
                for gv in grad_values.accumulated_grads
            ]
        )

    acc_grad = grad_values.get_accumulated_grads()
    assert all(
        [jnp.all(jnp.isclose(gv, 3.0)).item() for gv in acc_grad]
    ), "Gradient accumulation incorrect"


# def test_create_train_loop(): # this breaks other test cases bc of tracing errors
#     model_vars, grad_values = setup_fake_training()
#     opt = objax.optimizer.SGD(model_vars)
#     train_vars = model_vars + opt.vars() + grad_values.vars()
#     train_op_acc = create_train_op(
#         train_vars,
#         grad_values,
#         opt,
#         lambda x: x,
#         grad_accumulation=True,
#         noise=1.0,
#         effective_batch_size=10,
#     )
#     train_op_acc(*setup_fake_data(), 1.0, apply_norm_acc=False)
#     train_op_acc(*setup_fake_data(), 1.0, apply_norm_acc=True)


def test_grad_equality(utils):
    tf1 = TransformPipeline.from_dict_list(
        {
            "make_complex_both": None,
        }
    )
    tf1 = tf1.create_vectorized_image_transform()
    tf2 = TransformPipeline.from_dict_list(
        {
            "make_complex_both": None,
            "consecutive_augmentations": {
                "multiplicity": 4,
                "complex": True,
                "random_horizontal_flips": {"flip_prob": 0},
            },
        }
    )
    tf2 = tf2.create_vectorized_image_transform()
    config_dict = {
        "model": {
            "complex": True,
            "name": "cifar10model",
            "in_channels": 3,
            "num_classes": 1,
            "conv": "convws_nw",
            "activation": "conjmish",
            "normalization": "bn",
            "pooling": "avgpool",
            "extra_args": {
                "scale_norm": True,
            },
        }
    }
    config = utils.extend_base_config(config_dict)
    model = make_model_from_config(config.model)
    data, label = setup_fake_data((10, 3, 32, 32), (10, 1))
    _, gv = setup_fake_training(model)

    grads1, _ = gv.calc_per_sample_grads(
        tf1(data)[:, jnp.newaxis, ...], label[:, jnp.newaxis, ...]
    )
    grads2, _ = gv.calc_per_sample_grads(
        tf2(data), jnp.repeat(label[:, jnp.newaxis], 4, axis=1)
    )
    assert all([np.allclose(g1, g2, rtol=1e-5) for g1, g2 in zip(grads1, grads2)])
