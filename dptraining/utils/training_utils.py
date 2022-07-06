import wandb
import time
import contextlib

import numpy as np

import objax
import sys


from jax import numpy as jn
from jax.lax import rsqrt
from pathlib import Path
from sklearn import metrics
from tqdm import tqdm

sys.path.insert(0, str(Path.cwd()))

from dptraining.privacy import PrivateGradValuesAccumulation

# from dptraining.privacy import ComplexPrivateGradValues


def create_train_op(  # pylint:disable=too-many-arguments
    train_vars, loss_gv, opt, augment_op, grad_accumulation: bool, parallel=False,
):
    if grad_accumulation:
        assert isinstance(loss_gv, PrivateGradValuesAccumulation)

        @objax.Function.with_vars(train_vars)
        def calc_grads(image_batch, label_batch):
            image_batch = augment_op(image_batch)
            batch, stddev, clipped_grad, loss_value = loss_gv.setup_grad_step(
                image_batch, label_batch
            )
            loss_gv.accumulate_grad(clipped_grad, batch, loss_value)
            return stddev

        @objax.Function.with_vars(train_vars)
        def apply_grads(stddev, learning_rate):
            grads, loss = loss_gv.apply_accumulated_grads(stddev)
            if parallel:
                grads = objax.functional.parallel.pmean(grads)
                loss = objax.functional.parallel.pmean(loss)
            opt(learning_rate, grads)
            return loss, grads

        if parallel:
            calc_grads = objax.Parallel(
                calc_grads, reduce=lambda x: x[0], vc=train_vars,
            )
            apply_grads = objax.Parallel(apply_grads, reduce=np.mean, vc=train_vars,)
        else:
            # pass
            calc_grads = objax.Jit(calc_grads)
            apply_grads = objax.Jit(apply_grads)

        # @objax.Function.with_vars(train_vars)
        def train_op(  # pylint:disable=inconsistent-return-statements
            image_batch, label_batch, learning_rate: float, apply_norm_acc: bool
        ):
            stddev = calc_grads(image_batch, label_batch)
            if apply_norm_acc:
                return apply_grads(stddev, learning_rate)

    else:

        @objax.Function.with_vars(train_vars)
        def train_op(
            image_batch,
            label_batch,
            learning_rate,
            # apply_norm_acc: bool,  # pylint:disable=unused-argument
        ):
            image_batch = augment_op(image_batch)
            grads, loss = loss_gv(image_batch, label_batch)
            if parallel:
                grads = objax.functional.parallel.pmean(grads)
                loss = objax.functional.parallel.pmean(loss)
            opt(learning_rate, grads)
            # del apply_norm_acc
            return loss, grads

        if parallel:
            train_op = objax.Parallel(train_op, reduce=np.mean, vc=train_vars)
        else:
            train_op = objax.Jit(train_op, static_argnums=(3,))

    return train_op


def create_loss_gradient(config, model, model_vars, loss_fn, sigma):
    if config["DP"]["disable_dp"]:
        loss_gv = objax.GradValues(loss_fn, model.vars())
        # elif "complex" in config["model"] and config["model"]["complex"]:
        # loss_gv = ComplexPrivateGradValues(
        #     loss_fn,
        #     model_vars,
        #     sigma,
        #     config["DP"]["max_per_sample_grad_norm"],
        #     microbatch=1,
        #     batch_axis=(0, 0),
        #     use_norm_accumulation=config["DP"]["norm_acc"],
        # )
    else:
        loss_gv = PrivateGradValuesAccumulation(
            loss_fn,
            model_vars,
            sigma,
            config["DP"]["max_per_sample_grad_norm"],
            microbatch=1,
            batch_axis=(0, 0),
            use_norm_accumulation=config["DP"]["norm_acc"],
            gradient_accumulation_steps=config["DP"]["grad_acc_steps"]
            if "grad_acc_steps" in config["DP"]
            else 1,
            noise_scaling_factor=rsqrt(2.0)
            if "complex" in config["model"] and config["model"]["complex"]
            else 1.0,
        )

    return loss_gv


def train(  # pylint:disable=too-many-arguments,duplicate-code
    config,
    train_loader,
    train_op,
    learning_rate,
    train_vars,
    parallel,
    grad_acc: int,
    model_vars=None,
    ema=None,
):
    start_time = time.time()
    max_batches = (
        config["hyperparams"]["overfit"]
        if "overfit" in config["hyperparams"]
        else len(train_loader) + 1
    )
    for i, (img, label) in tqdm(
        enumerate(train_loader),  # pylint:disable=loop-invariant-statement
        total=len(train_loader),
        desc="Training",
        leave=False,
    ):
        with (train_vars).replicate() if parallel else contextlib.suppress():
            add_args = {}
            if grad_acc > 1:
                add_args["apply_norm_acc"] = (i + 1) % grad_acc == 0
            train_result = train_op(img, label, np.array(learning_rate), **add_args)
            if train_result is not None:
                train_loss, grads = train_result
        if ema is not None and train_result is not None:
            with model_vars.replicate() if parallel else contextlib.suppress():
                ema.update()
                ema.copy_to(model_vars)
        if (
            config["general"]["log_wandb"] and train_result is not None
        ):  # pylint:disable=loop-invariant-statement
            wandb.log(
                {
                    "train_loss": train_loss[0].item(),
                    "total_grad_norm": jn.linalg.norm(
                        [jn.linalg.norm(g) for g in grads]
                    ).item(),
                },
                commit=False,
            )
            if ema is not None:
                wandb.log(
                    {
                        k: (jn.abs(v) if jn.iscomplexobj(v) else v)
                        for k, v in ema.shadow_params.items()
                    },
                    commit=False,
                )

        if i > max_batches:
            break
    return time.time() - start_time


def test(  # pylint:disable=too-many-arguments
    config,
    test_loader,
    predict_op,
    test_aug,
    model_vars,
    parallel,
    score_fn=metrics.accuracy_score,
):
    ctx_mngr = (model_vars).replicate() if parallel else contextlib.suppress()
    with ctx_mngr:
        correct, predicted = [], []
        max_batches = (
            config["hyperparams"]["overfit"]
            if "overfit" in config["hyperparams"]
            else len(test_loader) + 1
        )
        for i, (image, label) in tqdm(
            enumerate(test_loader),  # pylint:disable=loop-invariant-statement
            total=len(test_loader),
            desc="Testing",
            leave=False,
        ):
            image = test_aug(image)
            y_pred = predict_op(image)
            correct.append(label)
            predicted.append(y_pred)
            if i > max_batches:
                break
    correct = np.concatenate(correct)
    predicted = np.concatenate(predicted).argmax(axis=1)

    if config["general"]["log_wandb"]:
        wandb.log(
            {
                "val": metrics.classification_report(
                    correct, predicted, output_dict=True, zero_division=0
                )
            }
        )
    else:
        print(
            metrics.classification_report(
                correct, predicted, output_dict=False, zero_division=0
            )
        )
    return score_fn(correct, predicted)
