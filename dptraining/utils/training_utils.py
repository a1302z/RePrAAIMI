import wandb
import time
import contextlib

import numpy as np

import objax
import sys


from pathlib import Path
from sklearn import metrics
from tqdm import tqdm

sys.path.insert(0, str(Path.cwd()))

from dptraining.privacy import ClipAndAccumulateGrads

# from dptraining.privacy import ComplexPrivateGradValues


def create_train_op(  # pylint:disable=too-many-arguments
    train_vars,
    loss_gv,
    opt,
    augment_op,
    grad_accumulation: bool,
    noise: float,
    effective_batch_size: int,
    parallel=False,
):
    if grad_accumulation:
        assert isinstance(loss_gv, ClipAndAccumulateGrads)

        @objax.Function.with_vars(train_vars)
        def calc_grads(image_batch, label_batch):
            image_batch = augment_op(image_batch)
            clipped_grad, loss_value = loss_gv.calc_per_sample_grads(
                image_batch, label_batch
            )
            loss_gv.accumulate_grad(clipped_grad, loss_value)
            if parallel:
                loss_value = objax.functional.parallel.psum(loss_value)
            loss_value = loss_value[0] / image_batch.shape[0]
            return loss_value

        @objax.Function.with_vars(train_vars)
        def apply_grads(learning_rate):
            grads = loss_gv.get_accumulated_grads()
            loss_gv.reset_accumulated_grads()
            if parallel:
                grads = objax.functional.parallel.psum(grads)
            # if isinstance(loss_gv, ClipAndAccumulateGrads):
            grads = loss_gv.add_noise(grads, noise, objax.random.DEFAULT_GENERATOR)
            grads = [gx / effective_batch_size for gx in grads]
            opt(learning_rate, grads)
            return grads

        if parallel:
            calc_grads = objax.Parallel(
                calc_grads, reduce=lambda x: x[0], vc=train_vars,
            )
            apply_grads = objax.Parallel(apply_grads, reduce=np.sum, vc=train_vars,)
        else:
            # pass
            calc_grads = objax.Jit(calc_grads)
            apply_grads = objax.Jit(apply_grads)

        # @objax.Function.with_vars(train_vars)
        def train_op(  # pylint:disable=inconsistent-return-statements
            image_batch, label_batch, learning_rate: float, apply_norm_acc: bool
        ):
            loss_value = calc_grads(image_batch, label_batch)
            if apply_norm_acc:
                return (
                    loss_value,
                    apply_grads(learning_rate),
                )  # theoretically the loss needs to be accumulated too, but who cares

    else:

        @objax.Function.with_vars(train_vars)
        def train_op(
            image_batch, label_batch, learning_rate,
        ):
            assert image_batch.shape[0] == effective_batch_size
            image_batch = augment_op(image_batch)
            grads, loss = loss_gv(image_batch, label_batch)
            if parallel:
                if isinstance(loss_gv, ClipAndAccumulateGrads):
                    grads = objax.functional.parallel.psum(grads)
                    loss = objax.functional.parallel.psum(loss)
                else:
                    grads = objax.functional.parallel.pmean(grads)
                    loss = objax.functional.parallel.pmean(loss)
            if isinstance(loss_gv, ClipAndAccumulateGrads):
                loss = loss[0] / image_batch.shape[0]
                grads = loss_gv.add_noise(grads, noise, objax.random.DEFAULT_GENERATOR)
                grads = [gx / effective_batch_size for gx in grads]
            else:
                loss = loss[0]
            opt(learning_rate, grads)
            return loss, grads

        if parallel:
            train_op = objax.Parallel(train_op, reduce=np.mean, vc=train_vars)
        else:
            train_op = objax.Jit(train_op, static_argnums=(3,))

    return train_op


def create_loss_gradient(config, model, model_vars, loss_fn):
    if config["DP"]["disable_dp"]:
        loss_gv = objax.GradValues(loss_fn, model.vars())
    else:
        loss_gv = ClipAndAccumulateGrads(
            loss_fn,
            model_vars,
            config["DP"]["max_per_sample_grad_norm"],
            batch_axis=(0, 0),
            use_norm_accumulation=config["DP"]["norm_acc"],
            gradient_accumulation_steps=config["DP"]["grad_acc_steps"]
            if "grad_acc_steps" in config["DP"]
            else 1,
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
    @objax.Function.with_vars(model_vars)
    def ema_update():
        ema.update(model_vars)
        ema.copy_to(model_vars)

    start_time = time.time()
    max_batches = (
        config["hyperparams"]["overfit"]
        if "overfit" in config["hyperparams"]
        else len(train_loader) + 1
    )
    for i, (img, label) in tqdm(
        enumerate(train_loader), total=len(train_loader), desc="Training", leave=False,
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
                ema_update()
        if config["general"]["log_wandb"] and train_result is not None:
            wandb.log(
                {
                    "train_loss": train_loss.item(),
                    "total_grad_norm": jn.linalg.norm(
                        [jn.linalg.norm(g) for g in grads]
                    ).item(),
                },
                commit=False,
            )
            if ema is not None:
                wandb.log(
                    {
                        k: (jn.abs(v.value) if jn.iscomplexobj(v) else v.value)
                        for k, v in model_vars.items()
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
            enumerate(test_loader), total=len(test_loader), desc="Testing", leave=False,
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
