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

from dptraining.privacy import ComplexPrivateGradValues


def create_train_op(  # pylint:disable=too-many-arguments
    train_vars, loss_gv, opt, augment_op, complex_valued=False, parallel=False
):
    @objax.Function.with_vars(train_vars)
    def train_op(image_batch, label_batch, learning_rate):
        image_batch = augment_op(image_batch)
        grads, loss = loss_gv(image_batch, label_batch)
        if parallel:
            grads = objax.functional.parallel.pmean(grads)
            loss = objax.functional.parallel.pmean(loss)
        if complex_valued:
            opt(learning_rate, [g.conj() for g in grads])
        else:
            opt(learning_rate, grads)
        return loss

    if parallel:
        train_op = objax.Parallel(
            train_op,
            reduce=np.mean,
            vc=train_vars,
        )
    else:

        # @objax.Function.with_vars(train_vars)
        # def train_op(image_batch, label_batch, learning_rate):
        #     image_batch = augment_op(image_batch)
        #     grads, loss = loss_gv(image_batch, label_batch)
        #     if complex_valued:
        #         opt(learning_rate, [g.conj() for g in grads])
        #     else:
        #         opt(learning_rate, grads)
        #     return loss

        train_op = objax.Jit(train_op)
    return train_op


def create_loss_gradient(config, model, model_vars, loss_fn, sigma):
    if config["DP"]["disable_dp"]:
        loss_gv = objax.GradValues(loss_fn, model.vars())
    elif "complex" in config["model"] and config["model"]["complex"]:
        loss_gv = ComplexPrivateGradValues(
            loss_fn,
            model_vars,
            sigma,
            config["DP"]["max_per_sample_grad_norm"],
            microbatch=1,
            batch_axis=(0, 0),
            use_norm_accumulation=config["DP"]["norm_acc"],
        )
    else:
        loss_gv = objax.privacy.dpsgd.PrivateGradValues(
            loss_fn,
            model_vars,
            sigma,
            config["DP"]["max_per_sample_grad_norm"],
            microbatch=1,
            batch_axis=(0, 0),
            use_norm_accumulation=config["DP"]["norm_acc"],
        )

    return loss_gv


def train(  # pylint:disable=too-many-arguments,duplicate-code
    config,
    train_loader,
    train_op,
    learning_rate,
    train_vars,
    parallel,
    model_vars=None,
    ema=None,
):
    ctx_mngr = (train_vars).replicate() if parallel else contextlib.suppress()
    with ctx_mngr:
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
            train_loss = train_op(img, label, np.array(learning_rate))
            if ema is not None:
                ema.update()
                ema.copy_to(model_vars)
            if config["general"][
                "log_wandb"
            ]:  # pylint:disable=loop-invariant-statement
                wandb.log({"train_loss": train_loss[0].item()}, commit=False)
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
