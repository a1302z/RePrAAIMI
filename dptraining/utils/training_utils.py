import wandb
import time
import contextlib
from typing import Callable

import numpy as np

import objax
import sys

from jax import numpy as jn, local_device_count
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path.cwd()))

from dptraining.config import Config, DatasetTask
from dptraining.privacy import ClipAndAccumulateGrads, BAM_ClipAndAccumulateGrads

N_DEVICES = local_device_count()


def create_train_op(  # pylint:disable=too-many-arguments,too-many-statements
    train_vars,
    grad_calc,
    opt,
    augment_op,
    label_op,
    grad_accumulation: bool,
    noise: float,
    effective_batch_size: int,
    n_augmentations: int = 1,
    parallel=False,
    ema=None,
):
    if grad_accumulation:
        assert isinstance(grad_calc, ClipAndAccumulateGrads)

        @objax.Function.with_vars(train_vars)
        def calc_grads(image_batch, label_batch):
            image_batch = augment_op(image_batch)
            label_batch = label_op(label_batch)
            if n_augmentations > 1:
                if image_batch.shape[1] != n_augmentations:
                    raise RuntimeError(
                        "number of augmentations different than augmentation axis"
                    )
                label_batch = jn.repeat(
                    label_batch[:, jn.newaxis], n_augmentations, axis=1
                )
            else:
                image_batch = image_batch[:, jn.newaxis, ...]
                label_batch = label_batch[:, jn.newaxis, ...]
            clipped_grad, loss_value = grad_calc.calc_per_sample_grads(
                image_batch, label_batch
            )
            grad_calc.accumulate_grad(clipped_grad, loss_value)
            if parallel:
                loss_value = objax.functional.parallel.psum(loss_value)
            loss_value = loss_value[0] / image_batch.shape[0]
            return loss_value

        @objax.Function.with_vars(train_vars)
        def apply_grads(learning_rate):
            grads = grad_calc.get_accumulated_grads()
            grad_calc.reset_accumulated_grads()
            if parallel:
                grads = objax.functional.parallel.psum(grads)
            grads = grad_calc.add_noise(grads, noise, objax.random.DEFAULT_GENERATOR)
            grads = [gx / effective_batch_size for gx in grads]
            opt(learning_rate, grads)
            if ema is not None:
                ema()
            return grads

        if parallel:
            calc_grads = objax.Parallel(
                calc_grads,
                reduce=lambda x: x[0],
                vc=train_vars,
            )
            apply_grads = objax.Parallel(
                apply_grads,
                reduce=np.sum,
                vc=train_vars,
            )
        else:
            # pass
            #calc_grads = objax.Jit(calc_grads)
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
            image_batch,
            label_batch,
            learning_rate,
        ):
            # assert image_batch.shape[0] == effective_batch_size
            image_batch = augment_op(image_batch)
            label_batch = label_op(label_batch)
            if n_augmentations > 1:
                label_batch = jn.repeat(
                    label_batch[:, jn.newaxis], n_augmentations, axis=1
                )
                if not isinstance(grad_calc, ClipAndAccumulateGrads):
                    ibs = image_batch.shape
                    image_batch = image_batch.reshape(ibs[0] * ibs[1], *ibs[2:])
                    label_batch = label_batch.flatten()
            elif isinstance(grad_calc, ClipAndAccumulateGrads):
                image_batch = image_batch[:, jn.newaxis, ...]
                label_batch = label_batch[:, jn.newaxis, ...]

            grads, loss = grad_calc(image_batch, label_batch)
            if parallel:
                if isinstance(grad_calc, ClipAndAccumulateGrads):
                    grads = objax.functional.parallel.psum(grads)
                    loss = objax.functional.parallel.psum(loss)
                else:
                    grads = objax.functional.parallel.pmean(grads)
                    loss = objax.functional.parallel.pmean(loss)
            if isinstance(grad_calc, ClipAndAccumulateGrads):
                loss = loss[0] / image_batch.shape[0]
                grads = grad_calc.add_noise(
                    grads, noise, objax.random.DEFAULT_GENERATOR
                )
                grads = [gx / effective_batch_size for gx in grads]
            else:
                loss = loss[0]
            opt(learning_rate, grads)
            if ema is not None:
                ema()
            return loss, grads

        if parallel:
            train_op = objax.Parallel(train_op, reduce=np.mean, vc=train_vars)
        else:
            train_op = objax.Jit(train_op, static_argnums=(3,))

    return train_op


def create_loss_gradient(config: Config, model_vars, loss_fn):
    if not config.DP:
        print("... CAREFUL! Not using DP")
        loss_gv = objax.GradValues(loss_fn, model_vars)
    elif config.DP.bam:
        print("... using BAM")
        loss_gv = BAM_ClipAndAccumulateGrads(
            loss_fn,
            model_vars,
            config.DP.max_per_sample_grad_norm,
            batch_axis=(0, 0),
            use_norm_accumulation=config.DP.norm_acc,
            gradient_accumulation_steps=config.DP.grad_acc_steps,
            r=config.DP.r,
            alpha=config.DP.alpha,
            log_grad_metrics=config.general.log_wandb
        )
    else:
        print("... not using BAM")
        loss_gv = ClipAndAccumulateGrads(
            loss_fn,
            model_vars,
            config.DP.max_per_sample_grad_norm,
            batch_axis=(0, 0),
            use_norm_accumulation=config.DP.norm_acc,
            gradient_accumulation_steps=config.DP.grad_acc_steps,
            log_grad_metrics=config.general.log_wandb
        )
        

    return loss_gv


def train(  # pylint:disable=too-many-arguments,duplicate-code
    config: Config,
    train_loader,
    train_op,
    learning_rate,
    train_vars,
    parallel,
    grad_acc: int,
):
    start_time = time.time()
    max_batches = (
        config.hyperparams.overfit
        if config.hyperparams.overfit is not None
        else len(train_loader)
    )
    if config.DP and max_batches % grad_acc != 0:
        # here we ensure that if a train loader is not evenly divisible
        # by the number of gradient accumulation steps we stop after
        # the maximum amount of batches that can be accmulated
        # otherwise the assertion fails
        assert max_batches // grad_acc > 0, (
            "The number of batches cannot be smaller than the number "
            "of gradient accumulation steps"
        )
        max_batches = max_batches - (max_batches % grad_acc)
    pbar = tqdm(
        enumerate(train_loader),
        total=max_batches,
        desc="Training",
        leave=False,
    )
    with (train_vars).replicate() if parallel else contextlib.suppress():
        for i, (img, label) in pbar:
            add_args = {}
            if grad_acc > 1:
                add_args["apply_norm_acc"] = (i + 1) % grad_acc == 0
            train_result = train_op(img, label, np.array(learning_rate), **add_args)
            if train_result is not None:
                train_loss, grads = train_result
                train_loss = train_loss.item()
                pbar.set_description(f"Train_loss: {train_loss:.2f}")
            if config.general.log_wandb and train_result is not None:
                log_dict = {
                    "train_loss": train_loss,
                    "total_grad_norm": jn.linalg.norm(
                        [jn.linalg.norm(g) for g in grads]
                    ).item(),
                }
                wandb.log(
                    log_dict,
                    commit=i % 10 == 0,
                )

            if i + 1 >= max_batches:
                break
    return time.time() - start_time


def calculate_metrics(task, metrics, loss_fn, correct, predicted):
    loss = loss_fn(predicted, correct).item()
    if task == DatasetTask.classification:
        predicted = predicted.argmax(axis=1)
    correct, predicted = correct.squeeze(), predicted.squeeze()
    if np.iscomplexobj(correct):
        correct = np.abs(correct)
    if np.iscomplexobj(predicted):
        predicted = np.abs(predicted)

    main_metric_fn, logging_fns = metrics
    main_metric = (
        main_metric_fn[0],
        loss
        if main_metric_fn[0] == "loss" and main_metric_fn[1] is None
        else main_metric_fn[1](correct, predicted),
    )
    logging_metrics = {
        func_name: lfn(correct, predicted) for func_name, lfn in logging_fns.items()
    }
    if main_metric[0] != "loss":
        logging_metrics["loss"] = loss
    logging_metrics[f"{main_metric[0]}"] = main_metric[1]
    return main_metric, logging_metrics


def test(  # pylint:disable=too-many-arguments,too-many-branches
    config: Config,
    test_loader,
    predict_op,
    test_aug,
    test_label_aug,
    model_vars,
    parallel,
    dataset_split: str,
    metrics: tuple,
    loss_fn: Callable,
):
    ctx_mngr = (model_vars).replicate() if parallel else contextlib.suppress()
    per_batch_metrics = (
        config.metrics.per_batch_metrics
        if "per_batch_metrics" in config.metrics
        else False
    )
    if per_batch_metrics:
        main_metric_list, logging_metric_list = [], []
    else:
        correct, scores = [], []
    with ctx_mngr:
        max_batches = (
            config.hyperparams.overfit
            if config.hyperparams.overfit is not None
            else len(test_loader)
        )
        for i, (image, label) in tqdm(
            enumerate(test_loader),
            total=max_batches,
            desc="Testing",
            leave=False,
        ):
            image = test_aug(image)
            label = test_label_aug(label)
            n_images = image.shape[0]
            if parallel and not (n_images % N_DEVICES) == 0:
                max_samples = n_images - (n_images % N_DEVICES)
                image = image[:max_samples]
                label = label[:max_samples]
            y_pred = predict_op(image)
            y_pred, label = np.array(y_pred), np.array(label)
            if per_batch_metrics:
                main_metric_batch, logging_metric_batch = calculate_metrics(
                    config.dataset.task, metrics, loss_fn, label, y_pred
                )
                main_metric_list.append(main_metric_batch)
                logging_metric_list.append(logging_metric_batch)
            else:
                correct.append(label)
                scores.append(y_pred)
            if i + 1 >= max_batches:
                break
    if per_batch_metrics:
        main_metric = (
            metrics[0][0],
            np.mean([batch_metric[1] for batch_metric in main_metric_list]),
        )
        logging_metrics = summarise_batch_metrics(
            metrics[1].keys(), logging_metric_list
        )
    else:
        correct = np.concatenate(correct)
        predicted = np.concatenate(scores)
        main_metric, logging_metrics = calculate_metrics(
            config.dataset.task, metrics, loss_fn, correct, predicted
        )

    if config.general.log_wandb:
        if config.general.log_wandb:
            wandb.log({dataset_split: logging_metrics})
    else:
        print(f"{dataset_split} evaluation:")
        for name, value in logging_metrics.items():
            print(f"\t{name}: {value}")
    return main_metric[1]


def summarise_batch_metrics(metric_names, metric_list):
    return {
        func_name: np.mean([metric[func_name] for metric in metric_list])
        for func_name in metric_names
    }
