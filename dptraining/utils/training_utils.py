import wandb
import time
import contextlib
from typing import Callable, Union, Any

import numpy as np

import objax
import sys

from jax import numpy as jn, local_device_count
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path.cwd()))

from dptraining.config import Config, DatasetTask
from dptraining.privacy import ClipAndAccumulateGrads
from dptraining.optim import AccumulateGrad
from dptraining.utils import NEED_RAW_PREDICTIONS

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
        assert isinstance(grad_calc, (ClipAndAccumulateGrads, AccumulateGrad))

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
                if not isinstance(grad_calc, ClipAndAccumulateGrads):
                    ibs = image_batch.shape
                    image_batch = image_batch.reshape(ibs[0] * ibs[1], *ibs[2:])
                    label_batch = label_batch.flatten()
            elif isinstance(grad_calc, ClipAndAccumulateGrads):
                image_batch = image_batch[:, jn.newaxis, ...]
                label_batch = label_batch[:, jn.newaxis, ...]
            clipped_grad, loss_value = grad_calc(image_batch, label_batch)
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
            if isinstance(grad_calc, ClipAndAccumulateGrads):
                grads = grad_calc.add_noise(
                    grads, noise, objax.random.DEFAULT_GENERATOR
                )
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
            calc_grads = objax.Jit(calc_grads, vc=train_vars)
            apply_grads = objax.Jit(apply_grads, vc=train_vars)

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
            train_op = objax.Parallel(
                train_op, reduce=np.mean, vc=train_vars, static_argnums=(2,)
            )
        else:
            train_op = objax.Jit(train_op, vc=train_vars, static_argnums=(2,))

    return train_op


def create_loss_gradient(config: Config, model_vars, loss_fn):
    if config.hyperparams.grad_acc_steps > 1 and not config.DP:
        loss_gv = AccumulateGrad(
            loss_fn,
            model_vars,
        )
    elif config.DP:
        loss_gv = ClipAndAccumulateGrads(
            loss_fn,
            model_vars,
            config.DP.max_per_sample_grad_norm,
            batch_axis=(0, 0),
            use_norm_accumulation=config.DP.norm_acc,
            gradient_accumulation_steps=config.hyperparams.grad_acc_steps,
        )
    else:
        loss_gv = objax.GradValues(loss_fn, model_vars)

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
            train_result = train_op(img, label, np.float32(learning_rate), **add_args)
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


def calculate_metrics(
    task, metrics, loss_fn, correct, raw_prediction, binary_reduction
):
    loss = loss_fn(raw_prediction, correct).item()
    if task in [DatasetTask.classification, DatasetTask.reconstruction]:
        if binary_reduction:
            predicted = np.where(raw_prediction > 0.5, 1, 0)
        else:
            predicted = raw_prediction.argmax(axis=1)
        correct, predicted = correct.squeeze(), predicted.squeeze()
    elif task == DatasetTask.segmentation:
        if binary_reduction:
            predicted = np.where(raw_prediction > 0.5, 1.0, 0.0)
        else:
            predicted = np.argmax(raw_prediction, axis=1, keepdims=True)
    if np.iscomplexobj(correct):
        correct = np.abs(correct)
    if np.iscomplexobj(predicted):
        predicted = np.abs(predicted)

    main_metric_fn, logging_fns = metrics
    main_metric = (
        main_metric_fn[0],
        loss
        if main_metric_fn[0] == "loss" and main_metric_fn[1] is None
        else (
            main_metric_fn[1](correct, raw_prediction)
            if main_metric_fn[0] in NEED_RAW_PREDICTIONS
            else main_metric_fn[1](correct, predicted)
        ),
    )
    logging_metrics = {
        func_name: (
            lfn(correct, raw_prediction)
            if func_name in NEED_RAW_PREDICTIONS
            else lfn(correct, predicted)
        )
        for func_name, lfn in logging_fns.items()
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
) -> float:
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
                    config.dataset.task,
                    metrics,
                    loss_fn,
                    label,
                    y_pred,
                    config.loss.binary_loss,
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
            summarise_dict_metrics(
                [batch_metric[1] for batch_metric in main_metric_list]
            )
            if isinstance(main_metric_list[0][1], dict)
            else np.mean([batch_metric[1] for batch_metric in main_metric_list]),
        )
        logging_metrics = summarise_batch_metrics(
            metrics[1].keys(), logging_metric_list
        )
    else:
        correct = np.concatenate(correct)
        predicted = np.concatenate(scores)
        main_metric, logging_metrics = calculate_metrics(
            config.dataset.task,
            metrics,
            loss_fn,
            correct,
            predicted,
            config.loss.binary_loss,
        )

    if config.general.log_wandb:
        if config.general.log_wandb:
            wandb.log({dataset_split: logging_metrics})
    else:
        print(f"{dataset_split} evaluation:")
        for name, value in logging_metrics.items():
            print(f"\t{name}: {value}")
    return main_metric[1]


def summarise_dict_metrics(metric_dict_list: list[dict[Any, float]]) -> dict:
    assert all(
        (
            set(metric_dict_list[0].keys()) == set(metric_dict_list[i].keys())
            for i in range(1, len(metric_dict_list))
        )
    ), "Cannot summarise metrics with different keys"
    metric_dict = {
        k: np.mean([m[k] for m in metric_dict_list]) for k in metric_dict_list[0].keys()
    }
    return metric_dict


def summarise_batch_metrics(
    metric_names: list[str], metric_list: list[Union[float, dict[Any, float]]]
) -> dict[str, Union[float, dict[Any, float]]]:
    return {
        func_name: (
            summarise_dict_metrics([metric[func_name] for metric in metric_list])
            if isinstance(metric_list[0][func_name], dict)
            else np.mean([metric[func_name] for metric in metric_list])
        )
        for func_name in metric_names
    }
