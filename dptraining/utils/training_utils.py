import wandb
import contextlib
from typing import Callable, Optional, Iterable

import numpy as np

import objax
import sys
from time import time
from jax import numpy as jn, local_device_count
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path.cwd()))

from dptraining.config import Config
from dptraining.privacy import ClipAndAccumulateGrads, analyse_epsilon
from dptraining.optim import AccumulateGrad, make_optim_from_config
from dptraining.utils.metrics import (
    calculate_metrics,
    summarise_batch_metrics,
    summarise_dict_metrics,
)
from dptraining.utils import ExponentialMovingAverage
from dptraining.utils.misc import get_num_params


N_DEVICES = local_device_count()



def make_train_op(
    model,
    model_vars,
    config: Config,
    loss_class,
    augment_op,
    label_augment_op,
    grad_acc,
    total_noise,
    effective_batch_size,
    n_augmentations,
):
    ema: Optional[ExponentialMovingAverage] = None
    if config.ema.use_ema:
        ema = ExponentialMovingAverage(
            model_vars, config.ema.decay, update_every=config.ema.update_every
        )
    opt = make_optim_from_config(config, model_vars)
    train_loss_fn = loss_class.create_train_loss_fn(model_vars, model)
    loss_gv = create_loss_gradient(config, model_vars, train_loss_fn)
    train_vars = (
        model_vars + loss_gv.vars() + opt.vars() + objax.random.DEFAULT_GENERATOR.vars()
    )
    if ema is not None:
        train_vars += ema.vars()
    train_op = create_train_op(
        train_vars,
        loss_gv,
        opt,
        augment_op,
        label_augment_op,
        grad_accumulation=grad_acc > 1,
        noise=total_noise,
        effective_batch_size=effective_batch_size,
        n_augmentations=n_augmentations,
        parallel=config.general.parallel,
        ema=ema,
    )

    return train_op, train_vars


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
    start_time = time()
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
    return time() - start_time


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


def train_loop(
    config: Config,
    train_loader,
    val_loader,
    model,
    unfreeze_schedule,
    checkpoint,
    predict_op_parallel,
    predict_op_jit,
    grad_acc,
    sampling_rate,
    delta,
    sigma,
    total_noise,
    batch_expansion_factor,
    effective_batch_size,
    accountant,
    metric_fns,
    n_augmentations,
    augment_op,
    label_augment_op,
    test_aug,
    test_label_aug,
    scheduler,
    stopper,
    loss_class,
    test_loss_fn,
    train_op,
    train_vars,
):
    epoch_time = []
    epoch_iter: Iterable
    if config.general.log_wandb:
        epoch_iter = tqdm(
            range(config.hyperparams.epochs),
            total=config.hyperparams.epochs,
            desc="Epoch",
            leave=True,
        )
    else:
        epoch_iter = range(config.hyperparams.epochs)
    if config.general.eval_init:
        if config.general.eval_train:
            test(
                config,
                train_loader,
                predict_op_parallel if config.general.parallel else predict_op_jit,
                test_aug,
                test_label_aug,
                model.vars(),
                config.general.parallel,
                "train",
                metrics=metric_fns,
                loss_fn=test_loss_fn,
            )
        if val_loader is not None:
            metric = test(
                config,
                val_loader,
                predict_op_parallel if config.general.parallel else predict_op_jit,
                test_aug,
                test_label_aug,
                model.vars(),
                config.general.parallel,
                "val",
                metrics=metric_fns,
                loss_fn=test_loss_fn,
            )
    for epoch, learning_rate in zip(epoch_iter, scheduler):
        cur_epoch_time = train(
            config,
            train_loader,
            train_op,
            learning_rate,
            train_vars,
            config.general.parallel,
            grad_acc,
        )
        if config.general.log_wandb:
            wandb.log({"epoch": epoch, "lr": learning_rate})
        else:
            print(f"Train Epoch: {epoch+1} \t took {cur_epoch_time} seconds")
        epoch_time.append(cur_epoch_time)
        if config.general.eval_train:
            test(
                config,
                train_loader,
                predict_op_parallel if config.general.parallel else predict_op_jit,
                test_aug,
                test_label_aug,
                model.vars(),
                config.general.parallel,
                "train",
                metrics=metric_fns,
                loss_fn=test_loss_fn,
            )
        if val_loader is not None:
            metric = test(
                config,
                val_loader,
                predict_op_parallel if config.general.parallel else predict_op_jit,
                test_aug,
                test_label_aug,
                model.vars(),
                config.general.parallel,
                "val",
                metrics=metric_fns,
                loss_fn=test_loss_fn,
            )
            scheduler.update_score(metric)
        else:
            metric = None
        if config.DP:
            epsilon = analyse_epsilon(
                accountant,
                (len(train_loader) // batch_expansion_factor) * (epoch + 1),
                sigma,
                sampling_rate,
                delta,
            )
            if config.general.log_wandb:
                wandb.log({"epsilon": epsilon})
            else:
                print(f"\tPrivacy: (ε = {epsilon:.2f}, δ = {delta})")
        if config.unfreeze_schedule is not None:
            model_vars, fresh_model_vars = unfreeze_schedule(epoch + 1)
            if fresh_model_vars:
                train_op, train_vars = make_train_op(
                    model,
                    model_vars,
                    config,
                    loss_class,
                    augment_op,
                    label_augment_op,
                    grad_acc,
                    total_noise,
                    effective_batch_size,
                    n_augmentations,
                )
            n_train_vars_cur = get_num_params(model_vars)
            if config.general.log_wandb:
                wandb.log({"num_trained_vars": n_train_vars_cur})
            else:
                print(f"\tNum Train Vars: {n_train_vars_cur:,}")
        if checkpoint is not None:
            checkpoint.save(model.vars(), idx=epoch)
        if metric is not None and stopper(metric):
            print("Early Stopping was activated -> Stopping Training")
            break
    return epoch_time
