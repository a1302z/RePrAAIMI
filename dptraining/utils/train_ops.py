from typing import Callable, List

import numpy as np
import pandas as pd

import objax
import sys

from jax import numpy as jn, local_device_count
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path.cwd()))

from dptraining.config import Config, DatasetTask
from dptraining.privacy import ClipAndAccumulateGrads

N_DEVICES = local_device_count()


def create_train_op_accumulate(  # pylint:disable=too-many-arguments,too-many-statements
    train_vars,
    grad_calc,
    opt,
    augment_op,
    label_op,
    noise: float,
    effective_batch_size: int,
    n_augmentations: int = 1,
    parallel=False,
    ema=None,
):
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
            label_batch = jn.repeat(label_batch[:, jn.newaxis], n_augmentations, axis=1)
        else:
            image_batch = image_batch[:, jn.newaxis, ...]
            label_batch = label_batch[:, jn.newaxis, ...]        
        clipped_grad, values = grad_calc.calc_per_sample_grads(
            image_batch, label_batch
        )
        loss_value = values[0]
        grad_calc.accumulate_grad(clipped_grad)
        if parallel:
            loss_value = objax.functional.parallel.psum(loss_value)
        # loss_value = loss_value[0] / image_batch.shape[0]
        return loss_value[0], values[1]

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
        #pass
        calc_grads = objax.Jit(calc_grads)
        apply_grads = objax.Jit(apply_grads)

    # @objax.Function.with_vars(train_vars)
    def train_op(  # pylint:disable=inconsistent-return-statements
        image_batch, label_batch, learning_rate: float, is_update_step: bool
    ):  
        values = calc_grads(image_batch, label_batch)
        if is_update_step:
            return (
                values,
                apply_grads(learning_rate),
            )

    return train_op


def create_train_op_normal(
    train_vars,
    grad_calc,
    opt,
    augment_op,
    label_op,
    noise: float,
    effective_batch_size: int,
    n_augmentations: int = 1,
    parallel=False,
    ema=None,
):
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
            label_batch = jn.repeat(label_batch[:, jn.newaxis], n_augmentations, axis=1)
            if not isinstance(grad_calc, ClipAndAccumulateGrads):
                ibs = image_batch.shape
                image_batch = image_batch.reshape(ibs[0] * ibs[1], *ibs[2:])
                label_batch = label_batch.flatten()
        elif isinstance(grad_calc, ClipAndAccumulateGrads):
            image_batch = image_batch[:, jn.newaxis, ...]
            label_batch = label_batch[:, jn.newaxis, ...]

        grads, values = grad_calc(image_batch, label_batch)
        if parallel:
            if isinstance(grad_calc, ClipAndAccumulateGrads):
                loss = values[0]
                grads = objax.functional.parallel.psum(grads)
                loss = objax.functional.parallel.psum(loss)
            else:
                grads = objax.functional.parallel.pmean(grads)
                loss = objax.functional.parallel.pmean(values)
        if isinstance(grad_calc, ClipAndAccumulateGrads):
            # loss = loss[0] / image_batch.shape[0]
            grads = grad_calc.add_noise(grads, noise, objax.random.DEFAULT_GENERATOR)
            grads = [gx / effective_batch_size for gx in grads]
        else:
            loss = values[0]
        opt(learning_rate, grads)
        if ema is not None:
            ema()
        return values, grads

    if parallel:
        train_op = objax.Parallel(train_op, reduce=np.mean, vc=train_vars)
    else:
        train_op = objax.Jit(train_op, static_argnums=(3,))

    return train_op


def create_train_op_SAT(
    train_vars,
    grad_calc,
    opt,
    augment_op,
    label_op,
    noise: float,
    effective_batch_size: int,
    n_augmentations: int = 1,
    parallel=False,
    ema=None,
):
    assert isinstance(grad_calc, ClipAndAccumulateGrads)

    @objax.Function.with_vars(train_vars)
    def calc_grads(image_batch, label_batch, prev_grad, prev_grad_norm):
        image_batch = augment_op(image_batch)
        label_batch = label_op(label_batch)
        if n_augmentations > 1:
            if image_batch.shape[1] != n_augmentations:
                raise RuntimeError(
                    "number of augmentations different than augmentation axis"
                )
            label_batch = jn.repeat(label_batch[:, jn.newaxis], n_augmentations, axis=1)
        else:
            image_batch = image_batch[:, jn.newaxis, ...]
            label_batch = label_batch[:, jn.newaxis, ...]

        clipped_grad, values = grad_calc.calc_per_sample_grads(
            image_batch, label_batch, prev_grad, prev_grad_norm
        )
        loss_value = values[0]
        grad_calc.accumulate_grad(clipped_grad)
        if parallel:
            loss_value = objax.functional.parallel.psum(loss_value)
        # loss_value = loss_value[0] / image_batch.shape[0]
        return loss_value[0], values[1]

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
        #pass
        calc_grads = objax.Jit(calc_grads)
        apply_grads = objax.Jit(apply_grads)

    # @objax.Function.with_vars(train_vars)
    def train_op(  # pylint:disable=inconsistent-return-statements
        image_batch, label_batch, learning_rate: float, is_update_step: bool, prev_grad, prev_grad_norm
    ):
        values = calc_grads(image_batch, label_batch, prev_grad=prev_grad, prev_grad_norm=prev_grad_norm)
        if is_update_step:
            private_grads = apply_grads(learning_rate)
            return (
                values,
                private_grads,
            )

    return train_op
