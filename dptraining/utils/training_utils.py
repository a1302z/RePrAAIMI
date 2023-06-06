import wandb
import time
import contextlib
from typing import Callable, List

import numpy as np

import objax
import sys

from jax import numpy as jn, local_device_count
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path.cwd()))

from dptraining.config import Config, DatasetTask
from dptraining.privacy import ClipAndAccumulateGrads

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
                grads = grad_calc.add_noise(
                    grads, noise, objax.random.DEFAULT_GENERATOR
                )
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


def create_loss_gradient(config: Config, model_vars, loss_fn):
    if not config.DP:
        loss_gv = objax.GradValues(loss_fn, model_vars)
    else:
        loss_gv = ClipAndAccumulateGrads(
            loss_fn,
            model_vars,
            config.DP.max_per_sample_grad_norm,
            batch_axis=(0, 0),
            use_norm_accumulation=config.DP.norm_acc,
            gradient_accumulation_steps=config.DP.grad_acc_steps,
            bam=config.DP.bam,
            r=config.DP.r,
            alpha=config.DP.alpha,
            log_grad_metrics=config.general.log_wandb,
        )

    return loss_gv


def train(  # pylint:disable=too-many-arguments,duplicate-code
    config: Config,
    train_loader,
    train_op,
    learning_rate,
    train_vars,
    val_loader,
    predict_op,
    metrics,
    eval_loss_fn,
    test_aug,
    test_label_aug,
    parallel: bool,
    grad_acc: int,
    eval_every_n: int,
):
    start_time = time.time()
    max_batches = (
        config.hyperparams.overfit
        if config.hyperparams.overfit is not None
        else len(train_loader)
    )
    val_iter = iter(val_loader)
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
        total=max_batches,
        desc="Training",
        leave=False,
    )
    train_iter = enumerate(iter(train_loader))
    with (train_vars).replicate() if parallel else contextlib.suppress():
        while True:
            try:
                if not config.dataset.has_group_attributes:
                    i, (img, label) = next(train_iter)
                else:
                    i, (img, attr_dict, label) = next(train_iter)
                    assert isinstance(attr_dict, dict), "attributes should be a dict"
            except StopIteration:
                break
            add_args = {}
            if grad_acc > 1:
                add_args["is_update_step"] = (i + 1) % grad_acc == 0
            train_result = train_op(img, label, np.array(learning_rate), **add_args)
            if train_result is not None:
                values, grads = train_result
                train_loss = values[0]
                if isinstance(train_loss, List):
                    train_loss = train_loss[0]
                pbar.set_description(f"Train_loss: {float(train_loss):.2f}")
            if config.general.log_wandb and train_result is not None:
                log_dict = {
                    "train_loss": train_loss,
                    "total_grad_norm": jn.linalg.norm(
                        [jn.linalg.norm(g) for g in grads]
                    ),
                }
                if len(values) > 1:
                    grad_metric_dict = values[1]
                    log_dict = log_dict | grad_metric_dict
                wandb.log(log_dict)
            if eval_every_n != -1 and i % eval_every_n == 0:
                # train_metrics
                train_pred = predict_op(img)
                main_metric_batch, logging_metric_batch = calculate_metrics(
                    config.dataset.task, metrics, eval_loss_fn, label, train_pred
                )
                if config.general.log_wandb:
                    wandb.log({"train": logging_metric_batch})
                # validation metrics
                try:
                    if config.dataset.has_group_attributes:
                        eval_img, eval_attr_dict, eval_label = next(val_iter)
                        assert isinstance(eval_attr_dict, dict), "attributes should be a dict"
                    else:
                        eval_img, eval_label = next(val_iter)
                except StopIteration:
                    break
                eval_img = test_aug(eval_img)
                eval_label = test_label_aug(eval_label)
                y_pred = predict_op(eval_img)
                main_metric_batch, logging_metric_batch = calculate_metrics(
                    config.dataset.task, metrics, eval_loss_fn, eval_label, y_pred
                )
                if config.general.log_wandb:
                    wandb.log({"val": logging_metric_batch})
            if i + 1 >= max_batches:
                break
            pbar.update(1)
    pbar.close()
    return time.time() - start_time


def calculate_metrics(task, metrics, loss_fn, correct, logits):
    loss = loss_fn(logits, correct)
    if task == DatasetTask.classification:
        predicted = logits.argmax(axis=1)
    elif task == DatasetTask.binary_classification:
        predicted = np.array(logits > 0.5)
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
        val_iter = enumerate(iter(test_loader))
        pbar = tqdm(
            total=max_batches,
            desc="Testing",
            leave=False,
        )
        num_samples = 0
        attr_dict_aggregated = {}
        while True:
            try:
                if not config.dataset.has_group_attributes:
                    i, (image, label) = next(val_iter)
                else:
                    i, (image, attr_dict, label) = next(val_iter)
                    if attr_dict_aggregated == {}:
                        attr_dict_aggregated = attr_dict
                    assert isinstance(attr_dict, dict), "attributes should be a dict"
            except StopIteration:
                break
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
                if config.dataset.has_group_attributes:
                    raise NotImplementedError("per-batch metrics are not currently supported with group attributes")
                main_metric_batch, logging_metric_batch = calculate_metrics(
                    config.dataset.task, metrics, loss_fn, label, y_pred
                )
                main_metric_list.append(main_metric_batch)
                logging_metric_list.append(logging_metric_batch)
            else:
                correct.append(label)
                scores.append(y_pred)
                if i != 0:
                    # aggregate attribute indices
                    for attr_name, agg_attr_idcs in attr_dict_aggregated.items():
                        to_aggregate_idcs = attr_dict[attr_name]
                        to_aggregate_idcs += num_samples
                        attr_dict_aggregated[attr_name] = np.concatenate(
                            (agg_attr_idcs, to_aggregate_idcs)
                        )
                num_samples += image.shape[0]
            pbar.update(1)
            if i + 1 >= max_batches:
                break
    if per_batch_metrics:
        if config.dataset.has_group_attributes:
                raise NotImplementedError("per-batch metrics are not currently supported with group attributes")
        main_metric = (
            metrics[0][0],
            np.mean([batch_metric[1] for batch_metric in main_metric_list]),
        )
        logging_metrics = summarise_batch_metrics(
            metrics[1].keys(), logging_metric_list
        )
    else:
        logging_metrics = {}
        correct = np.concatenate(correct)
        predicted = np.concatenate(scores)
        if config.dataset.has_group_attributes:
            # calculate metrics for sub-groups
            for attr_name, idcs in attr_dict_aggregated.items():
                attr_correct = correct[idcs]
                attr_predicted = predicted[idcs]
                attr_main_metric, attr_log_metric = calculate_metrics(
                    config.dataset.task, metrics, loss_fn, attr_correct, attr_predicted
                )
                logging_metrics[attr_name] = attr_log_metric 
            # calculate metrics for the everyone
            main_metric, log_metrics = calculate_metrics(
                    config.dataset.task, metrics, loss_fn, correct, predicted
                )
            logging_metrics["all"] = log_metrics
        else:
            # only calculate general metrics
            main_metric, logging_metrics = calculate_metrics(
                config.dataset.task, metrics, loss_fn, correct, predicted
            )

    if config.general.log_wandb:
        if config.dataset.has_group_attributes:
            for attr_name, attr_log_metrics in logging_metrics.items():
                wandb.log({f"{dataset_split}_{attr_name}": logging_metrics[attr_name]})
                for name, value in logging_metrics.items():
                    print(name)
                    if name == "classification_report":
                        df = pd.DataFrame(value)
                        table = wandb.Table(dataframe=df)
                        wandb.log({f"cr{dataset_split}_{attr_name}": table})
        else:
            wandb.log({f"{dataset_split}_": logging_metrics})
            for name, value in logging_metrics.items():
                if name == "classification_report":
                    df = pd.DataFrame(value)
                    table = wandb.Table(dataframe=df)
                    wandb.log({f"cr{dataset_split}": table})
    else:
        print(f"-----------------------------------------\n{dataset_split}_:")
        if config.dataset.has_group_attributes:
            for attr_name, attr_log_metrics in logging_metrics.items():
                print(f"{attr_name}:")
                for name, value in attr_log_metrics.items():
                    if name == "classification_report":
                        print(f"\t{name}: \n{value}")
                    else:
                        print(f"\t{name}: {value}")
        else:
            for name, value in logging_metrics.items():
                if name == "classification_report":
                        print(f"\t{name}: \n{value}")
                else:
                    print(f"\t{name}: {value}")
        print("-----------------------------------------")
    return main_metric[1]


def summarise_batch_metrics(metric_names, metric_list):
    return {
        func_name: np.mean([metric[func_name] for metric in metric_list])
        for func_name in metric_names
    }
