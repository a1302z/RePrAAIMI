import numpy as np
from tqdm import tqdm
from time import time
from contextlib import suppress
from jax import numpy as jn, local_device_count
from typing import Callable
import wandb
from objax import TrainRef, Module
from objax.io import load_var_collection

from dptraining.config import Config
from dptraining.utils.metrics import (
    calculate_metrics,
    summarise_batch_metrics,
    summarise_dict_metrics,
)
from dptraining.models import (
    make_model_from_config,
    modify_architecture_from_pretrained_model,
)

N_DEVICES = local_device_count()


def make_train_ops(models, make_train_op):
    attack_train_train_ops_and_vars = [
        make_train_op(model_vars, model) for model, model_vars, _ in models
    ]
    train_ops, train_vars = [tov[0] for tov in attack_train_train_ops_and_vars], [
        tov[1] for tov in attack_train_train_ops_and_vars
    ]
    model_vars = [model.vars() for model, _, _ in models]

    return train_ops, train_vars, model_vars


def create_N_models(config: Config, N: int):
    models = []
    for _ in tqdm(
        range(N),
        total=N,
        leave=False,
        desc="Creating shadow models",
    ):
        model = make_model_from_config(config.model)
        ######################################################################
        if config.general.use_pretrained_model is not None:
            print(f"Loading model from {config.general.use_pretrained_model}")
            load_var_collection(config.general.use_pretrained_model, model.vars())
            if config.model.pretrained_model_changes is not None:
                (
                    model_vars,
                    must_train_vars,
                ) = modify_architecture_from_pretrained_model(config, model)
            else:  # assuming pretrained model is exactly like current model
                model_vars, must_train_vars = model.vars(), model.vars()
        else:
            model_vars, must_train_vars = model.vars(), model.vars()
        models.append((model, model_vars, must_train_vars))
    del model, model_vars, must_train_vars
    return models


def flatten_weights(model: Module) -> tuple[dict[str, np.array], list[str], np.array]:
    data, names, seen, flat_data = {}, [], set(), []
    for k, v in model.vars().items():
        if isinstance(v, TrainRef):
            v = v.ref
        if id(v) not in seen:
            names.append(k)
            data[str(len(data))] = v.value
            flat_data.append(v.value.flatten())
            seen.add(id(v))
    flat_data = np.concatenate(flat_data)
    return data, names, flat_data


def train(  # pylint:disable=too-many-arguments,duplicate-code
    config: Config,
    train_loader,
    train_ops,
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
    with (sum(train_vars)).replicate() if parallel else suppress():
        for i, data in pbar:
            add_args = {}
            if grad_acc > 1:
                add_args["apply_norm_acc"] = (i + 1) % grad_acc == 0
            train_losses = []
            for train_op, (img, label) in tqdm(  # TODO: vectorize
                zip(train_ops, data),
                total=len(train_ops),
                leave=False,
                desc="Training shadow models",
            ):
                train_result = train_op(
                    img, label, np.float32(learning_rate), **add_args
                )
                if train_result is not None:
                    # train_loss, grads = train_result
                    train_losses.append(train_result[0].item())
            pbar.set_description(f"Average train_loss: {np.mean(train_losses):.2f}")
            # if config.general.log_wandb and train_result is not None:
            #     log_dict = {
            #         "train_loss": train_loss,
            #         "total_grad_norm": jn.linalg.norm(
            #             [jn.linalg.norm(g) for g in grads]
            #         ).item(),
            #     }
            #     wandb.log(
            #         log_dict,
            #         commit=i % 10 == 0,
            #     )

            # if i + 1 >= max_batches:
            #     break
    return time() - start_time


def test_multi_dataset(  # pylint:disable=too-many-arguments,too-many-branches
    config: Config,
    test_loader,
    predict_ops,
    test_aug,
    test_label_aug,
    model_vars,
    parallel,
    dataset_split: str,
    metrics: tuple,
    loss_fn: Callable,
    multi_dataset: bool = False,
) -> float:
    ctx_mngr = (sum(model_vars)).replicate() if parallel else suppress()
    per_batch_metrics = (
        config.metrics.per_batch_metrics
        if "per_batch_metrics" in config.metrics
        else False
    )
    main_metric_lists, logging_metric_lists, correct_lists, scores_lists = (
        None,
        None,
        None,
        None,
    )
    if per_batch_metrics:
        main_metric_lists, logging_metric_lists = [
            [] for _ in range(len(predict_ops))
        ], [[] for _ in range(len(predict_ops))]
    else:
        correct_lists, scores_lists = [[] for _ in range(len(predict_ops))], [
            [] for _ in range(len(predict_ops))
        ]
    with ctx_mngr:
        max_batches = (
            config.hyperparams.overfit
            if config.hyperparams.overfit is not None
            else len(test_loader)
        )

        for i, data in tqdm(
            enumerate(test_loader),
            total=max_batches,
            desc="Testing",
            leave=False,
        ):
            if multi_dataset:
                for j, ((image, label), predict_op) in tqdm(
                    enumerate(zip(data, predict_ops)),
                    total=len(predict_ops),
                    desc="Eval shadow models",
                    leave=False,
                ):
                    image = test_aug(image)
                    label = test_label_aug(label)
                    do_eval(
                        config,
                        parallel,
                        metrics,
                        loss_fn,
                        per_batch_metrics,
                        main_metric_lists,
                        logging_metric_lists,
                        correct_lists,
                        scores_lists,
                        j,
                        image,
                        label,
                        predict_op,
                    )
            else:
                image, label = data
                image = test_aug(image)
                label = test_label_aug(label)
                for j, predict_op in tqdm(
                    enumerate(predict_ops),
                    total=len(predict_ops),
                    desc="Eval shadow models",
                    leave=False,
                ):
                    do_eval(
                        config,
                        parallel,
                        metrics,
                        loss_fn,
                        per_batch_metrics,
                        main_metric_lists,
                        logging_metric_lists,
                        correct_lists,
                        scores_lists,
                        j,
                        image,
                        label,
                        predict_op,
                    )

            if i + 1 >= max_batches:
                break
    per_model_metrics = []
    if per_batch_metrics:
        for main_metric_list, logging_metric_list in zip(
            main_metric_lists, logging_metric_lists
        ):
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
            per_model_metrics.append((main_metric, logging_metrics))
    else:
        for correct, scores in zip(correct_lists, scores_lists):
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
            per_model_metrics.append((main_metric, logging_metrics))

    if config.general.log_wandb:
        if config.general.log_wandb:
            wandb.log(
                {
                    f"{dataset_split}_{i}": logging_metrics
                    for i, (_, logging_metrics) in enumerate(per_model_metrics)
                }
            )
    else:
        for i, (_, logging_metrics) in enumerate(per_model_metrics):
            print(f"{dataset_split}_{i} evaluation:")
            for name, value in logging_metrics.items():
                print(f"\t{name}: {value}")
        # print("Average metrics:") # TODO this might be helpful
        # avg_metrics = summarise_dict_metrics(
        #     [logging_metrics for _, logging_metrics in per_model_metrics],
        # )
        # for key, metric in avg_metrics.items():
        #     print(f"\t{key}: {metric}")
    return main_metric[1]


def do_eval(
    config,
    parallel,
    metrics,
    loss_fn,
    per_batch_metrics,
    main_metric_lists,
    logging_metric_lists,
    correct_lists,
    scores_lists,
    j,
    image,
    label,
    predict_op,
):
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
        main_metric_lists[j].append(main_metric_batch)
        logging_metric_lists[j].append(logging_metric_batch)
    else:
        correct_lists[j].append(label)
        scores_lists[j].append(y_pred)
