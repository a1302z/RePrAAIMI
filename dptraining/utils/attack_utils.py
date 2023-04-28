from contextlib import suppress, ExitStack
from functools import partial
from itertools import chain
from time import time
from typing import Callable, Iterable, Optional

import numpy as np
from jax import local_device_count
from objax import Jit, Module, TrainRef, functional
from sklearn import decomposition, preprocessing
from tqdm import tqdm

import wandb
from dptraining.config import Config, DatasetTask
from dptraining.models import (
    make_model_from_config,
    setup_pretrained_model,
)
from dptraining.privacy import analyse_epsilon
from dptraining.utils.metrics import (
    calculate_metrics,
    summarise_batch_metrics,
    summarise_dict_metrics,
)

N_DEVICES = local_device_count()


def rescale_and_shrink_network_params(
    rescale: bool,
    pca_dim: Optional[int],
    train_data,
    eval_data,
    include_eval: bool,
):
    if rescale:

        def scaler_fn(train_x, test_x, include_eval=True):
            old_shape = None
            if len(train_x.shape) > 2:
                old_shape = (train_x.shape, test_x.shape)
                train_x = train_x.reshape(-1, np.prod(train_x.shape[1:]))
                test_x = test_x.reshape(-1, np.prod(test_x.shape[1:]))
            if include_eval:
                scaler = preprocessing.StandardScaler().fit(
                    np.concatenate([train_x, test_x])
                )
            else:
                scaler = preprocessing.StandardScaler().fit(train_x)
            params_train_scaled = scaler.transform(train_x)
            params_test_scaled = scaler.transform(test_x)
            if old_shape:
                train_x = train_x.reshape(old_shape[0])
                test_x = test_x.reshape(old_shape[1])
            return params_train_scaled, params_test_scaled

        train_data, eval_data = scaler_fn(
            train_data,
            eval_data,
            include_eval=include_eval,
        )
    if pca_dim:
        if len(train_data.shape) > 2:
            train_data = train_data.reshape(-1, np.prod(train_data.shape[1:]))
            eval_data = eval_data.reshape(-1, np.prod(eval_data.shape[1:]))
        pca = decomposition.PCA(n_components=pca_dim)
        if include_eval:
            pca.fit(np.concatenate([train_data, eval_data]))
        else:
            pca.fit(train_data)
        train_data = pca.transform(train_data)
        eval_data = pca.transform(eval_data)

    return train_data, eval_data


def setup_predict_ops(config: Config, train_models, eval_models):
    if config.dataset.task in (
        DatasetTask.classification,
        DatasetTask.segmentation,
    ):
        if config.loss.binary_loss:
            predict_lambda = lambda model, x: functional.sigmoid(  # pylint:disable=unnecessary-lambda-assignment
                model(x, training=False)
            )
        else:
            predict_lambda = lambda model, x: functional.softmax(  # pylint:disable=unnecessary-lambda-assignment
                model(x, training=False),
                axis=1 if config.dataset.task == DatasetTask.segmentation else -1,
            )
    else:
        predict_lambda = lambda model, x: model(x, training=False)
        # predict_op_parallel = objax.Parallel(
        #     predict_lambda, model.vars(), reduce=np.concatenate
        # )
    predict_ops_jit_train = [
        Jit(partial(predict_lambda, model), model.vars())  # pylint:disable=not-callable
        for model, _, _ in train_models
    ]
    predict_ops_jit_eval = [
        Jit(partial(predict_lambda, model), model.vars())  # pylint:disable=not-callable
        for model, _, _ in eval_models
    ]

    return predict_ops_jit_train, predict_ops_jit_eval


def make_train_ops(
    models,
    make_train_op,
    config: Config,
    loss_class,
    augment_op,
    label_augment_op,
    grad_acc,
    total_noise,
    effective_batch_size,
    n_augmentations,
):
    attack_train_train_ops_and_vars = [
        make_train_op(
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
        for model, model_vars, _ in tqdm(
            models, total=len(models), desc="Create train ops", leave=False
        )
    ]
    train_ops, individual_vars = [tov[0] for tov in attack_train_train_ops_and_vars], [
        tov[2] for tov in attack_train_train_ops_and_vars
    ]
    model_vars = [model.vars() for model, _, _ in models]

    return train_ops, individual_vars, model_vars


def create_N_models(config: Config, N: int):
    models = []
    for _ in tqdm(
        range(N),
        total=N,
        leave=False,
        desc="Creating shadow models",
    ):
        model = make_model_from_config(config)
        model_vars, _, must_train_vars = setup_pretrained_model(config, model)
        models.append((model, model_vars, must_train_vars))
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
    with ExitStack() as es:
        if parallel:
            for tv in [x.replicate() for x in train_vars]:
                es.enter_context(tv)
        for i, data in pbar:
            add_args = {}
            if grad_acc > 1:
                add_args["apply_norm_acc"] = (i + 1) % grad_acc == 0
            train_losses = []
            for j, train_op in tqdm(  # TODO: vectorize
                enumerate(train_ops),
                total=len(train_ops),
                leave=False,
                desc="Training shadow models",
            ):
                img, label = data[j] if isinstance(data, list) else data
                train_result = train_op(
                    img, label, np.float32(learning_rate), **add_args
                )
                if train_result is not None:
                    # train_loss, grads = train_result
                    train_losses.append(train_result[0].item())
            if len(train_losses) > 0:
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
    ctx_mngr = ExitStack()
    if parallel:
        for tv in [x.replicate() for x in model_vars]:
            ctx_mngr.enter_context(tv)
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


def train_loop(
    config: Config,
    shadow_train_loader,
    shadow_eval_loader,
    attack_eval_loader,
    predict_ops_jit_train,
    predict_ops_jit_eval,
    grad_acc,
    sampling_rate,
    delta,
    sigma,
    batch_expansion_factor,
    accountant,
    metric_fns,
    test_aug,
    test_label_aug,
    scheduler,
    test_loss_fn,
    attack_train_train_ops,
    attack_train_train_vars,
    train_model_vars,
    attack_eval_train_ops,
    attack_eval_train_vars,
    eval_model_vars,
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
            metrics = test_multi_dataset(
                config,
                shadow_train_loader,
                predict_ops_jit_train,
                test_aug,
                test_label_aug,
                train_model_vars,
                config.general.parallel,
                "train_attack_train",
                metrics=metric_fns,
                loss_fn=test_loss_fn,
                multi_dataset=True,
            )
            metrics = test_multi_dataset(
                config,
                attack_eval_loader,
                predict_ops_jit_eval,
                test_aug,
                test_label_aug,
                eval_model_vars,
                config.general.parallel,
                "train_attack_eval",
                metrics=metric_fns,
                loss_fn=test_loss_fn,
                multi_dataset=True,
            )
        if shadow_eval_loader is not None:
            metric = test_multi_dataset(
                config,
                shadow_eval_loader,
                list(chain([predict_ops_jit_train, predict_ops_jit_eval])),
                test_aug,
                test_label_aug,
                train_model_vars + eval_model_vars,
                config.general.parallel,
                "val",
                metrics=metric_fns,
                loss_fn=test_loss_fn,
                multi_dataset=False,
            )
    for epoch, learning_rate in zip(epoch_iter, scheduler):
        cur_epoch_time = train(
            config,
            shadow_train_loader,
            attack_train_train_ops,
            learning_rate,
            attack_train_train_vars,
            config.general.parallel,
            grad_acc,
        )
        cur_epoch_time += train(
            config,
            attack_eval_loader,
            attack_eval_train_ops,
            learning_rate,
            attack_eval_train_vars,
            config.general.parallel,
            grad_acc,
        )
        if config.general.log_wandb:
            wandb.log({"epoch": epoch, "lr": learning_rate})
        else:
            print(f"Train Epoch: {epoch+1} \t took {cur_epoch_time} seconds")
        epoch_time.append(cur_epoch_time)
        if config.general.eval_train:
            metrics = test_multi_dataset(
                config,
                shadow_train_loader,
                predict_ops_jit_train,
                test_aug,
                test_label_aug,
                train_model_vars,
                config.general.parallel,
                "train_attack_train",
                metrics=metric_fns,
                loss_fn=test_loss_fn,
                multi_dataset=True,
            )
            metrics = test_multi_dataset(
                config,
                attack_eval_loader,
                predict_ops_jit_eval,
                test_aug,
                test_label_aug,
                eval_model_vars,
                config.general.parallel,
                "train_attack_eval",
                metrics=metric_fns,
                loss_fn=test_loss_fn,
                multi_dataset=True,
            )
        if shadow_eval_loader is not None:
            metric = test_multi_dataset(
                config,
                shadow_eval_loader,
                predict_ops_jit_train + predict_ops_jit_eval,
                test_aug,
                test_label_aug,
                train_model_vars + eval_model_vars,
                config.general.parallel,
                "val",
                metrics=metric_fns,
                loss_fn=test_loss_fn,
                multi_dataset=False,
            )
            scheduler.update_score(metric)
        else:
            metric = None
        if config.DP:
            epsilon = analyse_epsilon(
                accountant,
                (len(shadow_train_loader) // batch_expansion_factor) * (epoch + 1),
                sigma,
                sampling_rate,
                delta,
                add_alphas=config.DP.alphas,
            )
            if config.general.log_wandb:
                wandb.log({"epsilon": epsilon})
            else:
                print(f"\tPrivacy: (ε = {epsilon:.2f}, δ = {delta})")
    return epoch_time
