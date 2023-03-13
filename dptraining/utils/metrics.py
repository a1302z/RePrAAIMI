from functools import partial
from types import FunctionType
from typing import Any, Union
from warnings import warn

import numpy as np
from jax import numpy as jnp
from omegaconf import DictConfig, OmegaConf
from skimage import metrics as skimagemetrics
from sklearn import metrics as sklearnmetrics

from dptraining.config import Config, DatasetTask, LossType
from dptraining.utils.loss import f_score


def activate_fn(func, *args, **kwargs):
    assert callable(func)
    if not isinstance(func, FunctionType):
        return func(*args, **kwargs)
    if len(args) > 0 or len(kwargs) > 0:
        return partial(func, *args, **kwargs)
    return func


# def seperate_args_kwargs(args: Union[dict, list]):
#     kwargs = {k: v for a in args for k, v in a.items() if isinstance(args, dict)}
#     args = [a for a in args if isinstance(a, list)]
#     return args, kwargs


def retrieve_func_dict(func):
    return {name: getattr(func, name) for name in dir(func) if name[0] != "_"}


NEED_RAW_PREDICTIONS: tuple[str] = ("fscore", "fscore_avg", "weighted_fscore_avg")


def make_metrics(config: Config):
    if not config.metrics:
        warn("no metrics defined in config", category=UserWarning)
        return sklearnmetrics.accuracy_score, (sklearnmetrics.classification_report,)
    assert isinstance(config.metrics.main, (str, dict, DictConfig))
    assert isinstance(config.metrics.logging, (list, dict, DictConfig))
    all_funcs = {
        **retrieve_func_dict(sklearnmetrics),
        **retrieve_func_dict(skimagemetrics),
    }  # torchmetrics of course leads to problems -.-
    if config.loss.type == LossType.dice:

        def fscore(*args, **kwargs) -> float:
            f_scores = f_score(
                *args,
                **kwargs,
                as_loss_fn=False,
            )
            return {i: f_scores[i] for i in range(f_scores.shape[0])}

        def fscore_avg(*args, **kwargs) -> float:
            return f_score(
                *args,
                **kwargs,
                as_loss_fn=False,
            ).mean()

        def weighted_fscore_avg(*args, **kwargs) -> float:
            return 1.0 - f_score(
                *args,
                **kwargs,
                class_weights=np.array(config.loss.class_weights)
                if config.loss.class_weights
                else None,
                as_loss_fn=True,
            )

        all_funcs["fscore"] = fscore
        all_funcs["fscore_avg"] = fscore_avg
        all_funcs["weighted_fscore_avg"] = weighted_fscore_avg

    if isinstance(config.metrics.main, str):
        if config.metricsFalse.main == "loss":
            main_metric = ("loss", None)
        else:
            main_metric = (
                config.metrics.main,
                activate_fn(all_funcs[config.metrics.main]),
            )
    else:
        main_metric_config = OmegaConf.to_container(config.metrics.main)
        assert len(main_metric_config) == 1
        fn_name = list(main_metric_config.keys())[0]
        args = list(main_metric_config.values())[0]
        kwargs = args if isinstance(args, dict) else {}
        args = args if isinstance(args, list) else []
        main_metric = (fn_name, activate_fn(all_funcs[fn_name], *args, **kwargs))

    logging_metrics = {}
    logging_conf = OmegaConf.to_container(config.metrics.logging)
    if isinstance(logging_conf, list):
        logging_metrics.update(
            {func_name: activate_fn(all_funcs[func_name]) for func_name in logging_conf}
        )
    else:
        for func_name, args in logging_conf.items():
            # args, kwargs = seperate_args_kwargs(args) if args is not None else [], {}
            kwargs = args if isinstance(args, dict) else {}
            args = args if isinstance(args, list) else []
            logging_metrics[func_name] = activate_fn(
                all_funcs[func_name], *args, **kwargs
            )

    return main_metric, logging_metrics


def calculate_metrics(
    task, metrics, loss_fn, correct, raw_prediction, binary_reduction
):
    match task:
        case DatasetTask.classification:
            if binary_reduction:
                predicted = np.where(raw_prediction > 0.5, 1, 0)
            else:
                predicted = raw_prediction.argmax(axis=1)
            correct, predicted = correct.squeeze(), predicted.squeeze()
        case DatasetTask.segmentation:
            if binary_reduction:
                predicted = np.where(raw_prediction > 0.5, 1.0, 0.0)
            else:
                predicted = np.argmax(raw_prediction, axis=1, keepdims=True)
        case DatasetTask.reconstruction:
            predicted = raw_prediction = raw_prediction.reshape(correct.shape)
        case other:
            raise ValueError(f"DatasetTask {other} not defined")

    loss = loss_fn(jnp.array(raw_prediction), jnp.array(correct)).item()
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
