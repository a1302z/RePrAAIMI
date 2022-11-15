from warnings import warn
from sklearn import metrics as sklearnmetrics
from skimage import metrics as skimagemetrics
from functools import partial
from typing import Union
from types import FunctionType

# import torchmetrics
from dptraining.utils.loss import (
    CSELogitsSparse,
    CombinedLoss,
    L1Loss,
    L2Regularization,
)
from dptraining.utils.scheduler import (
    CosineSchedule,
    ConstantSchedule,
    LinearSchedule,
    ReduceOnPlateau,
)
from dptraining.utils.earlystopping import EarlyStopping
from dptraining.utils.ema import ExponentialMovingAverage

SUPPORTED_SCHEDULES = ("cosine", "const", "reduceonplateau")


def make_scheduler_from_config(config):
    scheduler: LinearSchedule
    if config["scheduler"]["normalize_lr"]:
        config["hyperparams"]["lr"] *= config["hyperparams"]["batch_size"]
    if config["scheduler"]["type"] == "cosine":
        scheduler = CosineSchedule(
            config["hyperparams"]["lr"], config["hyperparams"]["epochs"]
        )
    elif config["scheduler"]["type"] == "const":
        scheduler = ConstantSchedule(
            config["hyperparams"]["lr"], config["hyperparams"]["epochs"]
        )
    elif config["scheduler"]["type"] == "reduceonplateau":
        scheduler = ReduceOnPlateau(
            lr=config["hyperparams"]["lr"],
            patience=config["scheduler"]["patience"],
            factor=config["scheduler"]["factor"],
            min_delta=config["scheduler"]["min_delta"],
            cumulative_delta=config["scheduler"]["cumulative_delta"]
            if "cumulative_delta" in config["scheduler"]
            else True,
            mode=config["scheduler"]["mode"]
            if "mode" in config["scheduler"]
            else "maximize",
        )
    else:
        raise ValueError(
            f"{config['scheduler']['type']} scheduler not supported. "
            f"Supported Schedulers: {SUPPORTED_SCHEDULES}"
        )
    return scheduler


SUPPORTED_LOSSES = ("cse", "l1")
SUPPORTED_REDUCTION = ("sum", "mean")


def make_loss_from_config(config):  # pylint:disable=unused-argument
    if (
        not "loss" in config
        or not "type" in config["loss"]
        or not "reduction" in config["loss"]
    ):
        raise ValueError("Loss not specified. (Needs type and reduction)")
    loss_config = config["loss"]
    assert (
        loss_config["type"] in SUPPORTED_LOSSES
    ), f"Loss {loss_config['type']} not supported. (Only {SUPPORTED_LOSSES})"
    assert (
        loss_config["reduction"] in SUPPORTED_REDUCTION
    ), f"Loss {loss_config['reduction']} not supported. (Only {SUPPORTED_REDUCTION})"
    if loss_config["type"] == "cse":
        loss_fn = CSELogitsSparse(config)
    elif loss_config["type"] == "l1":
        loss_fn = L1Loss(config)
    else:
        raise ValueError(f"Unknown loss type ({loss_config['type']})")

    if (
        "l2regularization" in config["hyperparams"]
        and config["hyperparams"]["l2regularization"] > 0
    ):
        regularization = L2Regularization(config)
        loss_fn = CombinedLoss(config, [loss_fn, regularization])

    return loss_fn


def make_stopper_from_config(config):
    if "earlystopping" in config:
        return EarlyStopping(
            patience=config["earlystopping"]["patience"],
            min_delta=config["earlystopping"]["min_delta"],
            mode=config["earlystopping"]["mode"]
            if "mode" in config["earlystopping"]
            else "maximize",
            cumulative_delta=config["earlystopping"]["cumulative_delta"]
            if "cumulative_delta" in config["earlystopping"]
            else True,
        )
    return lambda _: False


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


def make_metrics(config):
    if not "metrics" in config:
        warn("no metrics defined in config", category=UserWarning)
        return sklearnmetrics.accuracy_score, (sklearnmetrics.classification_report,)
    metric_config = config["metrics"]
    assert "main" in metric_config and isinstance(metric_config["main"], (str, dict))
    assert "logging" in metric_config and isinstance(
        metric_config["logging"], (list, dict)
    )
    all_funcs = {
        **retrieve_func_dict(sklearnmetrics),
        **retrieve_func_dict(skimagemetrics),
    }  # torchmetrics of course leads to problems -.-
    if isinstance(metric_config["main"], str):
        if metric_config["main"] == "loss":
            main_metric = ("loss", None)
        else:
            main_metric = (
                metric_config["main"],
                activate_fn(all_funcs[metric_config["main"]]),
            )
    else:
        assert len(metric_config["main"]) == 1
        fn_name = list(metric_config["main"].keys())[0]
        args = list(metric_config["main"].values())[0]
        kwargs = args if isinstance(args, dict) else {}
        args = args if isinstance(args, list) else []
        main_metric = (fn_name, activate_fn(all_funcs[fn_name], *args, **kwargs))

    logging_metrics = {}
    if isinstance(metric_config["logging"], list):
        logging_metrics.update(
            {
                func_name: activate_fn(all_funcs[func_name])
                for func_name in metric_config["logging"]
            }
        )
    else:
        for func_name, args in metric_config["logging"].items():
            # args, kwargs = seperate_args_kwargs(args) if args is not None else [], {}
            kwargs = args if isinstance(args, dict) else {}
            args = args if isinstance(args, list) else []
            logging_metrics[func_name] = activate_fn(
                all_funcs[func_name], *args, **kwargs
            )

    return main_metric, logging_metrics
