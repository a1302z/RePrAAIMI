from warnings import warn
from sklearn import metrics as sklearnmetrics
from skimage import metrics as skimagemetrics
from functools import partial
from types import FunctionType
from numpy import array

from omegaconf import OmegaConf, DictConfig

# import torchmetrics
from dptraining.utils.loss import (
    CrossEntropy,
    CombinedLoss,
    L1Loss,
    L2Regularization,
    DiceLoss,
    f_score,
)
from dptraining.config import Config, SchedulerType, LossType
from dptraining.utils.scheduler import (
    CosineSchedule,
    ConstantSchedule,
    LinearSchedule,
    ReduceOnPlateau,
    ManualSchedule,
)
from dptraining.utils.earlystopping import EarlyStopping
from dptraining.utils.ema import ExponentialMovingAverage


def make_scheduler_from_config(config: Config):
    scheduler: LinearSchedule
    if config.scheduler.normalize_lr:
        config.hyperparams.lr *= config.hyperparams.batch_size
    match config.scheduler.type:
        case SchedulerType.const:
            scheduler = ConstantSchedule(
                config.hyperparams.lr, config.hyperparams.epochs
            )
        case SchedulerType.cosine:
            scheduler = CosineSchedule(config.hyperparams.lr, config.hyperparams.epochs)
        case SchedulerType.reduceonplateau:
            scheduler = ReduceOnPlateau(
                base_lr=config.hyperparams.lr,
                patience=config.scheduler.patience,
                factor=config.scheduler.factor,
                min_delta=config.scheduler.min_delta,
                cumulative_delta=config.scheduler.cumulative_delta,
                mode=config.scheduler.mode,
            )
        case SchedulerType.manual:
            scheduler = ManualSchedule(
                base_lr=config.hyperparams.lr,
                lr_list=config.scheduler.lr_list,
                epochs=config.scheduler.epoch_triggers,
            )
        case _:
            raise ValueError(f"{config.scheduler.type} scheduler not supported.")
    return scheduler


def make_loss_from_config(config: Config):  # pylint:disable=unused-argument
    match config.loss.type:
        case LossType.cse:
            loss_fn = CrossEntropy(config)
        case LossType.l1:
            loss_fn = L1Loss(config)
        case LossType.dice:
            loss_fn = DiceLoss(config)
        case other:
            raise ValueError(f"{other} loss not supported")
    if config.hyperparams.l2regularization and config.hyperparams.l2regularization > 0:
        regularization = L2Regularization(config)
        loss_fn = CombinedLoss(config, [loss_fn, regularization])

    return loss_fn


def make_stopper_from_config(config: Config):
    if config.earlystopping is not None:
        return EarlyStopping(
            patience=config.earlystopping.patience,
            min_delta=config.earlystopping.min_delta,
            mode=config.earlystopping.mode,
            cumulative_delta=config.earlystopping.cumulative_delta,
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
                class_weights=array(config.loss.class_weights)
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
