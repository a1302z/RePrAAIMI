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
    MSELoss,
    L2Regularization,
    DiceLoss,
    f_score,
    calc_class_weights,
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
from dptraining.utils.metrics import make_metrics


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
        case LossType.mse:
            loss_fn = MSELoss(config)
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
