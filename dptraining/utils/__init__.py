from dptraining.config import Config, SchedulerType, LossType
from dptraining.utils.loss import CSELogitsSparse, CombinedLoss, L2Regularization
from dptraining.utils.scheduler import (
    CosineSchedule,
    ConstantSchedule,
    LinearSchedule,
    ReduceOnPlateau,
)
from dptraining.utils.earlystopping import EarlyStopping
from dptraining.utils.ema import ExponentialMovingAverage


def make_scheduler_from_config(config: Config):
    scheduler: LinearSchedule
    if config.scheduler.normalize_lr:
        config.hyperparams.lr *= config.hyperparams.batch_size
    match config.scheduler.type:
        case SchedulerType.cosine:
            scheduler = CosineSchedule(config.hyperparams.lr, config.hyperparams.epochs)
        case SchedulerType.const:
            scheduler = ConstantSchedule(
                config.hyperparams.lr, config.hyperparams.epochs
            )
        case SchedulerType.reduceonplateau:
            scheduler = ReduceOnPlateau(
                base_lr=config.hyperparams.lr,
                patience=config.scheduler.patience,
                factor=config.scheduler.factor,
                min_delta=config.scheduler.min_delta,
                cumulative_delta=config.scheduler.cumulative_delta,
                mode=config.scheduler.mode,
            )
        case _:
            raise ValueError(f"{config.scheduler.type} scheduler not supported.")
    return scheduler


def make_loss_from_config(config: Config):  # pylint:disable=unused-argument
    match config.loss.type:
        case LossType.cse:
            loss_fn = CSELogitsSparse(config)
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
