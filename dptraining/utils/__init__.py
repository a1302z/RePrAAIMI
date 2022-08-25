from dptraining.utils.loss import CSELogitsSparse, CombinedLoss, L2Regularization
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


SUPPORTED_LOSSES = ("cse",)
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
