from dptraining.utils.loss import CSELogitsSparse
from dptraining.utils.scheduler import CosineSchedule, ConstantSchedule, LinearSchedule

SUPPORTED_SCHEDULES = ["cosine", "const"]


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
    else:
        raise ValueError(
            f"{config['scheduler']['type']} scheduler not supported. "
            f"Supported Schedulers: {SUPPORTED_SCHEDULES}"
        )
    return scheduler


def make_loss_from_config(config):  # pylint:disable=unused-argument
    return CSELogitsSparse
