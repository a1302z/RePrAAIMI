from dptraining.utils.loss import CSELogitsSparse
from dptraining.utils.scheduler import CosineSchedule, ConstantSchedule, LinearSchedule

SUPPORTED_SCHEDULES = ["cos", "const"]


def make_scheduler_from_args(args):
    scheduler: LinearSchedule
    if args.lr_schedule == "cos":
        scheduler = CosineSchedule(args.lr, args.epochs)
    elif args.lr_schedule == "const":
        scheduler = ConstantSchedule(args.lr, args.epochs)
    else:
        raise ValueError(
            f"{args.lr_schedule} scheduler not supported. "
            f"Supported Schedulers: {SUPPORTED_SCHEDULES}"
        )
    return scheduler


def make_loss_from_args(args):
    return CSELogitsSparse
