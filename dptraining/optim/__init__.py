from objax import optimizer
from dptraining.optim.nadam import NAdam

SUPPORTED_OPTIMS = ("momentum", "adam", "nadam")


def make_optim_from_config(config, model_vars):
    if config["optim"]["name"] not in SUPPORTED_OPTIMS:
        raise ValueError(
            f"{config['optim']['name']} not supported yet. "
            f"Currently supported normalizations: {SUPPORTED_OPTIMS}"
        )
    if config["optim"]["name"] == "momentum":
        opt = optimizer.Momentum
    elif config["optim"]["name"] == "adam":
        opt = optimizer.Adam
    elif config["optim"]["name"] == "nadam":
        opt = NAdam
    else:
        raise ValueError(
            f"This shouldn't happen. "
            f"{SUPPORTED_OPTIMS} includes not supported optimizer."
        )
    opt_args = {k: v for k, v in config["optim"].items() if not k == "name"}
    opt = opt(model_vars, **opt_args)
    return opt
