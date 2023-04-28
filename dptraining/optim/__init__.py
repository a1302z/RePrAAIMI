from objax import optimizer

from dptraining.config import Config, OptimName
from dptraining.optim.nadam import NAdam
from dptraining.optim.gradaccopt import AccumulateGrad


def make_optim_from_config(config: Config, model_vars):
    match config.optim.name:
        case OptimName.sgd:
            opt = optimizer.SGD
        case OptimName.momentum:
            opt = optimizer.Momentum
        case OptimName.adam:
            opt = optimizer.Adam
        case OptimName.nadam:
            opt = NAdam
        case _:
            raise ValueError(
                f"This shouldn't happen. "
                f"Optimizer {config.optim.name} is not supported."
            )
    opt = opt(model_vars, **config.optim.args)
    return opt
