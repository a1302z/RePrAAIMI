from enum import Enum
from functools import partial

from jax.lax import rsqrt
from scipy.optimize import minimize_scalar

from dptraining.config import Config
from dptraining.privacy.find_noise_mult import new_noise_multi


from opacus.accountants.utils import get_noise_multiplier


class NoiseCalcMode(Enum):
    SIGMA = 1
    EPOCHS = 2


def epsilon_opt_func_opacus(
    *args, accountant, epsilon=None, opt_keyword=None, **kwargs
):
    kwargs = {opt_keyword: args[0], **kwargs}
    accountant.history = [(kwargs["sigma"], kwargs["sampling_rate"], kwargs["steps"])]
    return abs(epsilon - accountant.get_epsilon(delta=kwargs["delta"]))


class EpsCalculator:
    def __init__(self, config: Config, train_loader) -> None:
        self._config = config
        self._eps = config.DP.epsilon
        self._delta = config.DP.delta
        if (
            config.DP.sigma is not None
            # and config.DP.grad_acc_steps is not None
            and config.hyperparams.epochs is None
        ):
            self._mode = NoiseCalcMode.EPOCHS
            self._sigma = config.DP.sigma
            effective_bs = EpsCalculator.calc_effective_batch_size(config)
            self._sampling_rate = effective_bs / len(train_loader.dataset)
            self._eff_batch_size = len(train_loader) // EpsCalculator.get_grad_acc(
                config
            )
        elif (
            # "grad_acc_steps" in config.DP and
            config.hyperparams.epochs is not None
            and config.DP.sigma is None
        ):
            self._mode = NoiseCalcMode.SIGMA
            effective_bs = EpsCalculator.calc_effective_batch_size(config)
            self._sampling_rate = effective_bs / len(train_loader.dataset)
            self._steps = (
                len(train_loader) // EpsCalculator.get_grad_acc(config)
            ) * config.hyperparams.epochs
        else:
            raise ValueError(
                "You need to specify either one of sigma or epochs in the config"
            )

    def fill_config(self, accountant, tol=1e-5) -> float:
        if self._mode == NoiseCalcMode.SIGMA:
            self._config.DP.sigma = get_noise_multiplier(
                target_epsilon=self._eps,
                target_delta=self._delta,
                sample_rate=self.sampling_rate,
                steps=self._steps,
                accountant=self._config.DP.mechanism,
                epsilon_tolerance=tol,
            )
        elif self._mode == NoiseCalcMode.EPOCHS:
            result = minimize_scalar(
                partial(
                    epsilon_opt_func_opacus,
                    accountant=accountant,
                    epsilon=self._eps,
                    sigma=self._sigma,
                    sampling_rate=self._sampling_rate,
                    delta=self._delta,
                    opt_keyword="steps",
                ),
                tol=tol,
            )
            self._steps = result.x
            self._config.hyperparams.epochs = int(self._steps // self._eff_batch_size)
        else:
            raise RuntimeError("Mode not implemented")

    @staticmethod
    def get_grad_acc(config: Config):
        # devices = device_count() if config.general.parallel else 1
        return config.hyperparams.grad_acc_steps

    @staticmethod
    def calc_effective_batch_size(config: Config):
        effective_batch_size = (
            EpsCalculator.get_grad_acc(config) * config.hyperparams.batch_size
        )
        return effective_batch_size

    def adapt_sigma(self):
        rsqrt2_correction_factor = (
            rsqrt(2.0) if self._config.DP.rsqrt_noise_adapt else 1.0
        )
        adapted_sigma = (
            new_noise_multi(
                self._config.DP.sigma,
                self.steps,
                self.sampling_rate,
                mode="complex" if self._config.model.complex else "real",
            )
            if self._config.DP.glrt_assumption
            else self._config.DP.sigma
        )
        total_noise = (
            adapted_sigma
            * rsqrt2_correction_factor
            * self._config.DP.max_per_sample_grad_norm
        )
        return total_noise, adapted_sigma

    @property
    def steps(self):
        return self._steps

    @property
    def sampling_rate(self):
        return self._sampling_rate
