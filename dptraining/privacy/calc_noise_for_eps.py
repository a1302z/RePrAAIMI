from enum import Enum
from functools import partial
from objax.privacy.dpsgd import analyze_dp
from scipy.optimize import minimize_scalar
from jax.lax import rsqrt

from dptraining.privacy.find_noise_mult import new_noise_multi


class NoiseCalcMode(Enum):
    SIGMA = 1
    EPOCHS = 2


def epsilon_opt_func(*args, epsilon=None, opt_keyword=None, **kwargs):
    kwargs = {opt_keyword: args[0], **kwargs}
    calc_eps = analyze_dp(**kwargs)
    return abs(epsilon - calc_eps)


class EpsCalculator:
    def __init__(self, config, train_loader) -> None:
        assert "epsilon" in config["DP"]
        self._config = config
        self._eps = config["DP"]["epsilon"]
        self._delta = config["DP"]["delta"]
        if (
            "sigma" in config["DP"]
            # and "grad_acc_steps" in config["DP"]
            and not "epochs" in config["hyperparams"]
        ):
            self._mode = NoiseCalcMode.EPOCHS
            self._sigma = config["DP"]["sigma"]
            effective_bs = EpsCalculator.calc_effective_batch_size(config)
            self._sampling_rate = effective_bs / len(train_loader.dataset)
            self._eff_batch_size = len(train_loader) // EpsCalculator.get_grad_acc(
                config
            )
        elif (
            # "grad_acc_steps" in config["DP"] and
            "epochs" in config["hyperparams"]
            and not "sigma" in config["DP"]
        ):
            self._mode = NoiseCalcMode.SIGMA
            effective_bs = EpsCalculator.calc_effective_batch_size(config)
            self._sampling_rate = effective_bs / len(train_loader.dataset)
            self._steps = (
                len(train_loader) // EpsCalculator.get_grad_acc(config)
            ) * config["hyperparams"]["epochs"]
        else:
            raise ValueError(
                "You need to specify either one of sigma or epochs in the config"
            )

    def fill_config(self, tol=1e-5) -> float:
        if self._mode == NoiseCalcMode.SIGMA:
            result = minimize_scalar(
                partial(
                    epsilon_opt_func,
                    epsilon=self._eps,
                    q=self._sampling_rate,
                    steps=self._steps,
                    delta=self._delta,
                    opt_keyword="noise_multiplier",
                ),
                tol=tol,
            )
            self._config["DP"]["sigma"] = result.x
        elif self._mode == NoiseCalcMode.EPOCHS:
            result = minimize_scalar(
                partial(
                    epsilon_opt_func,
                    epsilon=self._eps,
                    noise_multiplier=self._sigma,
                    q=self._sampling_rate,
                    delta=self._delta,
                    opt_keyword="steps",
                ),
                tol=tol,
            )
            self._steps = result.x
            self._config["hyperparams"]["epochs"] = int(
                self._steps // self._eff_batch_size
            )
        else:
            raise RuntimeError("Mode not implemented")

    @staticmethod
    def get_grad_acc(config):
        grad_acc = (
            config["DP"]["grad_acc_steps"]
            if "DP" in config and "grad_acc_steps" in config["DP"]
            else 1
        )
        # devices = device_count() if config["general"]["parallel"] else 1
        return grad_acc

    @staticmethod
    def calc_effective_batch_size(config):
        effective_batch_size = (
            EpsCalculator.get_grad_acc(config) * config["hyperparams"]["batch_size"]
        )
        return effective_batch_size

    def adapt_sigma(self):
        rsqrt2_correction_factor = (
            rsqrt(2.0)
            if "rsqrt_noise_adapt" in self._config["DP"]
            and self._config["DP"]["rsqrt_noise_adapt"]
            else 1.0
        )
        adapted_sigma = (
            new_noise_multi(
                self._config["DP"]["sigma"],
                self.steps,
                self.sampling_rate,
                mode="complex"
                if "complex" in self._config["model"]
                and self._config["model"]["complex"]
                else "real",
            )
            if "glrt_assumption" in self._config["DP"]
            and self._config["DP"]["glrt_assumption"]
            else self._config["DP"]["sigma"]
        )
        total_noise = (
            adapted_sigma
            * rsqrt2_correction_factor
            * self._config["DP"]["max_per_sample_grad_norm"]
        )
        return total_noise, adapted_sigma

    @property
    def steps(self):
        return self._steps

    @property
    def sampling_rate(self):
        return self._sampling_rate
