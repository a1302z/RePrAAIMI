from functools import partial
from objax.privacy.dpsgd import analyze_dp
from scipy.optimize import minimize_scalar


def optimisation_f(sigma, epsilon, sampling_rate, steps, delta):
    calc_eps = analyze_dp(
        q=sampling_rate,
        noise_multiplier=sigma,
        steps=steps,
        delta=delta,
    )
    return abs(epsilon - calc_eps)


class EpsCalculator:
    def __init__(self, config, train_loader) -> None:
        self._eps = config["DP"]["epsilon"]
        self._sampling_rate = config["hyperparams"]["batch_size"] / len(
            train_loader.dataset
        )
        self._delta = config["DP"]["delta"]
        self._steps = len(train_loader) * config["hyperparams"]["epochs"]

    def calc_noise_for_eps(self, tol=1e-3) -> float:
        result = minimize_scalar(
            partial(
                optimisation_f,
                epsilon=self._eps,
                sampling_rate=self._sampling_rate,
                steps=self._steps,
                delta=self._delta,
            ),
            tol=tol,
        )
        return result.x
