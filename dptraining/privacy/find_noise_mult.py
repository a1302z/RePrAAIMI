from scipy.stats import norm
from scipy.integrate import quad
import numpy as np
from scipy import optimize


def gaussian_tradeoff(alpha, mu):
    return norm.cdf(norm.isf(alpha) - mu)


def sgm_poisson(alpha, noise_multiplier, N, p):
    "Trade-off for DP SGM"
    mu_tilde = np.sqrt(np.exp(noise_multiplier ** (-2)) - 1) * np.sqrt(N) * p
    return gaussian_tradeoff(alpha, mu_tilde)


def glrt_poisson_worst_case(alpha, noise_multiplier, N, p):
    "GLRT trade-off curve for df=1"
    mu_tilde = np.sqrt(np.exp(noise_multiplier ** (-2)) - 1) * np.sqrt(N) * p
    return np.minimum(
        1 - alpha,
        norm.cdf(norm.isf(alpha / 2) - (mu_tilde / 2))
        - norm.sf(norm.isf(alpha / 2) + (mu_tilde / 2)),
    )


def get_glrt_noise_multiplier(dp_noise_multiplier, N, p):
    area_under = quad(sgm_poisson, 0, 1, args=(dp_noise_multiplier, N, p))[0]

    def auc_for_noise(noise):
        return quad(glrt_poisson_worst_case, 0, 1, args=(noise, N, p))[0] - area_under

    return optimize.root_scalar(
        auc_for_noise, bracket=(0.05, 100), method="brentq"
    ).root
