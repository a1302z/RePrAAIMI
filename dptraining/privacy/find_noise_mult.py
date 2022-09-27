from scipy.stats import norm, chi2, ncx2
from scipy.integrate import quad
import numpy as np
from scipy import optimize

# pylint:disable=all #this is schorschis stuff, not gonna lint it

q = norm().sf
qinv = norm().isf


def compute_mu_uniform(steps, noise_multiplier, sample_rate):
    c = sample_rate * np.sqrt(steps)
    return (
        np.sqrt(2)
        * c
        * np.sqrt(
            np.exp(noise_multiplier ** (-2)) * norm.cdf(1.5 / noise_multiplier)
            + 3 * norm.cdf(-0.5 / noise_multiplier)
            - 2
        )
    )


def real_roc(x, Delta, sigma):
    return norm().sf(norm().isf(x) - Delta / sigma)


def glrt_real_roc(x, Delta, sigma):
    return q(qinv(x / 2) - Delta / sigma) + q(qinv(x / 2) + Delta / sigma)


def glrt_complex_roc(x, Delta, sigma):
    return ncx2(df=2, scale=sigma**2 / 2, nc=2 * Delta**2 / sigma**2).sf(
        chi2(df=2, scale=sigma**2 / 2).isf(x)
    )


def real_roc_auc(Delta, sigma):
    return quad(real_roc, 0, 1, args=(Delta, sigma))[0]


def glrt_real_roc_auc(Delta, sigma):
    return quad(glrt_real_roc, 0, 1, args=(Delta, sigma))[0]


def glrt_complex_roc_auc(Delta, sigma):
    return quad(glrt_complex_roc, 0, 1, args=(Delta, sigma))[0]


auc_funcs = {"real": glrt_real_roc_auc, "complex": glrt_complex_roc_auc}


def new_noise_multi(old_noise_multi, steps, sample_rate, mode="complex"):
    assert mode in ("complex", "real")
    # find effective mu for the given parameters and uniform sampling
    mu = compute_mu_uniform(steps, old_noise_multi, sample_rate)
    # find corresponding AUC
    auc = real_roc_auc(mu, 1)
    # find new mu for the same AUC
    def roc_of_mu(mu):
        return auc_funcs[mode](mu, 1) - auc

    new_mu = optimize.root_scalar(roc_of_mu, bracket=(0.1, 6), method="brentq")
    # find new multiplier for the mu
    def mu_of_noise(noise):
        return compute_mu_uniform(steps, noise, sample_rate) - new_mu.root

    new_multi = optimize.root_scalar(
        mu_of_noise, bracket=(0.1, 6), method="brentq"
    ).root
    return new_multi
