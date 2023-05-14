# -*- coding: utf-8 -*-
"""reconstruct_with_prior_public.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1xe4vkpKYB5NNTxswOGwA0PinlftFjD4O

## Description

This is a minimal implementation of a (prior aware) training data reconstruction attack against a model trained with differential privacy, as described in [Hayes et al. (2022)](https://arxiv.org/abs/2302.07225).

The adversary is given a group of model parameters $\{\theta_1, \theta_2, \ldots, \theta_T\}$ and a set of privatized gradients $\{g_1, g_2, \ldots, g_T\}$, where each $\theta_i$ and $g_i$ denotes model parameters and privatized gradients at update step $i$. The adversary also has access to a prior set of inputs $\{z_1, z_2, \ldots, z_n\}$; the model is trained on $Z\cup\{z_i\}$, where $Z$ is a set of inputs known the adversary, which we refer to as the *fixed set*, and $z_i$ is sampled randomly from the prior. The goal of the attack is to infer which $z_i$ was used in training. This is achieved by iterating over each $z_i$ and computing the sum $\sum_{k=1}^T \langle g_{\theta_{k}}(z_i), g_k\rangle$, where $g_{\theta_{k}}(z_i)$ is the model parameter gradients given input $z_i$ with respect to model parameters $\theta_{k}$. The adversary selects the $z_i$ that maximizes this sum as their guess for the sample from the prior that was included in training.

## Imports
"""

# !pip install dm-haiku
# !pip install dp-accounting
# !pip install ml-collections
# !pip install optax
# %%
# from dp_accounting import dp_event
# from dp_accounting.rdp import rdp_privacy_accountant

# import functools
# import haiku as hk
# import jax

# import optax
# from ml_collections import config_dict
import numpy as np

# import tensorflow as tf
# import tensorflow_datasets as tfds
# from matplotlib import pyplot as plt

# %%

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import hydra
from copy import deepcopy
from pathlib import Path
from omegaconf import OmegaConf
from dptraining.config import Config
from dptraining.privacy import setup_privacy
from dptraining.datasets import make_loader_from_config
from datetime import datetime
import pandas as pd
from tqdm import trange


from dptraining.config.config_store import load_config_store
import jax.numpy as jnp

from matplotlib import pyplot as plt, ticker as mtick
import seaborn as sn

sn.set_theme(
    context="notebook",
    style="white",
    font="Times New Roman",
    font_scale=1.25,
    palette="viridis",
)
sn.despine()
sn.set(rc={"figure.figsize": (12, 12)})
colors = {
    "red": "firebrick",
    "blue": "steelblue",
    "green": "forestgreen",
    "purple": "darkorchid",
    "orange": "darkorange",
    "gray": "lightslategray",
    "black": "black",
}

load_config_store()


"""## Set up DP accounting"""
# %%

# RdpAccountant = rdp_privacy_accountant.RdpAccountant


# def get_rdp_epsilon(sampling_probability, noise_multiplier, steps, delta, orders):
#     """Get privacy budget from Renyi DP."""
#     event = dp_event.PoissonSampledDpEvent(
#         sampling_probability, event=dp_event.GaussianDpEvent(noise_multiplier)
#     )
#     rdp_accountant = RdpAccountant(orders=orders)
#     rdp_accountant.compose(event, steps)
#     rdp_epsilon, opt_order = rdp_accountant.get_epsilon_and_optimal_order(delta)
#     return rdp_epsilon, opt_order


"""## Model training hyperparameters

The config below should give a privacy budget of $\epsilon\approx 20$.
"""


def reconstruction_upper_bound(pmode, q, noise_mul, steps, mc_samples=10000):
    x = np.random.normal(0.0, noise_mul, (mc_samples, steps))
    per_step_log_ratio = np.log(
        1 - q + q * (np.exp((-((x - 1.0) ** 2) + (x) ** 2) / (2 * noise_mul**2)))
    )
    log_ratio = np.sum(per_step_log_ratio, axis=1)
    log_ratio = np.sort(log_ratio)
    r = np.exp(log_ratio)
    upper_bound = max(
        0.0, 1 - (1 - pmode) * np.mean(r[: int(mc_samples * (1 - pmode))])
    )
    return min(1.0, upper_bound)


# %%
# base_config = OmegaConf.structured(Config)
# config = OmegaConf.load("configs/imagenet_dp.yaml")
# del config.defaults
# config: Config = OmegaConf.merge(base_config, config)
@hydra.main(version_base=None, config_path=Path.cwd() / "configs")
def main(
    train_config: Config,
):
    upper_bounds = []
    if "N_REPS" in train_config.keys():
        N_REPS = train_config.N_REPS
    else:
        N_REPS = 100
    if "N_SAMPLES" in train_config.keys():
        N_SAMPLES = int(train_config.N_SAMPLES)
    else:
        N_SAMPLES = int(1e3)  # 200000
    if "eps_values" in train_config.keys():
        eps_values = train_config.eps_values
    else:
        eps_values = list(range(1000, 100001, 2000))
    train_loader, _, _ = make_loader_from_config(train_config)

    save_folder = (
        Path.cwd()
        / f"rero/{str(train_config.dataset.name).split('.')[-1]}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/"
    )

    # %%
    for eps in eps_values:
        config = deepcopy(train_config)
        config.DP.epsilon = eps
        if config.DP:
            (
                grad_acc,
                accountant,
                sampling_rate,
                delta,
                sigma,
                total_noise,
                batch_expansion_factor,
                effective_batch_size,
            ) = setup_privacy(config, train_loader)
        else:
            raise ValueError(
                "This script is only designed for DP trainings. All other are critically non-private."
            )
        # %%

        # config = config_dict.ConfigDict()

        # # Noise multiplier used in DP training.
        # config.noise_multiplier = 3.108635  # @param

        # # All individual sample gradients will be clipped to have a maximum L2 norm.
        # config.l2_norm_clip = 1.0  # @param

        # # Number of epochs.
        # config.epochs = 50  # @param

        # # Learning rate.
        # config.learning_rate = 1.0  # @param

        # # Total number of examples in the prior set.
        # # Attack base rate will be 1 / config.num_in_prior.
        # num_in_prior = len(train_loader.dataset)  # @param

        # # Batch size used in training.
        # config.batch_size = 64  # @param

        # # Training data sub-sampling probability.
        # config.q = 0.53  # @param

        # Total size of the training dataset. Determined by config.batch_size and
        # config.q. For convenience of the attack, which requires some conditions on
        # batch sizes, we require config.batch_size to be divisible by config.total_num.
        total_num = len(train_loader.dataset)
        print(total_num)

        # Number of update training steps.
        steps = (
            len(train_loader) // batch_expansion_factor
        ) * config.hyperparams.epochs

        # Probability of DP failure.
        # config.DP.delta = 8e-7

        # Seed used to initialize parameters and random noise used in DP training.
        # config.general.seed = 1  # @param

        # Generate orders used in RDP accounting
        # orders = list(jnp.linspace(1.1, 10.9, 99)) + list(range(11, 64))

        # Get privacy budget for the above configuration.
        # eps, opt_order = get_rdp_epsilon(
        #     config.q,
        #     config.noise_multiplier,
        #     config.steps,
        #     config.delta,
        #     orders,
        # )
        print(f"Epsilon: {config.DP.epsilon:.10f}")

        # @title Reconstruction hyperparameters (for gradient attack)

        # attack_config = config_dict.ConfigDict()

        # If True, adversary can subtract fixed set clipped gradients from the
        # # privatised gradient.
        # deduct_fixed_set_grads = True  # @param

        # # If True, adversary can rescale gradients by the batch size.
        # # Technically, there is a mismatch between theory and practice here since in DP
        # # accounting there isn't a "fixed" batch size, rather a probability with which
        # # an example is included in a batch.
        # rescale_by_batch_size = True  # @param

        # @title Get prior and fixed data

        # %%

        """## Reconstruction attack set up"""

        # %%
        p = total_num
        variance = []
        for _ in trange(N_REPS, leave=False, desc="repeating experiment"):
            rub = reconstruction_upper_bound(
                1 / p, sampling_rate, total_noise, steps, mc_samples=N_SAMPLES
            )
            variance.append(100.0 * rub)
        # %%
        print(
            f"Reconstruction upper bound for {p} samples: {np.mean(variance).item():.2f}%"
            f"\nLowest: {np.min(variance).item():.2f}%\t Highest: {np.max(variance).item():.2f}%"
        )

        upper_bounds.append(variance)

    upper_bounds = np.array(upper_bounds)
    means = upper_bounds.mean(axis=1)
    stds = upper_bounds.std(axis=1)
    print(upper_bounds.max(axis=1))
    err = np.stack(
        [means - upper_bounds.min(axis=1), upper_bounds.max(axis=1) - means], axis=0  #
    )
    # plt.errorbar(
    #     eps_values,
    #     means,
    #     yerr=err,
    #     color="blue",
    # )
    # plt.errorbar(eps_values, means, yerr=stds, color="blue", label="success mean/StD")
    plt.plot(
        [int(e) for e in eps_values],
        means,
        color="blue",
        label="reconstruction success",
    )
    # plt.fill_between(
    #     eps_values,
    #     means + err[1],
    #     means - err[0],
    #     # means + stds,
    #     # means - stds,
    #     color="blue",
    #     alpha=0.25,
    #     label="Success range",
    # )
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.xlabel("epsilon")
    plt.ylabel("Prior aware upper bound")
    plt.legend(loc="best")

    if not save_folder.is_dir():
        save_folder.mkdir(parents=True)
    plt.savefig(
        str(save_folder / "priorawareupper_bounds.svg"),
        bbox_inches="tight",
    )
    plt.savefig(
        (save_folder / "priorawareupper_bounds.png"),
        bbox_inches="tight",
        dpi=600,
    )
    result_dict = {}
    for eps, mv, std, lower, upper, all in zip(
        eps_values,
        means,
        stds,
        upper_bounds.min(axis=1),
        upper_bounds.max(axis=1),
        upper_bounds,
    ):
        print(f"{eps}: {mv:.2f}")
        result_dict[eps] = {
            "mean": mv,
            "std": std,
            "lower": lower,
            "upper": upper,
            **{i: res for i, res in enumerate(all)},
        }
    df = pd.DataFrame.from_dict(result_dict)
    df.to_csv(str(save_folder / "results.csv"), index=False)


if __name__ == "__main__":
    main()
