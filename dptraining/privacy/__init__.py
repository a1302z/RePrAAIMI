from typing import Optional
from opacus.accountants import create_accountant, IAccountant

from dptraining.privacy.calc_noise_for_eps import EpsCalculator
from dptraining.privacy.grad_clipper import ClipAndAccumulateGrads
from dptraining.config import Config


def analyse_epsilon(
    accountant: IAccountant,
    steps: int,
    sigma: float,
    sampling_rate: float,
    delta: float,
):
    accountant.history = [(sigma, sampling_rate, steps)]
    return accountant.get_epsilon(delta=delta)


def setup_privacy(config: Config, train_loader):
    grad_acc = config.hyperparams.grad_acc_steps
    accountant: Optional[IAccountant] = None
    if not config.DP:
        sampling_rate, delta, sigma, final_epsilon, total_noise = 0, 0, 0, 0, 0
        batch_expansion_factor = 1
        effective_batch_size = (
            config.hyperparams.batch_size * config.hyperparams.grad_acc_steps
        )
    else:
        accountant = create_accountant(mechanism=config.DP.mechanism)
        delta = config.DP.delta
        eps_calc = EpsCalculator(config, train_loader)
        try:
            eps_calc.fill_config(accountant=accountant, tol=config.DP.eps_tol)
        except ValueError as e:
            if (
                config.DP.mechanism == "gdp"
                and str(e) == "f(a) and f(b) must have different signs"
            ):
                raise ValueError(
                    f"Value error probably due to high epsilon while using GDP."
                )
            else:
                raise e
        sigma = config.DP.sigma

        total_noise, adapted_sigma = eps_calc.adapt_sigma()

        effective_batch_size = EpsCalculator.calc_effective_batch_size(config)
        batch_expansion_factor = EpsCalculator.get_grad_acc(config)
        sampling_rate: float = effective_batch_size / len(train_loader.dataset)

        final_epsilon = analyse_epsilon(
            accountant,
            (len(train_loader) // batch_expansion_factor) * config.hyperparams.epochs,
            sigma,
            sampling_rate,
            delta,
        )

        print(
            f"This training will lead to a final epsilon of {final_epsilon:.2f}"
            f" for {config.hyperparams.epochs} epochs"
            f" at a noise multiplier of {sigma:5f} and a delta of {delta}"
        )
        if config.DP.glrt_assumption:
            print(f"Effective sigma due to glrt assumption is {adapted_sigma}")
        max_batches = (
            config.hyperparams.overfit
            if config.hyperparams.overfit is not None
            else len(train_loader)
        )
        if max_batches % grad_acc != 0:
            reduced_max_batches = max_batches - (max_batches % grad_acc)
            assert reduced_max_batches > 0, (
                f"The number of batches ({max_batches}) cannot be smaller "
                f"than the number of gradient accumulation steps ({grad_acc})"
            )
            print(
                f"The number of batches per epoch will be reduced to "
                f"{reduced_max_batches} as it's the highest number of "
                f"batches ({max_batches}) which is evenly divisble by "
                f"the number of gradient accumulation steps ({grad_acc})"
            )

    return (
        grad_acc,
        accountant,
        sampling_rate,
        delta,
        sigma,
        total_noise,
        batch_expansion_factor,
        effective_batch_size,
    )
