import os
import sys
from pathlib import Path
from typing import Callable, Iterable, Optional

import hydra
import numpy as np
import omegaconf
import wandb
from omegaconf import OmegaConf
from tqdm import tqdm

sys.path.insert(0, str(Path.cwd()))


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"


# pylint:disable=import-outside-toplevel


@hydra.main(
    version_base=None,
    config_path=Path.cwd() / "configs",
)
def main(
    config,
):  # pylint:disable=too-many-locals,too-many-branches,too-many-statements
    if isinstance(config, omegaconf.dictconfig.DictConfig):
        config = OmegaConf.to_container(config)
    if "cpu" in config["general"] and config["general"]["cpu"]:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # This is absolutely disgusting but necessary to disable gpu training
    import objax
    from jax import device_count, lax

    from dptraining.datasets import make_loader_from_config
    from dptraining.models import make_model_from_config
    from dptraining.optim import make_optim_from_config
    from dptraining.privacy import EpsCalculator
    from dptraining.utils import (
        ExponentialMovingAverage,
        make_loss_from_config,
        make_scheduler_from_config,
        make_stopper_from_config,
    )
    from dptraining.utils.augment import Transformation
    from dptraining.utils.training_utils import (
        create_loss_gradient,
        create_train_op,
        test,
        train,
    )

    parallel = "parallel" in config["general"] and config["general"]["parallel"]
    if parallel:
        n_devices = device_count()
        assert (
            config["hyperparams"]["batch_size"] > n_devices
            and config["hyperparams"]["batch_size"] > n_devices
        ), "Batch size must be larger than number of devices"
        if config["hyperparams"]["batch_size"] % n_devices != 0:
            config["hyperparams"]["batch_size"] -= (
                config["hyperparams"]["batch_size"] % n_devices
            )
    if config["general"]["log_wandb"]:
        run = wandb.init(
            project=config["project"],
            config=config,
            settings=wandb.Settings(start_method="thread"),
            reinit=True,
        )
    train_loader, val_loader, test_loader = make_loader_from_config(config)
    if "overfit" in config["hyperparams"] and isinstance(
        config["hyperparams"]["overfit"], int
    ):
        val_loader = train_loader
        test_loader = train_loader
    model: Callable = make_model_from_config(config)
    model_vars = model.vars()

    opt = make_optim_from_config(config, model_vars)
    if parallel:
        predict_op = objax.Parallel(
            lambda x: objax.functional.softmax(
                model(x, training=False)  # pylint:disable=not-callable
            ),
            model_vars,
            reduce=np.concatenate,
        )
    else:
        predict_op = objax.Jit(
            lambda x: objax.functional.softmax(
                model(x, training=False)  # pylint:disable=not-callable
            ),
            model_vars,
        )
    ema: Optional[ExponentialMovingAverage] = None
    if "ema" in config and config["ema"]["use_ema"]:
        ema = ExponentialMovingAverage(
            model_vars,
            config["ema"]["decay"],
            config["ema"]["use_num_updates"]
            if "use_num_updates" in config["ema"]
            else True,
        )

    if config["DP"]["disable_dp"]:
        sampling_rate, delta, sigma, final_epsilon, total_noise = 0, 0, 0, 0, 0
        batch_expansion_factor, grad_acc = 1, 1
        effective_batch_size = config["hyperparams"]["batch_size"]
    else:
        grad_acc = (
            config["DP"]["grad_acc_steps"]
            if "DP" in config and "grad_acc_steps" in config["DP"]
            else 1
        )
        delta = config["DP"]["delta"]
        eps_calc = EpsCalculator(config, train_loader)
        sigma = eps_calc.calc_noise_for_eps()

        complex_correction_factor = (
            lax.rsqrt(2.0)
            if "complex" in config["model"] and config["model"]["complex"]
            else 1.0
        )
        total_noise = (
            sigma * complex_correction_factor * config["DP"]["max_per_sample_grad_norm"]
        )

        effective_batch_size = EpsCalculator.calc_effective_batch_size(config)
        batch_expansion_factor = EpsCalculator.calc_artificial_batch_expansion_factor(
            config
        )
        sampling_rate: float = effective_batch_size / len(train_loader.dataset)
        final_epsilon = objax.privacy.dpsgd.analyze_dp(
            q=sampling_rate,
            noise_multiplier=sigma,
            steps=(len(train_loader) // batch_expansion_factor)
            * config["hyperparams"]["epochs"],
            delta=delta,
        )

        print(
            f"This training will lead to a final epsilon of {final_epsilon:.2f}"
            f" for {config['hyperparams']['epochs']} epochs"
            f" at a noise multiplier of {sigma:.2f} and a delta of {delta:2f}"
        )
        max_batches = (
            config["hyperparams"]["overfit"]
            if "overfit" in config["hyperparams"]
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

    loss_class = make_loss_from_config(config)
    loss_fn = loss_class.create_loss_fn(model_vars, model)
    loss_gv = create_loss_gradient(config, model, model_vars, loss_fn)

    augmenter = Transformation.from_dict_list(config["augmentations"])
    augment_op = augmenter.create_vectorized_transform()
    if "test_augmentations" in config:
        test_augmenter = Transformation.from_dict_list(config["test_augmentations"])
        test_aug = test_augmenter.create_vectorized_transform()
    else:
        test_aug = lambda x: x
    scheduler = make_scheduler_from_config(config)
    stopper = make_stopper_from_config(config)

    train_vars = (
        model_vars + loss_gv.vars() + opt.vars() + objax.random.DEFAULT_GENERATOR.vars()
    )
    if ema is not None:
        train_vars += ema.vars()
    train_op = create_train_op(
        train_vars,
        loss_gv,
        opt,
        augment_op,
        # complex_valued="complex" in config["model"] and config["model"]["complex"],
        grad_accumulation=grad_acc > 1,
        noise=total_noise,
        effective_batch_size=effective_batch_size,
        parallel=parallel,
    )

    epoch_time = []
    epoch_iter: Iterable
    if config["general"]["log_wandb"]:
        epoch_iter = tqdm(
            range(config["hyperparams"]["epochs"]),
            total=config["hyperparams"]["epochs"],
            desc="Epoch",
            leave=True,
        )
    else:
        epoch_iter = range(config["hyperparams"]["epochs"])
    for epoch, learning_rate in zip(epoch_iter, scheduler):
        cur_epoch_time = train(
            config,
            train_loader,
            train_op,
            learning_rate,
            train_vars,
            parallel,
            grad_acc,
            model_vars,
            ema,
        )
        if config["general"]["log_wandb"]:
            wandb.log({"epoch": epoch, "lr": learning_rate})
        else:
            print(f"Train Epoch: {epoch+1} \t took {cur_epoch_time} seconds")
        epoch_time.append(cur_epoch_time)
        if "eval_train" in config["general"] and config["general"]["eval_train"]:
            test(
                config,
                train_loader,
                predict_op,
                test_aug,
                model_vars,
                parallel,
                "train",
            )
        metric = test(
            config, val_loader, predict_op, test_aug, model_vars, parallel, "val"
        )
        scheduler.update_score(metric)
        if not config["DP"]["disable_dp"]:
            epsilon = objax.privacy.dpsgd.analyze_dp(
                q=sampling_rate,
                noise_multiplier=sigma,
                steps=((len(train_loader) // batch_expansion_factor) * (epoch + 1)),
                delta=delta,
            )
            if config["general"]["log_wandb"]:
                wandb.log({"epsilon": epsilon})
            else:
                print(f"\tPrivacy: (ε = {epsilon:.2f}, δ = {delta})")
        if stopper(metric):
            print("Early Stopping was activated -> Stopping Training")
            break

    metric = test(
        config, test_loader, predict_op, test_aug, model_vars, parallel, "test"
    )
    if config["general"]["log_wandb"]:
        run.finish()
    else:
        print("Average epoch time (all epochs): ", np.average(epoch_time))
        print("Median epoch time (all epochs): ", np.median(epoch_time))
        if len(epoch_time) > 1:
            print("Average epoch time (except first): ", np.average(epoch_time[1:]))
            print("Median epoch time (except first): ", np.median(epoch_time[1:]))
        print("Total training time (excluding evaluation): ", np.sum(epoch_time))


if __name__ == "__main__":
    main()  # pylint:disable=no-value-for-parameter
