import os
import sys
from pathlib import Path
from typing import Callable, Iterable, Optional

import hydra
import numpy as np
import wandb
from omegaconf import OmegaConf
from tqdm import tqdm
from datetime import datetime

sys.path.insert(0, str(Path.cwd()))

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
os.environ["XLA_FLAGS"] = "--xla_gpu_force_compilation_parallelism=1"

from dptraining.config import Config, DatasetTask
from dptraining.config.config_store import load_config_store

# pylint:disable=import-outside-toplevel

load_config_store()


@hydra.main(version_base=None, config_path=Path.cwd() / "configs")
def main(
    config: Config,
):  # pylint:disable=too-many-locals,too-many-branches,too-many-statements
    if config.general.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # This is absolutely disgusting but necessary to disable gpu training
    import objax
    from jax import device_count
    from torch.random import manual_seed as torch_manual_seed

    from dptraining.datasets import make_loader_from_config
    from dptraining.models import make_model_from_config
    from dptraining.optim import make_optim_from_config
    from dptraining.privacy import EpsCalculator
    from dptraining.utils import (
        ExponentialMovingAverage,
        make_loss_from_config,
        make_scheduler_from_config,
        make_stopper_from_config,
        make_metrics,
    )
    from dptraining.utils.augment import Transformation
    from dptraining.utils.training_utils import (
        create_loss_gradient,
        create_train_op,
        test,
        train,
    )

    np.random.seed(config.general.seed)
    objax.random.DEFAULT_GENERATOR.seed(config.general.seed)
    torch_manual_seed(config.general.seed)

    if config.general.parallel:
        n_devices = device_count()
        assert (
            config.hyperparams.batch_size > n_devices
        ), "Batch size must be larger than number of devices"
        if config.hyperparams.batch_size % n_devices != 0:
            config.hyperparams.batch_size -= config.hyperparams.batch_size % n_devices
    if config.general.log_wandb:
        config_dict = OmegaConf.to_container(config)
        run = wandb.init(
            project=config_dict["project"],
            config=config_dict,
            settings=wandb.Settings(start_method="thread"),
            reinit=True,
        )
    checkpoint = None
    train_loader, val_loader, test_loader = make_loader_from_config(config)
    if config.hyperparams.overfit is not None:
        val_loader = train_loader
        test_loader = train_loader
    model: Callable = make_model_from_config(config)
    if config.general.use_pretrained_model is not None:
        print(f"Loading model from {config.general.use_pretrained_model}")
        objax.io.load_var_collection(config.general.use_pretrained_model, model.vars())
    model_vars = model.vars()

    identifying_model_str = ""
    if config.general.make_save_str_unique:
        if config.general.log_wandb:
            identifying_model_str += (
                f"_{wandb.run.name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
            )
        else:
            identifying_model_str += f"_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    if config.checkpoint:
        config.checkpoint.logdir += identifying_model_str
        checkpoint = objax.io.Checkpoint(**OmegaConf.to_container(config.checkpoint))
        checkpoint = None

    ema: Optional[ExponentialMovingAverage] = None
    if config.ema.use_ema:
        ema = ExponentialMovingAverage(
            model_vars, config.ema.decay, update_every=config.ema.update_every
        )
    opt = make_optim_from_config(config, model_vars)

    predict_lambda = (
        lambda x: objax.functional.softmax(  # pylint:disable=unnecessary-lambda-assignment
            model(x, training=False)
        )
        if config.dataset.task == DatasetTask.classification
        else model(x, training=False)
    )

    predict_op_parallel = objax.Parallel(
        predict_lambda, model_vars, reduce=np.concatenate
    )
    predict_op_jit = objax.Jit(
        predict_lambda, model_vars  # pylint:disable=not-callable
    )

    if not config.DP or config.DP.clip_only:
        sampling_rate, delta, sigma, final_epsilon, total_noise = 0, 0, 0, 0, 0
        batch_expansion_factor, grad_acc = 1, 1
        effective_batch_size = config.hyperparams.batch_size
        print("Careful! Not training with DP")
    else:
        delta = config.DP.delta
        eps_calc = EpsCalculator(config, train_loader)
        eps_calc.fill_config()
        sigma = config.DP.sigma
        grad_acc = config.DP.grad_acc_steps

        total_noise, adapted_sigma = eps_calc.adapt_sigma()

        effective_batch_size = EpsCalculator.calc_effective_batch_size(config)
        batch_expansion_factor = EpsCalculator.get_grad_acc(config)
        sampling_rate: float = effective_batch_size / len(train_loader.dataset)
        final_epsilon = objax.privacy.dpsgd.analyze_dp(
            q=sampling_rate,
            noise_multiplier=sigma,
            steps=(len(train_loader) // batch_expansion_factor)
            * config.hyperparams.epochs,
            delta=delta,
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

    loss_class = make_loss_from_config(config)
    train_loss_fn = loss_class.create_train_loss_fn(model_vars, model)
    test_loss_fn = loss_class.create_test_loss_fn()
    loss_gv = create_loss_gradient(config, model_vars, train_loss_fn)

    metric_fns = make_metrics(config)

    augmenter = Transformation.from_dict_list(
        OmegaConf.to_container(config.augmentations)
    )
    n_augmentations = augmenter.get_n_augmentations()
    augment_op = augmenter.create_vectorized_transform()
    if n_augmentations > 1:
        print(f"Augmentation multiplicity of {n_augmentations}")
    if config.label_augmentations:
        label_augmenter = Transformation.from_dict_list(
            OmegaConf.to_container(config.label_augmentations)
        )
        label_augment_op = label_augmenter.create_vectorized_transform()
    else:
        label_augment_op = lambda _: _  # pylint:disable=unnecessary-lambda-assignment
    # augment_op = augment_op.create_vectorized_transform()
    if config.test_augmentations:
        test_augmenter = Transformation.from_dict_list(
            OmegaConf.to_container(config.test_augmentations)
        )
        test_aug = test_augmenter.create_vectorized_transform()
    else:
        test_aug = lambda x: x  # pylint:disable=unnecessary-lambda-assignment
    if config.test_label_augmentations:
        test_label_augmenter = Transformation.from_dict_list(
            OmegaConf.to_container(config.test_label_augmentations)
        )
        test_label_aug = test_label_augmenter.create_vectorized_transform()
    else:
        test_label_aug = lambda _: _  # pylint:disable=unnecessary-lambda-assignment
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
        label_augment_op,
        grad_accumulation=grad_acc > 1,
        noise=total_noise,
        effective_batch_size=effective_batch_size,
        n_augmentations=n_augmentations,
        parallel=config.general.parallel,
        ema=ema,
    )

    epoch_time = []
    epoch_iter: Iterable
    if config.general.log_wandb:
        epoch_iter = tqdm(
            range(config.hyperparams.epochs),
            total=config.hyperparams.epochs,
            desc="Epoch",
            leave=True,
        )
    else:
        epoch_iter = range(config.hyperparams.epochs)
    for epoch, learning_rate in zip(epoch_iter, scheduler):
        cur_epoch_time = train(
            config=config,
            train_loader=train_loader,
            train_op=train_op,
            learning_rate=learning_rate,
            train_vars=train_vars,
            val_loader=val_loader,
            predict_op=predict_op_parallel if config.general.parallel else predict_op_jit,
            metrics=metric_fns,
            eval_loss_fn=test_loss_fn,
            test_aug=test_aug,
            test_label_aug=test_label_aug,
            parallel=config.general.parallel,
            grad_acc=grad_acc,
            eval_every_n=config.hyperparams.eval_every_n,
        )
        if config.general.log_wandb:
            wandb.log({"epoch": epoch, "lr": learning_rate})
        else:
            print(f"Train Epoch: {epoch+1} \t took {cur_epoch_time} seconds")
        epoch_time.append(cur_epoch_time)
        if config.general.eval_train:
            test(
                config,
                train_loader,
                predict_op_parallel if config.general.parallel else predict_op_jit,
                test_aug,
                test_label_aug,
                model_vars,
                config.general.parallel,
                "train",
                metrics=metric_fns,
                loss_fn=test_loss_fn,
            )
        if val_loader is not None:
            metric = test(
                config,
                val_loader,
                predict_op_parallel if config.general.parallel else predict_op_jit,
                test_aug,
                test_label_aug,
                model_vars,
                config.general.parallel,
                "val",
                metrics=metric_fns,
                loss_fn=test_loss_fn,
            )
            scheduler.update_score(metric)
        else:
            metric = None
        if config.DP:
            epsilon = objax.privacy.dpsgd.analyze_dp(
                q=sampling_rate,
                noise_multiplier=sigma,
                steps=((len(train_loader) // batch_expansion_factor) * (epoch + 1)),
                delta=delta,
            )
            if config.general.log_wandb:
                wandb.log({"epsilon": epsilon})
            else:
                print(f"\tPrivacy: (ε = {epsilon:.2f}, δ = {delta})")
        if checkpoint is not None:
            checkpoint.save(model.vars(), idx=epoch)
        if metric is not None and stopper(metric):
            print("Early Stopping was activated -> Stopping Training")
            break

    # never do test parallel as this could emit some test samples and thus distort results
    metric = test(
        config,
        test_loader,
        predict_op_jit,
        test_aug,
        test_label_aug,
        model_vars,
        False,
        "test",
        metrics=metric_fns,
        loss_fn=test_loss_fn,
    )
    if config.general.save_path:
        save_path = (
            config.general.save_path.parent / config.general.save_path.stem
            + identifying_model_str
            + config.general.save_path.suffix
        )
        print(f"Saving model to {config.general.save_path}")
        objax.io.save_var_collection(save_path, model.vars())
    if config.general.log_wandb:
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
