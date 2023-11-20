import os
import sys
from pathlib import Path
from functools import partial

import hydra
import numpy as np
import wandb
from omegaconf import OmegaConf

sys.path.insert(0, str(Path.cwd()))

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

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

    from dptraining.datasets import make_loader_from_config
    from dptraining.models import make_model_from_config, setup_pretrained_model
    from dptraining.privacy import setup_privacy
    from dptraining.utils import (
        make_loss_from_config,
        make_scheduler_from_config,
        make_stopper_from_config,
        make_metrics,
        calc_class_weights,
    )
    from dptraining.utils.augment import make_augs
    from dptraining.utils.training_utils import (
        test,
        make_train_op,
        train_loop,
        fix_seeds,
    )
    from dptraining.utils.misc import get_num_params, make_unique_str

    fix_seeds(config)

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
            config=config_dict,
            settings=wandb.Settings(start_method="thread"),
            reinit=True,
            **config_dict["wandb"],
        )
    train_loader, val_loader, test_loader = make_loader_from_config(config)
    if config.hyperparams.overfit is not None:
        val_loader = train_loader
        test_loader = train_loader
    model = make_model_from_config(config)
    model_vars, unfreeze_schedule, _ = setup_pretrained_model(config, model)
    n_train_vars_total: int = get_num_params(model.vars())
    if config.general.log_wandb:
        wandb.log({"total_model_vars": n_train_vars_total})
    elif config.general.print_info:
        print(f"Total model params: {n_train_vars_total:,}")
    n_train_vars_cur: int = get_num_params(model_vars)
    if config.general.log_wandb:
        wandb.log({"num_trained_vars": n_train_vars_cur})
    elif config.general.print_info:
        print(f"Num trained params: {n_train_vars_cur:,}")

    identifying_model_str = make_unique_str(config)

    if config.checkpoint:
        config.checkpoint.logdir += identifying_model_str
        checkpoint = objax.io.Checkpoint(**OmegaConf.to_container(config.checkpoint))
    else:
        checkpoint = None

    if config.dataset.task in (DatasetTask.classification, DatasetTask.segmentation):
        if config.loss.binary_loss:
            predict_lambda = lambda x: objax.functional.sigmoid(  # pylint:disable=unnecessary-lambda-assignment
                model(x, training=False)
            )
        else:
            predict_lambda = lambda x: objax.functional.softmax(  # pylint:disable=unnecessary-lambda-assignment
                model(x, training=False),
                axis=1 if config.dataset.task == DatasetTask.segmentation else -1,
            )
    else:
        predict_lambda = partial(model, training=False)

    predict_op_parallel = objax.Parallel(
        predict_lambda, model.vars(), reduce=np.concatenate
    )
    predict_op_jit = objax.Jit(
        predict_lambda, model.vars()  # pylint:disable=not-callable
    )

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

    metric_fns = make_metrics(config)

    (
        n_augmentations,
        augment_op,
        label_augment_op,
        test_aug,
        test_label_aug,
    ) = make_augs(config)
    if n_augmentations > 1 and config.general.print_info:
        print(f"Augmentation multiplicity of {n_augmentations}")

    if config.loss.calculate_class_weights:
        class_weight_file = (
            Path.cwd() / f"class_weights_{str(config.dataset.name).split('.')[-1]}.npy"
        )
        if class_weight_file.is_file():
            print(f"Using existing class weight file. ({class_weight_file})")
            config.loss.class_weights = np.load(class_weight_file).tolist()
        else:
            config.loss.class_weights = calc_class_weights(train_loader)
            np.save(class_weight_file, config.loss.class_weights)

    scheduler = make_scheduler_from_config(config)
    stopper = make_stopper_from_config(config)
    loss_class = make_loss_from_config(config)
    test_loss_fn = loss_class.create_test_loss_fn()

    train_op, train_vars, _ = make_train_op(
        model,
        model_vars,
        config,
        loss_class,
        augment_op,
        label_augment_op,
        grad_acc,
        total_noise,
        effective_batch_size,
        n_augmentations,
    )

    epoch_time = train_loop(
        config,
        train_loader,
        val_loader,
        model,
        unfreeze_schedule,
        checkpoint,
        predict_op_parallel,
        predict_op_jit,
        grad_acc,
        sampling_rate,
        delta,
        sigma,
        total_noise,
        batch_expansion_factor,
        effective_batch_size,
        accountant,
        metric_fns,
        n_augmentations,
        augment_op,
        label_augment_op,
        test_aug,
        test_label_aug,
        scheduler,
        stopper,
        loss_class,
        test_loss_fn,
        train_op,
        train_vars,
    )

    # never do test parallel as this could emit some test samples and thus distort results
    metric = test(
        config,
        test_loader,
        predict_op_jit,
        test_aug,
        test_label_aug,
        model.vars(),
        False,
        "test",
        metrics=metric_fns,
        loss_fn=test_loss_fn,
    )
    if config.general.save_path:
        save_path: Path = Path(config.general.save_path)
        save_path = save_path.parent / (
            save_path.stem + identifying_model_str + save_path.suffix
        )
        if config.general.print_info:
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
