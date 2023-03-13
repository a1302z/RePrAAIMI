import os
import sys
from pathlib import Path
from typing import Union

import hydra
import numpy as np
import wandb
from omegaconf import OmegaConf
from torch import save as torchsave, load as torchload
from torch.utils.data import Dataset, DataLoader

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
    from torch.random import manual_seed as torch_manual_seed

    from dptraining.datasets import make_attack_dataloader, collate_np_reconstruction
    from dptraining.privacy import setup_privacy
    from dptraining.utils import (
        make_loss_from_config,
        make_scheduler_from_config,
        make_stopper_from_config,
        make_metrics,
    )
    from dptraining.utils.augment import make_augs
    from dptraining.utils.training_utils import (
        make_train_op,
        # test,
        # train,
    )
    from dptraining.utils.attack_utils import (
        flatten_weights,
        create_N_models,
        make_train_ops,
        train_loop,
        setup_predict_ops,
    )
    from dptraining.utils.misc import get_num_params, make_unique_str

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
            config=config_dict,
            settings=wandb.Settings(start_method="thread"),
            reinit=True,
            **config_dict["wandb"],
        )
    ######################################################################
    (
        shadow_train_loader,
        shadow_eval_loader,
        attack_eval_loader,
    ) = make_attack_dataloader(config)
    ######################################################################

    ######################################################################
    # TODO: option to init all equally (e.g. deepcopy)
    # TODO: allow unfreezing schedules
    train_models = create_N_models(
        config, len(shadow_train_loader.dataset.shadow_indices)
    )
    eval_models = create_N_models(
        config, len(attack_eval_loader.dataset.shadow_indices)
    )
    n_train_vars_total: int = get_num_params(train_models[0][1])
    if config.general.log_wandb:
        wandb.log({"total_model_vars": n_train_vars_total})
    elif config.general.print_info:
        print(f"Total model params: {n_train_vars_total:,}")

    # n_train_vars_cur: int = get_num_params(model_vars)
    # if config.general.log_wandb:
    #     wandb.log({"num_trained_vars": n_train_vars_cur})
    # else:
    #     print(f"Num trained params: {n_train_vars_cur:,}")

    identifying_model_str = make_unique_str(
        config, id_str=wandb.run.name if config.general.log_wandb else ""
    )

    # if config.checkpoint:
    #     config.checkpoint.logdir += identifying_model_str
    #     checkpoint = objax.io.Checkpoint(**OmegaConf.to_container(config.checkpoint))
    # else:
    #     checkpoint = None

    predict_ops_jit_train, predict_ops_jit_eval = setup_predict_ops(
        config, train_models, eval_models
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
    ) = setup_privacy(config, shadow_train_loader)

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
    scheduler = make_scheduler_from_config(config)
    stopper = make_stopper_from_config(config)
    loss_class = make_loss_from_config(config)
    test_loss_fn = loss_class.create_test_loss_fn()

    (
        attack_train_train_ops,
        attack_train_train_vars,
        train_model_vars,
    ) = make_train_ops(
        train_models,
        make_train_op,
        config,
        loss_class,
        augment_op,
        label_augment_op,
        grad_acc,
        total_noise,
        effective_batch_size,
        n_augmentations,
    )
    attack_eval_train_ops, attack_eval_train_vars, eval_model_vars = make_train_ops(
        eval_models,
        make_train_op,
        config,
        loss_class,
        augment_op,
        label_augment_op,
        grad_acc,
        total_noise,
        effective_batch_size,
        n_augmentations,
    )
    attack_train_train_vars.append(objax.random.DEFAULT_GENERATOR.vars())
    attack_eval_train_vars.append(objax.random.DEFAULT_GENERATOR.vars())

    epoch_time = train_loop(
        config,
        shadow_train_loader,
        shadow_eval_loader,
        attack_eval_loader,
        predict_ops_jit_train,
        predict_ops_jit_eval,
        grad_acc,
        sampling_rate,
        delta,
        sigma,
        batch_expansion_factor,
        accountant,
        metric_fns,
        test_aug,
        test_label_aug,
        scheduler,
        test_loss_fn,
        attack_train_train_ops,
        attack_train_train_vars,
        train_model_vars,
        attack_eval_train_ops,
        attack_eval_train_vars,
        eval_model_vars,
    )
    #     if config.unfreeze_schedule is not None:
    #         model_vars, fresh_model_vars = unfreeze_schedule(epoch + 1)
    #         if fresh_model_vars:
    #             train_op, train_vars = make_train_op(model_vars)
    #         n_train_vars_cur = get_num_params(model_vars)
    #         if config.general.log_wandb:
    #             wandb.log({"num_trained_vars": n_train_vars_cur})
    #         else:
    #             print(f"\tNum Train Vars: {n_train_vars_cur:,}")
    #     if checkpoint is not None:
    #         checkpoint.save(model.vars(), idx=epoch)
    #     if metric is not None and stopper(metric):
    #         print("Early Stopping was activated -> Stopping Training")
    #         break

    # no need for testing in attack scenario, val set must do
    # metric = test(
    #     config,
    #     test_loader,
    #     predict_op_jit,
    #     test_aug,
    #     test_label_aug,
    #     model.vars(),
    #     False,
    #     "test",
    #     metrics=metric_fns,
    #     loss_fn=test_loss_fn,
    # )
    if config.general.save_path:
        save_path: Path = Path(config.general.save_path)
        save_path = save_path.parent / (
            save_path.stem + identifying_model_str + save_path.suffix
        )
        # save shadow models and indices as training dataset
        attack_train_weights = np.stack(
            [flatten_weights(model)[2] for model, _, _ in train_models], axis=0
        )
        attack_eval_weights = np.stack(
            [flatten_weights(model)[2] for model, _, _ in eval_models], axis=0
        )
        train_target_imgs_labels = shadow_train_loader.dataset[-1]
        eval_imgs_labels = attack_eval_loader.dataset[-1]
        attack_train_targets = np.stack(
            [target_img for target_img, _ in train_target_imgs_labels], axis=0
        )
        attack_train_target_label = np.stack(
            [target_label for _, target_label in train_target_imgs_labels], axis=0
        )
        attack_eval_targets = np.stack(
            [target_img for target_img, _ in eval_imgs_labels], axis=0
        )
        attack_eval_target_label = np.stack(
            [target_label for _, target_label in eval_imgs_labels], axis=0
        )
        attack_data_dict = {
            "config": OmegaConf.to_container(config),
            "general": {
                "fixed_indices": shadow_train_loader.dataset.indices,
                "shadow_train_indices": shadow_train_loader.dataset.shadow_indices,
                "shadow_eval_indices": shadow_eval_loader.dataset.indices,
                "attack_eval_indices": attack_eval_loader.dataset.shadow_indices,
            },
            "attack_train_weights": attack_train_weights,
            "attack_eval_weights": attack_eval_weights,
            "reconstruction_train_targets": attack_train_targets,
            "reconstruction_train_labels": attack_train_target_label,
            "reconstruction_eval_targets": attack_eval_targets,
            "reconstruction_eval_labels": attack_eval_target_label,
        }
        torchsave(attack_data_dict, f"{save_path}.pt")
    #     print(f"Saving model to {config.general.save_path}")
    #     objax.io.save_var_collection(save_path, model.vars())
    print("Average epoch time (all epochs): ", np.average(epoch_time))
    print("Median epoch time (all epochs): ", np.median(epoch_time))
    if len(epoch_time) > 1:
        print("Average epoch time (except first): ", np.average(epoch_time[1:]))
        print("Median epoch time (except first): ", np.median(epoch_time[1:]))
    print("Total training time (excluding evaluation): ", np.sum(epoch_time))

    if config.general.log_wandb:
        run.finish()


if __name__ == "__main__":
    main()  # pylint:disable=no-value-for-parameter
