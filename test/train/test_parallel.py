## This test breaks the entire test pipeline due to jax side effects

# import os

# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
# import sys
# import pytest
# from pathlib import Path

# sys.path.insert(0, str(Path.cwd()))
# from dptraining.train import main


# def test_train_cifar_one_batch_parallel_none_private(utils):
#    config_dict = {
#        "project": "test",
#        "general": {"log_wandb": False, "cpu": True, "parallel": True},
#        "loader": {"num_workers": 2},
#        "model": {"name": "cifar10model", "num_classes": 10},
#        "augmentations": {
#            "random_vertical_flips": None,
#            "random_horizontal_flips": None,
#            "random_img_shift": {"img_shape": (3, 32, 32)},
#        },
#        "dataset": {"name": "CIFAR10", "root": "./data"},
#        "optim": {"name": "momentum", "args": {"momentum": 0.5}},
#        "hyperparams": {
#            "epochs": 1,
#            "batch_size": 128,
#            "batch_size_test": 2,
#            "lr": 0.1,
#            "overfit": 2,
#        },
#        "scheduler": {"type": "cosine", "normalize_lr": True},
#    }
#    config = utils.extend_base_config(config_dict)
#    main(config)


# def test_train_assert_device_fail(utils):
#    with pytest.raises(AssertionError):
#        config_dict = {
#            "project": "test",
#            "general": {"log_wandb": False, "cpu": True, "parallel": True},
#            "loader": {"num_workers": 2},
#            "model": {"name": "cifar10model", "num_classes": 10},
#            "augmentations": {
#                "random_vertical_flips": None,
#                "random_horizontal_flips": None,
#                "random_img_shift": {"img_shape": (3, 32, 32)},
#            },
#            "dataset": {"name": "CIFAR10", "root": "./data"},
#            "optim": {"name": "momentum", "args": {"momentum": 0.5}},
#            "hyperparams": {
#                "epochs": 1,
#                "batch_size": 1,
#                "batch_size_test": 1,
#                "lr": 0.1,
#                "overfit": 2,
#            },
#            "scheduler": {"type": "cosine", "normalize_lr": True},
#        }
#        config = utils.extend_base_config(config_dict)
#        main(config)
