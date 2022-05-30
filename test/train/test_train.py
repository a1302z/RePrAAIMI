import jax
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from dptraining.train import main

jax.config.update("jax_platform_name", "cpu")


def test_train_cifar_one_epoch():
    config = {
        "project": "test",
        "log_wandb": False,
        "loader": {"num_workers": 2},
        "model": {"name": "cifar10model", "num_classes": 10},
        "augmentations": {
            "random_vertical_flips": None,
            "random_horizontal_flips": None,
            "random_img_shift": {"img_shape": (3, 32, 32)},
        },
        "dataset": {"name": "CIFAR10", "root": "./data"},
        "hyperparams": {
            "epochs": 1,
            "batch_size": 128,
            "batch_size_test": 1,
            "lr": 0.1,
            "momentum": 0.9,
            "overfit": 1,
        },
        "scheduler": {"type": "cosine", "normalize_lr": True},
        "DP": {
            "disable_dp": False,
            "sigma": 1.5,
            "max_per_sample_grad_norm": 10.0,
            "delta": 1e-5,
            "norm_acc": False,
        },
    }
    main(config)
