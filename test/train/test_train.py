import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from dptraining.train import main


def test_train_cifar_one_batch():
    config = {
        "project": "test",
        "general": {"log_wandb": False, "cpu": True},
        "loader": {"num_workers": 2, "collate_fn": "numpy"},
        "model": {"name": "cifar10model", "num_classes": 10},
        "augmentations": {
            "random_vertical_flips": None,
            "random_horizontal_flips": None,
            "random_img_shift": None,
        },
        "dataset": {"name": "CIFAR10", "root": "./data", "train_val_split": 0.9},
        "optim": {"name": "momentum", "momentum": 0.5},
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
            "epsilon": 7.5,
            "max_per_sample_grad_norm": 10.0,
            "delta": 1e-5,
            "norm_acc": False,
        },
    }
    main(config)
