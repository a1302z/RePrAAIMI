import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from dptraining.train import main


def test_train_cifar_one_batch(utils):
    config_dict = {
        "project": "test",
        "general": {"log_wandb": False, "cpu": True, "eval_train": False},
        "loader": {"num_workers": 2, "collate_fn": "numpy"},
        "model": {"name": "cifar10model", "num_classes": 10},
        "augmentations": {
            "random_vertical_flips": None,
            "random_horizontal_flips": None,
            "random_img_shift": None,
        },
        "dataset": {"name": "CIFAR10", "root": "./data", "train_val_split": 0.9},
        "optim": {"name": "momentum", "args": {"momentum": 0.5}},
        "loss": {"type": "cse", "reduction": "mean"},
        "hyperparams": {
            "epochs": 1,
            "batch_size": 128,
            "batch_size_test": 1,
            "lr": 0.1,
            "overfit": 1,
        },
        "scheduler": {"type": "cosine", "normalize_lr": True},
        "DP": {
            "epsilon": 7.5,
            "max_per_sample_grad_norm": 10.0,
            "delta": 1e-5,
            "norm_acc": False,
        },
    }
    config = utils.extend_base_config(config_dict)
    main(config)
