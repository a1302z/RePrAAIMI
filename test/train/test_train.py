import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from dptraining.train import main


def test_train_cifar_one_epoch():
    config = {
        "dataset": "CIFAR10",
        "hyperparams": {
            "epochs": 1,
            "batch_size": 128,
            "batch_size_test": 1,
            "lr": 0.1,
            "momentum": 0.9,
        },
        "scheduler": {
            "type": "cosine",
        },
        "DP": {
            "disable_dp": False,
            "sigma": 1.5,
            "max_per_sample_grad_norm": 10.0,
            "delta": 1e-5,
            "norm_acc": False,
        },
    }
    main(config)
