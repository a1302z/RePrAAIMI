import sys
from pathlib import Path
from pytest import raises

sys.path.insert(0, str(Path.cwd()))

from dptraining.privacy import EpsCalculator
from torch.utils.data import TensorDataset, DataLoader
from torch import randn


def test_calc_sigma():
    c = EpsCalculator(
        {
            "DP": {
                "epsilon": 25,
                "delta": 1e-5,
            },
            "hyperparams": {"epochs": 10, "batch_size": 50},
        },
        DataLoader(
            TensorDataset(randn(100, 3, 224, 224), randn(100, 9)), batch_size=50
        ),
    )
    c.fill_config()  # TODO: check if correct


def test_calc_epochs():
    c = EpsCalculator(
        {
            "DP": {"epsilon": 25, "delta": 1e-5, "sigma": 3.0},
            "hyperparams": {"batch_size": 50},
        },
        DataLoader(
            TensorDataset(randn(100, 3, 224, 224), randn(100, 9)), batch_size=50
        ),
    )
    c.fill_config()


def test_fail():
    with raises(ValueError):
        c = EpsCalculator(
            {
                "DP": {"epsilon": 25, "delta": 1e-5, "sigma": 3.0},
                "hyperparams": {"epochs": 10, "batch_size": 50},
            },
            DataLoader(
                TensorDataset(randn(100, 3, 224, 224), randn(100, 9)), batch_size=50
            ),
        )
        c.fill_config()
