import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from dptraining.privacy import EpsCalculator
from torch.utils.data import TensorDataset, DataLoader
from torch import randn


def test_calc():
    c = EpsCalculator(
        {
            "DP": {"epsilon": 25, "delta": 1e-5,},
            "hyperparams": {"epochs": 10, "batch_size": 50},
        },
        DataLoader(
            TensorDataset(randn(100, 3, 224, 224), randn(100, 9)), batch_size=50
        ),
    )
    c.calc_noise_for_eps()  # TODO: check if correct
