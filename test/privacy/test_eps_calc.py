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


def test_adapt_noise_real():
    c = EpsCalculator(
        {
            "DP": {
                "epsilon": 25,
                "delta": 1e-5,
                "sigma": 3.0,
                "rsqrt_noise_adapt": False,
                "glrt_assumption": False,
                "max_per_sample_grad_norm": 1.0,
            },
            "hyperparams": {"batch_size": 50},
            "model": {"complex": False},
        },
        DataLoader(
            TensorDataset(randn(100, 3, 224, 224), randn(100, 9)), batch_size=50
        ),
    )
    c.fill_config()
    total_noise, adapted_sigma = c.adapt_sigma()
    assert abs(total_noise - adapted_sigma) < 1e-5
    assert abs(total_noise - 3.0) < 1e-5  # check sigma


def test_adapt_noise_real_glrt():
    c = EpsCalculator(
        {
            "DP": {
                "epsilon": 25,
                "delta": 1e-5,
                "sigma": 3.0,
                "rsqrt_noise_adapt": False,
                "glrt_assumption": True,
                "max_per_sample_grad_norm": 1.0,
            },
            "hyperparams": {"batch_size": 50},
            "model": {"complex": False},
        },
        DataLoader(
            TensorDataset(randn(100, 3, 224, 224), randn(100, 9)), batch_size=50
        ),
    )
    c.fill_config()
    total_noise, adapted_sigma = c.adapt_sigma()
    assert abs(total_noise - adapted_sigma) < 1e-5
    assert total_noise < 3.0


def test_adapt_noise_complex():
    c = EpsCalculator(
        {
            "DP": {
                "epsilon": 25,
                "delta": 1e-5,
                "sigma": 3.0,
                "rsqrt_noise_adapt": True,
                "glrt_assumption": False,
                "max_per_sample_grad_norm": 1.0,
            },
            "hyperparams": {"batch_size": 50},
            "model": {"complex": True},
        },
        DataLoader(
            TensorDataset(randn(100, 3, 224, 224), randn(100, 9)), batch_size=50
        ),
    )
    c.fill_config()
    total_noise, adapted_sigma = c.adapt_sigma()
    assert abs(total_noise - adapted_sigma / (2**0.5)) < 1e-5
    assert abs(total_noise - 3.0 / (2**0.5)) < 1e-5  # check sigma


def test_adapt_noise_complex_glrt():
    c = EpsCalculator(
        {
            "DP": {
                "epsilon": 25,
                "delta": 1e-5,
                "sigma": 3.0,
                "rsqrt_noise_adapt": True,
                "glrt_assumption": True,
                "max_per_sample_grad_norm": 1.0,
            },
            "hyperparams": {"batch_size": 50},
            "model": {"complex": True},
        },
        DataLoader(
            TensorDataset(randn(100, 3, 224, 224), randn(100, 9)), batch_size=50
        ),
    )
    c.fill_config()
    total_noise, adapted_sigma = c.adapt_sigma()
    assert total_noise < 3.0  # check sigma
