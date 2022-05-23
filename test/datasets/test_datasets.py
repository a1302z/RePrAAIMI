import pytest
from dptraining.datasets import CIFAR10Creator


def access_dataset(train_ds, test_ds):
    x_train = train_ds[0]
    x_test = test_ds[len(test_ds) - 1]


def test_cifar10():
    train_ds, test_ds = CIFAR10Creator.make_datasets(
        (),
        {"root": "./data", "download": True},
        (),
        {"root": "./data", "download": True},
    )
    access_dataset(train_ds, test_ds)


def test_cifar10_unnormalized():
    train_ds, test_ds = CIFAR10Creator.make_datasets(
        (),
        {"root": "./data", "download": True},
        (),
        {"root": "./data", "download": True},
        normalize_by_default=False,
    )
    access_dataset(train_ds, test_ds)


def test_cifar10_invalid_args():
    with pytest.raises(ValueError):
        train_ds, test_ds = CIFAR10Creator.make_datasets(
            (),
            {
                "root": "./data",
                "download": True,
            },
            (),
            {"root": "./data", "download": True},
            numpy_optimisation=False,
        )
        access_dataset(train_ds, test_ds)


def test_cifar10_standard_torchvision():
    train_ds, test_ds = CIFAR10Creator.make_datasets(
        (),
        {"root": "./data", "download": True},
        (),
        {"root": "./data", "download": True},
        normalize_by_default=False,
        numpy_optimisation=False,
    )
    access_dataset(train_ds, test_ds)
