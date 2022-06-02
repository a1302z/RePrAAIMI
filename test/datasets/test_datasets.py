import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from dptraining.datasets import CIFAR10Creator, ImageNetCreator


def access_dataset(train_ds, test_ds):
    train_ds[0]
    test_ds[len(test_ds) - 1]


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


def test_imagenet():
    train_ds, test_ds = ImageNetCreator.make_datasets(
        (),
        {"root": "./data/ILSVRC2012", "split": "train"},
        (),
        {"root": "./data/ILSVRC2012", "split": "val"},
    )
    access_dataset(train_ds, test_ds)
