import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from dptraining.datasets import CIFAR10Creator, ImageNetCreator, TinyImageNetCreator


def access_dataset(*args):
    for ds in args:
        ds[0]
        ds[len(ds) - 1]


def test_cifar10():
    train_ds, val_ds, test_ds = CIFAR10Creator.make_datasets(
        {"dataset": {"root": "./data", "download": True, "train_val_split": 0.9}},
        (None, None, None),
    )
    access_dataset(train_ds, val_ds, test_ds)


def test_cifar10_unnormalized():
    train_ds, val_ds, test_ds = CIFAR10Creator.make_datasets(
        {"dataset": {"root": "./data", "download": True, "train_val_split": 0.9}},
        (None, None, None),
        normalize_by_default=False,
    )
    access_dataset(train_ds, val_ds, test_ds)


def test_cifar10_invalid_args():
    with pytest.raises(ValueError):
        train_ds, val_ds, test_ds = CIFAR10Creator.make_datasets(
            {"dataset": {"root": "./data", "download": True, "train_val_split": 0.9}},
            (None, None, None),
            numpy_optimisation=False,
        )
        access_dataset(train_ds, val_ds, test_ds)


def test_cifar10_standard_torchvision():
    train_ds, val_ds, test_ds = CIFAR10Creator.make_datasets(
        {"dataset": {"root": "./data", "download": True, "train_val_split": 0.9}},
        (None, None, None),
        normalize_by_default=False,
        numpy_optimisation=False,
    )
    access_dataset(train_ds, val_ds, test_ds)


def test_cifar10_fft():
    train_ds, val_ds, test_ds = CIFAR10Creator.make_datasets(
        {
            "dataset": {
                "root": "./data",
                "download": True,
                "train_val_split": 0.9,
                "fft": True,
            }
        },
        (None, None, None),
    )
    access_dataset(train_ds, val_ds, test_ds)


def test_imagenet():
    train_ds, val_ds, test_ds = ImageNetCreator.make_datasets(
        {
            "dataset": {
                "root": "./data/ILSVRC2012",
                "split": "train",
                "train_val_split": 0.9,
            }
        },
        (None, None, None),
    )
    access_dataset(train_ds, val_ds, test_ds)


def test_tinyimagenet32():
    train_ds, val_ds, test_ds = TinyImageNetCreator.make_datasets(
        {
            "dataset": {
                "root": "./data/ImageNet32",
                "train_val_split": 0.9,
                "version": 32,
            }
        },
        (None, None, None),
    )
    access_dataset(train_ds, val_ds, test_ds)


# def test_tinyimagenet64(): # this is too slow to test each time
#     train_ds, val_ds, test_ds = TinyImageNetCreator.make_datasets(
#         {
#             "dataset": {
#                 "root": "./data/ImageNet64",
#                 "train_val_split": 0.9,
#                 "version": 64,
#             }
#         },
#         (None, None, None),
#     )
#     access_dataset(train_ds, val_ds, test_ds)
