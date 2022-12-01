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


def test_cifar10(utils):
    config_dict = {"dataset": {"root": "./data", "train_val_split": 0.9}}
    config = utils.extend_base_config(config_dict)
    train_ds, val_ds, test_ds = CIFAR10Creator.make_datasets(config, (None, None, None))
    access_dataset(train_ds, val_ds, test_ds)


def test_cifar10_unnormalized(utils):
    config_dict = {"dataset": {"root": "./data", "train_val_split": 0.9}}
    config = utils.extend_base_config(config_dict)
    train_ds, val_ds, test_ds = CIFAR10Creator.make_datasets(
        config,
        (None, None, None),
        normalize_by_default=False,
    )
    access_dataset(train_ds, val_ds, test_ds)


def test_cifar10_invalid_args(utils):
    config_dict = {"dataset": {"root": "./data", "train_val_split": 0.9}}
    config = utils.extend_base_config(config_dict)
    with pytest.raises(ValueError):
        train_ds, val_ds, test_ds = CIFAR10Creator.make_datasets(
            config,
            (None, None, None),
            normalize_by_default=True,
            numpy_optimisation=False,
        )


def test_cifar10_standard_torchvision(utils):
    config_dict = {"dataset": {"root": "./data", "train_val_split": 0.9}}
    config = utils.extend_base_config(config_dict)
    train_ds, val_ds, test_ds = CIFAR10Creator.make_datasets(
        config,
        (None, None, None),
        normalize_by_default=False,
        numpy_optimisation=False,
    )
    access_dataset(train_ds, val_ds, test_ds)


def test_cifar10_fft(utils):
    config_dict = {
        "dataset": {
            "root": "./data",
            "train_val_split": 0.9,
            "fft": True,
        }
    }
    config = utils.extend_base_config(config_dict)
    train_ds, val_ds, test_ds = CIFAR10Creator.make_datasets(config, (None, None, None))
    access_dataset(train_ds, val_ds, test_ds)


def test_imagenet(utils):
    config_dict = {
        "dataset": {
            "root": "./data/ILSVRC2012",
            # "split": "train",
            "train_val_split": 0.9,
        }
    }
    config = utils.extend_base_config(config_dict)
    train_ds, val_ds, test_ds = ImageNetCreator.make_datasets(
        config, (None, None, None)
    )
    access_dataset(train_ds, val_ds, test_ds)


def test_tinyimagenet32(utils):
    config_dict = {
        "dataset": {
            "root": "./data/ImageNet32",
            "train_val_split": 0.9,
            "version": 32,
        }
    }
    config = utils.extend_base_config(config_dict)
    train_ds, val_ds, test_ds = TinyImageNetCreator.make_datasets(
        config, (None, None, None)
    )
    access_dataset(train_ds, val_ds, test_ds)


# def test_tinyimagenet64(utils):  # this is too slow to test each time
#    config_dict = {
#        "dataset": {
#            "root": "./data/ImageNet64",
#            "train_val_split": 0.9,
#            "version": 64,
#        }
#    }
#    config = utils.extend_base_config(config_dict)
#    train_ds, val_ds, test_ds = TinyImageNetCreator.make_datasets(
#        config, (None, None, None)
#    )
#    access_dataset(train_ds, val_ds, test_ds)
