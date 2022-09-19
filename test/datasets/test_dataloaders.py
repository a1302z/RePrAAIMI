import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from dptraining.datasets import CIFAR10Creator, make_loader_from_config


def access_dataloader(*args):
    for dl in args:
        next(iter(dl))


def test_cifar10():
    train_ds, val_ds, test_ds = CIFAR10Creator.make_datasets(
        {"dataset": {"root": "./data", "download": True, "train_val_split": 0.9}},
        (None, None, None),
    )
    train_dl, val_dl, test_dl = CIFAR10Creator.make_dataloader(
        train_ds,
        val_ds,
        test_ds,
        {"batch_size": 2},
        {"batch_size": 1},
        {"batch_size": 1},
    )
    for _, _ in train_dl:
        pass
    for _, _ in val_dl:
        pass
    for _, _ in test_dl:
        pass


def test_cifar10_from_config():
    train_dl, val_dl, test_dl = make_loader_from_config(
        {
            "hyperparams": {"batch_size": 4, "batch_size_test": 1},
            "dataset": {"name": "cifar10", "root": "./data", "train_val_split": 0.9},
            "loader": {},
            "DP": {"disable_dp": False},
        }
    )
    access_dataloader(train_dl, val_dl, test_dl)


def test_imagenet_from_config():
    train_dl, val_dl, test_dl = make_loader_from_config(
        {
            "hyperparams": {"batch_size": 4, "batch_size_test": 1},
            "dataset": {
                "name": "imagenet",
                "root": "./data/ILSVRC2012",
                "train_val_split": 0.9,
            },
            "train_transforms": {
                "Resize": {"size": 224},
                "RandomCrop": {"size": 224},
                "ToTensor": None,
            },
            "test_transforms": {
                "Resize": {"size": 224},
                "CenterCrop": {"size": 224},
                "ToTensor": None,
            },
            "augmentations": {"random_vertical_flips": {"flip_prob": 0.5}},
            "loader": {},
            "DP": {"disable_dp": False},
        }
    )
    access_dataloader(train_dl, val_dl, test_dl)


def test_no_val_set():
    train_dl, val_dl, test_dl = make_loader_from_config(
        {
            "hyperparams": {"batch_size": 1, "batch_size_test": 1},
            "dataset": {
                "name": "CIFAR10",
                "root": "./data/",
                "train_val_split": 1.0,
            },
            "augmentations": {},
            "loader": {},
            "DP": {"disable_dp": False},
        }
    )
    assert len(train_dl) == 50000
    assert len(test_dl) == 10000
    assert val_dl is None


def test_imagenet_from_config():
    train_dl, val_dl, test_dl = make_loader_from_config(
        {
            "hyperparams": {"batch_size": 4, "batch_size_test": 1},
            "dataset": {
                "name": "tinyimagenet",
                "root": "./data/ImageNet32",
                "version": 32,
                "train_val_split": 0.9,
            },
            "train_transforms": {
                "ToTensor": None,
            },
            "test_transforms": {
                "ToTensor": None,
            },
            "augmentations": {"random_vertical_flips": {"flip_prob": 0.5}},
            "loader": {},
            "DP": {"disable_dp": False},
        }
    )
    access_dataloader(train_dl, val_dl, test_dl)
