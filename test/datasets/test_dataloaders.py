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


def test_cifar10(utils):
    config_dict = {
        "dataset": {"root": "./data", "download": True, "train_val_split": 0.9},
        "loader": {"num_workers": 16, "prefetch_factor": 16, "collate_fn": "numpy"},
    }
    config = utils.extend_base_config(config_dict)
    train_ds, val_ds, test_ds = CIFAR10Creator.make_datasets(config, (None, None, None))
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


def test_cifar10_from_config(utils):
    config_dict = {
        "hyperparams": {"batch_size": 4, "batch_size_test": 1},
        "dataset": {
            "name": "CIFAR10",
            "root": "./data",
            "train_val_split": 0.9,
            "task": "classification",
        },
        "loader": {"num_workers": 16, "prefetch_factor": 16, "collate_fn": "numpy"},
    }
    config = utils.extend_base_config(config_dict)
    train_dl, val_dl, test_dl = make_loader_from_config(config)
    access_dataloader(train_dl, val_dl, test_dl)


def test_imagenet_from_config(utils):
    config_dict = {
        "hyperparams": {"batch_size": 4, "batch_size_test": 1},
        "dataset": {
            "task": "classification",
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
        "loader": {"num_workers": 16, "prefetch_factor": 16, "collate_fn": "numpy"},
    }
    config = utils.extend_base_config(config_dict)
    train_dl, val_dl, test_dl = make_loader_from_config(config)
    access_dataloader(train_dl, val_dl, test_dl)


def test_no_val_set(utils):
    config_dict = {
        "hyperparams": {"batch_size": 1, "batch_size_test": 1},
        "dataset": {
            "task": "classification",
            "name": "CIFAR10",
            "root": "./data/",
            "train_val_split": 1.0,
        },
        "loader": {"num_workers": 16, "prefetch_factor": 16, "collate_fn": "numpy"},
    }
    config = utils.extend_base_config(config_dict)
    train_dl, val_dl, test_dl = make_loader_from_config(config)
    assert len(train_dl) == 50000
    assert len(test_dl) == 10000
    assert val_dl is None


def test_tiny_imagenet_from_config(utils):
    config_dict = {
        "hyperparams": {"batch_size": 4, "batch_size_test": 1},
        "dataset": {
            "task": "classification",
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
        "loader": {"num_workers": 16, "prefetch_factor": 16, "collate_fn": "numpy"},
    }
    config = utils.extend_base_config(config_dict)
    train_dl, val_dl, test_dl = make_loader_from_config(config)
    access_dataloader(train_dl, val_dl, test_dl)
