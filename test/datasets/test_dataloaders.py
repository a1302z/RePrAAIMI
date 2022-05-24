import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from dptraining.utils.augment import Augmentation
from dptraining.datasets import CIFAR10Creator


def test_cifar10():
    train_ds, test_ds = CIFAR10Creator.make_datasets(
        (),
        {"root": "./data", "download": True},
        (),
        {"root": "./data", "download": True},
    )
    train_dl, test_dl = CIFAR10Creator.make_dataloader(
        train_ds, test_ds, (), {"batch_size": 2}, (), {"batch_size": 1}
    )
    for x, y in train_dl:
        pass
    for x, y in test_dl:
        pass
