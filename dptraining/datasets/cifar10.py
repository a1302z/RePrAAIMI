import numpy as np
from typing import Tuple, Any
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from dptraining.datasets.base_creator import DataLoaderCreator


class NumpyCIFAR10(CIFAR10):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR10Creator(DataLoaderCreator):
    CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR_STDDEV = (0.24703279, 0.24348423, 0.26158753)

    @staticmethod
    def normalize_images(image: np.array):
        image = image.astype(np.float32)
        image = image / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = (
            image - np.reshape(CIFAR10Creator.CIFAR_MEAN, [1, 3, 1, 1])
        ) / np.reshape(CIFAR10Creator.CIFAR_STDDEV, [1, 3, 1, 1])
        return image

    @staticmethod
    def transpose_to_hwc(dataset: Dataset):
        dataset.data.transpose(0, 2, 3, 1)

    @staticmethod
    def make_datasets(  # pylint:disable=too-many-arguments
        config,
        transforms: Tuple,
        numpy_optimisation=True,
        normalize_by_default=True,
    ) -> Tuple[Dataset, Dataset]:
        train_tf, val_tf, test_tf = transforms
        train_kwargs = {
            "root": config["dataset"]["root"],
            "download": True,
            "transform": train_tf,
            "train": True,
        }
        val_kwargs = {
            "root": config["dataset"]["root"],
            "download": True,
            "transform": val_tf,
            "train": True,
        }
        test_kwargs = {
            "root": config["dataset"]["root"],
            "download": True,
            "transform": test_tf,
            "train": False,
        }
        if normalize_by_default and not numpy_optimisation:
            raise ValueError(
                "CIFAR10 Creator can only normalize by default if numpy optimisation is activated"
            )
        if numpy_optimisation:
            train_ds = NumpyCIFAR10(**train_kwargs)
            val_ds = NumpyCIFAR10(**val_kwargs)
            test_ds = NumpyCIFAR10(**test_kwargs)
        else:
            train_ds = CIFAR10(**train_kwargs)
            val_ds = CIFAR10(**val_kwargs)
            test_ds = CIFAR10(**test_kwargs)
        if normalize_by_default:
            train_ds.data = CIFAR10Creator.normalize_images(train_ds.data)
            val_ds.data = CIFAR10Creator.normalize_images(val_ds.data)
            test_ds.data = CIFAR10Creator.normalize_images(test_ds.data)
        train_val_split = config["dataset"]["train_val_split"]
        train_data, val_data, train_targets, val_targets = train_test_split(
            train_ds.data, train_ds.targets, train_size=train_val_split
        )
        train_ds.data = train_data
        train_ds.targets = train_targets
        val_ds.data = val_data
        val_ds.targets = val_targets
        return train_ds, val_ds, test_ds

    @staticmethod
    def make_dataloader(  # pylint:disable=too-many-arguments,duplicate-code
        train_ds: Dataset,
        val_ds: Dataset,
        test_ds: Dataset,
        train_kwargs,
        val_kwargs,
        test_kwargs,
    ) -> Tuple[DataLoader, DataLoader]:
        train_dl = DataLoader(train_ds, **train_kwargs)
        val_dl = DataLoader(val_ds, **val_kwargs)
        test_dl = DataLoader(test_ds, **test_kwargs)
        return train_dl, val_dl, test_dl
