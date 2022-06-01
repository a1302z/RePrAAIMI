import numpy as np
from typing import Tuple, Any
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset, DataLoader

from dptraining.datasets.base_creator import DataLoaderCreator
from dptraining.datasets.utils import collate_np_arrays


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
    CIFAR_STDDEV = (0.2023, 0.1994, 0.2010)

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
        train_args,
        train_kwargs,
        test_args,
        test_kwargs,
        numpy_optimisation=True,
        normalize_by_default=True,
    ) -> Tuple[Dataset, Dataset]:
        if normalize_by_default and not numpy_optimisation:
            raise ValueError(
                "CIFAR10 Creator can only normalize by default if numpy optimisation is activated"
            )
        if numpy_optimisation:
            train_ds = NumpyCIFAR10(*train_args, **train_kwargs)
            test_ds = NumpyCIFAR10(*test_args, **test_kwargs)
        else:
            train_ds = CIFAR10(*train_args, **train_kwargs)
            test_ds = CIFAR10(*test_args, **test_kwargs)
        if normalize_by_default:
            train_ds.data = CIFAR10Creator.normalize_images(train_ds.data)
            test_ds.data = CIFAR10Creator.normalize_images(test_ds.data)
        return train_ds, test_ds

    @staticmethod
    def make_dataloader(  # pylint:disable=too-many-arguments,duplicate-code
        train_ds: Dataset,
        test_ds: Dataset,
        train_args,
        train_kwargs,
        test_args,
        test_kwargs,
        numpy_collate=True,
    ) -> Tuple[DataLoader, DataLoader]:
        if numpy_collate:
            if not "collate_fn" in train_kwargs:
                train_kwargs["collate_fn"] = collate_np_arrays
            if not "collate_fn" in test_kwargs:
                test_kwargs["collate_fn"] = collate_np_arrays
        train_dl = DataLoader(train_ds, *train_args, **train_kwargs)
        test_dl = DataLoader(test_ds, *test_args, **test_kwargs)
        return train_dl, test_dl
