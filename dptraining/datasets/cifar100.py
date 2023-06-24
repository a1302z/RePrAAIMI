from typing import Any, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR100

from dptraining.config import Config
from dptraining.datasets.base_creator import DataLoaderCreator


def fft_conversion(img, axes=None):
    return np.fft.fftshift(np.fft.fft2(img, axes=axes), axes=axes)



def undersample(image_arr:np.ndarray, label_arr:np.ndarray, class_index:int=8, factor:float=0.25, plot_hist:bool=False):
    print(f"... downsampling class {class_index} by a factor of {factor}")
    if not isinstance(label_arr, np.ndarray):
        label_arr = np.array(label_arr)
        image_arr = np.array(image_arr)
    idcs = np.where(label_arr.flatten()==class_index)[0]
    other = np.where(label_arr != class_index)[0]
    sub_indices = np.random.choice(idcs, size=int(np.count_nonzero(idcs) * factor), replace=False)
    out_idcs = np.concatenate([other, sub_indices])
    return image_arr[out_idcs], label_arr[out_idcs]


class NumpyCIFAR100(CIFAR100):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR100Creator(DataLoaderCreator):
    CIFAR_MEAN = (0.5074, 0.4867, 0.4411)
    CIFAR_STDDEV = (0.2011, 0.1987, 0.2025)

    @staticmethod
    def reshape_images(image: np.array):
        image = image.astype(np.float32)
        image = image.transpose(0, 3, 1, 2)
        return image

    @staticmethod
    def normalize_images(image: np.array):
        image /= 255.0
        image = (
            image - np.reshape(CIFAR100Creator.CIFAR_MEAN, [1, 3, 1, 1])
        ) / np.reshape(CIFAR100Creator.CIFAR_STDDEV, [1, 3, 1, 1])
        return image

    @staticmethod
    def transpose_to_hwc(dataset: Dataset):
        dataset.data.transpose(0, 2, 3, 1)

    @staticmethod
    def make_datasets(  # pylint:disable=too-many-arguments
        config: Config,
        transforms: Tuple,
        numpy_optimisation=True,
        normalize_by_default=True,
        undersample_class:bool=False,
        undersample_factor:float=0.0,
    ) -> Tuple[Dataset, Dataset, Dataset]:
        train_tf, val_tf, test_tf = transforms
        train_kwargs = {
            "root": config.dataset.root,
            "download": True,
            "transform": train_tf,
            "train": True,
        }
        val_kwargs = {
            "root": config.dataset.root,
            "download": True,
            "transform": val_tf,
            "train": True,
        }
        test_kwargs = {
            "root": config.dataset.root,
            "download": True,
            "transform": test_tf,
            "train": False,
        }
        if normalize_by_default and not numpy_optimisation:
            raise ValueError(
                "CIFAR100 Creator can only normalize by default if numpy optimisation is activated"
            )
        if numpy_optimisation:
            train_ds = NumpyCIFAR100(**train_kwargs)
            val_ds = NumpyCIFAR100(**val_kwargs)
            test_ds = NumpyCIFAR100(**test_kwargs)
            train_ds.data = CIFAR100Creator.reshape_images(train_ds.data)
            val_ds.data = CIFAR100Creator.reshape_images(val_ds.data)
            test_ds.data = CIFAR100Creator.reshape_images(test_ds.data)
        else:
            train_ds = CIFAR100(**train_kwargs)
            val_ds = CIFAR100(**val_kwargs)
            test_ds = CIFAR100(**test_kwargs)
        if normalize_by_default:
            train_ds.data = CIFAR100Creator.normalize_images(train_ds.data)
            val_ds.data = CIFAR100Creator.normalize_images(val_ds.data)
            test_ds.data = CIFAR100Creator.normalize_images(test_ds.data)
        if config.dataset.fft:
            if not numpy_optimisation:
                raise ValueError("FFT only works with numpy optimisation")
            train_ds.data = fft_conversion(train_ds.data, axes=(1, 2, 3))
            val_ds.data = fft_conversion(val_ds.data, axes=(1, 2, 3))
            test_ds.data = fft_conversion(test_ds.data, axes=(1, 2, 3))
        train_val_split = config.dataset.train_val_split
        assert 0.0 < train_val_split <= 1.0, "Train/Val split must be in (0,1]"
        if abs(train_val_split - 1.0) < 1e-5:
            val_ds = None
        else:
            train_data, val_data, train_targets, val_targets = train_test_split(
                train_ds.data, train_ds.targets, train_size=train_val_split
            )
            train_ds.data = train_data
            train_ds.targets = train_targets
            val_ds.data = val_data
            val_ds.targets = val_targets
        if undersample_class:
            train_ds.data, train_ds.targets = undersample(train_ds.data, train_ds.targets, factor=undersample_factor)

        return train_ds, val_ds, test_ds

    @staticmethod
    def make_dataloader(  # pylint:disable=too-many-arguments,duplicate-code
        train_ds: Dataset,
        val_ds: Dataset,
        test_ds: Dataset,
        train_kwargs,
        val_kwargs,
        test_kwargs,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        train_dl = DataLoader(train_ds, **train_kwargs)
        val_dl = DataLoader(val_ds, **val_kwargs) if val_ds is not None else None
        test_dl = DataLoader(test_ds, **test_kwargs)
        return train_dl, val_dl, test_dl