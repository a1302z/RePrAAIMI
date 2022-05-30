import numpy as np
from jax import numpy as jnp
from typing import Tuple, Any, List
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset, DataLoader

from dptraining.datasets.base_creator import DataLoaderCreator


# class NumpyDataset(Dataset):
#     def __init__(self, x, y, batch_size, shuffle=True):
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#         self.size = len(y)
#         self.x = np.array(x, dtype=np.float32)
#         self.y = np.array(y, dtype=int)

#     def __getitem__(self):
#         example_indices = np.arange(self.size)
#         if self.shuffle:
#             np.random.shuffle(example_indices)
#         for idx in range(0, self.size, self.batch_size):
#             indices = example_indices[idx : idx + self.batch_size]
#             yield (self.x[indices], self.y[indices])

#     def __len__(self):
#         return self.size


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
    def numpy_collate(batch: List[np.array]) -> Tuple[np.array, np.array]:
        return np.stack([b[0] for b in batch]), np.array(
            [b[1] for b in batch], dtype=int
        )

    @staticmethod
    def jax_collate(batch):
        return jnp.stack([b[0] for b in batch]), np.array(
            [b[1] for b in batch], dtype=int
        )

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
                train_kwargs["collate_fn"] = CIFAR10Creator.numpy_collate
            if not "collate_fn" in test_kwargs:
                test_kwargs["collate_fn"] = CIFAR10Creator.numpy_collate
        train_dl = DataLoader(train_ds, *train_args, **train_kwargs)
        test_dl = DataLoader(test_ds, *test_args, **test_kwargs)
        return train_dl, test_dl
