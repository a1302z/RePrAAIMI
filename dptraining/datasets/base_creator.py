import abc
from pathlib import Path
from tqdm import tqdm
from typing import Tuple
from torch.utils.data import Dataset, DataLoader
from numpy import newaxis, sqrt
from dptraining.config import Config


class DataLoaderCreator(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def make_datasets(
        config: Config, transforms: Tuple
    ) -> Tuple[Dataset, Dataset, Dataset]:
        pass

    @staticmethod
    def make_dataloader(  # pylint:disable=too-many-arguments
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


def mk_subdirectories(path: Path, subdirs: list[str]) -> list[Path]:
    new_dirs = []
    for subdir in subdirs:
        new_path: Path = path / subdir
        if not new_path.is_dir():
            new_path.mkdir()
        new_dirs.append(new_path)
    return new_dirs


def calc_mean_std(dataset: DataLoader):
    mean = 0.0
    for images, _ in tqdm(
        dataset, total=len(dataset), desc="calculating mean", leave=False
    ):
        batch_samples = images.shape[0]
        images = images.reshape((batch_samples, images.shape[1], -1))
        mean += images.mean(2).sum(0)
    mean = mean / len(dataset.dataset)

    var = 0.0
    reshaped_mean = mean[newaxis, ...]
    num_px = 0
    for images, _ in tqdm(
        dataset, total=len(dataset), desc="calculating std", leave=False
    ):
        batch_samples = images.shape[0]
        images = images.reshape(batch_samples, images.shape[1], -1)
        var += ((images - reshaped_mean) ** 2).sum(2).sum(0)
        num_px += images.shape[2]
    std = sqrt(var / (len(dataset.dataset) * num_px))
    return mean, std
