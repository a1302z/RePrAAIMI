from copy import deepcopy
from pathlib import Path
from pickle import load
from typing import Callable, Optional, Tuple

import numpy as np
from torch import Generator  # pylint:disable=no-name-in-module
from torch.utils.data import Dataset, random_split
from tqdm import tqdm

from dptraining.config import Config
from dptraining.datasets.base_creator import DataLoaderCreator


def unpickle(file):
    with open(file, "rb") as open_file:
        data_dict = load(open_file)
    return data_dict


class TinyImageNet(Dataset):
    def __init__(
        self,
        root: str,
        train: bool,
        version: int,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        assert version in [32, 64]
        self.transform = transform if transform is not None else lambda x: x
        self.target_tf = (
            target_transform if target_transform is not None else lambda x: x
        )
        root = Path(root)
        version_squared = version * version
        self.imgs, self.labels = None, None
        if train:
            self.imgs, self.labels = [], []
            for i in tqdm(
                range(9), total=10, leave=False, desc="Loading tiny imagenet..."
            ):
                data_dict = unpickle(root / f"train_data_batch_{i+1}")
                img_data = data_dict["data"]
                label_data = data_dict["labels"]
                img_data = img_data / np.float32(255)
                label_data = [i - 1 for i in label_data]
                img_data = np.dstack(
                    (
                        img_data[:, :version_squared],
                        img_data[:, version_squared : 2 * version_squared],
                        img_data[:, 2 * version_squared :],
                    )
                )
                img_data = img_data.reshape(
                    (img_data.shape[0], version, version, 3)
                )  # .transpose(0, 3, 1, 2)
                self.imgs.append(img_data)
                self.labels.extend(label_data)
            self.imgs = np.concatenate(self.imgs, axis=0)
        else:
            data_dict = unpickle(root / "val_data")
            img_data = data_dict["data"]
            label_data = data_dict["labels"]
            img_data = img_data / np.float32(255)
            label_data = [i - 1 for i in label_data]
            img_data = np.dstack(
                (
                    img_data[:, :version_squared],
                    img_data[:, version_squared : 2 * version_squared],
                    img_data[:, 2 * version_squared :],
                )
            )
            img_data = img_data.reshape(
                (img_data.shape[0], version, version, 3)
            )  # .transpose(0, 3, 1, 2)
            self.imgs = img_data
            self.labels = label_data
        if normalize:
            mean = np.array((0.485, 0.456, 0.406)).reshape(1, 1, 1, 3)
            std = np.array((0.229, 0.224, 0.225)).reshape(1, 1, 1, 3)
            self.imgs = (self.imgs - mean) / std

    def __len__(self):
        assert self.imgs.shape[0] == len(self.labels)
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple:
        return self.transform(self.imgs[index]), self.target_tf(self.labels[index])


class TinyImageNetCreator(DataLoaderCreator):
    @staticmethod
    def make_datasets(config: Config, transforms: Tuple) -> Tuple[Dataset, Dataset]:
        train_tf, val_tf, test_tf = transforms
        train_kwargs = {
            "root": config.dataset.root,
            "version": config.dataset.version,
            "train": True,
            "transform": train_tf,
            "normalize": config.dataset.normalization,
        }
        # val_kwargs = {
        #     "root": config.dataset.root,
        #     "version": config.dataset.version,
        #     "train": True,
        #     "transform": val_tf,
        # }
        test_kwargs = {
            "root": config.dataset.root,
            "version": config.dataset.version,
            "train": False,
            "transform": test_tf,
        }
        train_ds = TinyImageNet(**train_kwargs)
        val_ds = deepcopy(train_ds)
        val_ds.transform = val_tf if val_tf is not None else lambda x: x
        train_val_split = config.dataset.train_val_split
        # pylint:disable=duplicate-code
        assert 0.0 < train_val_split <= 1.0, "Train/Val split must be in (0,1]"
        if abs(train_val_split - 1.0) < 1e-5:
            val_ds = None
        else:
            n_train = int(train_val_split * len(train_ds))
            lengths = (n_train, len(train_ds) - n_train)
            train_ds, _ = random_split(
                train_ds, lengths, generator=Generator().manual_seed(42)
            )
            _, val_ds = random_split(
                val_ds, lengths, generator=Generator().manual_seed(42)
            )
        test_ds = TinyImageNet(**test_kwargs)
        return train_ds, val_ds, test_ds


# if __name__ == "__main__":
#     train_ds = TinyImageNet(root="./data/ImageNet32", train=True, version=32)
#     print(f"{len(train_ds)} train images")
#     val_ds = TinyImageNet(root="./data/ImageNet32", train=False, version=32)
#     print(f"{len(val_ds)} val images")
#     for i in tqdm(
#         range(len(train_ds)), total=len(train_ds), desc="Iterating train ds", leave=True
#     ):
#         train_ds[i]
#     for i in tqdm(
#         range(len(val_ds)), total=len(val_ds), desc="Iterating train ds", leave=True
#     ):
#         val_ds[i]
#     del train_ds, val_ds
#     train_ds = TinyImageNet(root="./data/ImageNet64", train=True, version=64)
#     print(f"{len(train_ds)} train images")
#     val_ds = TinyImageNet(root="./data/ImageNet64", train=False, version=64)
#     print(f"{len(val_ds)} val images")
#     for i in tqdm(
#         range(len(train_ds)), total=len(train_ds), desc="Iterating train ds", leave=True
#     ):
#         train_ds[i]
#     for i in tqdm(
#         range(len(val_ds)), total=len(val_ds), desc="Iterating train ds", leave=True
#     ):
#         val_ds[i]
