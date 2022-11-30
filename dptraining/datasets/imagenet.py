from typing import Tuple

from torch import Generator  # pylint:disable=no-name-in-module
from torch.utils.data import Dataset, random_split
from torchvision.datasets import ImageNet

from dptraining.config import Config
from dptraining.datasets.base_creator import DataLoaderCreator


class ImageNetCreator(DataLoaderCreator):
    @staticmethod
    def make_datasets(config: Config, transforms) -> Tuple[Dataset, Dataset]:
        train_tf, val_tf, test_tf = transforms
        train_kwargs = {
            "root": config.dataset.root,
            "split": "train",
            "transform": train_tf,
        }
        val_kwargs = {
            "root": config.dataset.root,
            "split": "train",
            "transform": val_tf,
        }
        test_kwargs = {
            "root": config.dataset.root,
            "split": "val",
            "transform": test_tf,
        }
        train_ds = ImageNet(**train_kwargs)
        val_ds = ImageNet(**val_kwargs)
        train_val_split = config.dataset.train_val_split
        # pylint:disable=duplicate-code
        assert 0.0 < train_val_split <= 1.0, "Train/Val split must be in (0,1]"
        if abs(train_val_split - 1.0) < 1e-5:
            val_ds = None
        else:
            # if isinstance(train_val_split, float) and 0 < train_val_split < 1.0:
            n_train = int(train_val_split * len(train_ds))
            lengths = (n_train, len(train_ds) - n_train)
            # elif isinstance(train_val_split, int):
            #     lengths = (train_val_split, len(train_ds) - train_val_split)
            # else:
            #     raise ValueError(f"train_val_split {train_val_split} could not be parsed.")
            # We do it like that to avoid problems when train and val transforms are different
            train_ds, _ = random_split(
                train_ds, lengths, generator=Generator().manual_seed(42)
            )
            _, val_ds = random_split(
                val_ds, lengths, generator=Generator().manual_seed(42)
            )
        test_ds = ImageNet(**test_kwargs)
        return train_ds, val_ds, test_ds
