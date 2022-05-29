from typing import Tuple
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageNet
from dptraining.datasets.base_creator import DataLoaderCreator


class ImageNetCreator(DataLoaderCreator):
    @staticmethod
    def make_datasets(
        train_args, train_kwargs, test_args, test_kwargs
    ) -> Tuple[Dataset, Dataset]:
        train_ds = ImageNet(*train_args, **train_kwargs)
        test_ds = ImageNet(*test_args, **test_kwargs)
        return train_ds, test_ds
