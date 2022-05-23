from abc import ABC, abstractclassmethod
from typing import Tuple
from torch.utils.data import Dataset, DataLoader


class DataLoaderCreator(ABC):
    @abstractclassmethod
    def make_datasets(
        train_args, train_kwargs, test_args, test_kwargs
    ) -> Tuple[Dataset, Dataset]:
        pass

    @abstractclassmethod
    def make_dataloader(
        train_ds: Dataset,
        test_ds: Dataset,
        train_args,
        train_kwargs,
        test_args,
        test_kwargs,
    ) -> Tuple[DataLoader, DataLoader]:
        pass
