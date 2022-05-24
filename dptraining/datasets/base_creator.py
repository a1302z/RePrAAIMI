from abc import ABC, abstractstaticmethod
from typing import Tuple
from torch.utils.data import Dataset, DataLoader


class DataLoaderCreator(ABC):
    @abstractstaticmethod
    def make_datasets(
        train_args, train_kwargs, test_args, test_kwargs
    ) -> Tuple[Dataset, Dataset]:
        pass

    @abstractstaticmethod
    def make_dataloader(
        train_ds: Dataset,
        test_ds: Dataset,
        train_args,
        train_kwargs,
        test_args,
        test_kwargs,
    ) -> Tuple[DataLoader, DataLoader]:
        pass
