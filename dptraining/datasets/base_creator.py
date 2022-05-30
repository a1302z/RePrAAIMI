import abc
from typing import Tuple
from torch.utils.data import Dataset, DataLoader


class DataLoaderCreator(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def make_datasets(
        train_args, train_kwargs, test_args, test_kwargs
    ) -> Tuple[Dataset, Dataset]:
        pass

    @staticmethod
    def make_dataloader(  # pylint:disable=too-many-arguments
        train_ds: Dataset,
        test_ds: Dataset,
        train_args,
        train_kwargs,
        test_args,
        test_kwargs,
    ) -> Tuple[DataLoader, DataLoader]:
        train_dl = DataLoader(train_ds, *train_args, **train_kwargs)
        test_dl = DataLoader(test_ds, *test_args, **test_kwargs)
        return train_dl, test_dl
