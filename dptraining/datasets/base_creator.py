import abc
from typing import Tuple
from torch.utils.data import Dataset, DataLoader


class DataLoaderCreator(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def make_datasets(
        config: dict, transforms: Tuple
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
