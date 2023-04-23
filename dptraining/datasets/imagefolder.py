from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split

from dptraining.config import Config
from dptraining.datasets.base_creator import DataLoaderCreator
from dptraining.datasets.subset import DataSubset


class ImageFolderCreator(DataLoaderCreator):
    @staticmethod
    def make_datasets(
        config: Config, transforms: tuple
    ) -> tuple[Dataset, Dataset, Dataset]:
        base_dataset = ImageFolder(config.dataset.root)
        classes = base_dataset.targets
        idcs_train, idcs_val, idcs_test = (
            [i for i in range(len(base_dataset))],
            None,
            None,
        )
        if abs(config.dataset.test_split) > 0.0:
            idcs_test, idcs_train = train_test_split(
                idcs_train,
                stratify=classes,
                random_state=config.dataset.datasplit_seed,
                train_size=config.dataset.test_split,
            )
        if abs(config.dataset.train_val_split - 1.0) > 0.0:
            idcs_train, idcs_val = train_test_split(
                idcs_train,
                stratify=[classes[c] for c in idcs_train],
                random_state=config.dataset.datasplit_seed,
                train_size=config.dataset.train_val_split,
            )
        train_ds = DataSubset(base_dataset, idcs_train, transforms[0])
        val_ds = DataSubset(base_dataset, idcs_val, transforms[1]) if idcs_val else None
        test_ds = (
            DataSubset(base_dataset, idcs_test, transforms[2]) if idcs_test else None
        )
        return train_ds, val_ds, test_ds
