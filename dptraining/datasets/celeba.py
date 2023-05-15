from torch.utils.data import Dataset
from torchvision.datasets import CelebA
from dptraining.config import Config


from dptraining.datasets.base_creator import DataLoaderCreator


class CelebACreator(DataLoaderCreator):
    @staticmethod
    def make_datasets(
        config: Config, transforms: tuple
    ) -> tuple[Dataset, Dataset, Dataset]:
        train_ds = CelebA(
            config.dataset.root,
            split="train",
            transform=transforms[0],
            download=True,
            target_type=config.dataset.celeba.target_type,
        )
        val_ds = CelebA(
            config.dataset.root,
            split="valid",
            transform=transforms[1],
            download=True,
            target_type=config.dataset.celeba.target_type,
        )
        test_ds = CelebA(
            config.dataset.root,
            split="test",
            transform=transforms[2],
            download=True,
            target_type=config.dataset.celeba.target_type,
        )
        return train_ds, val_ds, test_ds
