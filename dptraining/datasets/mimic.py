import pandas as pd
from pandas import DataFrame
from pathlib import Path
from PIL import Image
import numpy as np
from torchvision.datasets import VisionDataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from typing import Tuple, Union

from dptraining.config import Config
from dptraining.datasets.base_creator import DataLoaderCreator


class MIMICDataset(VisionDataset):
    def __init__(
        self,
        df: DataFrame,
        root: Path,
        img_size: int = 64,
        transform = None,
        target_transform = None,
    ):
        super().__init__(root=root, transform=transform, target_transform=target_transform)
        self.dataframe = pd.read_csv(df) if not isinstance(df, pd.DataFrame) else df
        self.img_size = img_size
        self.MIMIC_MEAN = 0.47411758
        self.MIMIC_STD = 0.29748
        self.DEFAULT_SIZE = 224

    def __getitem__(self, index: int):
        """Returns a tuple of (image, attributes, target)"""

        row = self.dataframe.iloc[index]
        path_parts = Path(row["path"]).parts[-2:]
        path_str = f"{path_parts[0]}_{path_parts[1]}"
        img_path = self.root / Path(path_str)
        assert img_path.exists(), f"Image path {img_path} does not exist"
        # load in scan image
        img = Image.open(img_path)
        if self.img_size != self.DEFAULT_SIZE:
            img = transforms.Resize((self.img_size, self.img_size))(img)
        img = self.normalize(np.array(img).astype(np.float32))
        img = np.expand_dims(img, axis=0)  # add channel axis
        # group attributes
        sex, race = row["sex_label"], row["race_label"]
        attrs = {"sex": (sex, [0, 1]), "race": (race, [0, 1, 2])}
        # target
        target = row[
            "disease_label"
        ]
        # merge disease labels
        if target == 2:
            target = 1
        if self.transform is not None:
            img = self.transform(img)
        return img, attrs, target

    def normalize(self, img):
        img /= 255.0
        return (img - self.MIMIC_MEAN) / self.MIMIC_STD

    def __len__(self) -> int:
        return len(self.dataframe)


class MIMICCreator(DataLoaderCreator):
    TRAIN_DF = "mimic.sample.train.csv"
    VAL_DF = "mimic.sample.val.csv"
    TEST_DF = "mimic.sample.test.csv"

    @staticmethod
    def make_datasets(
        config: Config, transforms: Tuple
    ) -> Tuple[Dataset, Dataset, Dataset]:
        train_tf, val_tf, test_tf = transforms
        metadata_dir = Path(config.dataset.root) / "meta"
        data_dir = Path(config.dataset.root) / "data" / "preproc_224x224"
        # get dataframes
        train_df = metadata_dir / MIMICCreator.TRAIN_DF
        val_df = metadata_dir / MIMICCreator.VAL_DF
        test_df = metadata_dir / MIMICCreator.TEST_DF
        # create datasets
        train_ds = MIMICDataset(train_df, data_dir, transform=train_tf)
        val_ds = MIMICDataset(val_df, data_dir, transform=val_tf)
        test_ds = MIMICDataset(test_df, data_dir, transform=test_tf)
        return train_ds, val_ds, test_ds

    @staticmethod
    def make_dataloaders(
        train_ds,
        val_ds,
        test_ds,
        train_kwargs,
        val_kwargs,
        test_kwargs,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        train_dl = DataLoader(train_ds, **train_kwargs)
        val_dl = DataLoader(val_ds, **val_kwargs)
        test_dl = DataLoader(test_ds, **test_kwargs)
        return train_dl, val_dl, test_dl
