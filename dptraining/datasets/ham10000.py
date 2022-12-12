# %%
from pathlib import Path
from typing import Callable, Optional, Tuple
import numpy as np

import pandas as pd
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from sklearn.model_selection import train_test_split

import sys

sys.path.insert(0, str(Path.cwd()))

from dptraining.datasets.base_creator import DataLoaderCreator

IDENTITY = lambda _: _


class HAM10000(Dataset):
    def __init__(
        self,
        root_dir: Path,
        metadata: Optional[pd.DataFrame] = None,
        merge_labels: bool = True,
        transform: Optional[Callable] = None,
        label_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.root = root_dir
        self.metadata = (
            pd.read_csv(root_dir / "HAM10000_metadata.csv")
            if metadata is None
            else metadata
        )
        self.imgs = {
            img.stem: img
            for img in (root_dir / "HAM10000_images").rglob("*.jpg")
            if img.is_file()
        }
        if merge_labels:
            # seperated in needs attention vs no urgent attention required
            self.metadata["label"] = (
                self.metadata["dx"].isin(["akiec", "bcc", "mel"]).astype(int)
            )
        else:
            label_assignment = {val: i for i, val in enumerate(df.dx.unique())}
            self.metadata["label"] = self.metadata.dx.map(label_assignment).astype(int)

        self.transform = transform if transform is not None else IDENTITY
        self.label_transform = (
            label_transform if label_transform is not None else IDENTITY
        )

    def __getitem__(self, index: int):
        entry = self.metadata.iloc[index]
        img_name = self.imgs[entry.image_id]
        label = entry.label
        img = default_loader(img_name)

        img = self.transform(img)
        label = self.label_transform(label)

        return img, label

    def __len__(self) -> int:
        return len(self.metadata)


class HAM10000Creator(DataLoaderCreator):
    @staticmethod
    def make_datasets(config: dict, transforms: Tuple) -> Tuple[Dataset, Dataset]:
        root = config.dataset.root
        test_split = config.dataset.test_split
        train_split = config.dataset.train_val_split
        merge_labels = config.dataset.merge_labels
        df = pd.read_csv(root / "HAM10000_metadata.csv")
        label_assignment = {val: i for i, val in enumerate(df.dx.unique())}
        df["label"] = df.dx.map(label_assignment).astype(int)
        idcs, lbls = np.arange(len(df)), df.label
        idcs_train, idcs_test, lbls_train, _ = train_test_split(
            idcs, lbls, stratify=True, test_size=test_split
        )
        idcs_train, idcs_val, _, _ = train_test_split(
            idcs_train, lbls_train, stratify=True, train_size=train_split
        )
        train_df = df.iloc[idcs_train]
        val_df = df.iloc[idcs_val]
        test_df = df.iloc[idcs_test]

        return (
            HAM10000(root, metadata=split_df, merge_labels=merge_labels, transform=tf)
            for split_df, tf in zip([train_df, val_df, test_df], transforms)
        )


# %%
if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from torchvision import transforms
    from tqdm import tqdm
    from time import time

    df = pd.read_csv(
        Path.cwd().parent.parent / "./data/dataverse_files/HAM10000_metadata.csv"
    )
    print(df.dx.value_counts())

    label_assignment = {val: i for i, val in enumerate(df.dx.unique())}
    print(label_assignment)
    df["label"] = df.dx.map(label_assignment)
    print(df["label"].value_counts())

    df["needsattention"] = df["dx"].isin(["akiec", "bcc", "mel"]).astype(int)
    print(df.needsattention.value_counts())

    ds = HAM10000(Path.cwd().parent.parent / "./data/dataverse_files/")
    print(len(ds))
    img, label = ds[0]
    print(type(img))
    plt.imshow(img)
    plt.show()

    def iterate_ds(ds):
        for i in tqdm(range(len(ds)), total=len(ds)):
            ds[i]

    t0 = time()
    iterate_ds(ds)
    t1 = time()
    print(f"Iterating without transforms takes {t1-t0:.1f} seconds")

    ds = HAM10000(
        Path.cwd().parent.parent / "./data/dataverse_files/",
        transform=transforms.Compose(
            [transforms.Resize(224), transforms.CenterCrop(224), lambda x: np.array(x)]
        ),
    )
    img, label = ds[0]
    plt.imshow(img)
    plt.show()
    ds = HAM10000(
        Path.cwd().parent.parent / "./data/dataverse_files/",
        transform=transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                lambda x: np.array(x).transpose(2, 0, 1).astype(np.float32) / 255.0,
            ]
        ),
    )
    img, label = ds[0]
    print(img.shape)
    print(img.min())
    print(img.max())
    t0 = time()
    iterate_ds(ds)
    t1 = time()
    print(f"Iterating with transforms takes {t1-t0:.1f} seconds")

# %%
