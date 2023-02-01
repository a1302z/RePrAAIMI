from typing import Tuple
from torch import Generator  # pylint:disable=no-name-in-module
from torch.utils.data import Dataset, random_split
import medmnist
from medmnist import INFO
from dptraining.datasets.base_creator import DataLoaderCreator
import torchvision as tv

class pneumoniamnistCreator(DataLoaderCreator):
    @staticmethod
    def make_datasets(config, transforms) -> Tuple[Dataset, Dataset]:
        transform = tv.transforms.Compose([tv.transforms.ToTensor()])
        download = False
        train_kwargs = {
            "root": config["dataset"]["root"],
            'split': "train",
            "transform": transform,
            "download": download,
        }
        val_kwargs = {
            "root": config["dataset"]["root"],
            "split": "val",
            "transform": transform,
            "download": download,
        }
        test_kwargs = {
            "root": config["dataset"]["root"],
            "split": "test",
            "transform": transform,
            "download": download,
        }
        data_flag = 'pneumoniamnist'
        info = INFO[data_flag]
        DataClass = getattr(medmnist, info['python_class'])
        # load the data
        train_ds = DataClass(**train_kwargs)
        val_ds = DataClass(**val_kwargs)
        test_ds = DataClass(**test_kwargs)
        #print('train', type(train_ds))
        #print('validation', type(train_ds[0]))
        #print('test', train_ds[0])
        return train_ds, val_ds, test_ds
