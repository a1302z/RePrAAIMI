from pathlib import Path
from tqdm import tqdm
from typing import Tuple
import pickle
import random
import sys

from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from torch import manual_seed
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision import transforms
import numpy as np
import pandas as pd
import torch


sys.path.insert(0, str(Path.cwd())) # TODO remove, only for testing purposes
from dptraining.datasets.base_creator import DataLoaderCreator


# adjusted from https://github.com/gkaissis/PriMIA/blob/master/torchlib/dataloader.py
class PPPP(Dataset):
    
    def __init__(
        self, 
        data_dir,
        train = False,
        single_channel = True,
        transform = None, 
        seed = 1,
    ):
        super().__init__()
        random.seed(seed)
        manual_seed(seed)
        self.train = train
        self.data_dir = data_dir
        self.label_df = pd.read_csv(data_dir/'Labels.csv')
        self.labels = self.label_df[
            self.label_df["Dataset_type"] == ("TRAIN" if train else "TEST")
        ]
        self.single_channel = single_channel
        self.transform = transform
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        row = self.labels.iloc[index]
        label = torch.tensor(row["Numeric_Label"])
        
        infection_type = 'normal'
        if 'bacteria' in row['X_ray_image_name']:
            infection_type = 'bacterial pneumonia'
        elif 'virus' in row['X_ray_image_name']:
            infection_type = 'viral pneumonia'
        path =  self.data_dir / ('train' if self.train else 'test') / infection_type / row["X_ray_image_name"]
        img = Image.open(path)

        if self.single_channel:
            img = ImageOps.grayscale(img)

        if self.transform:
            img = self.transform(img)

        return img, label

    def get_labels(self) -> np.ndarray:
        return np.array(self.labels["Numeric_Label"])

    def stratified_split(self, train_size=0.9):
        labels = self.get_labels()
        indices = list(range(len(labels)))

        indices_train, indices_val, labels_train, labels_val = train_test_split(
                indices, labels, stratify=labels, train_size=train_size, random_state=0
                )
        
        train = Subset(self, indices_train)
        val = Subset(self, indices_val)

        return train, val


class CustomDataLoader():
    def __init__(self, data_dir):
        self.data_dir = data_dir

        self._train_data = None
        self._val_data = None
        self._test_data = None

    @property
    def train_data(self, train_size):
        if self._train_data == None:
            self.make_datasets(train_size)
        return self._train_data

    @property
    def val_data(self, train_size):
        if self._val_data == None:
            self.make_datasets(train_size)
        return self._val_data

    @property
    def test_data(self, train_size):
        if self._test_data == None:
            self.make_datasets(train_size)
        return self._test_data

    def calc_mean_std(self, dataset):
        """ 
        Expects the dataset to consist of tensors of the same size
        """
        images = []
        for data in tqdm(dataset, total=len(dataset), leave=False, desc="calculate mean and std"):
            img = data[0]
            images.append(img)
        
        images = stack(images)

        if images.shape[1] in [1, 3]:  # ugly hack
            dims = (0, *range(2, len(images.shape)))
        else:
            dims = (*range(len(images.shape)),)

        std, mean = std_mean(images, dim=dims)

        return mean, std

    def get_mean_std(self, train_size):
        file_name = "trainsize=" + str(train_size) + ".pkl"
        stats_file = self.data_dir/"stats"/file_name
        
        if stats_file.is_file():
            with open(stats_file, "rb") as f:   # load stats
                mean, std = pickle.load(f)    
                # print(mean, std)

        else: 
            tf = transforms.Compose([
                transforms.Resize(224), 
                transforms.CenterCrop(224), 
                transforms.ToTensor(),
                ])

            train_val = PPPP(train=True, data_dir=self.data_dir, transform=tf)
            train, _ = train_val.stratified_split(train_size=train_size)
            mean, std = self.calc_mean_std(train)

            with open(stats_file, "wb") as f:    # save stats
                pickle.dump(f)

        return mean, std

    def make_datasets(self, train_size, transform):
        """ 
        Returns standardized training, validation and test data.
        Stratified split is used to get train and val datasets.
        """
        mean, std = self.get_mean_std(train_size=train_size)
        
        # basic_transform = transforms.Compose([
        #     transforms.Resize(224), 
        #     transforms.CenterCrop(224), 
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean, std) 
        # ])

        train_val = PPPP(train=True, data_dir=self.data_dir, transform=transform[0])
        test = PPPP(train=False, data_dir=self.data_dir, transform=transform[2])

        train, val = train_val.stratified_split(train_size=train_size)

        self._train_data = train
        self._val_data = val
        self._test_data = test

        return train, val, test


class PPPPCreator(DataLoaderCreator):
    @staticmethod
    def make_datasets(config, transform):
        data_dir = Path("/media/sarah/data/raw/pppp")

        dataloader = CustomDataLoader(data_dir=data_dir)
        train, val, test = dataloader.make_datasets(train_size=0.9, transform=transform)

        return train, val, test

    # @staticmethod
    # def make_dataloader():
    #     train_dl = DataLoader(train, batch_size=16, shuffle=True)
    #     val_dl = DataLoader(val, batch_size=16, shuffle=False)
    #     test_dl = DataLoader(test, batch_size=16, shuffle=False)

    #     return train_dl, val_dl, test_dl


if __name__ == "__main__":
    creator = PPPPCreator()
    train, val, test = creator.make_datasets()
    train_dl, val_dl, test_dl = creator.make_dataloader()