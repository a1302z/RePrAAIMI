from typing import Tuple, Any
from torch import Generator  # pylint:disable=no-name-in-module
from torch.utils.data import Dataset, random_split, DataLoader
from dptraining.datasets.base_creator import DataLoaderCreator
import torchvision as tv
import torchxrayvision as xrv

import collections
import os
import os.path
import pprint
import random
import sys
import tarfile
import warnings
import zipfile

import imageio
import numpy as np
import pandas as pd
import skimage
from typing import Dict
import skimage.transform
from skimage.transform import resize
from skimage.io import imread
import torch
from torchvision import transforms
from torchvision.transforms import Resize


class MIMIC_dataset(xrv.datasets.Dataset):
    """MIMIC-CXR Dataset
    Johnson AE, Pollard TJ, Berkowitz S, Greenbaum NR, Lungren MP, Deng CY, Mark RG, Horng S.
    MIMIC-CXR: A large publicly available database of labeled chest radiographs.
    arXiv preprint arXiv:1901.07042. 2019 Jan 21.
    https://arxiv.org/abs/1901.07042
    Dataset website here:
    https://physionet.org/content/mimic-cxr-jpg/2.0.0/
    """

    def __init__(self,
                 imgpath,
                 csvpath,
                 metacsvpath,
                 views=["PA"],
                 transform=None,
                 data_aug=None,
                 flat_dir=True,
                 seed=0,
                 unique_patients=True
                 ):

        super(MIMIC_dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.

        self.pathologies = ["Enlarged Cardiomediastinum",
                            "Cardiomegaly",
                            "Lung Opacity",
                            "Lung Lesion",
                            "Edema",
                            "Consolidation",
                            "Pneumonia",
                            "Atelectasis",
                            "Pneumothorax",
                            "Pleural Effusion",
                            "Pleural Other",
                            "Fracture",
                            "Support Devices"]

        self.pathologies = sorted(self.pathologies)

        self.imgpath = imgpath
        self.transform = transform
        self.data_aug = data_aug
        self.csvpath = csvpath
        self.csv = pd.read_csv(self.csvpath)
        self.metacsvpath = metacsvpath
        self.metacsv = pd.read_csv(self.metacsvpath)

        self.csv = self.csv.set_index(['subject_id', 'study_id'])
        self.metacsv = self.metacsv.set_index(['subject_id', 'study_id'])

        self.csv = self.csv.join(self.metacsv).reset_index()

        # Keep only the desired view
        self.csv["view"] = self.csv["ViewPosition"]
        self.limit_to_selected_views(views)

        if unique_patients:
            self.csv = self.csv.groupby("subject_id").first().reset_index()

        # Get our classes.
        healthy = self.csv["No Finding"] == 1
        self.labels = []
        for pathology in self.pathologies:
            if pathology in self.csv.columns:
                self.csv.loc[healthy, pathology] = 0
                mask = self.csv[pathology]

            self.labels.append(mask.values)
        self.labels = np.asarray(self.labels).T
        self.labels = self.labels.astype(np.int) #np.float32

        # Make all the -1 values into nans to keep things simple
        #self.labels[self.labels == -1] = -99999

        # Rename pathologies
        self.pathologies = np.char.replace(self.pathologies, "Pleural Effusion", "Effusion")

        # add consistent csv values

        # offset_day_int
        self.csv["offset_day_int"] = self.csv["StudyDate"]

        # patientid
        self.csv["patientid"] = self.csv["subject_id"].astype(str)

    def string(self):
        return self.__class__.__name__ + " num_samples={} views={} data_aug={}".format(len(self), self.views, self.data_aug)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx) -> Tuple[Any,Any]:
        subjectid = str(self.csv.iloc[idx]["subject_id"])
        studyid = str(self.csv.iloc[idx]["study_id"])
        dicom_id = str(self.csv.iloc[idx]["dicom_id"])

        img_path = os.path.join(self.imgpath, "p" + subjectid[:2], "p" + subjectid, "s" + studyid, dicom_id + ".jpg")
        img = imread(img_path)
        img = resize(img,(256,256))
        img = xrv.datasets.normalize(img, maxval=255, reshape=True)
        img = xrv.datasets.apply_transforms(img, self.transform)
        img = xrv.datasets.apply_transforms(img, self.data_aug)
        return img, self.labels[idx]

class MIMICCreator(DataLoaderCreator):
    @staticmethod
    def make_datasets(config, transforms) -> Tuple[Dataset, Dataset]:
        mimic_dataset = MIMIC_dataset(
            imgpath = config['dataset']['root']['img'],
            csvpath = config['dataset']['root']['csv'],
            metacsvpath = config['dataset']['root']['meta'], 
            views = ['PA', 'AP'], 
            unique_patients = False
        )
        #print(mimic_dataset)
        #print('imgpath', config['dataset']['root']['img'])
        #print('csvpath', config['dataset']['root']['csv'])
        #print('metacsvpath', config['dataset']['root']['meta'])
        # split the data
        train_size = int(0.8*len(mimic_dataset))
        val_size = int(0.1*len(mimic_dataset))
        test_size = int(len(mimic_dataset)-train_size-val_size)
        train_data, val_data, test_data = random_split(
                mimic_dataset, [train_size, val_size, test_size]
        )
        '''for data in train_data:
            print(type(data))
            print(data)'''
        train_ds = train_data
        val_ds = val_data
        test_ds = test_data
        return train_ds, val_ds , test_ds

    @staticmethod
    def make_dataloader(  # pylint:disable=too-many-arguments,duplicate-code
        train_ds: Dataset,
        val_ds: Dataset,
        test_ds: Dataset,
        train_kwargs,
        val_kwargs,
        test_kwargs,
    ) -> Tuple[DataLoader, DataLoader]:
        train_dl = DataLoader(train_ds, **train_kwargs)
        val_dl = DataLoader(val_ds, **val_kwargs) if val_ds is not None else None
        test_dl = DataLoader(test_ds, **test_kwargs)

        return train_dl, val_dl, test_dl
