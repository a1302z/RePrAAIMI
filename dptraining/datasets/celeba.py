from typing import Any, Tuple

import numpy as np
import PIL, os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CelebA
import torchvision.transforms as transforms

from dptraining.config import Config
from dptraining.datasets.base_creator import DataLoaderCreator


def fft_conversion(img, axes=None):
    return np.fft.fftshift(np.fft.fft2(img, axes=axes), axes=axes)


class NumpyCelebA(CelebA):
    def __init__(
        self,
        img_size: int = 64,
        load_into_mem: bool = False,
        use_pickled: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.img_size = img_size
        self.use_pickled = use_pickled
        self.pickled_path = os.path.join(
            self.root,
            self.base_folder,
            "pickled",
            f"pickled_{img_size}_{self.split}.npy",
        )
        self.load_into_mem = load_into_mem
        if self.load_into_mem:
            self.load_imgs_into_mem()

    def load_imgs_into_mem(self):
        data = []
        if self.use_pickled and os.path.isfile(self.pickled_path):
            data = np.load(self.pickled_path)
        else:
            for f_name in self.filename:
                img = PIL.Image.open(
                    os.path.join(
                        self.root, self.base_folder, "img_align_celeba", f_name
                    )
                )
                img = transforms.Resize((self.img_size, self.img_size))(img)
                data.append(img)
            if self.use_pickled:
                np.save(self.pickled_path, np.stack(data, axis=0))
        self.data = np.stack(data, axis=0)

    def reshape_img(self, img):
        img = np.array(img)
        img = img.astype(np.float32)
        img = img.transpose(2, 0, 1)
        return img

    def normalize_img(self, img):
        img = (img - np.reshape(CelebACreator.CELEB_MEAN, [3, 1, 1])) / np.reshape(
            CelebACreator.CELEB_STDDEV, [3, 1, 1]
        )
        return img

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        if self.load_into_mem:
            img = self.data[index]
        else:
            img = PIL.Image.open(
                os.path.join(
                    self.root,
                    self.base_folder,
                    "img_align_celeba",
                    self.filename[index],
                )
            )
            img = transforms.Resize((self.img_size, self.img_size))(img)
        img = self.reshape_img(img)
        img = self.normalize_img(img)
        t_type = self.target_type[0]
        if t_type == "attr":
            target = self.attr[index, :]
        else:
            raise ValueError(f'Target type "{t}" is not supported.')

        if self.transform is not None:
            img = self.transform(img)

        attr = target[20]  # gender attribute (our auxiliary attribute)
        label = target[9]  # blond attribute (our target label)

        if self.target_transform is not None:
            label = self.target_transform(label)
            attr = self.target_transform(attr)

        return img, attr, label


class CelebACreator(DataLoaderCreator):
    CELEB_MEAN = (0.485, 0.456, 0.406)
    CELEB_STDDEV = (0.229, 0.224, 0.225)

    @staticmethod
    def reshape_images(image: np.array):
        image = image.astype(np.float32)
        image = image.transpose(0, 3, 1, 2)  # convert to CHW
        return image

    @staticmethod
    def make_datasets(  # pylint:disable=too-many-arguments
        config: Config,
        transforms: Tuple,
        numpy_optimisation=True,
        normalize_by_default=True,
    ) -> Tuple[Dataset, Dataset, Dataset]:
        train_tf, val_tf, test_tf = transforms
        train_kwargs = {
            "root": config.dataset.root,
            "download": True,
            "transform": train_tf,
            "split": "train",
        }
        val_kwargs = {
            "root": config.dataset.root,
            "download": True,
            "transform": val_tf,
            "split": "valid",
        }
        test_kwargs = {
            "root": config.dataset.root,
            "download": True,
            "transform": test_tf,
            "split": "test",
        }
        if normalize_by_default and not numpy_optimisation:
            raise ValueError(
                "CelebA Creator can only normalize by default if numpy optimisation is activated"
            )
        if numpy_optimisation:
            train_ds = NumpyCelebA(**train_kwargs)
            val_ds = NumpyCelebA(**val_kwargs)
            test_ds = NumpyCelebA(**test_kwargs)
        else:
            train_ds = CelebA(**train_kwargs)
            val_ds = CelebA(**val_kwargs)
            test_ds = CelebA(**test_kwargs)

        return train_ds, val_ds, test_ds

    @staticmethod
    def make_dataloader(  # pylint:disable=too-many-arguments,duplicate-code
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
