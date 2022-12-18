import sys
from enum import Enum
from pathlib import Path
from random import seed, shuffle
from typing import Callable, Optional, Tuple

from time import time

import ctypes
import multiprocessing as mp

import nibabel as nib
import numpy as np
from nilearn.image import resample_img
from torch.utils.data import Dataset

sys.path.insert(0, str(Path.cwd()))

from dptraining.config import Config, Normalization
from dptraining.datasets.base_creator import DataLoaderCreator, mk_subdirectories

# We are assuming that all tasks are structured as Task03_Liver


def scale_array_zero_one(array: np.array) -> np.array:
    """Scales a numpy array from 0 to 1. Works in 3D
    Return np.array"""
    assert array.max() - array.min() > 0

    return ((array - array.min()) / (array.max() - array.min())).astype(np.float32)


def scale_array_unit_gaussian(
    array: np.array, mean: np.array, std: np.array
) -> np.array:
    return (array - mean) / std


def rotate_label(label_volume) -> np.array:
    """Rotates and flips the label in the same way the scans were rotated and flipped
    Return: np.array"""

    label_volume = np.rot90(label_volume)
    label_volume = np.fliplr(label_volume)

    return label_volume.astype(np.float32)


class MSD(Dataset):
    # raw_stats = np.array([-20.76621642152709, 10.325427899115375])
    raw_stats = np.array([-41.432015233869066, 9.509903769499498])
    consecutive_stats = np.array([0.06497363541417449, 0.027557578189309158])

    def __init__(
        self,
        matched_labeled_scans: list[tuple[str, Path, Path]],
        transform: Optional[Callable] = None,
        label_transform: Optional[Callable] = None,
        resolution: Optional[int] = None,
        slice_thickness: Optional[int] = None,
        normalization: Normalization = Normalization.raw,
        data_mean_std: Optional[tuple[np.array, np.array]] = None,
        cache_files: bool = False,
    ) -> None:
        super().__init__()
        self.matched_labeled_scans: list[tuple[str, Path, Path]] = matched_labeled_scans
        self.transform = transform
        self.label_transform = label_transform
        self.resolution: Optional[int] = resolution
        self.slice_thickness: Optional[int] = slice_thickness
        self.normalization: Normalization = normalization
        self.cache: bool = cache_files
        self.cached_files: Optional[mp.Array]
        if self.cache:
            # self.cached_files = [False for _ in self.matched_labeled_scans]
            self.cached_files = mp.Array(
                ctypes.c_bool, [False] * len(self.matched_labeled_scans), lock=True
            )
        else:
            self.cached_files = None
        if self.normalization == Normalization.gaussian:
            if data_mean_std is not None:
                self.mean = data_mean_std[0]
                self.std = data_mean_std[1]
            else:
                raise ValueError(
                    "Normalization supposed to be gaussian "
                    "but no mean and variance provided"
                )

    def __len__(self) -> int:
        return len(self.matched_labeled_scans)

    def __getitem__(self, index: int) -> Tuple[np.array, np.array]:
        _, img_file, label_file = self.matched_labeled_scans[index]
        # print(f"{index} cached: {self.cached_files[index]}")
        # t0 = time()
        if self.cache and self.cached_files[index]:
            scan_path, label_path = self.create_new_filenames(img_file, label_file)
            scan, label = self.load_np_files(scan_path, label_path)
        else:
            scan, label = self.load_nifti_files(index, img_file, label_file)
        scan = self.transform(scan) if self.transform is not None else scan
        label = (
            self.label_transform(label) if self.label_transform is not None else label
        )
        # t1 = time()
        # print(f"\t Loading took {t1-t0:.1f} seconds")
        return scan, label

    def load_nifti_files(self, index, img_file, label_file):
        if self.cache:
            scan_path, label_path = self.create_new_filenames(img_file, label_file)
        if img_file.is_symlink():
            img_file = img_file.readlink()
        if label_file.is_symlink():
            label_file = label_file.readlink()
        scan = nib.load(img_file)
        label = nib.load(label_file)
        if self.resolution or self.slice_thickness:
            scan = self.resize_scan(scan, label=False)
            label = self.resize_scan(label, label=True)

        scan, label = self.preprocess_and_convert_to_numpy(scan, label)
        if self.cache:
            np.save(scan_path, scan)
            np.save(label_path, label)
            # self.matched_labeled_scans[index] = (
            #     self.matched_labeled_scans[index][0],
            #     scan_path,
            #     label_path,
            # )
            self.cached_files[index] = np.array(True, dtype=np.bool8)
        return scan, label

    def load_np_files(self, scan_path, label_path):
        scan = np.load(scan_path)
        label = np.load(label_path)
        return scan, label

    def create_new_filenames(self, img_file, label_file):
        scan_path = img_file.parent / f"preprocessed_scan_{img_file.stem}.npy"
        label_path = label_file.parent / f"preprocessed_label_{label_file.stem}.npy"

        return scan_path, label_path

    def preprocess_scan(self, scan) -> np.array:
        """Performs Preprocessing:
        - clips vales to -150 to 200,
        - peforms rotations and flipping to move patient into reference position
        Return: np.array"""
        scan = np.clip(scan, -150, 200)
        match self.normalization:
            case Normalization.zeroone:
                scan = scale_array_zero_one(scan)
            case Normalization.gaussian:
                scan = scale_array_unit_gaussian(scan, self.mean, self.std)
            case Normalization.consecutive:
                scan = scale_array_zero_one(scan)
                scan = scale_array_unit_gaussian(scan, self.mean, self.std)
        scan = np.rot90(scan)
        scan = np.fliplr(scan)

        return scan

    def preprocess_and_convert_to_numpy(
        self, nifti_scan: nib.Nifti1Image, nifti_mask: nib.Nifti1Image
    ) -> list:
        """Convert scan and label to numpy arrays and perform preprocessing
        Return: Tuple(np.array, np.array)"""
        np_scan = nifti_scan.get_fdata()
        np_label = nifti_mask.get_fdata()
        nifti_mask.uncache()
        nifti_scan.uncache()
        np_scan = self.preprocess_scan(np_scan)
        np_label = rotate_label(np_label)
        assert np_scan.shape == np_label.shape

        return np_scan, np_label

    def resize_scan(self, scan, label: bool):
        data_shape = scan.header.get_data_shape()
        zooms = scan.header.get_zooms()
        print(
            f"Actual size: {data_shape[0]*zooms[0]:.1f}mm x {data_shape[1]*zooms[1]:.1f}mm x {data_shape[2]*zooms[2]:.1f}mm"
        )
        new_shape = (
            self.resolution if self.resolution else data_shape[0],
            self.resolution if self.resolution else data_shape[0],
            int(data_shape[-1] * zooms[2] / self.slice_thickness)
            if self.slice_thickness
            else data_shape[-1],
        )
        new_affine = np.copy(scan.affine)
        for i in range(2):
            new_affine[i, i] *= data_shape[i] / new_shape[i]
        # print(zooms)
        if self.slice_thickness:
            new_affine[2, 2] = self.slice_thickness
        else:
            new_affine[2, 2] = data_shape[-1]
        scan = resample_img(
            scan,
            target_affine=new_affine,
            target_shape=new_shape,
            interpolation="nearest" if label else "continuous",
        )
        new_zooms = scan.header.get_zooms()

        print(
            f"New real size: {new_shape[0]*new_zooms[0]:.1f}mm x {new_shape[1]*new_zooms[1]:.1f}mm x {new_shape[2]*new_zooms[2]:.1f}mm"
        )
        return scan


class MSDCreator(DataLoaderCreator):

    subtask_paths = {
        1: "Task01_BrainTumour",
        2: "Task02_Heart",
        3: "Task03_Liver",
        4: "Task04_Hippocampus",
        5: "Task05_Prostate",
        6: "Task06_Lung",
        7: "Task07_Pancreas",
        8: "Task08_HepaticVessel",
        9: "Task09_Spleen",
        10: "Task10_Colon",
    }

    @staticmethod
    def make_datasets(
        config: Config, transforms: Tuple
    ) -> Tuple[Dataset, Dataset, Dataset]:
        root = Path(config.dataset.root)
        if config.dataset.subtask:
            root = root / MSDCreator.subtask_paths[config.dataset.subtask.value]
        seed(config.dataset.datasplit_seed)

        train_split, test_split = (
            config.dataset.train_val_split,
            config.dataset.test_split,
        )

        labeled_scans_path = root / "imagesTr"
        label_path = root / "labelsTr"
        labeled_scans = {
            scan.stem.replace(".nii", ""): scan
            for scan in labeled_scans_path.rglob("*.nii.gz")
            if not scan.stem.startswith("._")
        }
        label_files = {
            scan.stem.replace(".nii", ""): scan
            for scan in label_path.rglob("*.nii.gz")
            if not scan.stem.startswith("._")
        }

        assert set(labeled_scans.keys()) == set(label_files.keys())

        matched_labeled_scans = [
            (name, labeled_scans[name], label_files[name])
            for name in labeled_scans.keys()
        ]
        shuffle(matched_labeled_scans)
        L_train = int(round(len(matched_labeled_scans) * test_split))
        train_files = matched_labeled_scans[L_train:]
        test_files = matched_labeled_scans[:L_train]
        L_train = int(round(len(train_files) * train_split))
        val_files = train_files[L_train:]
        train_files = train_files[:L_train]
        subdirs = mk_subdirectories(root, ["train_split", "val_split", "test_split"])
        split_files = (train_files, val_files, test_files)

        split_matched_labeled_scans = []
        for directory, files in zip(subdirs, split_files):
            scan_subdir, label_subdir = mk_subdirectories(
                directory, ["images", "labels"]
            )
            sub_matched_labeled_scans = []
            for (name, scan_path, label_path) in files:
                new_scan_path = scan_subdir / name
                new_label_path = label_subdir / name
                if not new_scan_path.is_symlink():
                    new_scan_path.symlink_to(scan_path)
                if not new_label_path.is_symlink():
                    new_label_path.symlink_to(label_path)
                sub_matched_labeled_scans.append((name, new_scan_path, new_label_path))
            split_matched_labeled_scans.append(sub_matched_labeled_scans)
        train_ds, val_ds, test_ds = (
            MSD(
                smls,
                transform=tf,
                resolution=config.dataset.resolution,
                slice_thickness=config.dataset.slice_thickness,
                normalization=config.dataset.normalization_type,
                data_mean_std=MSD.raw_stats,
                cache_files=config.dataset.cache,
            )
            for smls, tf in zip(split_matched_labeled_scans, transforms)
        )
        return train_ds, val_ds, test_ds


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    def extend_base_config(overrides: dict):
        base_conf = OmegaConf.structured(Config)
        merged_conf = OmegaConf.merge(base_conf, overrides)
        return merged_conf

    def collate_fn(list_of_data_tuples: list[tuple[np.array, np.array]]):
        scans = [item[0] for item in list_of_data_tuples]
        labels = [item[1] for item in list_of_data_tuples]
        scans = np.concatenate(scans, axis=2)[np.newaxis, ...].transpose(3, 0, 1, 2)
        labels = np.concatenate(labels, axis=2)[np.newaxis, ...].transpose(3, 0, 1, 2)
        return scans, labels

    def calc_mean_std(dataset: DataLoader):
        mean = 0.0
        for images, _ in tqdm(
            dataset, total=len(dataset), desc="calculating mean", leave=False
        ):
            mean += np.mean(images)
        mean = mean / len(dataset.dataset)

        var = 0.0
        N_px = 0
        for images, _ in tqdm(
            dataset, total=len(dataset), desc="calculating std", leave=False
        ):
            var += ((images - mean) ** 2).sum()
            N_px += np.prod(images.shape)
        std = np.sqrt(var / (len(dataset.dataset) * N_px))
        return mean, std

    # config = extend_base_config(
    #     {
    #         "project": "test",
    #         "general": {"log_wandb": False, "cpu": True, "eval_train": False},
    #         "loader": {"num_workers": 0, "collate_fn": "numpy"},
    #         "dataset": {
    #             "name": "msd",
    #             "root": "./data/MSD/",
    #             "subtask": "liver",
    #             "train_val_split": 0.9,
    #             "test_split": 0.1,
    #             "datasplit_seed": 0,
    #             "task": "segmentation",
    #             "resolution": 256,
    #         },
    #     }
    # )
    # train_ds, val_ds, test_ds = MSDCreator.make_datasets(config, (None, None, None))
    # print(
    #     f"{len(train_ds)} train files\t {len(val_ds)} val files\t {len(test_ds)} test files"
    # )
    # scan, label = train_ds[0]
    # print(scan.shape)
    # print(label.shape)

    # N = int(np.ceil(np.sqrt(scan.shape[-1])))
    # fig, axs = plt.subplots(N, N, sharex=True, sharey=True)
    # for i in range(N):
    #     for j in range(N):
    #         if i * N + j >= scan.shape[-1]:
    #             break
    #         axs[i, j].imshow(scan[:, :, i * N + j], cmap="gray")
    #         img = axs[i, j].imshow(
    #             label[:, :, i * N + j] / 2.0,
    #             cmap="inferno",
    #             alpha=0.3,
    #             vmin=0,
    #             vmax=1.0,
    #         )
    #         axs[i, j].set_axis_off()
    # plt.savefig("test_msd.png", dpi=600)

    # config = extend_base_config(
    #     {
    #         "project": "test",
    #         "general": {"log_wandb": False, "cpu": True, "eval_train": False},
    #         "loader": {"num_workers": 0, "collate_fn": "numpy"},
    #         "dataset": {
    #             "name": "msd",
    #             "root": "./data/MSD/",
    #             "subtask": "liver",
    #             "train_val_split": 1.0,
    #             "test_split": 0.0,
    #             "datasplit_seed": 0,
    #             "task": "segmentation",
    #             "cache": False,
    #             "normalization": "raw",
    #             "slice_thickness": None,
    #         },
    #     }
    # )
    # train_ds, val_ds, test_ds = MSDCreator.make_datasets(config, (None, None, None))
    # train_dl = DataLoader(
    #     train_ds,
    #     batch_size=2,
    #     shuffle=False,
    #     num_workers=4,
    #     prefetch_factor=2,
    #     collate_fn=collate_fn,
    # )
    # print(calc_mean_std(train_dl))
    # print(MSD.raw_stats)
    # N = 4
    # t0 = time()
    # for i in tqdm(range(N), total=N, leave=False):
    #     train_ds[i]
    # t1 = time()
    # print(f"First access took {t1-t0:.2f}s")
    # t0 = time()
    # for i in tqdm(range(N), total=N, leave=False):
    #     train_ds[i]
    # t1 = time()
    # print(f"Second access took {t1-t0:.2f}s")

    config = extend_base_config(
        {
            "project": "test",
            "general": {"log_wandb": False, "cpu": True, "eval_train": False},
            "loader": {"num_workers": 0, "collate_fn": "numpy"},
            "dataset": {
                "name": "msd",
                "root": "./data/MSD/",
                "subtask": "liver",
                "train_val_split": 1.0,
                "test_split": 0.0,
                "datasplit_seed": 0,
                "task": "segmentation",
                "cache": False,
                "normalization": "gaussian",
                "slice_thickness": None,
            },
        }
    )
    train_ds, val_ds, test_ds = MSDCreator.make_datasets(config, (None, None, None))

    train_dl = DataLoader(
        train_ds,
        batch_size=2,
        shuffle=False,
        num_workers=4,
        prefetch_factor=2,
        collate_fn=collate_fn,
    )
    print(calc_mean_std(train_dl))

    # N = 16
    # t0 = time()
    # for i, _ in tqdm(enumerate(train_dl), total=N, leave=False):
    #     if i > N:
    #         break
    # t1 = time()
    # print(f"First access took {t1-t0:.2f}s")
    # t0 = time()
    # for i, _ in tqdm(enumerate(train_dl), total=N, leave=False):
    #     if i > N:
    #         break
    # t1 = time()
    # print(f"Second access took {t1-t0:.2f}s")
