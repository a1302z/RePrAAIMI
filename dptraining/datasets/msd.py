import sys
from pathlib import Path
from random import seed, shuffle
from typing import Callable, Optional, Tuple


import ctypes
import multiprocessing as mp

import nibabel as nib
import numpy as np
from nilearn.image import resample_img
from torch.utils.data import Dataset

sys.path.insert(0, str(Path.cwd()))

from dptraining.config import Config, Normalization, DataStats, CTWindow
from dptraining.datasets.base_creator import DataLoaderCreator, mk_subdirectories

# We are assuming that all tasks are structured as Task03_Liver


def scale_array_zero_one(array: np.array) -> np.array:
    """Scales a numpy array from 0 to 1. Works in 3D
    Return np.array"""
    assert array.max() - array.min() > 0

    return ((array - array.min()) / (array.max() - array.min())).astype(np.float32)


def scale_array_unit_gaussian(
    array: np.array,
    mean: np.array,
    std: np.array,
) -> np.array:
    return (array - mean) / std


def rotate_label(label_volume) -> np.array:
    """Rotates and flips the label in the same way the scans were rotated and flipped
    Return: np.array"""

    label_volume = np.rot90(label_volume)
    label_volume = np.fliplr(label_volume)

    return label_volume.astype(np.float32)


class MSD(Dataset):

    # resolution: 256 / window: (-150, 200) / thickness: 3mm
    # Mean: -81.909645        Std: 8.735866

    def __init__(
        self,
        matched_labeled_scans: list[tuple[str, Path, Path]],
        transform: Optional[Callable] = None,
        label_transform: Optional[Callable] = None,
        resolution: Optional[int] = None,
        slice_thickness: Optional[float] = None,
        n_slices: Optional[int] = None,
        normalization: Normalization = Normalization.raw,
        data_stats: DataStats = None,
        cache_files: bool = False,
        ct_window: CTWindow = CTWindow(-150, 200),
        assume_same_settings: bool = False,
    ) -> None:
        super().__init__()
        assert (slice_thickness is not None) or (
            n_slices is not None
        ), "You can only set either slice_thickness or n_slices"
        self.matched_labeled_scans: list[tuple[str, Path, Path]] = matched_labeled_scans
        self.transform = transform
        self.label_transform = label_transform
        self.resolution: Optional[int] = resolution
        self.slice_thickness: Optional[float] = slice_thickness
        self.n_slices: Optional[int] = n_slices
        self.normalization: Normalization = normalization
        self.cache: bool = cache_files
        self.cached_files: Optional[mp.Array]
        self.ct_window: CTWindow = ct_window
        self.assume_same_settings: bool = assume_same_settings
        if self.assume_same_settings:
            assert all(
                [
                    file.is_file()
                    for _, img_file, label_file in self.matched_labeled_scans
                    for file in self.create_new_filenames(img_file, label_file)
                ]
            ), "Not all data files are already precomputed. Set assume_same_settings to False"
        # self.slice_thicknesses = mp.Array(
        #     ctypes.c_float, [0.0] * len(self.matched_labeled_scans), lock=True
        # )

        if self.cache:
            # self.cached_files = [False for _ in self.matched_labeled_scans]
            self.cached_files = mp.Array(
                ctypes.c_bool, [False] * len(self.matched_labeled_scans), lock=True
            )
        else:
            self.cached_files = None
        if self.normalization in [Normalization.gaussian, Normalization.consecutive]:
            if data_stats is not None:
                self.mean = data_stats.mean
                self.std = data_stats.std
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
        new_scan_path, new_label_path = self.create_new_filenames(img_file, label_file)
        if (self.cache and self.cached_files[index]) or (
            self.assume_same_settings
            and new_scan_path.is_file()
            and new_label_path.is_file()
        ):
            scan, label = self.load_np_files(new_scan_path, new_label_path)
        else:
            scan, label = self.load_nifti_files(index, img_file, label_file)
        scan = self.transform(scan) if self.transform is not None else scan
        label = (
            self.label_transform(label) if self.label_transform is not None else label
        )
        # t1 = time()
        # print(f"\t Loading took {t1-t0:.1f} seconds")
        return (
            scan[np.newaxis, ...],
            label[np.newaxis, ...],
        )  # add channel dimension

    def load_nifti_files(self, index, img_file, label_file):
        if self.cache:
            scan_path, label_path = self.create_new_filenames(img_file, label_file)
        if img_file.is_symlink():
            img_file = img_file.readlink()
        if label_file.is_symlink():
            label_file = label_file.readlink()
        scan = nib.load(img_file)
        label = nib.load(label_file)
        if self.resolution or self.slice_thickness or self.n_slices:
            scan, label = self.resize_scan(scan, label)
            assert scan.header.get_data_shape() == label.header.get_data_shape(), (
                f"Index {index} has not matching shapes"
                f"\nShapes: {scan.header.get_data_shape()}\t{label.header.get_data_shape()}"
            )
            # if self.n_slices:
            #     new_zooms = scan.header.get_zooms()
            #     # print(
            #     #     f"Old num slices: {data_shape[2]}\tOld thickness: {zooms[2]:.2f}"
            #           f"\tNew num slices: {self.n_slices}\tNew thickness: {new_zooms[2]:.2f}"
            #     # )
            #     self.slice_thicknesses[index] = new_zooms[2]

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

    def create_new_filenames(self, img_file, label_file) -> tuple[Path, Path]:
        scan_path = img_file.parent / f"preprocessed_scan_{img_file.stem}.npy"
        label_path = label_file.parent / f"preprocessed_label_{label_file.stem}.npy"

        return scan_path, label_path

    def preprocess_scan(self, scan) -> np.array:
        """Performs Preprocessing:
        - clips vales to -150 to 200,
        - peforms rotations and flipping to move patient into reference position
        Return: np.array"""
        scan = np.clip(scan, self.ct_window.low, self.ct_window.high)
        match self.normalization:
            case Normalization.zeroone:
                scan = scale_array_zero_one(scan)
            case Normalization.gaussian:
                scan = scale_array_unit_gaussian(
                    scan,
                    self.mean,
                    self.std,
                )
            case Normalization.consecutive:
                # not really necessary but may be useful if
                # stats were calculated over 0,1 interval
                scan = scale_array_zero_one(scan)
                scan = scale_array_unit_gaussian(
                    scan,
                    self.mean,
                    self.std,
                )
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
        assert np_scan.shape == np_label.shape, (
            f"Scan has shape {np_scan.shape} while label has {np_label.shape}"
            f"\nNifti data: {nifti_scan.header.get_data_shape()}"
            f"\t {nifti_mask.header.get_data_shape()}"
        )

        return np_scan, np_label

    def resize_scan(
        self, scan: nib.Nifti1Header, label: nib.Nifti1Header
    ) -> tuple[nib.Nifti1Image, nib.Nifti1Image]:
        data_shape = scan.header.get_data_shape()
        zooms = scan.header.get_zooms()
        # print(
        #     f"Actual size: {data_shape[0]*zooms[0]:.1f}mm "
        #     f"x {data_shape[1]*zooms[1]:.1f}mm "
        #     f"x {data_shape[2]*zooms[2]:.1f}mm"
        # )
        z_shape = data_shape[2]
        if self.slice_thickness:
            z_shape = int(data_shape[2] * zooms[2] / self.slice_thickness)
        if self.n_slices:
            z_shape = self.n_slices
        new_shape = (
            self.resolution if self.resolution else data_shape[0],
            self.resolution if self.resolution else data_shape[1],
            z_shape,
        )
        new_affine = np.copy(scan.affine)
        for i, new_shape_i in enumerate(new_shape):
            new_affine[i, i] *= data_shape[i] / new_shape_i
        # print(zooms)
        if self.slice_thickness:
            new_affine[2, 2] = self.slice_thickness
        elif self.n_slices:
            new_affine[2, 2] = (data_shape[2] * zooms[2]) / self.n_slices
        else:
            new_affine[2, 2] = data_shape[2]
        scan = resample_img(
            scan,
            target_affine=new_affine,
            target_shape=new_shape,
            interpolation="continuous",
        )
        label = resample_img(
            label,
            target_affine=new_affine,
            target_shape=new_shape,
            interpolation="nearest",
        )
        # if self.n_slices:
        #     new_zooms = scan.header.get_zooms()
        # print(
        #     f"Old num slices: {data_shape[2]}\tOld thickness: {zooms[2]:.2f}"
        #     f"\tNew num slices: {self.n_slices}\tNew thickness: {new_zooms[2]:.2f}"
        # )

        # print(
        #     f"New real size: {new_shape[0]*new_zooms[0]:.1f}mm "
        #     f"x {new_shape[1]*new_zooms[1]:.1f}mm "
        #     f"x {new_shape[2]*new_zooms[2]:.1f}mm"
        # )
        return scan, label

    # def __del__(self):
    #     if self.cache:
    #         for i in range(len(self)):
    #             if self.cached_files[i]:
    #                 _, img_file, label_file = self.matched_labeled_scans[i]
    #                 scan_path, label_path = self.create_new_filenames(
    #                     img_file, label_file
    #                 )
    #                 print(scan_path)
    #                 # scan_path.unlink()
    #                 # label_path.unlink()


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
        if config.dataset.msd.subtask:
            root = root / MSDCreator.subtask_paths[config.dataset.msd.subtask.value]
        seed(config.dataset.msd.datasplit_seed)

        train_split, test_split = (
            config.dataset.train_val_split,
            config.dataset.msd.test_split,
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
        num_train = int(round(len(matched_labeled_scans) * test_split))
        train_files = matched_labeled_scans[num_train:]
        test_files = matched_labeled_scans[:num_train]
        num_train = int(round(len(train_files) * train_split))
        val_files = train_files[num_train:]
        train_files = train_files[:num_train]
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
                resolution=config.dataset.msd.resolution,
                slice_thickness=config.dataset.msd.slice_thickness,
                n_slices=config.dataset.msd.n_slices,
                normalization=config.dataset.msd.normalization_type,
                data_stats=config.dataset.msd.data_stats,
                cache_files=config.dataset.msd.cache,
                ct_window=config.dataset.msd.ct_window,
                assume_same_settings=config.dataset.msd.assume_same_settings,
            )
            for smls, tf in zip(split_matched_labeled_scans, transforms)
        )
        return train_ds, val_ds, test_ds


# if __name__ == "__main__":
#     from matplotlib import pyplot as plt
#     from omegaconf import OmegaConf
#     from torch.utils.data import DataLoader
#     from tqdm import tqdm

#     from time import time

#     def extend_base_config(overrides: dict):
#         base_conf = OmegaConf.structured(Config)
#         merged_conf = OmegaConf.merge(base_conf, overrides)
#         return merged_conf

#     def collate_fn(list_of_data_tuples: list[tuple[np.array, np.array]]):
#         scans = [item[0] for item in list_of_data_tuples]
#         labels = [item[1] for item in list_of_data_tuples]
#         scans = np.concatenate(scans, axis=-1)
#         labels = np.concatenate(labels, axis=-1)
#         return scans, labels

#     def calc_mean_std(dataset: DataLoader):
#         mean = 0.0
#         for images, _ in tqdm(
#             dataset, total=len(dataset), desc="calculating mean", leave=False
#         ):
#             mean += np.mean(images)
#         mean = mean / len(dataset.dataset)

#         var = 0.0
#         N_px = 0
#         for images, _ in tqdm(
#             dataset, total=len(dataset), desc="calculating std", leave=False
#         ):
#             var += ((images - mean) ** 2).sum()
#             N_px += np.prod(images.shape)
#         std = np.sqrt(var / (len(dataset.dataset) * N_px))
#         return mean, std

#     # N = int(np.ceil(np.sqrt(scan.shape[-1])))
#     # fig, axs = plt.subplots(N, N, sharex=True, sharey=True)
#     # for i in range(N):
#     #     for j in range(N):
#     #         if i * N + j >= scan.shape[-1]:
#     #             break
#     #         axs[i, j].imshow(scan[:, :, i * N + j], cmap="gray")
#     #         img = axs[i, j].imshow(
#     #             label[:, :, i * N + j] / 2.0,
#     #             cmap="inferno",
#     #             alpha=0.3,
#     #             vmin=0,
#     #             vmax=1.0,
#     #         )
#     #         axs[i, j].set_axis_off()
#     # plt.savefig("test_msd.png", dpi=600)

#     config = extend_base_config(
#         {
#             "project": "test",
#             "general": {"log_wandb": False, "cpu": True, "eval_train": False},
#             "loader": {"num_workers": 0, "collate_fn": "numpy"},
#             "dataset": {
#                 "name": "msd",
#                 "root": "./data/MSD/",
#                 "subtask": "liver",
#                 "train_val_split": 1.0,
#                 "test_split": 0.0,
#                 "datasplit_seed": 0,
#                 "task": "segmentation",
#                 "cache": True,
#                 # "slice_thickness": 3.0,
#                 "n_slices": 100,
#                 "normalization_type": "raw",
#                 "resolution": 128,
#                 "ct_window": {"low": -150, "high": 250},
#             },
#         }
#     )
#     train_ds, val_ds, test_ds = MSDCreator.make_datasets(config, (None, None, None))

#     train_dl = DataLoader(
#         train_ds,
#         batch_size=1,
#         shuffle=False,
#         num_workers=16,
#         prefetch_factor=2,
#         collate_fn=collate_fn,
#     )
#     # for _ in tqdm(train_dl, total=len(train_dl)):
#     #     pass
#     # x = np.array(train_dl.dataset.slice_thicknesses)
#     # plt.hist(x, bins=list(range(10)), density=True)
#     # plt.savefig("slice_thicknesses.png")
#     # exit()

#     # mean, std = calc_mean_std(train_dl)
#     # print(f"Mean: {mean:.6f}\tStd: {std:.6f}")

#     # config = extend_base_config(
#     #     {
#     #         "project": "test",
#     #         "general": {"log_wandb": False, "cpu": True, "eval_train": False},
#     #         "loader": {"num_workers": 0, "collate_fn": "numpy"},
#     #         "dataset": {
#     #             "task": "segmentation",
#     #             "name": "msd",
#     #             "root": "./data/MSD/",
#     #             "train_val_split": 0.9,
#     #             "test_split": 0.1,
#     #             "resolution": 128,
#     #             "subtask": "liver",
#     #             "cache": True,
#     #             "slice_thickness": 3.0,
#     #             "normalization_type": "gaussian",
#     #             "data_stats": {"mean": float(mean), "std": float(std)},
#     #             "ct_window": {"low": -150, "high": 250}
#     #             # "data_stats": {"mean": -63.59883264405043, "std": 110.56125220959515},
#     #         },
#     #     }
#     # )
#     # train_ds, val_ds, test_ds = MSDCreator.make_datasets(config, (None, None, None))

#     # t0 = time()
#     # img, label = train_ds[0]
#     # t1 = time()
#     # print(f"First access took {t1-t0:.1f}s")

#     # print(f"{img.mean()}\t{img.std()}")

#     # train_dl = DataLoader(
#     #     train_ds,
#     #     batch_size=1,
#     #     shuffle=False,
#     #     num_workers=16,
#     #     prefetch_factor=2,
#     #     collate_fn=collate_fn,
#     # )
#     # mean, std = calc_mean_std(train_dl)
#     # print(f"Mean: {mean:.6f}\tStd: {std:.6f}")

#     # t0 = time()
#     # img, label = train_ds[0]
#     # t1 = time()
#     # print(f"Second access took {t1-t0:.3f}s")
#     # pass
