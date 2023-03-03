from pathlib import Path
from typing import Callable, Optional, Tuple


import ctypes
import multiprocessing as mp

import nibabel as nib
import numpy as np
from torch.utils.data import Dataset
from h5py import File as h5file


from dptraining.config import Normalization, DataStats, CTWindow
from dptraining.datasets.nifti.nifti_utils import (
    preprocess_and_convert_to_numpy,
    resize_scan,
)

# We are assuming that all tasks are structured as Task03_Liver


class NiftiSegmentationDataset(Dataset):
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
        normalize_per_ct: bool = False,
        database_file: Optional[h5file] = None,
    ) -> None:
        super().__init__()
        assert (slice_thickness is None) or (
            n_slices is None
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
        self.normalize_per_ct: bool = normalize_per_ct
        self.assume_same_settings: bool = assume_same_settings
        self.database: Optional[h5file] = database_file
        self.database_needs_correction: bool = not (
            self.resolution is None and self.n_slices is None and data_stats is None
        )
        self.corrected_database_entry: bool = False
        assert not (
            self.cache and self.database is not None
        ), "numpy caching and h5database are mutually exclusive"
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

        if self.cache and not self.database:
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
        else:
            self.mean, self.std = None, None

    def __len__(self) -> int:
        return len(self.matched_labeled_scans)

    def __getitem__(self, index: int) -> Tuple[np.array, np.array]:
        file_key, img_file, label_file = self.matched_labeled_scans[index]
        # print(f"{index} cached: {self.cached_files[index]}")
        # t0 = time()
        new_scan_path, new_label_path = self.create_new_filenames(img_file, label_file)
        if (self.cache and self.cached_files[index]) or (
            self.assume_same_settings
            and new_scan_path.is_file()
            and new_label_path.is_file()
        ):
            scan, label = self.load_np_files(new_scan_path, new_label_path)
        elif self.database and (
            self.corrected_database_entry or not self.database_needs_correction
        ):
            data = self.database[file_key][
                "corrected_img_label_pair"
                if self.database_needs_correction
                else "img_label_pair"
            ]
            scan, label = data[0], data[1]
        else:
            scan, label = self.load_nifti_files(index, img_file, label_file)
            if self.database:
                data = np.stack([scan, label], axis=0)
                self.database[file_key].create_dataset(
                    "corrected_img_label_pair", data.shape, data.dtype, data
                )
                self.corrected_database_entry = True
        scan = self.transform(scan) if self.transform is not None else scan
        label = (
            self.label_transform(label) if self.label_transform is not None else label
        )
        if self.normalize_per_ct:
            scan = (scan - scan.mean()) / scan.std()
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
            scan, label = resize_scan(
                scan, label, self.slice_thickness, self.resolution, self.n_slices
            )
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

        scan, label = preprocess_and_convert_to_numpy(
            scan, label, self.ct_window, self.normalization, (self.mean, self.std)
        )
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
