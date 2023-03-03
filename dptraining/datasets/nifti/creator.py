from pathlib import Path
from random import seed, shuffle
from typing import Tuple, Optional, Callable
from itertools import compress
from tqdm import tqdm
import nibabel as nib
import numpy as np
from pickle import dump, load
from torch.utils.data import Dataset
import h5py
import nibabel as nib


from dptraining.config import Config, DatasetName
from dptraining.datasets.base_creator import DataLoaderCreator, mk_subdirectories
from dptraining.datasets.nifti.nifti_seg_dataset import NiftiSegmentationDataset


def find_niftis_file_id(
    labeled_scans_path: Path, search_term: str = "*.nii.gz", exclude_term: str = "._"
):
    return {
        scan.stem.replace(".nii", ""): scan
        for scan in labeled_scans_path.rglob(search_term)
        if not scan.stem.startswith(exclude_term)
    }


def find_niftis_folder_id(
    labeled_scans_path, search_term: str = "*.nii.gz", exclude_term: str = "._"
):
    return {
        scan.parent.name: scan
        for scan in labeled_scans_path.rglob(search_term)
        if not scan.stem.startswith(exclude_term)
    }


def reduce_to_available_files(labeled_scans, available_files):
    return {k: v for k, v in labeled_scans.items() if k in available_files}


def filter_niftis(
    config: Config,
    scan_files: dict[str, Path],
    label_files: dict[str, Path],
    database_file: Optional[h5py.File] = None,
):
    assert scan_files.keys() == label_files.keys()
    if config.dataset.nifti_seg_options.filter_options:
        if config.dataset.nifti_seg_options.filter_options.reuse_filtered_files:
            with open(
                config.dataset.nifti_seg_options.filter_options.reuse_filtered_files,
                "rb",
            ) as fp:
                valid_keys = load(fp)
                old_size, new_size = len(scan_files), len(valid_keys)
                scan_files = {idf: scan_files[idf] for idf in valid_keys}
                label_files = {idf: label_files[idf] for idf in valid_keys}
        else:
            keep_entries = []
            for file_id, img_path in tqdm(
                scan_files.items(),
                total=len(scan_files),
                leave=False,
                desc="Filtering niftis",
            ):
                keep_entry = True
                if database_file:
                    data = database_file[file_id]["img_label_pair"]
                    img_shape = label_shape = data.shape[1:]
                    label_fdata = np.asarray(data)[1:].squeeze(0)
                else:
                    label_path = label_files[file_id]
                    img_path, label_path = Path(img_path), Path(label_path)
                    if img_path.is_symlink():
                        img_path = img_path.readlink()
                    if label_path.is_symlink():
                        label_path = label_path.readlink()
                    img_file: nib.Nifti1Image = nib.load(img_path)
                    label_file: nib.Nifti1Image = nib.load(label_path)
                    img_shape, label_shape = (
                        img_file.header.get_data_shape(),
                        label_file.header.get_data_shape(),
                    )
                    label_fdata: Optional[np.array] = None
                if (
                    config.dataset.nifti_seg_options.filter_options.resolution
                    is not None
                ):
                    keep_entry &= img_shape == label_shape
                    keep_entry &= (
                        img_shape
                        == config.dataset.nifti_seg_options.filter_options.resolution
                    )
                if (
                    keep_entry
                    and config.dataset.nifti_seg_options.filter_options.min_pixels_per_organ
                ):
                    if label_fdata is None:
                        label_fdata = label_file.get_fdata()
                    labels, counts = np.unique(
                        label_fdata.flatten(), return_counts=True
                    )
                    keep_entry &= np.all(
                        labels
                        == np.array(
                            range(
                                len(
                                    config.dataset.nifti_seg_options.filter_options.min_pixels_per_organ
                                )
                            )
                        )
                    )
                    if keep_entry:
                        keep_entry &= np.all(
                            np.greater_equal(
                                counts,
                                np.array(
                                    config.dataset.nifti_seg_options.filter_options.min_pixels_per_organ
                                ),
                            )
                        )
                if (
                    keep_entry
                    and config.dataset.nifti_seg_options.filter_options.length_threshold
                ):
                    if label_fdata is None:
                        label_fdata = label_file.get_fdata()
                    first_non_zero = np.nonzero(label_fdata.sum(axis=(0, 1)))[0][0]
                    keep_entry &= (
                        first_non_zero
                        > config.dataset.nifti_seg_options.filter_options.length_threshold
                    )

                keep_entries.append(keep_entry)
            old_size, new_size = len(keep_entries), sum(keep_entries)
            if new_size < old_size:
                scan_files = {
                    idf: scan_files[idf]
                    for idf in list(compress(scan_files, keep_entries))
                }
                label_files = {
                    idf: label_files[idf]
                    for idf in list(compress(label_files, keep_entries))
                }
            if config.dataset.nifti_seg_options.filter_options.save_filtered_files:
                with open(
                    config.dataset.nifti_seg_options.filter_options.save_filtered_files,
                    "wb",
                ) as fp:
                    dump(list(scan_files.keys()), fp)

        if new_size < old_size:
            print(f"\tFiltering reduced dataset from {old_size} to {new_size}")
            assert scan_files.keys() == label_files.keys()
        else:
            print(f"\tNo files were removed due to filtering")
    return scan_files, label_files
def create_database(config, scan_files, label_files):
    database_file = None
    if config.dataset.nifti_seg_options.database:
        if config.dataset.nifti_seg_options.filter_options.resolution is None:
            raise ValueError(
                f"We do not support database creation for datasets with different image sizes yet"
            )
        if config.dataset.nifti_seg_options.database is not None:
            if config.dataset.nifti_seg_options.database.is_file():
                print(
                    f"Reusing {config.dataset.nifti_seg_options.database}. If this is not intended "
                    "please change the filename"
                )
            else:
                with h5py.File(config.dataset.nifti_seg_options.database, "w") as database_file:
                    for file_name in tqdm(
                        scan_files.keys(),
                        total=len(scan_files),
                        leave=False,
                        desc="Create database",
                    ):
                        img_path, label_path = (
                            scan_files[file_name],
                            label_files[file_name],
                        )
                        data = np.stack(
                            [
                                nib.load(img_path).get_fdata(),
                                nib.load(label_path).get_fdata(),
                            ],
                            axis=0,
                        )
                        grp = database_file.create_group(file_name)
                        grp.create_dataset(
                            "img_label_pair",
                            shape=data.shape,
                            dtype=data.dtype,
                            data=data,
                        )

            database_file = h5py.File(
                config.dataset.nifti_seg_options.database, "r"
            )
            
    return database_file


def make_dataset(
    config: Config,
    transforms: Optional[Callable],
    scan_files: dict[str, Path],
    label_files: dict[str, Path],
    database_file: Optional[h5py.File] = None,
):
    seed(config.dataset.nifti_seg_options.datasplit_seed)

    train_split, test_split = (
        config.dataset.train_val_split,
        config.dataset.nifti_seg_options.test_split,
    )

    assert set(scan_files.keys()) == set(label_files.keys())

    matched_labeled_scans = [
        (name, scan_files[name], label_files[name]) for name in scan_files.keys()
    ]
    shuffle(matched_labeled_scans)
    num_train = int(round(len(matched_labeled_scans) * test_split))
    train_files = matched_labeled_scans[num_train:]
    test_files = matched_labeled_scans[:num_train]
    num_train = int(round(len(train_files) * train_split))
    val_files = train_files[num_train:]
    train_files = train_files[:num_train]
    subdirs = mk_subdirectories(
        Path(
            config.dataset.nifti_seg_options.new_data_root
            if config.dataset.nifti_seg_options.new_data_root is not None
            else config.dataset.root
        ),
        ["train_split", "val_split", "test_split"],
    )
    split_files = (train_files, val_files, test_files)

    split_matched_labeled_scans = []
    for directory, files in zip(subdirs, split_files):
        scan_subdir, label_subdir = mk_subdirectories(directory, ["images", "labels"])
        sub_matched_labeled_scans = []
        for name, scan_path, label_path in files:
            new_scan_path = scan_subdir / name
            new_label_path = label_subdir / name
            if not new_scan_path.is_symlink():
                new_scan_path.symlink_to(scan_path)
            if not new_label_path.is_symlink():
                new_label_path.symlink_to(label_path)
            sub_matched_labeled_scans.append((name, new_scan_path, new_label_path))
        split_matched_labeled_scans.append(sub_matched_labeled_scans)
    train_ds, val_ds, test_ds = (
        NiftiSegmentationDataset(
            smls,
            transform=tf,
            resolution=config.dataset.nifti_seg_options.resolution,
            slice_thickness=config.dataset.nifti_seg_options.slice_thickness,
            n_slices=config.dataset.nifti_seg_options.n_slices,
            normalization=config.dataset.nifti_seg_options.normalization_type,
            data_stats=config.dataset.nifti_seg_options.data_stats,
            cache_files=config.dataset.nifti_seg_options.cache,
            ct_window=config.dataset.nifti_seg_options.ct_window,
            assume_same_settings=config.dataset.nifti_seg_options.assume_same_settings,
            normalize_per_ct=config.dataset.nifti_seg_options.normalize_per_scan,
            database_file=database_file,
        )
        for smls, tf in zip(split_matched_labeled_scans, transforms)
    )

    return train_ds, val_ds, test_ds


class NiftiSegCreator(DataLoaderCreator):
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
        if config.dataset.name == DatasetName.msd:
            if config.dataset.nifti_seg_options.msd_subtask:
                root = (
                    Path(config.dataset.root)
                    / NiftiSegCreator.subtask_paths[
                        config.dataset.nifti_seg_options.msd_subtask.value
                    ]
                )
                config.dataset.root = root

            scans_path = root / "imagesTr"
            label_path = root / "labelsTr"
            scan_files = find_niftis_file_id(scans_path)
            label_files = find_niftis_file_id(label_path)
        elif config.dataset.name == DatasetName.ukbb_seg:
            scans_path = Path(config.dataset.nifti_seg_options.image_file_root)
            label_path = Path(config.dataset.nifti_seg_options.label_file_root)
            scan_files = find_niftis_folder_id(scans_path, search_term="*wat.nii.gz")
            label_files = find_niftis_folder_id(label_path)
        else:
            raise ValueError(f"{config.dataset.name} not supported")

        available_files = set(scan_files.keys()).intersection(set(label_files.keys()))
        if len(available_files) < max(len(scan_files), len(label_files)):
            print(
                f"Dataset was reduced to labeled scans:"
                f"\n\tBefore: {len(scan_files)} files\t {len(label_files)} labels"
                f"\n\tNow: {len(available_files)} matched files"
            )
        scan_files = reduce_to_available_files(scan_files, available_files)
        label_files = reduce_to_available_files(label_files, available_files)

        database_file: Optional[h5py.File] = None
        database_file = create_database(config, scan_files, label_files)
        scan_files, label_files = filter_niftis(
            config, scan_files, label_files, database_file
        )

        if config.dataset.nifti_seg_options.limit_dataset:
            keys = list(scan_files.keys())[
                : config.dataset.nifti_seg_options.limit_dataset
            ]
            scan_files = {key: scan_files[key] for key in keys}
            label_files = {key: label_files[key] for key in keys}

        train_ds, val_ds, test_ds = make_dataset(
            config, transforms, scan_files, label_files, database_file
        )
        return train_ds, val_ds, test_ds

    