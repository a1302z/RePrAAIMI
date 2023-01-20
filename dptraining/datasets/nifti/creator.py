from pathlib import Path
from random import seed, shuffle
from typing import Tuple

from torch.utils.data import Dataset


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


def make_dataset(config: Config, transforms, scan_files, label_files):
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

        train_ds, val_ds, test_ds = make_dataset(
            config, transforms, scan_files, label_files
        )
        return train_ds, val_ds, test_ds
