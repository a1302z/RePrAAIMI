"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
# pylint: skip-file

from pathlib import Path
from typing import Callable, Optional, Union

import torch
from random import shuffle, seed as seed_fn
from warnings import warn

from dptraining.datasets.fmri.mri_data import CombinedSliceDataset, SliceDataset

from tqdm import tqdm


def _check_both_not_none(val1, val2):
    if (val1 is not None) and (val2 is not None):
        return True

    return False


class FastMriDataModule:
    def __init__(
        self,
        data_path: Path,
        challenge: str,
        train_transform: Callable,
        val_transform: Callable,
        test_transform: Callable,
        combine_train_val: bool = False,
        test_split: str = "test",
        test_path: Optional[Path] = None,
        sample_rate: Optional[float] = None,
        val_sample_rate: Optional[float] = None,
        test_sample_rate: Optional[float] = None,
        volume_sample_rate: Optional[float] = None,
        val_volume_sample_rate: Optional[float] = None,
        test_volume_sample_rate: Optional[float] = None,
        train_filter: Optional[Callable] = None,
        val_filter: Optional[Callable] = None,
        test_filter: Optional[Callable] = None,
        use_dataset_cache_file: bool = True,
        batch_size: int = 1,
        num_workers: int = 4,
        distributed_sampler: bool = False,
        split_train_dataset: Optional[float] = None,
        seed: int = 0,
        new_data_root: Optional[Path] = None,
    ):
        """
        Args:
            data_path: Path to root data directory. For example, if knee/path
                is the root directory with subdirectories multicoil_train and
                multicoil_val, you would input knee/path for data_path.
            challenge: Name of challenge from ('multicoil', 'singlecoil').
            train_transform: A transform object for the training split.
            val_transform: A transform object for the validation split.
            test_transform: A transform object for the test split.
            combine_train_val: Whether to combine train and val splits into one
                large train dataset. Use this for leaderboard submission.
            test_split: Name of test split from ("test", "challenge").
            test_path: An optional test path. Passing this overwrites data_path
                and test_split.
            sample_rate: Fraction of slices of the training data split to use.
                Can be set to less than 1.0 for rapid prototyping. If not set,
                it defaults to 1.0. To subsample the dataset either set
                sample_rate (sample by slice) or volume_sample_rate (sample by
                volume), but not both.
            val_sample_rate: Same as sample_rate, but for val split.
            test_sample_rate: Same as sample_rate, but for test split.
            volume_sample_rate: Fraction of volumes of the training data split
                to use. Can be set to less than 1.0 for rapid prototyping. If
                not set, it defaults to 1.0. To subsample the dataset either
                set sample_rate (sample by slice) or volume_sample_rate (sample
                by volume), but not both.
            val_volume_sample_rate: Same as volume_sample_rate, but for val
                split.
            test_volume_sample_rate: Same as volume_sample_rate, but for val
                split.
            train_filter: A callable which takes as input a training example
                metadata, and returns whether it should be part of the training
                dataset.
            val_filter: Same as train_filter, but for val split.
            test_filter: Same as train_filter, but for test split.
            use_dataset_cache_file: Whether to cache dataset metadata. This is
                very useful for large datasets like the brain data.
            batch_size: Batch size.
            num_workers: Number of workers for PyTorch dataloader.
        """
        super().__init__()

        if _check_both_not_none(sample_rate, volume_sample_rate):
            raise ValueError("Can set sample_rate or volume_sample_rate, but not both.")
        if _check_both_not_none(val_sample_rate, val_volume_sample_rate):
            raise ValueError(
                "Can set val_sample_rate or val_volume_sample_rate, but not both."
            )
        if _check_both_not_none(test_sample_rate, test_volume_sample_rate):
            raise ValueError(
                "Can set test_sample_rate or test_volume_sample_rate, but not both."
            )

        self.data_path = data_path
        self.challenge = challenge
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.combine_train_val = combine_train_val
        self.test_split = test_split
        self.test_path = test_path
        self.sample_rate = sample_rate
        self.val_sample_rate = val_sample_rate
        self.test_sample_rate = test_sample_rate
        self.volume_sample_rate = volume_sample_rate
        self.val_volume_sample_rate = val_volume_sample_rate
        self.test_volume_sample_rate = test_volume_sample_rate
        self.train_filter = train_filter
        self.val_filter = val_filter
        self.test_filter = test_filter
        self.use_dataset_cache_file = use_dataset_cache_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.distributed_sampler = distributed_sampler
        self.split_train = split_train_dataset
        self.seed = seed

        train_folder = self.data_path / f"{self.challenge}_train"
        val_folder = self.data_path / f"{self.challenge}_val"
        if self.split_train is not None:
            seed_fn(seed)
            train_files = [f for f in train_folder.rglob("*.h5") if f.is_file()]
            shuffle(train_files)
            L_train = int(round(self.split_train * len(train_files)))
            train_split_files, val_split_files = (
                train_files[:L_train],
                train_files[L_train:],
            )
            new_root_folder = (
                new_data_root if new_data_root is not None else self.data_path
            )
            new_train_folder = (
                new_root_folder
                / f"{self.challenge}_seed={seed}_split={self.split_train}_train"
            )
            new_val_folder = (
                new_root_folder
                / f"{self.challenge}_seed={seed}_split={self.split_train}_val"
            )
            if not new_train_folder.is_dir():
                new_train_folder.mkdir(exist_ok=True)
            if not new_val_folder.is_dir():
                new_val_folder.mkdir(exist_ok=True)
            for f in tqdm(
                train_split_files,
                total=len(train_split_files),
                desc="symlinking train files",
                leave=False,
            ):
                new_file = new_train_folder / f"{f.stem}_symlink"
                if not (new_file.is_file() or new_file.is_symlink()):
                    new_file.symlink_to(f)
            for f in tqdm(
                val_split_files,
                total=len(train_split_files),
                desc="symlinking train files",
                leave=False,
            ):
                new_file = new_val_folder / f"{f.stem}_symlink"
                if not (new_file.is_file() or new_file.is_symlink()):
                    new_file.symlink_to(f)

            self.train_path = new_train_folder
            self.val_path = new_val_folder
            self.test_path = val_folder
        else:
            self.train_path = train_folder
            self.val_path = val_folder
            if self.test_path is None:
                warn(
                    "Using validation set as test as "
                    "set as no data split is provided"
                )
                self.test_path = self.val_path

    def _create_dataset(
        self,
        data_transform: Callable,
        data_partition: str,
        sample_rate: Optional[float] = None,
        volume_sample_rate: Optional[float] = None,
        overfit: Optional[int] = None,
    ) -> torch.utils.data.DataLoader:
        if data_partition == "train":
            is_train = True
            sample_rate = self.sample_rate if sample_rate is None else sample_rate
            volume_sample_rate = (
                self.volume_sample_rate
                if volume_sample_rate is None
                else volume_sample_rate
            )
            raw_sample_filter = self.train_filter
        else:
            is_train = False
            if data_partition == "val":
                sample_rate = (
                    self.val_sample_rate if sample_rate is None else sample_rate
                )
                volume_sample_rate = (
                    self.val_volume_sample_rate
                    if volume_sample_rate is None
                    else volume_sample_rate
                )
                raw_sample_filter = self.val_filter
            elif data_partition == "test":
                sample_rate = (
                    self.test_sample_rate if sample_rate is None else sample_rate
                )
                volume_sample_rate = (
                    self.test_volume_sample_rate
                    if volume_sample_rate is None
                    else volume_sample_rate
                )
                raw_sample_filter = self.test_filter

        # if desired, combine train and val together for the train split
        dataset: Union[SliceDataset, CombinedSliceDataset]
        if is_train and self.combine_train_val:
            data_paths = [
                self.train_path,
                self.val_path,
            ]
            data_transforms = [data_transform, data_transform]
            challenges = [self.challenge, self.challenge]
            sample_rates, volume_sample_rates = None, None  # default: no subsampling
            if sample_rate is not None:
                sample_rates = [sample_rate, sample_rate]
            if volume_sample_rate is not None:
                volume_sample_rates = [volume_sample_rate, volume_sample_rate]
            dataset = CombinedSliceDataset(
                roots=data_paths,
                transforms=data_transforms,
                challenges=challenges,
                sample_rates=sample_rates,
                volume_sample_rates=volume_sample_rates,
                use_dataset_cache=self.use_dataset_cache_file,
                raw_sample_filter=raw_sample_filter,
                overfit=overfit,
            )
        else:
            if data_partition in ("test", "challenge") and self.test_path is not None:
                data_path = self.test_path
            elif data_partition == "train":
                data_path = self.train_path
            elif data_partition == "val":
                data_path = self.val_path

            dataset = SliceDataset(
                root=data_path,
                transform=data_transform,
                sample_rate=sample_rate,
                volume_sample_rate=volume_sample_rate,
                challenge=self.challenge,
                use_dataset_cache=self.use_dataset_cache_file,
                raw_sample_filter=raw_sample_filter,
                overfit=overfit,
            )

        add_kwargs = {}
        if self.num_workers > 0:
            add_kwargs["prefetch_factor"] = 10

        # def numpy_collate_fn(list_of_samples):
        #     if len(list_of_samples) > 1:
        #         list_of_outputs = tuple(
        #             stack([s[i] for s in list_of_samples], axis=0)
        #             for i in range(len(list_of_samples[0]))
        #         )
        #     else:
        #         list_of_outputs = tuple(list_of_samples)
        #     return list_of_outputs

        # if self.split_train is not None and "split" in original_data_partition:
        #     L_train = int(round(self.split_train * len(dataset)))
        #     L_val = len(dataset) - L_train
        #     train_ds, val_ds = torch.utils.data.random_split(
        #         dataset,
        #         lengths=[L_train, L_val],
        #         generator=torch.Generator().manual_seed(0),
        #     )
        #     dataset = train_ds if "train_split" in original_data_partition else val_ds

        return dataset

    def prepare_data(self):
        # call dataset for each split one time to make sure the cache is set up on the
        # rank 0 ddp process. if not using cache, don't do this
        if self.use_dataset_cache_file:
            assert self.test_path is not None
            data_paths = [
                self.train_path,
                self.val_path,
                self.test_path,
            ]
            data_transforms = [
                self.train_transform,
                self.val_transform,
                self.test_transform,
            ]
            for i, (data_path, data_transform) in enumerate(
                zip(data_paths, data_transforms)
            ):
                # NOTE: Fixed so that val and test use correct sample rates
                sample_rate = self.sample_rate  # if i == 0 else 1.0
                volume_sample_rate = self.volume_sample_rate  # if i == 0 else None
                _ = SliceDataset(
                    root=data_path,
                    transform=data_transform,
                    sample_rate=sample_rate,
                    volume_sample_rate=volume_sample_rate,
                    challenge=self.challenge,
                    use_dataset_cache=self.use_dataset_cache_file,
                )

    def train_dataset(self, overfit: Optional[int] = None):
        return self._create_dataset(
            self.train_transform, data_partition="train", overfit=overfit
        )

    def val_dataloader(self):
        return self._create_dataset(self.val_transform, data_partition="val")

    def test_dataloader(self):
        return self._create_dataset(self.test_transform, data_partition="test")
