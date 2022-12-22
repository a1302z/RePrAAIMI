from pathlib import Path

from dptraining.datasets.fmri.subsample import create_mask_for_mask_type
from dptraining.datasets.fmri.transforms import UnetDataTransform
from dptraining.datasets.fmri.data_module import FastMriDataModule
from dptraining.datasets.base_creator import DataLoaderCreator

from typing import Tuple
from torch.utils.data import Dataset


class FMRICreator(DataLoaderCreator):
    @staticmethod
    def make_datasets(
        config: dict, transforms: Tuple
    ) -> Tuple[Dataset, Dataset, Dataset]:
        if not (
            transforms is None
            or (isinstance(transforms, tuple) and all((t is None for t in transforms)))
        ):
            raise ValueError(
                "No direct specification of transforms for fastmri supported"
            )

        mask = create_mask_for_mask_type(
            config.dataset.fmri.mask_type,
            config.dataset.fmri.center_fractions,
            config.dataset.fmri.accelerations,
        )
        train_transform = UnetDataTransform(
            config.dataset.fmri.challenge,
            mask_func=mask,
            use_seed=False,
            size=config.dataset.fmri.resolution,
        )
        val_transform = UnetDataTransform(
            config.dataset.fmri.challenge,
            mask_func=mask,
            size=config.dataset.fmri.resolution,
        )
        test_transform = UnetDataTransform(
            config.dataset.fmri.challenge, size=config.dataset.fmri.resolution
        )
        data_module = FastMriDataModule(
            data_path=Path(config.dataset.root),
            challenge=config.dataset.fmri.challenge,
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            test_split="val",
            test_path=None,
            sample_rate=None,
            batch_size=config.hyperparams.batch_size,
            num_workers=config.loader.num_workers,
            split_train_dataset=config.dataset.train_val_split,
            new_data_root=Path(config.dataset.fmri.new_data_root)
            if config.dataset.fmri.new_data_root
            else None,
        )
        train_set, val_set, test_set = (
            data_module.train_dataset(),
            data_module.val_dataloader(),
            data_module.test_dataloader(),
        )
        return train_set, val_set, test_set
