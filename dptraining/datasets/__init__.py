from typing import Any
import numpy as np
from copy import deepcopy
from warnings import warn
from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader

from deepee.dataloader import UniformWORSubsampler

from dptraining.config import (
    Config,
    DatasetName,
    LoaderCollateFn,
    DatasetTask,
)
from dptraining.datasets.cifar10 import CIFAR10Creator
from dptraining.datasets.imagenet import ImageNetCreator
from dptraining.datasets.radimagenet import RadImageNetCreator
from dptraining.datasets.fmri import FMRICreator
from dptraining.datasets.tinyimagenet import TinyImageNetCreator
from dptraining.datasets.nifti.creator import NiftiSegCreator
from dptraining.datasets.ham10000 import HAM10000Creator
from dptraining.datasets.attack_datasets import AttackCreator
from dptraining.datasets.mnist import MNISTCreator
from dptraining.datasets.imagefolder import ImageFolderCreator
from dptraining.datasets.subset import (
    FixedAndShadowDatasetFromOneSet,
    DataSubset,
    FixedAndShadowDatasetFromTwoSets,
)
from dptraining.datasets.utils import (
    collate_np_classification,
    collate_np_reconstruction,
    create_collate_fn_lists,
    collate_mia,
)
from dptraining.utils.augment import Transformation


SUPPORTED_FFT = (DatasetName.CIFAR10, DatasetName.mnist)
SUPPORTED_NORMALIZATION = (
    DatasetName.CIFAR10,
    DatasetName.mnist,
    DatasetName.tinyimagenet,
)


def select_creator(config):
    match config.dataset.name:
        case DatasetName.CIFAR10:
            creator = CIFAR10Creator
        case DatasetName.tinyimagenet:
            creator = TinyImageNetCreator
        case DatasetName.imagenet:
            creator = ImageNetCreator
        case DatasetName.fastmri:
            creator = FMRICreator
        case DatasetName.radimagenet:
            creator = RadImageNetCreator
        case DatasetName.msd:
            creator = NiftiSegCreator
        case DatasetName.ukbb_seg:
            creator = NiftiSegCreator
        case DatasetName.ham10000:
            creator = HAM10000Creator
        case DatasetName.attack:
            creator = AttackCreator
        case DatasetName.mnist:
            creator = MNISTCreator
        case DatasetName.imagefolder:
            creator = ImageFolderCreator
        case _ as unsupported:
            raise ValueError(f"Unsupported dataset '{unsupported}'.")
    return creator


def get_collate_fn(loader_config, task):
    if (
        loader_config["collate_fn"] == LoaderCollateFn.numpy
        and task == DatasetTask.classification
    ):
        return collate_np_classification
    elif loader_config["collate_fn"] == LoaderCollateFn.numpy and task in [
        DatasetTask.reconstruction,
        DatasetTask.segmentation,
    ]:
        return collate_np_reconstruction
    elif loader_config["collate_fn"] == LoaderCollateFn.mia:
        assert task == DatasetTask.classification
        return collate_mia
    else:
        raise ValueError(
            f"collate_fn {loader_config['collate_fn']} for {task} not supported."
        )


def modify_collate_fn_config(loader_config, task):
    if "collate_fn" in loader_config:
        loader_config["collate_fn"] = get_collate_fn(loader_config, task)


def make_dataset(config: Config) -> tuple[Dataset]:
    if config.dataset.fft and config.dataset.name not in SUPPORTED_FFT:
        raise ValueError(f"Direct FFT conversion only supported for {SUPPORTED_FFT}")
    if (
        config.dataset.normalization
        and config.dataset.name not in SUPPORTED_NORMALIZATION
    ):
        raise ValueError(
            f"Direct normalization only supported for {SUPPORTED_NORMALIZATION}"
        )

    train_tf = (
        Transformation.from_dict_list(OmegaConf.to_container(config.train_transforms))
        if config.train_transforms
        else None
    )
    test_tf = (
        Transformation.from_dict_list(OmegaConf.to_container(config.test_transforms))
        if config.test_transforms
        else None
    )
    val_tf = (
        Transformation.from_dict_list(OmegaConf.to_container(config.val_transforms))
        if config.val_transforms
        else test_tf
    )
    add_kwargs = {}
    if config.dataset.name in [DatasetName.CIFAR10, DatasetName.mnist]:
        add_kwargs["normalize_by_default"] = config.dataset.normalization
    creator = select_creator(config)
    train_ds, val_ds, test_ds = creator.make_datasets(
        config, (train_tf, val_tf, test_tf), **add_kwargs
    )
    return train_ds, val_ds, test_ds


def get_random_idcs(max_idx: int):
    idcs = np.arange(max_idx)
    np.random.shuffle(idcs)
    return idcs


def make_attack_datasets(
    config: Config,
) -> tuple[
    FixedAndShadowDatasetFromOneSet, DataSubset, FixedAndShadowDatasetFromTwoSets
]:
    train_ds, shadow_eval_dataset, attack_eval_dataset = make_dataset(config)
    attack_samples: int = config.attack.N_fixed_dataset + config.attack.N_shadow_train
    assert len(train_ds) > attack_samples, (
        "Attack train datasets must be summed up smaller than entire train dataset",
        f"\t({len(train_ds)} vs {attack_samples})",
    )
    data_indices = get_random_idcs(len(train_ds))
    fixed_idcs, shadow_train_idcs = (
        data_indices[: config.attack.N_fixed_dataset],
        data_indices[config.attack.N_fixed_dataset : attack_samples],
    )
    shadow_train_dataset = FixedAndShadowDatasetFromOneSet(
        train_ds, fixed_idcs, shadow_train_idcs
    )
    attack_eval_dataset = FixedAndShadowDatasetFromTwoSets(
        train_ds,
        fixed_idcs,
        attack_eval_dataset,
        get_random_idcs(len(attack_eval_dataset))[: config.attack.N_attack_eval]
        if config.attack.N_attack_eval
        else np.arange(len(attack_eval_dataset)),
    )
    shadow_eval_dataset = DataSubset(
        shadow_eval_dataset,
        get_random_idcs(len(shadow_eval_dataset))[: config.attack.N_shadow_eval]
        if config.attack.N_shadow_eval
        else np.arange(len(shadow_eval_dataset)),
    )

    return shadow_train_dataset, shadow_eval_dataset, attack_eval_dataset


def make_attack_dataloader(config: Config) -> tuple[DataLoader]:
    (
        shadow_train_dataset,
        shadow_eval_dataset,
        attack_eval_dataset,
    ) = make_attack_datasets(config)
    assert not config.hyperparams.overfit, "No overfitting for attacks permitted"
    train_loader_kwargs, test_loader_kwargs, val_loader_kwargs = modify_loader_kwargs(
        config
    )
    train_loader_kwargs["collate_fn"] = create_collate_fn_lists(
        train_loader_kwargs["collate_fn"]
    )
    if config.DP:  # TODO sampler should be over all datasets
        train_loader_kwargs["batch_sampler"] = UniformWORSubsampler(
            shadow_train_dataset, config.hyperparams.batch_size
        )
    else:
        train_loader_kwargs["batch_size"] = config.hyperparams.batch_size
        train_loader_kwargs["shuffle"] = True
    fixed_dataset_loader = DataLoader(shadow_train_dataset, **train_loader_kwargs)
    shadow_eval_loader = DataLoader(shadow_eval_dataset, **val_loader_kwargs)
    attack_eval_loader = DataLoader(attack_eval_dataset, **train_loader_kwargs)
    return fixed_dataset_loader, shadow_eval_loader, attack_eval_loader


def make_loader_from_config(config: Config) -> tuple[DataLoader]:
    train_ds, val_ds, test_ds = make_dataset(config)
    train_loader_kwargs, test_loader_kwargs, val_loader_kwargs = modify_loader_kwargs(
        config
    )
    overfitting = config.hyperparams.overfit is not None
    if not config.DP or overfitting or not config.DP.use_batch_sampling:
        train_loader_kwargs["batch_size"] = config.hyperparams.batch_size
        train_loader_kwargs["shuffle"] = not overfitting
        if overfitting and config.DP:
            warn("Due to overfitting argument we turn off correct sampling for DP")
    else:
        train_loader_kwargs["batch_sampler"] = UniformWORSubsampler(
            train_ds, config.hyperparams.batch_size
        )
    creator = select_creator(config)
    train_loader, val_loader, test_loader = creator.make_dataloader(
        train_ds,
        val_ds,
        test_ds,
        train_loader_kwargs,
        val_loader_kwargs,
        test_loader_kwargs,
    )
    return train_loader, val_loader, test_loader


def modify_loader_kwargs(config: Config) -> tuple[dict[str, Any]]:
    loader_kwargs = deepcopy(OmegaConf.to_container(config.loader))

    if loader_kwargs["prefetch_factor"] is None:
        del loader_kwargs["prefetch_factor"]

    if loader_kwargs["num_workers"] is None:
        del loader_kwargs["num_workers"]

    modify_collate_fn_config(loader_kwargs, config.dataset.task)
    if "train_loader" in loader_kwargs and "test_loader" in loader_kwargs:
        train_loader_kwargs = loader_kwargs["train_loader"]
        test_loader_kwargs = loader_kwargs["test_loader"]
        val_loader_kwargs = (
            loader_kwargs["val_loader"]
            if "val_loader" in loader_kwargs
            else deepcopy(test_loader_kwargs)
        )
    else:
        train_loader_kwargs = deepcopy(loader_kwargs)
        val_loader_kwargs = deepcopy(loader_kwargs)
        test_loader_kwargs = deepcopy(loader_kwargs)
    test_loader_kwargs["batch_size"] = config.hyperparams.batch_size_test
    val_loader_kwargs["batch_size"] = (
        config.hyperparams.batch_size_val
        if config.hyperparams.batch_size_val is not None
        else config.hyperparams.batch_size_test
    )
    test_loader_kwargs["shuffle"] = False
    val_loader_kwargs["shuffle"] = False
    return train_loader_kwargs, test_loader_kwargs, val_loader_kwargs
