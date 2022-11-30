from copy import deepcopy
from warnings import warn
from omegaconf import OmegaConf

from deepee.dataloader import UniformWORSubsampler

from dptraining.config import Config, DatasetName, LoaderCollateFn
from dptraining.datasets.cifar10 import CIFAR10Creator
from dptraining.datasets.imagenet import ImageNetCreator
from dptraining.datasets.tinyimagenet import TinyImageNetCreator
from dptraining.datasets.utils import collate_np_arrays
from dptraining.utils.augment import Transformation


SUPPORTED_FFT = (DatasetName.CIFAR10,)
SUPPORTED_NORMALIZATION = (DatasetName.CIFAR10, DatasetName.tinyimagenet)


def modify_collate_fn_config(config: dict):
    if config["collate_fn"] is not None:
        if config["collate_fn"] == LoaderCollateFn.numpy:
            config["collate_fn"] = collate_np_arrays
        else:
            raise ValueError(f"collate_fn {config.collate_fn} not supported.")


def make_dataset(config: Config):
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
    match config.dataset.name:
        case DatasetName.CIFAR10:
            train_ds, val_ds, test_ds = CIFAR10Creator.make_datasets(
                config,
                (train_tf, val_tf, test_tf),
                normalize_by_default=config.dataset.normalization,
            )
        case DatasetName.tinyimagenet:
            train_ds, val_ds, test_ds = TinyImageNetCreator.make_datasets(
                config, (train_tf, val_tf, test_tf)
            )
        case DatasetName.imagenet:
            train_ds, val_ds, test_ds = ImageNetCreator.make_datasets(
                config, (train_tf, val_tf, test_tf)
            )
        case _ as unsupported:
            raise ValueError(f"Unsupported dataset '{unsupported}'.")
    return train_ds, val_ds, test_ds


def make_loader_from_config(config: Config):
    train_ds, val_ds, test_ds = make_dataset(config)
    loader_kwargs = deepcopy(OmegaConf.to_container(config.loader))

    if loader_kwargs["prefetch_factor"] is None:
        del loader_kwargs["prefetch_factor"]

    if loader_kwargs["num_workers"] is None:
        del loader_kwargs["num_workers"]

    modify_collate_fn_config(loader_kwargs)
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
    overfitting = config.hyperparams.overfit is not None
    if not config.DP or overfitting:
        train_loader_kwargs["batch_size"] = config.hyperparams.batch_size
        train_loader_kwargs["shuffle"] = not overfitting
        if overfitting and config.DP:
            warn("Due to overfitting argument we turn off correct sampling for DP")
    else:
        train_loader_kwargs["batch_sampler"] = UniformWORSubsampler(
            train_ds, config.hyperparams.batch_size
        )
    test_loader_kwargs["batch_size"] = config.hyperparams.batch_size_test
    val_loader_kwargs["batch_size"] = (
        config.hyperparams.batch_size_val
        if config.hyperparams.batch_size_val is not None
        else config.hyperparams.batch_size_test
    )
    test_loader_kwargs["shuffle"] = False
    val_loader_kwargs["shuffle"] = False

    match config.dataset.name:
        case DatasetName.CIFAR10:
            train_loader, val_loader, test_loader = CIFAR10Creator.make_dataloader(
                train_ds,
                val_ds,
                test_ds,
                train_loader_kwargs,
                val_loader_kwargs,
                test_loader_kwargs,
            )
        case DatasetName.tinyimagenet:
            train_loader, val_loader, test_loader = TinyImageNetCreator.make_dataloader(
                train_ds,
                val_ds,
                test_ds,
                train_loader_kwargs,
                val_loader_kwargs,
                test_loader_kwargs,
            )
        case DatasetName.imagenet:
            train_loader, val_loader, test_loader = ImageNetCreator.make_dataloader(
                train_ds,
                val_ds,
                test_ds,
                train_loader_kwargs,
                val_loader_kwargs,
                test_loader_kwargs,
            )
        case _ as unsupported:
            raise ValueError(
                f"This shouldn't happen. " f"Unsupported dataset {unsupported}."
            )
    return train_loader, val_loader, test_loader
