from copy import deepcopy
from warnings import warn

from deepee.dataloader import UniformWORSubsampler

from dptraining.datasets.cifar10 import CIFAR10Creator
from dptraining.datasets.imagenet import ImageNetCreator
from dptraining.datasets.radimagenet import RadImageNetCreator
from dptraining.datasets.fmri import FMRICreator
from dptraining.datasets.tinyimagenet import TinyImageNetCreator
from dptraining.datasets.utils import (
    collate_np_classification,
    collate_np_reconstruction,
)
from dptraining.utils.augment import Transformation


SUPPORTED_DATASETS = ("cifar10", "imagenet", "tinyimagenet", "radimagenet", "fastmri")

SUPPORTED_FFT = ("cifar10",)
SUPPORTED_NORMALIZATION = ("cifar10", "tinyimagenet")


def select_creator(config):
    match config["dataset"]["name"].lower():
        case "cifar10":
            creator = CIFAR10Creator
        case "tinyimagenet":
            creator = TinyImageNetCreator
        case "imagenet":
            creator = ImageNetCreator
        case "fastmri":
            creator = FMRICreator
        case "radimagenet":
            creator = RadImageNetCreator
        case other:
            raise ValueError(
                f"This shouldn't happen. "
                f"{SUPPORTED_DATASETS} includes not supported datasets. "
                f"Got {other}"
            )

    return creator


def modify_collate_fn_config(config):

    if "collate_fn" in config["loader"]:
        if (
            config["loader"]["collate_fn"] == "numpy"
            and config["dataset"]["task"] == "classification"
        ):
            config["loader"]["collate_fn"] = collate_np_classification
        elif (
            config["loader"]["collate_fn"] == "numpy"
            and config["dataset"]["task"] == "reconstruction"
        ):
            config["loader"]["collate_fn"] = collate_np_reconstruction
        else:
            raise ValueError(f"collate_fn {config['collate_fn']} not supported.")


def make_dataset(config):
    fft = "fft" in config["dataset"] and config["dataset"]["fft"]
    normalize = (
        "normalization" in config["dataset"] and config["dataset"]["normalization"]
    )
    if fft and config["dataset"]["name"].lower() not in SUPPORTED_FFT:
        raise ValueError(f"Direct FFT conversion only supported for {SUPPORTED_FFT}")
    if normalize and config["dataset"]["name"].lower() not in SUPPORTED_NORMALIZATION:
        raise ValueError(
            f"Direct normalization only supported for {SUPPORTED_NORMALIZATION}"
        )

    train_tf = (
        Transformation.from_dict_list(config["train_transforms"])
        if "train_transforms" in config
        else None
    )
    test_tf = (
        Transformation.from_dict_list(config["test_transforms"])
        if "test_transforms" in config
        else None
    )
    val_tf = (
        Transformation.from_dict_list(config["val_transforms"])
        if "val_transforms" in config
        else test_tf
    )
    add_kwargs = {}
    match config["dataset"]["name"].lower():
        case "cifar10":
            add_kwargs["normalize_by_default"] = normalize
    creator = select_creator(config)
    train_ds, val_ds, test_ds = creator.make_datasets(
        config, (train_tf, val_tf, test_tf), **add_kwargs
    )
    return train_ds, val_ds, test_ds


def make_loader_from_config(config):
    dataset_name = config["dataset"]["name"].lower()
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Dataset {config['dataset']['name']} not supported yet. "
            f"Currently supported datasets: {SUPPORTED_DATASETS}"
        )
    train_ds, val_ds, test_ds = make_dataset(config)
    loader_kwargs = deepcopy(config)
    modify_collate_fn_config(loader_kwargs)
    loader_kwargs = loader_kwargs["loader"]
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
    overfitting = "overfit" in config["hyperparams"] and isinstance(
        config["hyperparams"]["overfit"], int
    )
    if config["DP"]["disable_dp"] or overfitting:
        train_loader_kwargs["batch_size"] = config["hyperparams"]["batch_size"]
        train_loader_kwargs["shuffle"] = not overfitting
        if overfitting and not config["DP"]["disable_dp"]:
            warn("Due to overfitting argument we turn off correct sampling for DP")
    else:
        train_loader_kwargs["batch_sampler"] = UniformWORSubsampler(
            train_ds, config["hyperparams"]["batch_size"]
        )
    test_loader_kwargs["batch_size"] = config["hyperparams"]["batch_size_test"]
    val_loader_kwargs["batch_size"] = (
        config["hyperparams"]["batch_size_val"]
        if "batch_size_val" in config["hyperparams"]
        else config["hyperparams"]["batch_size_test"]
    )
    test_loader_kwargs["shuffle"] = False
    val_loader_kwargs["shuffle"] = False

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
