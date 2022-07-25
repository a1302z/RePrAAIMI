from copy import deepcopy

from deepee.dataloader import UniformWORSubsampler

from dptraining.datasets.cifar10 import CIFAR10Creator
from dptraining.datasets.imagenet import ImageNetCreator
from dptraining.datasets.utils import collate_np_arrays
from dptraining.utils.augment import Transformation


SUPPORTED_DATASETS = ("cifar10", "imagenet")


def modify_collate_fn_config(config):
    if "collate_fn" in config:
        if config["collate_fn"] == "numpy":
            config["collate_fn"] = collate_np_arrays
        else:
            raise ValueError(f"collate_fn {config['collate_fn']} not supported.")


def make_dataset(config):

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
    if config["dataset"]["name"].lower() == "cifar10":
        train_ds, val_ds, test_ds = CIFAR10Creator.make_datasets(
            config,
            (train_tf, val_tf, test_tf),
        )
    elif config["dataset"]["name"].lower() == "imagenet":
        train_ds, val_ds, test_ds = ImageNetCreator.make_datasets(
            config, (train_tf, val_tf, test_tf)
        )
    return train_ds, val_ds, test_ds


def make_loader_from_config(config):
    if config["dataset"]["name"].lower() not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Dataset {config['dataset']['name']} not supported yet. "
            f"Currently supported datasets: {SUPPORTED_DATASETS}"
        )
    train_ds, val_ds, test_ds = make_dataset(config)
    loader_kwargs = deepcopy(config["loader"])
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
    if config["DP"]["disable_dp"]:
        train_loader_kwargs["batch_size"] = config["hyperparams"]["batch_size"]
        train_loader_kwargs["shuffle"] = not (
            "overfit" in config["hyperparams"]
            and isinstance(config["hyperparams"]["overfit"], int)
        )
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

    if config["dataset"]["name"].lower() == "cifar10":
        train_loader, val_loader, test_loader = CIFAR10Creator.make_dataloader(
            train_ds,
            val_ds,
            test_ds,
            train_loader_kwargs,
            val_loader_kwargs,
            test_loader_kwargs,
        )
    elif config["dataset"]["name"].lower() == "imagenet":
        train_loader, val_loader, test_loader = ImageNetCreator.make_dataloader(
            train_ds,
            val_ds,
            test_ds,
            train_loader_kwargs,
            val_loader_kwargs,
            test_loader_kwargs,
        )
    else:
        raise ValueError(
            f"This shouldn't happen. "
            f"{SUPPORTED_DATASETS} includes not supported datasets."
        )
    return train_loader, val_loader, test_loader
