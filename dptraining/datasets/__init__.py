from dptraining.datasets.cifar10 import CIFAR10Creator
from dptraining.datasets.imagenet import ImageNetCreator
from dptraining.datasets.utils import collate_np_arrays
from dptraining.utils.augment import Transformation


SUPPORTED_DATASETS = ("cifar10", "imagenet")


def make_collate_fn(config):
    if "collate_fn" in config["loader"]:
        if config["loader"]["collate_fn"] == "numpy":
            config["loader"]["collate_fn"] = collate_np_arrays
        else:
            raise ValueError(
                f"collate_fn {config['loader']['collate_fn']} not supported."
            )


def make_loader_from_config(config):
    if config["dataset"]["name"].lower() not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Dataset {config['dataset']['name']} not supported yet. "
            f"Currently supported datasets: {SUPPORTED_DATASETS}"
        )

    train_tf = (
        Transformation.from_dict_list(config["train_transforms"])
        if "train_transforms" in config
        else None
    )
    test_tf = (
        Transformation.from_dict_list(config["test_transforms"])
        if "train_transforms" in config
        else None
    )
    make_collate_fn(config)
    if config["dataset"]["name"].lower() == "cifar10":
        train_ds, test_ds = CIFAR10Creator.make_datasets(
            (),
            {
                "root": config["dataset"]["root"],
                "download": True,
                "transform": train_tf,
            },
            (),
            {"root": config["dataset"]["root"], "download": True, "transform": test_tf},
        )
        train_loader, test_loader = CIFAR10Creator.make_dataloader(
            train_ds,
            test_ds,
            (),
            {
                "batch_size": config["hyperparams"]["batch_size"],
                "shuffle": True,
                "drop_last": True,
                **config["loader"],
            },
            (),
            {
                "batch_size": config["hyperparams"]["batch_size_test"],
                "shuffle": False,
            },
        )
    elif config["dataset"]["name"].lower() == "imagenet":
        train_ds, test_ds = ImageNetCreator.make_datasets(
            (),
            {
                "root": config["dataset"]["root"],
                "split": "train",
                "transform": train_tf,
            },
            (),
            {"root": config["dataset"]["root"], "split": "val", "transform": test_tf},
        )
        train_loader, test_loader = ImageNetCreator.make_dataloader(
            train_ds,
            test_ds,
            (),
            {
                "batch_size": config["hyperparams"]["batch_size"],
                "shuffle": True,
                "drop_last": True,
                **config["loader"],
            },
            (),
            {"batch_size": config["hyperparams"]["batch_size_test"], "shuffle": False},
        )
    else:
        raise ValueError(
            f"This shouldn't happen. "
            f"{SUPPORTED_DATASETS} includes not supported datasets."
        )
    return train_loader, test_loader
