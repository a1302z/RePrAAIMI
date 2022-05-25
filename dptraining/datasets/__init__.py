from dptraining.datasets.cifar10 import CIFAR10Creator


SUPPORTED_DATASETS = ["CIFAR10"]


def make_loader_from_config(config):
    if config["dataset"] == "CIFAR10":
        train_ds, test_ds = CIFAR10Creator.make_datasets(
            (),
            {"root": "./data", "download": True},
            (),
            {"root": "./data", "download": True},
        )
        train_loader, test_loader = CIFAR10Creator.make_dataloader(
            train_ds,
            test_ds,
            (),
            {"batch_size": config["hyperparams"]["batch_size"], "shuffle": True},
            (),
            {"batch_size": config["hyperparams"]["batch_size_test"], "shuffle": False},
        )
    else:
        raise ValueError(
            f"Dataset {config['dataset']} not supported yet. "
            f"Currently supported datasets: {SUPPORTED_DATASETS}"
        )
    return train_loader, test_loader
