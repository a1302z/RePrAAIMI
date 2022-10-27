from pathlib import Path

from dptraining.datasets.fmri.subsample import create_mask_for_mask_type
from dptraining.datasets.fmri.transforms import UnetDataTransform
from dptraining.datasets.fmri.data_module import FastMriDataModule


def make_fmri_dataset(config):

    mask = create_mask_for_mask_type(
        config["dataset"]["mask_type"],
        config["dataset"]["center_fractions"],
        config["dataset"]["accelerations"],
    )
    train_transform = UnetDataTransform(
        config["dataset"]["challenge"],
        mask_func=mask,
        use_seed=False,
        size=config["dataset"]["resolution"],
    )
    val_transform = UnetDataTransform(
        config["dataset"]["challenge"],
        mask_func=mask,
        size=config["dataset"]["resolution"],
    )
    test_transform = UnetDataTransform(
        config["dataset"]["challenge"], size=config["dataset"]["resolution"]
    )
    data_module = FastMriDataModule(
        data_path=Path(config["dataset"]["root"]),
        challenge=config["dataset"]["challenge"],
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        test_split="test",
        test_path=None,
        sample_rate=None,
        batch_size=config["hyperparams"]["batch_size"],
        num_workers=config["loader"]["num_workers"],
    )
    train_loader, val_loader = (
        data_module.train_dataloader(
            overfit=config["hyperparams"]["overfit"]
            if "overfit" in config["hyperparams"]
            and isinstance(config["hyperparams"]["overfit"], int)
            else None
        ),
        data_module.val_dataloader(),
        # data_module.test_dataloader(),
    )
    # if config["hyperparams"]["overfit"] is not None:
    #     val_loader = train_loader
    test_loader = val_loader  # TODO: make real test set
    return train_loader, val_loader, test_loader
