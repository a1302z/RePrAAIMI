from omegaconf import OmegaConf
from typing import Union
from pathlib import Path

import wandb

from breaching.breaching import get_config
from dptraining.config import Config


def make_configs(
    attack_overrides: list[str], train_config_path: Union[str, Path]
) -> tuple[OmegaConf, Config]:
    cfg = get_config(overrides=attack_overrides)
    base_config = OmegaConf.structured(Config)
    train_config = OmegaConf.load(train_config_path)
    del train_config.defaults
    train_config = OmegaConf.merge(base_config, train_config)
    return cfg, train_config


def init_wandb(cfg, train_config):
    run = None
    if hasattr(cfg.attack, "wandb"):
        cfg_dict = OmegaConf.to_container(cfg)
        train_dict = OmegaConf.to_container(train_config)
        del train_dict["wandb"]
        total_dict = {**cfg_dict, **train_dict}
        run = wandb.init(
            config=total_dict,
            settings=wandb.Settings(start_method="thread"),
            reinit=True,
            **cfg_dict["attack"]["wandb"],
        )
    return run
