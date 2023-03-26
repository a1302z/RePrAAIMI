from omegaconf import OmegaConf
from typing import Union
from pathlib import Path

import wandb

from breaching.breaching import get_config
from dptraining.config import Config


from dptraining.models import make_model_from_config, setup_pretrained_model
from dptraining.datasets import make_loader_from_config, make_dataset
from dptraining.utils import make_loss_from_config
from dptraining.utils.augment import make_augs
from dptraining.utils.training_utils import create_loss_gradient
from dptraining.config import Config
from dptraining.config.config_store import load_config_store
from dptraining.privacy.grad_clipper import ClipAndAccumulateGrads
from dptraining.optim.gradaccopt import AccumulateGrad


def make_configs(attack_overrides: list[str], train_config_path: Union[str, Path]):
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


def make_make_fns(train_config):
    def make_model():
        model = make_model_from_config(train_config)
        setup_pretrained_model(train_config, model)
        return model

    def make_loss_gv_from_model(model):
        loss_cls = make_loss_from_config(train_config)
        loss_fn = loss_cls.create_train_loss_fn(model.vars(), model)
        loss_gv = create_loss_gradient(
            config=Config, model_vars=model.vars(), loss_fn=loss_fn
        )
        return loss_fn, loss_gv

    return make_model, make_loss_gv_from_model
