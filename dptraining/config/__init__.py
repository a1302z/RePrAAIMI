from omegaconf import OmegaConf

from dptraining.config.config import (
    Config,
    GeneralConfig,
    DatasetConfig,
    DatasetName,
    LoaderConfig,
    LoaderCollateFn,
    ModelConfig,
    OptimConfig,
    OptimName,
    LossConfig,
    LossType,
    LossReduction,
    EmaConfig,
    HyperparamsConfig,
    SchedulerConfig,
    SchedulerType,
    EarlyStoppingConfig,
    DPConfig,
    DatasetTask,
    Normalization,
    DataStats,
    CTWindow,
    AttackType,
    AttackInput,
)

from dptraining.config.utils import get_allowed_names, get_allowed_values


def dict_to_config(overrides: dict):
    base_conf = OmegaConf.structured(Config)
    merged_conf = OmegaConf.merge(base_conf, overrides)
    return merged_conf
