from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Optional
from psutil import cpu_count
from pathlib import Path

from omegaconf import MISSING

from dptraining.config.model.model import ModelConfig

# We have to use Enums here as Literals are not supported by OmegaConf yet
# (see https://github.com/omry/omegaconf/issues/422).

# pylint:disable=invalid-name


@dataclass
class GeneralConfig:
    log_wandb: bool = False
    parallel: bool = True
    cpu: bool = False
    seed: int = 0
    use_pretrained_model: Optional[str] = None
    save_path: Optional[str] = None
    eval_train: bool = MISSING
    make_save_str_unique: Optional[str] = None
    eval_init: bool = False


@dataclass
class WandBConfig:
    project: str = MISSING
    entity: Optional[str] = None
    notes: Optional[str] = None
    group: Optional[str] = None
    magic: Optional[bool] = None
    config_exclude_keys: Optional[list[str]] = None
    config_include_keys: Optional[list[str]] = None
    tags: Optional[list[str]] = None
    name: Optional[str] = None
    save_code: Optional[bool] = None


class DatasetName(Enum):
    CIFAR10 = 1
    imagenet = 2
    tinyimagenet = 3
    fastmri = 4
    radimagenet = 5
    msd = 6
    ukbb_seg = 7
    ham10000 = 8


class MSDSubtask(Enum):
    braintumour = 1
    heart = 2
    liver = 3
    hippocampus = 4
    prostate = 5
    lung = 6
    pancreas = 7
    hepaticvessel = 8
    spleen = 9
    colon = 10


class DatasetTask(Enum):
    classification = 1
    reconstruction = 2
    segmentation = 3


class Normalization(Enum):
    raw = 0
    zeroone = 1
    gaussian = 2
    consecutive = 3


@dataclass
class DataStats:
    mean: float = MISSING
    std: float = MISSING


@dataclass
class CTWindow:
    low: int = MISSING
    high: int = MISSING


@dataclass
class FmriConfig:
    mask_type: str = "random"
    center_fractions: tuple[float] = (0.08,)
    accelerations: tuple[float] = (4,)
    challenge: str = "knee"
    resolution: int = 320
    new_data_root: Optional[str] = None


@dataclass
class RadimagenetConfig:
    datasplit_seed: Optional[int] = 0
    modality: str = "all"
    normalize_by_modality: bool = False
    allowed_body_regions: str = "all"
    allowed_labels: str = "all"
    split_folder: Optional[str] = None


@dataclass
class FilterOptionsNifti:
    resolution: Optional[tuple[int, int, int]] = None
    min_pixels_per_organ: Optional[tuple[int]] = None
    length_threshold: Optional[int] = None
    save_filtered_files: Optional[Path] = None
    reuse_filtered_files: Optional[Path] = None


@dataclass
class NiftiSegmentationConfig:
    slice_thickness: Optional[float] = None
    n_slices: Optional[int] = None
    cache: bool = False
    normalization_type: Normalization = MISSING
    data_stats: Optional[DataStats] = None
    ct_window: Optional[CTWindow] = None
    test_split: float = MISSING
    resolution: Optional[int] = None
    datasplit_seed: Optional[int] = 0
    assume_same_settings: bool = False
    msd_subtask: MSDSubtask = MISSING
    new_data_root: Optional[str] = None
    image_file_root: Optional[str] = None
    label_file_root: Optional[str] = None
    normalize_per_scan: bool = False
    filter_options: Optional[FilterOptionsNifti] = None
    limit_dataset: Optional[int] = None
    database: Optional[Path] = None


@dataclass
class HAM10000:
    merge_labels: bool = True  # only for HAM10000


@dataclass
class DatasetConfig:
    name: DatasetName = MISSING
    root: str = MISSING
    version: Optional[int] = None
    train_val_split: float = MISSING
    normalization: bool = False
    download: Optional[bool] = False
    fft: bool = False
    task: DatasetTask = MISSING
    test_split: float = 0.1
    radimagenet: Optional[RadimagenetConfig] = None
    fmri: Optional[FmriConfig] = None
    nifti_seg_options: Optional[NiftiSegmentationConfig] = None
    ham: Optional[HAM10000] = None


class LoaderCollateFn(Enum):
    numpy = 1


@dataclass
class LoaderConfig:
    num_workers: Optional[int] = cpu_count()
    prefetch_factor: Optional[int] = None
    collate_fn: LoaderCollateFn = LoaderCollateFn.numpy
    pin_memory: bool = MISSING


class OptimName(Enum):
    sgd = 1
    momentum = 2
    adam = 3
    nadam = 4


@dataclass
class OptimConfig:
    name: OptimName = OptimName.nadam
    args: dict[str, Any] = field(default_factory=dict)


class LossType(Enum):
    cse = 1
    l1 = 2
    dice = 3


class LossReduction(Enum):
    sum = 1
    mean = 2


@dataclass
class LossConfig:
    type: LossType = MISSING
    reduction: LossReduction = MISSING
    binary_loss: bool = MISSING
    class_weights: Optional[list[float]] = None


@dataclass
class EmaConfig:
    use_ema: bool = False
    decay: float = MISSING
    update_every: int = 1


@dataclass
class HyperparamsConfig:
    epochs: Optional[int] = None
    batch_size: int = MISSING
    batch_size_test: int = MISSING
    batch_size_val: Optional[int] = None
    lr: float = MISSING
    l2regularization: Optional[float] = None
    overfit: Optional[int] = None
    grad_acc_steps: int = 1


class SchedulerType(Enum):
    const = 1
    cosine = 2
    reduceonplateau = 3
    manual = 4


@dataclass
class SchedulerConfig:
    type: SchedulerType = SchedulerType.const
    normalize_lr: bool = False
    mode: str = "maximize"
    cumulative_delta: bool = True
    min_delta: float = MISSING
    patience: int = MISSING
    factor: float = MISSING
    epoch_triggers: list[int] = MISSING
    lr_list: list[float] = MISSING


@dataclass
class EarlyStoppingConfig:
    mode: str = "maximize"
    cumulative_delta: bool = True
    min_delta: float = MISSING
    patience: int = MISSING


@dataclass
class UnfreezingSchedule:
    trigger_points: list[int] = MISSING


@dataclass
class DPConfig:
    epsilon: float = MISSING
    max_per_sample_grad_norm: float = MISSING
    delta: float = MISSING
    sigma: Optional[float] = None
    norm_acc: bool = MISSING
    rsqrt_noise_adapt: bool = False
    glrt_assumption: bool = False
    mechanism: str = "gdp"
    eps_tol: float = 1e-5


@dataclass
class Config:
    general: GeneralConfig = GeneralConfig()
    wandb: WandBConfig = WandBConfig()
    dataset: DatasetConfig = DatasetConfig()
    checkpoint: dict[str, Any] = field(default_factory=dict)
    train_transforms: dict[str, Any] = field(default_factory=dict)
    test_transforms: dict[str, Any] = field(default_factory=dict)
    val_transforms: dict[str, Any] = field(default_factory=dict)
    loader: LoaderConfig = LoaderConfig()
    augmentations: dict[str, Any] = field(default_factory=dict)
    test_augmentations: dict[str, Any] = field(default_factory=dict)
    label_augmentations: dict[str, Any] = field(default_factory=dict)
    test_label_augmentations: dict[str, Any] = field(default_factory=dict)
    model: ModelConfig = ModelConfig()
    optim: OptimConfig = OptimConfig()
    loss: LossConfig = LossConfig()
    ema: EmaConfig = EmaConfig()
    hyperparams: HyperparamsConfig = HyperparamsConfig()
    earlystopping: Optional[EarlyStoppingConfig] = None
    scheduler: SchedulerConfig = SchedulerConfig()
    unfreeze_schedule: Optional[UnfreezingSchedule] = None
    metrics: dict[str, Any] = field(default_factory=dict)
    DP: Optional[DPConfig] = None
