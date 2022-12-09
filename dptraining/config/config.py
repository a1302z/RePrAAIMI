from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Optional
from psutil import cpu_count

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


class DatasetName(Enum):
    CIFAR10 = 1
    imagenet = 2
    tinyimagenet = 3
    fastmri = 4
    radimagenet = 5
    msd = 6


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
class DatasetConfig:
    name: DatasetName = MISSING
    root: str = MISSING
    version: Optional[int] = None
    train_val_split: float = MISSING
    normalization: bool = False
    download: Optional[bool] = False
    fft: bool = False
    task: DatasetTask = MISSING
    datasplit_seed: Optional[int] = 0  # only for radimagenet
    modality: str = "all"  # only for radimagenet
    normalize_by_modality: bool = False  # only for radimagenet
    test_split: float = 0.1  # only for radimagenet
    allowed_body_regions: str = "all"  # only for radimagenet
    allowed_labels: str = "all"  # only for radimagenet
    split_folder: Optional[str] = None  # only for radimagenet
    mask_type: str = "random"  # only for fmri
    center_fractions: tuple[float] = (0.08,)  # only for fmri
    accelerations: tuple[float] = (4,)  # only for fmri
    challenge: str = "knee"  # only for fmri
    resolution: Optional[int] = None  # only for fmri and msd so far
    new_data_root: Optional[str] = None  # only for fmri so far
    subtask: Optional[MSDSubtask] = None  # only for msd so far
    slice_thickness: Optional[float] = None  # only for msd
    cache: bool = False  # only for MSD
    normalization: Normalization = Normalization.gaussian


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


class LossReduction(Enum):
    sum = 1
    mean = 2


@dataclass
class LossConfig:
    type: LossType = MISSING
    reduction: LossReduction = MISSING


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


class SchedulerType(Enum):
    cosine = 1
    const = 2
    reduceonplateau = 3


@dataclass
class SchedulerConfig:
    type: SchedulerType = MISSING
    normalize_lr: bool = MISSING
    mode: str = "maximize"
    cumulative_delta: bool = True
    min_delta: float = MISSING
    patience: int = MISSING
    factor: float = MISSING


@dataclass
class EarlyStoppingConfig:
    mode: str = "maximize"
    cumulative_delta: bool = True
    min_delta: float = MISSING
    patience: int = MISSING


@dataclass
class DPConfig:
    epsilon: float = MISSING
    max_per_sample_grad_norm: float = MISSING
    delta: float = MISSING
    sigma: Optional[float] = None
    norm_acc: bool = MISSING
    grad_acc_steps: int = 1
    rsqrt_noise_adapt: bool = False
    glrt_assumption: bool = False


@dataclass
class Config:
    project: str = MISSING
    general: GeneralConfig = GeneralConfig()
    dataset: DatasetConfig = DatasetConfig()
    ckpt: dict[str, Any] = field(default_factory=dict)
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
    metrics: Optional[dict[str, Any]] = field(default_factory=dict)
    DP: Optional[DPConfig] = None
