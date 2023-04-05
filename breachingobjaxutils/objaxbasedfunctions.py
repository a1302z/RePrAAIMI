from warnings import warn

from numpy import array
from omegaconf import OmegaConf

from dptraining.utils.augment import make_augs

from dptraining.models import make_model_from_config, setup_pretrained_model
from dptraining.utils import make_loss_from_config
from dptraining.utils.training_utils import create_loss_gradient
from dptraining.config import Config


def make_make_fns(train_config: Config):
    def make_model():
        model = make_model_from_config(train_config)
        setup_pretrained_model(train_config, model)
        return model

    def make_loss_gv_from_model(model):
        loss_cls = make_loss_from_config(train_config)
        loss_fn = loss_cls.create_train_loss_fn(model.vars(), model)
        loss_gv = create_loss_gradient(
            config=train_config, model_vars=model.vars(), loss_fn=loss_fn
        )
        return loss_fn, loss_gv

    return make_model, make_loss_gv_from_model


def get_transform_normalization(config: Config) -> tuple[array, array]:
    dm, ds = None, None
    tf_entries = [key for key in config.keys() if "transform" in key]
    for tf_config in tf_entries:  # this is the easy case
        has_normalize = [key for key in config[tf_config].keys() if "normalize" in key]
        if len(has_normalize) > 0:
            assert len(has_normalize) == 1
            tf = config[tf_config][has_normalize[0]]
            dm, ds = extract_mean_std(tf)
            return dm, ds  # we assume that normalization is the same for train and test


def get_aug_normalization(config: Config) -> tuple[array, array]:
    aug_entries = [key for key in config.keys() if "augmentation" in key]
    results = [get_sub_augs(config[aug]) for aug in aug_entries]
    return boil_down(results)


def extract_mean_std(tf):
    dm = array(tf.mean)
    ds = array(tf.std)
    return dm, ds


def get_sub_augs(aug):
    has_normalize = [key for key in aug.keys() if "normalize" in key]
    if len(has_normalize) > 0:
        tf = aug[has_normalize[0]]
        dm, ds = extract_mean_std(tf)
        temp_cfg = Config()
        temp_cfg.augmentations = OmegaConf.create({has_normalize[0]: tf})
        n_augmentations, augment_op, _, _, _ = make_augs(temp_cfg)
        assert n_augmentations == 1
        return dm, ds, augment_op
    results = []
    for sub_aug in [key for key in aug.keys()]:
        res = get_sub_augs(aug[sub_aug])
        if res is not None:
            results.append(res)
    boil_down(results)


def boil_down(results):
    results = [res for res in results if res is not None]
    if len(results) == 0:
        return None
    else:
        if len(results) > 1:
            warn(
                f"Found multiple normalization results: {results}"
                "\n\t Returning first"
            )
        return results[0]
