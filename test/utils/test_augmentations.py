import numpy as np
import sys
from pathlib import Path
from functools import partial

sys.path.insert(0, str(Path.cwd()))
from dptraining.utils.augment import Augmentation


def test_augment_from_string():
    a = Augmentation.from_string_list(["random_vertical_flips", "random_img_shift"])


def test_augment_from_init():
    a = Augmentation(
        [Augmentation.random_horizontal_flips, Augmentation.random_img_shift]
    )


def test_append_augmentation_method():
    a = Augmentation([Augmentation.random_horizontal_flips])
    a.append_augmentation_method(Augmentation.random_cifar_shift)


def test_append_augmentation_method():
    a = Augmentation([Augmentation.random_horizontal_flips])
    a.append_augmentation_str("random_img_shift")


def test_create_method():
    a = Augmentation.from_string_list(["random_horizontal_flips", "random_img_shift"])
    op = a.create_augmentation_op()


def test_vertical_flips():
    a = Augmentation([partial(Augmentation.random_vertical_flips, flip_prob=1.0)])
    op = a.create_augmentation_op()
    x = np.random.randn(10, 3, 32, 32)
    y = op(x)
    assert np.allclose(x[:, :, :, 0], y[:, :, :, -1])
    a = Augmentation([partial(Augmentation.random_vertical_flips, flip_prob=0.0)])
    op = a.create_augmentation_op()
    x = np.random.randn(10, 3, 32, 32)
    y = op(x)
    assert np.allclose(x, y)


def test_horizontal_flips():
    a = Augmentation([partial(Augmentation.random_horizontal_flips, flip_prob=1.0)])
    op = a.create_augmentation_op()
    x = np.random.randn(10, 3, 32, 32)
    y = op(x)
    assert np.allclose(x[:, :, 0, :], y[:, :, -1, :])
    a = Augmentation([partial(Augmentation.random_horizontal_flips, flip_prob=0.0)])
    op = a.create_augmentation_op()
    x = np.random.randn(10, 3, 32, 32)
    y = op(x)
    assert np.allclose(x, y)


def test_img_shifts():
    a = Augmentation([partial(Augmentation.random_img_shift, img_size=(3, 32, 32))])
    op = a.create_augmentation_op()
    x = np.random.randn(10, 3, 32, 32)
    y = op(x)
