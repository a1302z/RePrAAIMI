import numpy as np
import sys
from pathlib import Path
from functools import partial

sys.path.insert(0, str(Path.cwd()))
from dptraining.utils.augment import Transformation
from dptraining.utils.transform import (
    RandomHorizontalFlipsJax,
    RandomVerticalFlipsJax,
    RandomImageShiftsJax,
)


def test_augment_from_string():
    a = Transformation.from_string_list(["random_vertical_flips", "random_img_shift"])


def test_augment_from_init():
    a = Transformation([RandomHorizontalFlipsJax(), RandomVerticalFlipsJax()])


def test_augment_from_dict():
    a = Transformation.from_dict_list(
        {
            "random_vertical_flips": {"flip_prob": 0.5},
            "random_horizontal_flips": {"flip_prob": 0.5},
            "ToTensor": None,
        }
    )


def test_append_augmentation_method():
    a = Transformation([RandomHorizontalFlipsJax()])
    a.append_transform(RandomVerticalFlipsJax())


def test_append_augmentation_method():
    a = Transformation([RandomHorizontalFlipsJax])
    a.append_str_transform("random_img_shift")


def test_create_method():
    a = Transformation.from_string_list(["random_horizontal_flips", "random_img_shift"])
    op = a.create_vectorized_transform()


def test_vertical_flips():
    a = Transformation([RandomVerticalFlipsJax(flip_prob=1.0)])
    op = a.create_vectorized_transform()
    x = np.random.randn(10, 3, 32, 32)
    y = op(x)
    assert np.allclose(x[:, :, :, 0], y[:, :, :, -1])
    a = Transformation([RandomVerticalFlipsJax(flip_prob=0.0)])
    op = a.create_vectorized_transform()
    x = np.random.randn(10, 3, 32, 32)
    y = op(x)
    assert np.allclose(x, y)


def test_horizontal_flips():
    a = Transformation([RandomHorizontalFlipsJax(flip_prob=1.0)])
    op = a.create_vectorized_transform()
    x = np.random.randn(10, 3, 32, 32)
    y = op(x)
    assert np.allclose(x[:, :, 0, :], y[:, :, -1, :])
    a = Transformation([RandomHorizontalFlipsJax(flip_prob=0.0)])
    op = a.create_vectorized_transform()
    x = np.random.randn(10, 3, 32, 32)
    y = op(x)
    assert np.allclose(x, y)


def test_img_shifts():
    a = Transformation([RandomImageShiftsJax(img_shape=(3, 32, 32))])
    op = a.create_vectorized_transform()
    x = np.random.randn(10, 3, 32, 32)
    y = op(x)
