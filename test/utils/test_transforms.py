import numpy as np
from jax import numpy as jnp
import sys
from pathlib import Path
from functools import partial

sys.path.insert(0, str(Path.cwd()))
from dptraining.utils.augment import Transformation
from dptraining.utils.transform import (
    RandomHorizontalFlipsJax,
    RandomVerticalFlipsJax,
    RandomImageShiftsJax,
    MakeComplexOnlyReal,
    MakeComplexRealAndImaginary,
    TransposeNumpyImgToCHW,
    TransposeNumpyBatchToCHW,
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


def test_complex_conversion_real():
    a = Transformation.from_string_list(["make_complex_real"])
    op = a.create_vectorized_transform()
    x = np.random.randn(10, 3, 32, 32)
    y = op(x)
    assert y.dtype == jnp.complex64
    assert y.imag.sum() < 1e-3


def test_complex_conversion_real():
    a = Transformation.from_string_list(["make_complex_both"])
    op = a.create_vectorized_transform()
    x = np.random.randn(10, 3, 32, 32)
    y = op(x)
    assert y.dtype == jnp.complex64
    assert jnp.sum(jnp.abs(y.real - y.imag)) < 1e-3


def test_transpose_to_hwc():
    tf = Transformation.from_string_list(["numpy_img_to_chw"])
    data = np.random.randn(224, 224, 3)
    out = tf(data)
    assert out.shape == (3, 224, 224)


def test_transpose_to_hwc_batch():
    tf = Transformation.from_string_list(["numpy_batch_to_chw"])
    data = np.random.randn(10, 224, 224, 3)
    out = tf(data)
    assert out.shape == (10, 3, 224, 224)
