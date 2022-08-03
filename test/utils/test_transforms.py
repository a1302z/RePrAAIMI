import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import numpy as np
from jax import numpy as jnp
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from dptraining.utils.augment import Transformation
from dptraining.utils.transform import (
    RandomHorizontalFlipsJax,
    RandomVerticalFlipsJax,
    RandomImageShiftsJax,
    # MakeComplexOnlyReal,
    # MakeComplexRealAndImaginary,
    # TransposeNumpyImgToCHW,
    # TransposeNumpyBatchToCHW,
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
    data = np.random.randn(32, 32, 3)
    out = tf(data)
    assert out.shape == (3, 32, 32)


def test_transpose_to_hwc_batch():
    tf = Transformation.from_string_list(["numpy_batch_to_chw"])
    data = np.random.randn(10, 32, 32, 3)
    out = tf(data)
    assert out.shape == (10, 3, 32, 32)


def test_complex_aug_pre():
    tf = Transformation.from_dict_list(
        {
            "random_horizontal_flips_batch": {"flip_prob": 0.5},
            "random_vertical_flips_batch": {"flip_prob": 0.5},
            "make_complex_both": None,
        }
    )
    data = np.random.randn(10, 3, 32, 32)
    out = tf(data)
    assert np.allclose(out.real, out.imag)


def test_complex_aug_same():
    tf = Transformation.from_dict_list(
        {
            "make_complex_both": None,
            "complex_augmentations": {
                "random_horizontal_flips_batch": {"flip_prob": 1.0},
                "random_vertical_flips_batch": {"flip_prob": 0.0},
            },
        }
    )
    data = np.random.randn(10, 3, 32, 32)
    out = tf(data)
    assert np.allclose(out.real, out.imag)


def test_complex_aug_diff():
    tf = Transformation.from_dict_list(
        {
            "make_complex_both": None,
            "complex_augmentations": {
                "random_horizontal_flips": {"flip_prob": 0.5},
                "random_vertical_flips": {"flip_prob": 0.5},
                # "random_img_shift_batch": {"max_shift": 4},
            },
        }
    )
    tf = tf.create_vectorized_transform()
    data = np.random.randn(10, 3, 32, 32)
    out = tf(data)
    assert not np.allclose(
        out.real, out.imag
    )  # chance that this fails although it's correct is 0.000000954
    # -> chance that it failed once after 1000 runs is still < 1%
    # -> chance that it failed once after 100.000 runs is < 10%


def test_multiplicity():
    tf = Transformation.from_dict_list(
        {
            "make_complex_both": None,
            "complex_augmentations": [
                "stack_augmentations",
                {"random_horizontal_flips": {"flip_prob": 0}},  # identity
                {"random_horizontal_flips": {"flip_prob": 1.0}},
                {"random_vertical_flips": {"flip_prob": 1.0}},
                {"random_horizontal_flips": {"flip_prob": 0.5}},
                {"random_horizontal_flips": {"flip_prob": 0.5}},
                {"random_horizontal_flips": {"flip_prob": 0.5}},
                {"random_vertical_flips": {"flip_prob": 0.5}},
                {"random_vertical_flips": {"flip_prob": 0.5}},
                {"random_vertical_flips": {"flip_prob": 0.5}},
                {"random_img_shift": {"max_shift": 4}},
                {"random_img_shift": {"max_shift": 4}},
                {"random_img_shift": {"max_shift": 4}},
                {"random_img_shift": {"max_shift": 8}},
                {"random_img_shift": {"max_shift": 8}},
                {"random_img_shift": {"max_shift": 8}},
                {
                    "complex_augmentations": {  # combined
                        "random_horizontal_flips": {"flip_prob": 0.5},
                        "random_vertical_flips": {"flip_prob": 0.5},
                        "random_img_shift": {"max_shift": 4},
                    }
                },
                {
                    "complex_augmentations": {  # combined
                        "random_horizontal_flips": {"flip_prob": 0.5},
                        "random_vertical_flips": {"flip_prob": 0.5},
                        "random_img_shift": {"max_shift": 6},
                    }
                },
                {
                    "complex_augmentations": {  # combined
                        "random_horizontal_flips": {"flip_prob": 0.5},
                        "random_vertical_flips": {"flip_prob": 0.5},
                        "random_img_shift": {"max_shift": 8},
                    }
                },
            ],
        }
    )
    tf = tf.create_vectorized_transform()
    data = np.random.randn(10, 3, 32, 32)
    out = tf(data)
    assert np.allclose(out[:, 0].real, data) and np.allclose(out[:, 0].imag, data)
    assert np.allclose(out[:, 1].real, out[:, 1].imag)
    assert np.allclose(out[:, 2].real, out[:, 2].imag)

    for i in range(out.shape[1]):
        for j in range(i + 1, out.shape[1]):
            if jnp.allclose(out[:, j], out[:, i]):
                raise ValueError()
            if jnp.allclose(out[:, j].real, out[:, i].real):
                raise ValueError()
            if jnp.allclose(out[:, j].imag, out[:, i].imag):
                raise ValueError()


def test_random_transform():
    tf = Transformation.from_dict_list(
        {
            "random_augmentations": [
                {"random_horizontal_flips": {"flip_prob": 1.0}},
                {"random_horizontal_flips": {"flip_prob": 1.0}},
            ],
        }
    )

    tf = tf.create_vectorized_transform()
    data = np.random.randn(10, 3, 32, 32)
    out = tf(data)
    assert np.allclose(data, out[:, :, ::-1, :])
