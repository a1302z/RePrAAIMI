import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import numpy as np
from jax import numpy as jnp
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from dptraining.transform import (
    TransformPipeline,
    RandomImageShiftsJax,
    RandomVerticalFlipsJax,
    RandomHorizontalFlipsJax,
)


def test_construct_transform_pipeline():
    tf = TransformPipeline(
        [
            {"apply": RandomHorizontalFlipsJax().unary_transform, "to": "images"},
            {"apply": RandomVerticalFlipsJax().unary_transform, "to": "images"},
        ]
    )
    assert len(tf.get_transformations()) == 2


def test_construct_transform_pipeline_defaults_to_images():
    tf = TransformPipeline(
        [
            {"apply": RandomHorizontalFlipsJax().unary_transform},
            {"apply": RandomVerticalFlipsJax().unary_transform},
        ]
    )
    assert len(tf.get_transformations()) == 2
    assert all(single_tf["to"] == "images" for single_tf in tf.get_transformations())


def test_construct_transform_pipeline_from_config_dict():
    tf = TransformPipeline.from_dict_list(
        {
            "random_vertical_flips": {"flip_prob": 0.5},
            "random_horizontal_flips": {"flip_prob": 0.5},
            "ToTensor": None,
        }
    )
    assert len(tf.get_transformations()) == 3


def test_construct_transform_pipeline_from_config_dict_defaults_to_images():
    tf = TransformPipeline.from_dict_list(
        {
            "random_vertical_flips": {"flip_prob": 0.5},
            "random_horizontal_flips": {"flip_prob": 0.5},
        }
    )
    assert len(tf.get_transformations()) == 2
    assert all(single_tf["to"] == "images" for single_tf in tf.get_transformations())


def test_transform_image_label():
    tf = TransformPipeline(
        [
            {"apply": (lambda x, y: (x + 1, y + 2)), "to": "images_and_labels"},
        ]
    )
    image = np.array([1, 1, 1])
    label = np.array([1, 1, 1])
    expected_transformed_image = np.array([2, 2, 2])
    expected_transformed_label = np.array([3, 3, 3])
    actual_transformed_image, actual_transformed_label = tf.transform_image_label(
        image, label
    )
    assert np.allclose(expected_transformed_image, actual_transformed_image)
    assert np.allclose(expected_transformed_label, actual_transformed_label)


def test_transform_image_label_vectorized():
    tf = TransformPipeline(
        [
            {"apply": (lambda x, y: (x + 1, y + 2)), "to": "images_and_labels"},
        ]
    )
    image = np.array([1, 1, 1])
    label = np.array([1, 1, 1])
    expected_transformed_image = np.array([2, 2, 2])
    expected_transformed_label = np.array([3, 3, 3])
    apply_transform_vectorized = tf.create_vectorized_image_label_transform()
    actual_transformed_image, actual_transformed_label = apply_transform_vectorized(
        image, label
    )
    assert np.allclose(expected_transformed_image, actual_transformed_image)
    assert np.allclose(expected_transformed_label, actual_transformed_label)


def test_transform_image_label_without_label_transformations():
    tf = TransformPipeline(
        [
            {"apply": (lambda x: x + 1), "to": "images"},
        ]
    )
    image = np.array([1, 1, 1])
    label = np.array([1, 1, 1])
    expected_transformed_image = np.array([2, 2, 2])
    expected_transformed_label = np.array([1, 1, 1])
    actual_transformed_image, actual_transformed_label = tf.transform_image_label(
        image, label
    )
    assert np.allclose(expected_transformed_image, actual_transformed_image)
    assert np.allclose(expected_transformed_label, actual_transformed_label)


def test_transform_image_label_without_image_transformations():
    tf = TransformPipeline(
        [
            {"apply": (lambda x: x + 1), "to": "labels"},
        ]
    )
    image = np.array([1, 1, 1])
    label = np.array([1, 1, 1])
    expected_transformed_image = np.array([1, 1, 1])
    expected_transformed_label = np.array([2, 2, 2])
    actual_transformed_image, actual_transformed_label = tf.transform_image_label(
        image, label
    )
    assert np.allclose(expected_transformed_image, actual_transformed_image)
    assert np.allclose(expected_transformed_label, actual_transformed_label)


def test_transform_image_label_plain_and_vectorized_with_mixed_transformations():
    tf = TransformPipeline(
        [
            {"apply": (lambda x: x + 1), "to": "images"},
            {"apply": (lambda x, y: (x + 2, y + 2)), "to": "images_and_labels"},
            {"apply": (lambda y: y + 4), "to": "labels"},
        ]
    )
    image = np.array([1, 1, 1])
    label = np.array([1, 1, 1])
    expected_transformed_image = np.array([4, 4, 4])
    expected_transformed_label = np.array([7, 7, 7])
    actual_transformed_image, actual_transformed_label = tf.transform_image_label(
        image, label
    )
    assert np.allclose(expected_transformed_image, actual_transformed_image)
    assert np.allclose(expected_transformed_label, actual_transformed_label)
    apply_transform_vectorized = tf.create_vectorized_image_label_transform()
    (
        actual_transformed_image_vectorized,
        actual_transformed_label_vectorized,
    ) = apply_transform_vectorized(image, label)
    assert np.allclose(expected_transformed_image, actual_transformed_image_vectorized)
    assert np.allclose(expected_transformed_label, actual_transformed_label_vectorized)


def test_vertical_flips():
    a = TransformPipeline(
        [{"apply": RandomVerticalFlipsJax(flip_prob=1.0).unary_transform}]
    )
    op = a.create_vectorized_image_transform()
    x = np.random.randn(10, 3, 32, 32)
    y = op(x)
    assert np.allclose(x[:, :, :, 0], y[:, :, :, -1])
    a = TransformPipeline(
        [{"apply": RandomVerticalFlipsJax(flip_prob=0.0).unary_transform}]
    )
    op = a.create_vectorized_image_transform()
    x = np.random.randn(10, 3, 32, 32)
    y = op(x)
    assert np.allclose(x, y)


def test_horizontal_flips():
    a = TransformPipeline(
        [{"apply": RandomHorizontalFlipsJax(flip_prob=1.0).unary_transform}]
    )
    op = a.create_vectorized_image_transform()
    x = np.random.randn(10, 3, 32, 32)
    y = op(x)
    assert np.allclose(x[:, :, 0, :], y[:, :, -1, :])
    a = TransformPipeline(
        [{"apply": RandomHorizontalFlipsJax(flip_prob=0.0).unary_transform}]
    )
    op = a.create_vectorized_image_transform()
    x = np.random.randn(10, 3, 32, 32)
    y = op(x)
    assert np.allclose(x, y)


def test_img_shifts():
    a = TransformPipeline(
        [{"apply": RandomImageShiftsJax(img_shape=(3, 32, 32)).unary_transform}]
    )
    op = a.create_vectorized_image_transform()
    x = np.random.randn(10, 3, 32, 32)
    y = op(x)


def test_complex_conversion_real():
    a = TransformPipeline.from_dict_list({"make_complex_real": None})
    op = a.create_vectorized_image_transform()
    x = np.random.randn(10, 3, 32, 32)
    y = op(x)
    assert y.dtype == jnp.complex64
    assert y.imag.sum() < 1e-3


def test_complex_conversion_real_and_imaginary():
    a = TransformPipeline.from_dict_list({"make_complex_both": None})
    op = a.create_vectorized_image_transform()
    x = np.random.randn(10, 3, 32, 32)
    y = op(x)
    assert y.dtype == jnp.complex64
    assert jnp.sum(jnp.abs(y.real - y.imag)) < 1e-3


def test_transpose_to_hwc():
    tf = TransformPipeline.from_dict_list({"numpy_img_to_chw": None})
    data = np.random.randn(32, 32, 3)
    out = tf.transform_image(data)
    assert out.shape == (3, 32, 32)


def test_transpose_to_hwc_batch():
    tf = TransformPipeline.from_dict_list({"numpy_batch_to_chw": None})
    data = np.random.randn(10, 32, 32, 3)
    out = tf.transform_image(data)
    assert out.shape == (10, 3, 32, 32)


def test_complex_aug_pre():
    tf = TransformPipeline.from_dict_list(
        {
            "random_horizontal_flips_batch": {"flip_prob": 0.5},
            "random_vertical_flips_batch": {"flip_prob": 0.5},
            "make_complex_both": None,
        }
    )
    data = np.random.randn(10, 3, 32, 32)
    out = tf.transform_image(data)
    assert np.allclose(out.real, out.imag)


def test_complex_aug_same():
    tf = TransformPipeline.from_dict_list(
        {
            "make_complex_both": None,
            "consecutive_augmentations": {
                "complex": True,
                "augmentations": {
                    "random_horizontal_flips_batch": {"flip_prob": 1.0},
                    "random_vertical_flips_batch": {"flip_prob": 0.0},
                },
            },
        }
    )
    data = np.random.randn(10, 3, 32, 32)
    out = tf.transform_image(data)
    assert np.allclose(out.real, out.imag)


def test_complex_aug_diff():
    tf = TransformPipeline.from_dict_list(
        {
            "make_complex_both": None,
            "consecutive_augmentations": {
                "complex": True,
                "augmentations": {
                    "random_horizontal_flips": {"flip_prob": 0.5},
                    "random_vertical_flips": {"flip_prob": 0.5},
                    # "random_img_shift_batch": {"max_shift": 4},
                },
            },
        }
    )
    tf = tf.create_vectorized_image_transform()
    data = np.random.randn(10, 3, 32, 32)
    out = tf(data)
    assert not np.allclose(
        out.real, out.imag
    )  # chance that this fails although it's correct is 0.000000954
    # -> chance that it failed once after 1000 runs is still < 1%
    # -> chance that it failed once after 100.000 runs is < 10%


def test_multiplicity():
    tf = TransformPipeline.from_dict_list(
        {
            "make_complex_both": None,
            "consecutive_augmentations": {
                "complex": True,
                "augmentations": [
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
                        "consecutive_augmentations": {  # combined
                            "complex": True,
                            "augmentations": {
                                "random_horizontal_flips": {"flip_prob": 0.5},
                                "random_vertical_flips": {"flip_prob": 0.5},
                                "random_img_shift": {"max_shift": 4},
                            },
                        }
                    },
                    {
                        "consecutive_augmentations": {  # combined
                            "complex": True,
                            "augmentations": {
                                "random_horizontal_flips": {"flip_prob": 0.5},
                                "random_vertical_flips": {"flip_prob": 0.5},
                                "random_img_shift": {"max_shift": 6},
                            },
                        }
                    },
                    {
                        "consecutive_augmentations": {  # combined
                            "complex": True,
                            "augmentations": {
                                "random_horizontal_flips": {"flip_prob": 0.5},
                                "random_vertical_flips": {"flip_prob": 0.5},
                                "random_img_shift": {"max_shift": 8},
                            },
                        }
                    },
                ],
            },
        }
    )
    tf = tf.create_vectorized_image_transform()
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
    tf = TransformPipeline.from_dict_list(
        {
            "random_augmentations": {
                "augmentations": {
                    "random_horizontal_flips": {"flip_prob": 1.0},
                },
            },
        }
    )
    tf = tf.create_vectorized_image_transform()
    data = np.random.randn(10, 3, 32, 32)
    out = tf(data)
    assert np.allclose(data, out[:, :, ::-1, :])


def test_multiplicity_2():
    augmenter = TransformPipeline.from_dict_list(
        {
            "consecutive_augmentations": {
                "multiplicity": 8,
                "augmentations": [
                    {
                        "consecutive_augmentations": {
                            "multiplicity": 3,
                            "augmentations": {
                                "random_img_shift": {"max_shift": 4},
                                "random_vertical_flips": {"flip_prob": 0.5},
                            },
                        }
                    },
                    {
                        "consecutive_augmentations": {
                            "multiplicity": 4,
                            "augmentations": {
                                "random_img_shift": {"max_shift": 4},
                                "random_vertical_flips": {"flip_prob": 0.5},
                            },
                        }
                    },
                    {
                        "consecutive_augmentations": {
                            "augmentations": {
                                "random_img_shift": {"max_shift": 4},
                                "random_vertical_flips": {"flip_prob": 0.5},
                            }
                        }
                    },
                    {"random_img_shift": {"max_shift": 4}},
                ],
            }
        }
    )
    aug_op = augmenter.create_vectorized_image_transform()
    n_transforms = augmenter.get_n_augmentations()
    assert n_transforms == 8 * 12
    data = np.random.randn(10, 3, 32, 32)
    aug_data = aug_op(data)
    assert aug_data.shape[1] == n_transforms


def test_fft():
    augmenter = TransformPipeline.from_dict_list({"jaxfft": (1, 2)})
    aug_op = augmenter.create_vectorized_image_transform()
    data = jnp.array(np.random.randn(10, 3, 32, 32))
    fft_data = aug_op(data)
    fft_data = np.array(fft_data)
    recon_data = np.fft.ifftshift(np.fft.ifft2(fft_data, axes=(2, 3)), axes=(2, 3)).real
    assert np.allclose(data, recon_data, atol=1e-5)
