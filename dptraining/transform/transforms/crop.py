import jax
from jax import numpy as jn
import numpy as np
import objax
from typing import Union

from dptraining.transform.unary_image_and_label_transform import (
    UnaryImageAndLabelTransform,
)
from dptraining.transform.image_label_transform import ImageLabelTransform


class CenterCrop(UnaryImageAndLabelTransform):
    def __init__(self, size: Union[tuple[int], int]) -> None:
        super().__init__()
        self._crop_size: Union[tuple[int], int] = size

    def unary_transform(self, x):
        s = x.shape[1:]
        center = (s_i // 2 for s_i in s)
        crop = (
            (self._crop_size // 2 for _ in s)
            if isinstance(self._crop_size, int)
            else (c_i // 2 for c_i in self._crop_size)
        )
        crop_coordinates = (slice(None),) + tuple(
            slice(c_i - cp_i, c_i + cp_i) for c_i, cp_i in zip(center, crop)
        )
        return x[crop_coordinates]


class RandomCrop(ImageLabelTransform):
    def __init__(
        self,
        size: list[int],
        generator=np.random.default_rng(),
    ) -> None:
        super().__init__()
        self._crop_size: list[int] = size
        self._generator = generator

    def transform_image_label(self, image, label):
        max_offsets = [
            dim_max - crop_dim_size
            for dim_max, crop_dim_size in zip(image.shape, self._crop_size)
        ]
        assert all(max_offset >= 0 for max_offset in max_offsets)
        crop_offsets = self._generator.integers([0, 0, 0], max_offsets)
        crop_slice = tuple(
            slice(crop_offset, crop_offset + crop_dim_size)
            for crop_offset, crop_dim_size in zip(crop_offsets, self._crop_size)
        )
        image_cropped = image[crop_slice]
        label_cropped = label[crop_slice]
        return image_cropped, label_cropped


class RandomCropJAX(ImageLabelTransform):
    def __init__(
        self,
        size: list[int],
        generator=objax.random.DEFAULT_GENERATOR,
    ) -> None:
        super().__init__()
        self._crop_size: list[int] = [1] + size
        self._generator = generator

    def transform_image_label(self, image, label):
        max_offsets = [
            dim_max - crop_dim_size
            for dim_max, crop_dim_size in zip(image.shape, self._crop_size)
        ]
        assert all(max_offset >= 0 for max_offset in max_offsets)
        crop_offsets_gen = (
            objax.random.randint(
                (1,), low=0, high=max_offset, generator=self._generator
            )
            for max_offset in max_offsets
        )
        crop_offsets = jn.concatenate(list(crop_offsets_gen))
        x_cropped = jax.lax.dynamic_slice(image, crop_offsets, self._crop_size)
        y_cropped = jax.lax.dynamic_slice(label, crop_offsets, self._crop_size)
        return x_cropped, y_cropped
