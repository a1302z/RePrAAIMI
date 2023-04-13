from typing import Optional

import jax
import objax
from jax import numpy as jn

from dptraining.transform.unary_image_and_label_transform import (
    UnaryImageAndLabelTransform,
)


class RandomImageShiftsJax(UnaryImageAndLabelTransform):
    def __init__(
        self,
        img_shape: Optional[tuple] = None,
        max_shift=4,
    ) -> None:
        super().__init__()
        self._img_shape = img_shape
        self._max_shift = max_shift

    def unary_transform(self, x):
        if self._img_shape is None:
            img_shape = x.shape
        else:
            img_shape = self._img_shape
        x_pad = jn.pad(
            x,
            [
                [0, 0],
                [self._max_shift, self._max_shift],
                [self._max_shift, self._max_shift],
            ],
            "reflect",
        )
        offset = objax.random.randint((2,), 0, self._max_shift)
        return jax.lax.dynamic_slice(x_pad, (0, offset[0], offset[1]), img_shape)


class RandomImageShiftsJaxBatch(UnaryImageAndLabelTransform):
    def __init__(
        self,
        img_shape: Optional[tuple] = None,
        max_shift=4,
    ) -> None:
        super().__init__()
        self._img_shape = img_shape
        self._max_shift = max_shift

    def unary_transform(self, x):
        if self._img_shape is None:
            img_shape = x.shape
        else:
            img_shape = self._img_shape
        x_pad = jn.pad(
            x,
            [
                [0, 0],
                [0, 0],
                [self._max_shift, self._max_shift],
                [self._max_shift, self._max_shift],
            ],
            "reflect",
        )
        offset = objax.random.randint((2,), 0, self._max_shift)
        return jax.lax.dynamic_slice(x_pad, (0, 0, offset[0], offset[1]), img_shape)
