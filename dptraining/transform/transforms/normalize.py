import numpy as np
from jax import numpy as jn

from dptraining.transform.unary_image_and_label_transform import (
    UnaryImageAndLabelTransform,
)


class NormalizeNumpyBatch(UnaryImageAndLabelTransform):
    def __init__(self, mean, std) -> None:
        super().__init__()
        self._mean = np.array(mean).reshape(  # pylint:disable=too-many-function-args
            1, -1, 1, 1
        )
        self._std = np.array(std).reshape(  # pylint:disable=too-many-function-args
            1, -1, 1, 1
        )

    def unary_transform(self, x):
        x = (x - self._mean) / self._std
        return x


class NormalizeJAX(UnaryImageAndLabelTransform):
    def __init__(self, mean, std) -> None:
        super().__init__()
        self._mean, self._std = jn.array(
            mean
        ).reshape(  # pylint:disable=too-many-function-args
            -1, 1, 1
        ), jn.array(
            std
        ).reshape(  # pylint:disable=too-many-function-args
            -1, 1, 1
        )

    def unary_transform(self, x):
        x = (x - self._mean) / self._std
        return x


class NormalizeJAXBatch(UnaryImageAndLabelTransform):
    def __init__(self, mean, std) -> None:
        super().__init__()
        self._mean, self._std = jn.array(
            mean
        ).reshape(  # pylint:disable=too-many-function-args
            1, -1, 1, 1
        ), jn.array(
            std
        ).reshape(  # pylint:disable=too-many-function-args
            1, -1, 1, 1
        )

    def unary_transform(self, x):
        x = (x - self._mean) / self._std
        return x


class NormalizeNumpyImg(UnaryImageAndLabelTransform):
    def __init__(self, mean, std) -> None:
        super().__init__()
        self._mean = np.array(mean).reshape(  # pylint:disable=too-many-function-args
            -1, 1, 1
        )
        self._std = np.array(std).reshape(  # pylint:disable=too-many-function-args
            -1, 1, 1
        )

    def unary_transform(self, x):
        return (x - self._mean) / self._std
