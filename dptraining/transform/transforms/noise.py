import numpy as np

from dptraining.transform.unary_image_and_label_transform import (
    UnaryImageAndLabelTransform,
)


class GaussianNoise(UnaryImageAndLabelTransform):
    def __init__(self, std=0.01) -> None:
        super().__init__()
        self._std = std

    def unary_transform(self, x):
        return x + np.random.randn(*x.shape) * self._std
