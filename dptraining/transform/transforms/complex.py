from jax import numpy as jn

from dptraining.transform.unary_image_and_label_transform import (
    UnaryImageAndLabelTransform,
)


class MakeComplexOnlyReal(UnaryImageAndLabelTransform):
    def unary_transform(self, x):
        return x + 1j * jn.zeros_like(x)


class MakeComplexRealAndImaginary(UnaryImageAndLabelTransform):
    def unary_transform(self, x):
        return x + 1j * x
