import numpy as np
from jax import numpy as jn

from dptraining.transform.unary_image_and_label_transform import (
    UnaryImageAndLabelTransform,
)


class PILToNumpy(UnaryImageAndLabelTransform):
    def unary_transform(self, x):
        return np.array(x, dtype=np.float32) / 255.0


class PILToJAXNumpy(UnaryImageAndLabelTransform):
    def unary_transform(self, x):
        x = jn.array(x, dtype=jn.float32) / 255.0
        return x
