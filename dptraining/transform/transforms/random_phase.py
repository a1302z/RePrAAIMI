import numpy as np
from PIL import Image
from jax import numpy as jn

from dptraining.transform.unary_image_and_label_transform import (
    UnaryImageAndLabelTransform,
)


class AddRandomPhase(UnaryImageAndLabelTransform):
    def __init__(self, control_points=4) -> None:
        super().__init__()
        self._control_points = control_points

    def unary_transform(self, x):
        img_shape = (x.shape[-2], x.shape[-1])
        assert self._control_points <= min(img_shape)
        control_points_2d = np.float32(
            np.random.uniform(0, 1, (self._control_points, self._control_points))
        )
        bias_field_2d = np.array(
            Image.fromarray(control_points_2d, mode="L").resize(
                img_shape, resample=Image.BICUBIC
            ),
            dtype=x.dtype,
        )
        bias_field_2d /= np.max(bias_field_2d)
        bias_field_2d *= 2 * np.pi * 0.999
        bias_field_2d -= np.pi
        return np.maximum(x, 1e-6) * np.exp(1j * bias_field_2d)


class AddRandomPhaseJAX(UnaryImageAndLabelTransform):
    def __init__(self, control_points=4) -> None:
        super().__init__()
        self._control_points = control_points

    def unary_transform(self, x):
        img_shape = (x.shape[-2], x.shape[-1])
        assert self._control_points <= min(img_shape)
        control_points_2d = np.float32(
            np.random.uniform(0, 1, (self._control_points, self._control_points))
        )
        bias_field_2d = jn.array(
            Image.fromarray(control_points_2d, mode="L").resize(
                img_shape, resample=Image.BICUBIC
            ),
            dtype=x.dtype,
        )
        bias_field_2d /= jn.max(bias_field_2d)
        bias_field_2d *= 2 * jn.pi * 0.999
        bias_field_2d -= jn.pi
        return jn.maximum(x, 1e-6) * jn.exp(1j * bias_field_2d)
