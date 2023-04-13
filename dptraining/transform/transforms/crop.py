from typing import Union

from dptraining.transform.unary_image_and_label_transform import (
    UnaryImageAndLabelTransform,
)


class CenterCrop(UnaryImageAndLabelTransform):
    def __init__(self, crop_size: Union[tuple[int], int]) -> None:
        super().__init__()
        self._crop_size: Union[tuple[int], int] = crop_size

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
