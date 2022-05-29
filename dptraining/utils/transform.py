from abc import ABC, abstractmethod
from typing import Optional
import objax, jax, jax.numpy as jn


class Transform(ABC):
    @abstractmethod
    def __call__(self, x):
        pass


class PILToJAXNumpy(Transform):
    def __call__(self, x):
        x = jn.array(x, dtype=jn.float32) / 255.0
        return x


class NormalizeJAX(Transform):
    def __init__(self, mean, std) -> None:
        super().__init__()
        self._mean, self._std = jn.array(mean), jn.array(std)

    def __call__(self, x):
        x = (x - self._mean) / self._std
        return x


class RandomVerticalFlipsJax(Transform):
    def __init__(self, flip_prob=0.5) -> None:
        super().__init__()
        self._flip_prob = flip_prob

    def __call__(self, x):
        return jax.lax.cond(
            objax.random.uniform(()) > self._flip_prob,
            lambda t: t,
            lambda t: t[:, :, ::-1],
            operand=x,
        )


class RandomHorizontalFlipsJax(Transform):
    def __init__(self, flip_prob=0.5) -> None:
        super().__init__()
        self._flip_prob = flip_prob

    def __call__(self, x):
        return jax.lax.cond(
            objax.random.uniform(()) > self._flip_prob,
            lambda t: t,
            lambda t: t[:, ::-1, :],
            operand=x,
        )


class RandomImageShiftsJax(Transform):
    def __init__(self, img_shape: Optional[tuple] = None, max_shift=4) -> None:
        super().__init__()
        self._img_shape = img_shape
        self._max_shift = max_shift

    def __call__(self, x):
        if self._img_shape is None:
            img_shape = x.shape[1:]
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
