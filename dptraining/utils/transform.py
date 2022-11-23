from abc import ABC, abstractmethod
from typing import Optional
import objax
import jax
import jax.numpy as jn
import numpy as np


class Transform(ABC):
    @abstractmethod
    def __call__(self, x):  # pylint:disable=invalid-name
        pass


class PILToNumpy(Transform):
    def __call__(self, x):
        x = np.array(x, dtype=np.float32) / 255.0
        return x


class NormalizeNumpyImg(Transform):
    def __init__(self, mean, std) -> None:
        super().__init__()
        self._mean = np.array(mean).reshape(  # pylint:disable=too-many-function-args
            -1, 1, 1
        )
        self._std = np.array(std).reshape(  # pylint:disable=too-many-function-args
            -1, 1, 1
        )

    def __call__(self, x):
        x = (x - self._mean) / self._std
        return x


class NormalizeNumpyBatch(Transform):
    def __init__(self, mean, std) -> None:
        super().__init__()
        self._mean = np.array(mean).reshape(  # pylint:disable=too-many-function-args
            1, -1, 1, 1
        )
        self._std = np.array(std).reshape(  # pylint:disable=too-many-function-args
            1, -1, 1, 1
        )

    def __call__(self, x):
        x = (x - self._mean) / self._std
        return x


class PILToJAXNumpy(Transform):
    def __call__(self, x):
        x = jn.array(x, dtype=jn.float32) / 255.0
        return x


class NormalizeJAX(Transform):
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

    def __call__(self, x):
        x = (x - self._mean) / self._std
        return x


class NormalizeJAXBatch(Transform):
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


class RandomVerticalFlipsJaxBatch(Transform):
    def __init__(self, flip_prob=0.5) -> None:
        super().__init__()
        self._flip_prob = flip_prob

    def __call__(self, x):
        return jax.lax.cond(
            objax.random.uniform(()) > self._flip_prob,
            lambda t: t,
            lambda t: t[:, :, :, ::-1],
            operand=x,
        )


class RandomHorizontalFlipsJaxBatch(Transform):
    def __init__(self, flip_prob=0.5) -> None:
        super().__init__()
        self._flip_prob = flip_prob

    def __call__(self, x):
        return jax.lax.cond(
            objax.random.uniform(()) > self._flip_prob,
            lambda t: t,
            lambda t: t[:, :, ::-1, :],
            operand=x,
        )


class RandomImageShiftsJax(Transform):
    def __init__(self, img_shape: Optional[tuple] = None, max_shift=4) -> None:
        super().__init__()
        self._img_shape = img_shape
        self._max_shift = max_shift

    def __call__(self, x):
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


class RandomImageShiftsJaxBatch(Transform):
    def __init__(self, img_shape: Optional[tuple] = None, max_shift=4) -> None:
        super().__init__()
        self._img_shape = img_shape
        self._max_shift = max_shift

    def __call__(self, x):
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


class MakeComplexOnlyReal(Transform):
    def __init__(self) -> None:
        pass

    def __call__(self, x):
        return x + 1j * jn.zeros_like(x)


class MakeComplexRealAndImaginary(Transform):
    def __init__(self) -> None:
        pass

    def __call__(self, x):
        return x + 1j * x


class TransposeNumpyBatchToCHW(Transform):
    def __init__(self) -> None:
        pass

    def __call__(self, data: np.array, labels=None):
        return data.transpose(0, 3, 1, 2)


class TransposeNumpyImgToCHW(Transform):
    def __init__(self) -> None:
        pass

    def __call__(self, data: np.array, labels=None):
        return data.transpose(2, 0, 1)


class FFT(Transform):
    def __init__(self, axes=(1, 2, 3)) -> None:
        super().__init__()
        self.axes = axes

    def __call__(self, x):
        return np.fft.fft2(np.fft.fftshift(x, axes=self.axes), axes=self.axes)


class JaxFFT(Transform):
    def __init__(self, axes=(1, 2, 3)) -> None:
        super().__init__()
        self.axes = axes

    def __call__(self, x):
        return jn.fft.fft2(jn.fft.fftshift(x, axes=self.axes), axes=self.axes)


class IFFT(Transform):
    def __init__(self, axes=(1, 2, 3)) -> None:
        super().__init__()
        self.axes = axes

    def __call__(self, x):
        return np.fft.ifft2(np.fft.ifftshift(x, axes=self.axes), axes=self.axes).real


class JaxIFFT(Transform):
    def __init__(self, axes=(1, 2, 3)) -> None:
        super().__init__()
        self.axes = axes

    def __call__(self, x):
        return jn.fft.ifft2(jn.fft.ifftshift(x, axes=self.axes), axes=self.axes).real


class GaussianNoise(Transform):
    def __init__(self, std=0.01) -> None:
        super().__init__()
        self._std = std

    def __call__(self, x):
        return x + np.random.randn(*x.shape) * self._std
