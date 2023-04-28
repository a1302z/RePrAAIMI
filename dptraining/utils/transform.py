from abc import ABC, abstractmethod
from typing import Optional, Union
import objax
import jax
import jax.numpy as jn
import numpy as np
from PIL import Image


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


class RandomZFlipsJax(Transform):
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


class CenterCrop(Transform):
    def __init__(self, crop_size: Union[tuple[int], int]) -> None:
        super().__init__()
        self._crop_size: Union[tuple[int], int] = crop_size

    def __call__(self, x):
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


class AddRandomPhase(Transform):
    def __init__(self, control_points=4) -> None:
        super().__init__()
        self._control_points = control_points

    def __call__(self, x):
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


class AddRandomPhaseJAX(Transform):
    def __init__(self, control_points=4) -> None:
        super().__init__()
        self._control_points = control_points

    def __call__(self, x):
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
