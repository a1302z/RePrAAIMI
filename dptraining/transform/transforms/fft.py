import numpy as np
from jax import numpy as jn

from dptraining.transform.unary_image_and_label_transform import (
    UnaryImageAndLabelTransform,
)


class FFT(UnaryImageAndLabelTransform):
    def __init__(self, axes=(1, 2, 3)) -> None:
        super().__init__()
        self.axes = axes

    def unary_transform(self, x):
        return np.fft.fft2(np.fft.fftshift(x, axes=self.axes), axes=self.axes)


class JaxFFT(UnaryImageAndLabelTransform):
    def __init__(self, axes=(1, 2, 3)) -> None:
        super().__init__()
        self.axes = axes

    def unary_transform(self, x):
        return jn.fft.fft2(jn.fft.fftshift(x, axes=self.axes), axes=self.axes)


class IFFT(UnaryImageAndLabelTransform):
    def __init__(self, axes=(1, 2, 3)) -> None:
        super().__init__()
        self.axes = axes

    def unary_transform(self, x):
        return np.fft.ifft2(np.fft.ifftshift(x, axes=self.axes), axes=self.axes).real


class JaxIFFT(UnaryImageAndLabelTransform):
    def __init__(self, axes=(1, 2, 3)) -> None:
        super().__init__()
        self.axes = axes

    def unary_transform(self, x):
        return jn.fft.ifft2(jn.fft.ifftshift(x, axes=self.axes), axes=self.axes).real
