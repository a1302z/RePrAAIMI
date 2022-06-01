from typing import Union, Tuple, Optional
from objax.typing import JaxArray, ConvPaddingInt
from objax.constants import ConvPadding
from objax import Module

from objax.functional import max_pool_2d


class ConjugateMaxPool2D(Module):
    """Applies Max Pooling in a conjugate way"""

    def __init__(
        self,
        size: Union[Tuple[int, int], int] = 2,
        strides: Optional[Union[Tuple[int, int], int]] = None,
        padding: Union[ConvPadding, str, ConvPaddingInt] = ConvPadding.VALID,
    ) -> JaxArray:
        self.size = size
        self.strides = strides
        self.padding = padding

    def __call__(self, x: JaxArray) -> JaxArray:
        real = max_pool_2d(x.real, self.size, self.strides, self.padding) - max_pool_2d(
            x.imag, self.size, self.strides, self.padding
        )
        imag = max_pool_2d(x.real, self.size, self.strides, self.padding) + max_pool_2d(
            x.imag, self.size, self.strides, self.padding
        )
        return real + 1j * imag


class SeparableMaxPool2D(Module):
    """Applies Max Pooling separately to real and complex parts"""

    def __init__(
        self,
        size: Union[Tuple[int, int], int] = 2,
        strides: Optional[Union[Tuple[int, int], int]] = None,
        padding: Union[ConvPadding, str, ConvPaddingInt] = ConvPadding.VALID,
    ) -> JaxArray:
        self.size = size
        self.strides = strides
        self.padding = padding

    def __call__(self, x: JaxArray) -> JaxArray:
        return max_pool_2d(
            x.real, self.size, self.strides, self.padding
        ) + 1j * max_pool_2d(x.imag, self.size, self.strides, self.padding)
