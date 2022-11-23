from typing import Callable, Union, Tuple, Optional
from objax.typing import JaxArray, ConvPaddingInt
from objax.constants import ConvPadding
from objax import Module

from objax.functional import max_pool_2d  # , average_pool_2d


class ConjugatePool2D(Module):
    """Applies Max Pooling in a conjugate way"""

    def __init__(
        self,
        size: Union[Tuple[int, int], int] = 2,
        strides: Optional[Union[Tuple[int, int], int]] = None,
        padding: Union[ConvPadding, str, ConvPaddingInt] = ConvPadding.VALID,
        pool_func: Callable = max_pool_2d,
    ) -> JaxArray:
        self.size = size
        self.strides = strides
        self.padding = padding
        self.pool_func = pool_func

    def __call__(self, x: JaxArray) -> JaxArray:
        real = self.pool_func(
            x.real, self.size, self.strides, self.padding
        ) - self.pool_func(x.imag, self.size, self.strides, self.padding)
        imag = self.pool_func(
            x.real, self.size, self.strides, self.padding
        ) + self.pool_func(x.imag, self.size, self.strides, self.padding)
        return real + 1j * imag


class SeparablePool2D(Module):
    """Applies Max Pooling separately to real and complex parts"""

    def __init__(
        self,
        size: Union[Tuple[int, int], int] = 2,
        strides: Optional[Union[Tuple[int, int], int]] = None,
        padding: Union[ConvPadding, str, ConvPaddingInt] = ConvPadding.VALID,
        pool_func: Callable = max_pool_2d,
    ) -> JaxArray:
        self.size = size
        self.strides = strides
        self.padding = padding
        self.pool_func = pool_func

    def __call__(self, x: JaxArray) -> JaxArray:
        return self.pool_func(
            x.real, self.size, self.strides, self.padding
        ) + 1j * self.pool_func(x.imag, self.size, self.strides, self.padding)
