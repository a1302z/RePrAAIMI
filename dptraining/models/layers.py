from functools import partial
from typing import Callable, Union, Tuple, Optional
from objax import Module, nn
from objax.typing import JaxArray
from objax.functional import flatten
from dptraining.models.complex import ComplexGroupNormWhitening

from jax import lax, numpy as jn


def is_groupnorm(instance):
    return issubclass(instance, (nn.GroupNorm2D, ComplexGroupNormWhitening))


class Flatten(Module):
    def __call__(self, x: JaxArray) -> JaxArray:  # pylint:disable=arguments-differ
        return flatten(x)


class ConvWS2D(nn.Conv2D):
    def __call__(self, x: JaxArray) -> JaxArray:
        self.w.assign(self.w.value - self.w.value.mean(axis=(0, 1, 2), keepdims=True))
        self.w.assign(self.w.value / self.w.value.std(axis=(0, 1, 2), keepdims=True))
        return super().__call__(x)


class ConvCentering2D(nn.Conv2D):
    def __call__(self, x: JaxArray) -> JaxArray:
        weight = self.w - jn.mean(self.w.value, axis=(0, 1, 2), keepdims=True)
        # return super().__call__(x)
        output = lax.conv_general_dilated(
            x,
            weight,
            self.strides,
            self.padding,
            rhs_dilation=self.dilations,
            feature_group_count=self.groups,
            dimension_numbers=("NCHW", "HWIO", "NCHW"),
        )
        return output


class AdaptivePooling(Module):
    def __init__(
        self,
        pool_func: Callable,
        output_size: Union[int, Tuple[int, int]],
        stride: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.pool_func = pool_func
        if isinstance(output_size, int):
            self.output_size_x = output_size
            self.output_size_y = output_size
        elif isinstance(output_size, tuple):
            self.output_size_x, self.output_size_y = output_size
        else:
            raise ValueError("output size must be either int or tuple of ints")
        self.stride = stride

    @staticmethod
    def _calc_kernel(inpt_size, outpt_size, stride) -> int:
        return inpt_size - (outpt_size - 1) * stride

    def __call__(self, x):
        _, _, size_x, size_y = x.shape
        if self.stride is None:
            stride = (size_x // self.output_size_x, size_y // self.output_size_y)
        else:
            stride = (self.stride, self.stride)
        k_x = AdaptivePooling._calc_kernel(size_x, self.output_size_x, stride[0])
        k_y = AdaptivePooling._calc_kernel(size_y, self.output_size_y, stride[1])
        return partial(self.pool_func, size=(k_x, k_y), strides=stride)(x)
