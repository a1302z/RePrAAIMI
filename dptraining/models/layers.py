from functools import partial
from typing import Callable, Union, Tuple, Optional
from objax import Module, nn, TrainVar, util
from objax.typing import JaxArray, ConvPaddingInt
from objax.constants import ConvPadding
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
        self.w.assign(
            self.w.value - jn.mean(self.w.value, axis=(0, 1, 2), keepdims=True)
        )
        return super().__call__(x)
        # output = lax.conv_general_dilated(
        #     x,
        #     weight,
        #     self.strides,
        #     self.padding,
        #     rhs_dilation=self.dilations,
        #     feature_group_count=self.groups,
        #     dimension_numbers=("NCHW", "HWIO", "NCHW"),
        # )
        # return output


class ConvWSTranspose2D(nn.ConvTranspose2D):
    def __call__(self, x: JaxArray) -> JaxArray:
        self.w.assign(self.w.value - self.w.value.mean(axis=(0, 1, 2), keepdims=True))
        self.w.assign(self.w.value / self.w.value.std(axis=(0, 1, 2), keepdims=True))
        return super().__call__(x)


class ConvCenteringTranspose2D(nn.ConvTranspose2D):
    def __call__(self, x: JaxArray) -> JaxArray:
        self.w.assign(self.w - jn.mean(self.w.value, axis=(0, 1, 2), keepdims=True))
        return super().__call__(x)


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


class Conv3D(Module):
    """Applies a 2D convolution on a 4D-input batch of shape (N,C,H,W,D)."""

    def __init__(
        self,
        nin: int,
        nout: int,
        k: Union[Tuple[int, int, int], int],
        strides: Union[Tuple[int, int, int], int] = 1,
        dilations: Union[Tuple[int, int, int], int] = 1,
        groups: int = 1,
        padding: Union[ConvPadding, str, ConvPaddingInt] = ConvPadding.SAME,
        use_bias: bool = True,
        w_init: Callable = nn.init.kaiming_normal,
    ):
        """Creates a Conv3D module instance.

        Args:
            nin: number of channels of the input tensor.
            nout: number of channels of the output tensor.
            k: size of the convolution kernel, either tuple (height, width, depth) or single number if they're the same.
            strides: convolution strides, either tuple (stride_y, stride_x, stride_z) or single number if they're the same.
            dilations: spacing between kernel points (also known as astrous convolution),
                       either tuple (dilation_y, dilation_x, dilation_z) or single number if they're the same.
            groups: number of input and output channels group. When groups > 1 convolution operation is applied
                    individually for each group. nin and nout must both be divisible by groups.
            padding: padding of the input tensor, either Padding.SAME, Padding.VALID or numerical values.
            use_bias: if True then convolution will have bias term.
            w_init: initializer for convolution kernel (a function that takes in a HWDIO shape and returns a 4D matrix).
        """
        super().__init__()
        assert nin % groups == 0, "nin should be divisible by groups"
        assert nout % groups == 0, "nout should be divisible by groups"
        self.b = TrainVar(jn.zeros((nout, 1, 1, 1))) if use_bias else None
        self.w = TrainVar(w_init((*util.to_tuple(k, 3), nin // groups, nout)))  # HWIO
        self.padding = util.to_padding(padding, 3)
        self.strides = util.to_tuple(strides, 3)
        self.dilations = util.to_tuple(dilations, 3)
        self.groups = groups
        self.w_init = w_init

    def __call__(self, x: JaxArray) -> JaxArray:
        """Returns the results of applying the convolution to input x."""
        nin = self.w.value.shape[3] * self.groups
        assert x.shape[1] == nin, (
            f"Attempting to convolve an input with {x.shape[1]} input channels "
            f"when the convolution expects {nin} channels. For reference, "
            f"self.w.value.shape={self.w.value.shape} and x.shape={x.shape}."
        )
        y = lax.conv_general_dilated(
            x,
            self.w.value,
            self.strides,
            self.padding,
            rhs_dilation=self.dilations,
            feature_group_count=self.groups,
            dimension_numbers=("NCHWD", "HWDIO", "NCHWD"),
        )
        if self.b is not None:
            y += self.b.value
        return y

    def __repr__(self):
        args = dict(
            nin=self.w.value.shape[3] * self.groups,
            nout=self.w.value.shape[4],
            k=self.w.value.shape[:3],
            strides=self.strides,
            dilations=self.dilations,
            groups=self.groups,
            padding=self.padding,
            use_bias=self.b is not None,
        )
        args = ", ".join(f"{k}={repr(v)}" for k, v in args.items())
        return (
            f"{util.class_name(self)}({args}, w_init={util.repr_function(self.w_init)})"
        )


class ConvTranspose3D(Conv3D):
    """Applies a 2D transposed convolution on a 4D-input batch of shape (N,C,H,W,D).

    This module can be seen as a transformation going in the opposite direction of a normal convolution, i.e.,
    from something that has the shape of the output of some convolution to something that has the shape of its input
    while maintaining a connectivity pattern that is compatible with said convolution.
    Note that ConvTranspose2D is consistent with
    `Conv2DTranspose <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2DTranspose>`_,
    of Tensorflow but is not consistent with
    `ConvTranspose2D <https://pytorch.org/docs/master/generated/torch.nn.ConvTranspose2d.html>`_
    of PyTorch due to kernel transpose and padding.
    """

    def __init__(
        self,
        nin: int,
        nout: int,
        k: Union[Tuple[int, int, int], int],
        strides: Union[Tuple[int, int, int], int] = 1,
        dilations: Union[Tuple[int, int, int], int] = 1,
        padding: Union[ConvPadding, str, ConvPaddingInt] = ConvPadding.SAME,
        use_bias: bool = True,
        w_init: Callable = nn.init.kaiming_normal,
    ):
        """Creates a ConvTranspose3D module instance.

        Args:
            nin: number of channels of the input tensor.
            nout: number of channels of the output tensor.
            k: size of the convolution kernel, either tuple (height, width, depth) or single number if they're the same.
            strides: convolution strides, either tuple (stride_y, stride_x, stride_z) or single number if they're the same.
            dilations: spacing between kernel points (also known as astrous convolution),
                       either tuple (dilation_y, dilation_x, dilation_z) or single number if they're the same.
            padding: padding of the input tensor, either Padding.SAME, Padding.VALID or numerical values.
            use_bias: if True then convolution will have bias term.
            w_init: initializer for convolution kernel (a function that takes in a HWDIO shape and returns a 4D matrix).
        """
        super().__init__(
            nin=nout,
            nout=nin,
            k=k,
            strides=strides,
            dilations=dilations,
            padding=padding,
            use_bias=False,
            w_init=w_init,
        )
        self.b = TrainVar(jn.zeros((nout, 1, 1, 1))) if use_bias else None

    def __call__(self, x: JaxArray) -> JaxArray:
        """Returns the results of applying the transposed convolution to input x."""
        y = lax.conv_transpose(
            x,
            self.w.value,
            self.strides,
            self.padding,
            rhs_dilation=self.dilations,
            dimension_numbers=("NCHWD", "HWDIO", "NCHWD"),
            transpose_kernel=True,
        )
        if self.b is not None:
            y += self.b.value
        return y

    def __repr__(self):
        args = dict(
            nin=self.w.value.shape[4],
            nout=self.w.value.shape[3],
            k=self.w.value.shape[:3],
            strides=self.strides,
            dilations=self.dilations,
            padding=self.padding,
            use_bias=self.b is not None,
        )
        args = ", ".join(f"{k}={repr(v)}" for k, v in args.items())
        return (
            f"{util.class_name(self)}({args}, w_init={util.repr_function(self.w_init)})"
        )


class ConvWS3D(Conv3D):
    def __call__(self, x: JaxArray) -> JaxArray:
        self.w.assign(
            self.w.value - self.w.value.mean(axis=(0, 1, 2, 3), keepdims=True)
        )
        self.w.assign(self.w.value / self.w.value.std(axis=(0, 1, 2, 3), keepdims=True))
        return super().__call__(x)


class ConvCentering3D(Conv3D):
    def __call__(self, x: JaxArray) -> JaxArray:
        self.w.assign(self.w - jn.mean(self.w.value, axis=(0, 1, 2, 3), keepdims=True))
        return super().__call__(x)


class ConvWSTranspose3D(ConvTranspose3D):
    def __call__(self, x: JaxArray) -> JaxArray:
        self.w.assign(
            self.w.value - self.w.value.mean(axis=(0, 1, 2, 3), keepdims=True)
        )
        self.w.assign(self.w.value / self.w.value.std(axis=(0, 1, 2, 3), keepdims=True))
        return super().__call__(x)


class ConvCenteringTranspose3D(ConvTranspose3D):
    def __call__(self, x: JaxArray) -> JaxArray:
        self.w.assign(self.w - jn.mean(self.w.value, axis=(0, 1, 2, 3), keepdims=True))
        return super().__call__(x)
