from functools import partial
from typing import Callable, Union, Tuple, Optional
from objax import Module, nn, TrainVar, util, TrainRef
from objax.typing import JaxArray, ConvPaddingInt
from objax.constants import ConvPadding
from objax.functional import flatten
from dptraining.models.complex import ComplexGroupNormWhitening
from numpy import prod

from jax import lax, numpy as jn


def is_groupnorm(instance):
    return issubclass(instance, (nn.GroupNorm2D, ComplexGroupNormWhitening))


class Flatten(Module):
    def __call__(self, x: JaxArray) -> JaxArray:  # pylint:disable=arguments-differ
        return flatten(x)


class ConvWS2D(nn.Conv2D):
    def __init__(
        self,
        nin: int,
        nout: int,
        k: Union[Tuple[int, int], int],
        strides: Union[Tuple[int, int], int] = 1,
        dilations: Union[Tuple[int, int], int] = 1,
        groups: int = 1,
        padding: Union[ConvPadding, str, ConvPaddingInt] = ConvPadding.SAME,
        use_bias: bool = True,
        w_init: Callable = nn.init.kaiming_normal,
    ):
        assert (k > 1 if isinstance(k, int) else any((k_i > 1 for k_i in k))) or (
            nin // groups > 1
        ), (
            "Normalizing weights of a conv layer with k=1 and nin=1 is not defined"
            "Use normal conv instead"
        )
        super().__init__(
            nin, nout, k, strides, dilations, groups, padding, use_bias, w_init
        )

    def __call__(self, x: JaxArray) -> JaxArray:
        weight = self.w.value
        weight -= weight.mean(axis=(0, 1, 2), keepdims=True)
        weight /= weight.std(axis=(0, 1, 2), keepdims=True)
        output = lax.conv_general_dilated(
            x,
            weight,
            self.strides,
            self.padding,
            rhs_dilation=self.dilations,
            feature_group_count=self.groups,
            dimension_numbers=("NCHW", "HWIO", "NCHW"),
        )
        if self.b is not None:
            output += self.b.value
        return output


class ConvCentering2D(nn.Conv2D):
    def __init__(
        self,
        nin: int,
        nout: int,
        k: Union[Tuple[int, int], int],
        strides: Union[Tuple[int, int], int] = 1,
        dilations: Union[Tuple[int, int], int] = 1,
        groups: int = 1,
        padding: Union[ConvPadding, str, ConvPaddingInt] = ConvPadding.SAME,
        use_bias: bool = True,
        w_init: Callable = nn.init.kaiming_normal,
    ):
        assert (k > 1 if isinstance(k, int) else any((k_i > 1 for k_i in k))) or (
            nin // groups > 1
        ), (
            "Normalizing weights of a conv layer with k=1 and nin=1 is not defined"
            "Use normal conv instead"
        )
        super().__init__(
            nin, nout, k, strides, dilations, groups, padding, use_bias, w_init
        )

    def __call__(self, x: JaxArray) -> JaxArray:
        weight = self.w.value - jn.mean(self.w.value, axis=(0, 1, 2), keepdims=True)
        output = lax.conv_general_dilated(
            x,
            weight,
            self.strides,
            self.padding,
            rhs_dilation=self.dilations,
            feature_group_count=self.groups,
            dimension_numbers=("NCHW", "HWIO", "NCHW"),
        )
        if self.b is not None:
            output += self.b.value
        return output


class ConvWSTranspose2D(nn.ConvTranspose2D):
    def __init__(
        self,
        nin: int,
        nout: int,
        k: Union[Tuple[int, int], int],
        strides: Union[Tuple[int, int], int] = 1,
        dilations: Union[Tuple[int, int], int] = 1,
        padding: Union[ConvPadding, str, ConvPaddingInt] = ConvPadding.SAME,
        use_bias: bool = True,
        w_init: Callable = nn.init.kaiming_normal,
    ):
        assert (k > 1 if isinstance(k, int) else any((k_i > 1 for k_i in k))) or (
            nin > 1
        ), (
            "Normalizing weights of a conv layer with k=1 and nin=1 is not defined"
            "Use normal conv instead"
        )
        super().__init__(nin, nout, k, strides, dilations, padding, use_bias, w_init)

    def __call__(self, x: JaxArray) -> JaxArray:
        weight = self.w.value
        weight -= weight.mean(axis=(0, 1, 2), keepdims=True)
        weight /= weight.std(axis=(0, 1, 2), keepdims=True)
        y = lax.conv_transpose(
            x,
            weight,
            self.strides,
            self.padding,
            rhs_dilation=self.dilations,
            dimension_numbers=("NCHW", "HWIO", "NCHW"),
            transpose_kernel=True,
        )
        if self.b is not None:
            y += self.b.value
        return y


class ConvCenteringTranspose2D(nn.ConvTranspose2D):
    def __init__(
        self,
        nin: int,
        nout: int,
        k: Union[Tuple[int, int], int],
        strides: Union[Tuple[int, int], int] = 1,
        dilations: Union[Tuple[int, int], int] = 1,
        padding: Union[ConvPadding, str, ConvPaddingInt] = ConvPadding.SAME,
        use_bias: bool = True,
        w_init: Callable = nn.init.kaiming_normal,
    ):
        assert (k > 1 if isinstance(k, int) else any((k_i > 1 for k_i in k))) or (
            nin > 1
        ), (
            "Normalizing weights of a conv layer with k=1 and nin=1 is not defined"
            "Use normal conv instead"
        )
        super().__init__(nin, nout, k, strides, dilations, padding, use_bias, w_init)

    def __call__(self, x: JaxArray) -> JaxArray:
        weight = self.w.value
        weight -= weight.mean(axis=(0, 1, 2), keepdims=True)
        y = lax.conv_transpose(
            x,
            weight,
            self.strides,
            self.padding,
            rhs_dilation=self.dilations,
            dimension_numbers=("NCHW", "HWIO", "NCHW"),
            transpose_kernel=True,
        )
        if self.b is not None:
            y += self.b.value
        return y


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
    """Applies a 3D convolution on a 4D-input batch of shape (N,C,H,W,D)."""

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
            k: size of the convolution kernel, either tuple (height, width, depth) or single
                number if they're the same.
            strides: convolution strides, either tuple (stride_y, stride_x, stride_z) or single
                number if they're the same.
            dilations: spacing between kernel points (also known as astrous convolution),
                either tuple (dilation_y, dilation_x, dilation_z) or
                single number if they're the same.
            groups: number of input and output channels group. When groups > 1 convolution
                operation is applied individually for each group. nin and nout must both
                be divisible by groups.
            padding: padding of the input tensor, either Padding.SAME, Padding.VALID
                or numerical values.
            use_bias: if True then convolution will have bias term.
            w_init: initializer for convolution kernel (a function that takes in a
                HWDIO shape and returns a 4D matrix).
        """
        super().__init__()
        assert nin % groups == 0, "nin should be divisible by groups"
        assert nout % groups == 0, "nout should be divisible by groups"
        self.bias = TrainVar(jn.zeros((nout, 1, 1, 1))) if use_bias else None
        self.weight = TrainVar(
            w_init((*util.to_tuple(k, 3), nin // groups, nout))
        )  # HWIO
        self.padding = util.to_padding(padding, 3)
        self.strides = util.to_tuple(strides, 3)
        self.dilations = util.to_tuple(dilations, 3)
        self.groups = groups
        self.w_init = w_init

    def __call__(self, inpt: JaxArray) -> JaxArray:
        """Returns the results of applying the convolution to input x."""
        nin = self.weight.value.shape[3] * self.groups
        assert inpt.shape[1] == nin, (
            f"Attempting to convolve an input with {inpt.shape[1]} input channels "
            f"when the convolution expects {nin} channels. For reference, "
            f"self.w.value.shape={self.weight.value.shape} and x.shape={inpt.shape}."
        )
        out = lax.conv_general_dilated(
            inpt,
            self.weight.value,
            self.strides,
            self.padding,
            rhs_dilation=self.dilations,
            feature_group_count=self.groups,
            dimension_numbers=("NCHWD", "HWDIO", "NCHWD"),
        )
        if self.bias is not None:
            out += self.bias.value
        return out

    def __repr__(self):
        args = dict(
            nin=self.weight.value.shape[3] * self.groups,
            nout=self.weight.value.shape[4],
            k=self.weight.value.shape[:3],
            strides=self.strides,
            dilations=self.dilations,
            groups=self.groups,
            padding=self.padding,
            use_bias=self.bias is not None,
        )
        args = ", ".join(f"{k}={repr(v)}" for k, v in args.items())
        return (
            f"{util.class_name(self)}({args}, w_init={util.repr_function(self.w_init)})"
        )


class ConvTranspose3D(Conv3D):
    """Applies a 2D transposed convolution on a 4D-input batch of shape (N,C,H,W,D).

    This module can be seen as a transformation going in the opposite direction of a normal
    convolution, i.e., from something that has the shape of the output of some convolution
    to something that has the shape of its input while maintaining a connectivity pattern
    that is compatible with said convolution.
    Note that ConvTranspose2D is consistent with
    `Conv2DTranspose <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2DTranspose>`_
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
            k: size of the convolution kernel, either tuple (height, width, depth) or single
                number if they're the same.
            strides: convolution strides, either tuple (stride_y, stride_x, stride_z) or single
                number if they're the same.
            dilations: spacing between kernel points (also known as astrous convolution),
                       either tuple (dilation_y, dilation_x, dilation_z) or single number if
                       they're the same.
            padding: padding of the input tensor, either Padding.SAME, Padding.VALID
                or numerical values.
            use_bias: if True then convolution will have bias term.
            w_init: initializer for convolution kernel (a function that takes in a HWDIO shape and
                returns a 4D matrix).
        """
        assert (k > 1 if isinstance(k, int) else any((k_i > 1 for k_i in k))) or (
            nin > 1
        ), (
            "Normalizing weights of a conv layer with k=1 and nin=1 is not defined"
            "Use normal conv instead"
        )
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
        self.bias = TrainVar(jn.zeros((nout, 1, 1, 1))) if use_bias else None

    def __call__(self, inpt: JaxArray) -> JaxArray:
        """Returns the results of applying the transposed convolution to input x."""
        out = lax.conv_transpose(
            inpt,
            self.weight.value,
            self.strides,
            self.padding,
            rhs_dilation=self.dilations,
            dimension_numbers=("NCHWD", "HWDIO", "NCHWD"),
            transpose_kernel=True,
        )
        if self.bias is not None:
            out += self.bias.value
        return out

    def __repr__(self):
        args = dict(
            nin=self.weight.value.shape[4],
            nout=self.weight.value.shape[3],
            k=self.weight.value.shape[:3],
            strides=self.strides,
            dilations=self.dilations,
            padding=self.padding,
            use_bias=self.bias is not None,
        )
        args = ", ".join(f"{k}={repr(v)}" for k, v in args.items())
        return (
            f"{util.class_name(self)}({args}, w_init={util.repr_function(self.w_init)})"
        )


class ConvWS3D(Conv3D):
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
        assert (k > 1 if isinstance(k, int) else any((k_i > 1 for k_i in k))) or (
            nin // groups > 1
        ), (
            "Normalizing weights of a conv layer with k=1 and nin=1 is not defined"
            "Use normal conv instead"
        )
        super().__init__(
            nin, nout, k, strides, dilations, groups, padding, use_bias, w_init
        )

    def __call__(self, inpt: JaxArray) -> JaxArray:
        weight = self.weight.value
        weight -= weight.mean(axis=(0, 1, 2, 3), keepdims=True)
        weight /= weight.std(axis=(0, 1, 2, 3), keepdims=True)
        out = lax.conv_general_dilated(
            inpt,
            self.weight.value,
            self.strides,
            self.padding,
            rhs_dilation=self.dilations,
            feature_group_count=self.groups,
            dimension_numbers=("NCHWD", "HWDIO", "NCHWD"),
        )
        if self.bias is not None:
            out += self.bias.value
        return out


class ConvCentering3D(Conv3D):
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
        assert (k > 1 if isinstance(k, int) else any((k_i > 1 for k_i in k))) or (
            nin // groups > 1
        ), (
            "Normalizing weights of a conv layer with k=1 and nin=1 is not defined"
            "Use normal conv instead"
        )
        super().__init__(
            nin, nout, k, strides, dilations, groups, padding, use_bias, w_init
        )

    def __call__(self, inpt: JaxArray) -> JaxArray:
        weight = self.weight.value
        weight -= weight.mean(axis=(0, 1, 2, 3), keepdims=True)
        out = lax.conv_general_dilated(
            inpt,
            self.weight.value,
            self.strides,
            self.padding,
            rhs_dilation=self.dilations,
            feature_group_count=self.groups,
            dimension_numbers=("NCHWD", "HWDIO", "NCHWD"),
        )
        if self.bias is not None:
            out += self.bias.value
        return out


class ConvWSTranspose3D(ConvTranspose3D):
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
        assert (k > 1 if isinstance(k, int) else any((k_i > 1 for k_i in k))) or (
            nin > 1
        ), (
            "Normalizing weights of a conv layer with k=1 and nin=1 is not defined"
            "Use normal conv instead"
        )
        super().__init__(nin, nout, k, strides, dilations, padding, use_bias, w_init)

    def __call__(self, x: JaxArray) -> JaxArray:
        weight = self.weight.value
        weight -= weight.mean(axis=(0, 1, 2, 3), keepdims=True)
        weight /= weight.std(axis=(0, 1, 2, 3), keepdims=True)
        out = lax.conv_transpose(
            x,
            weight,
            self.strides,
            self.padding,
            rhs_dilation=self.dilations,
            dimension_numbers=("NCHWD", "HWDIO", "NCHWD"),
            transpose_kernel=True,
        )
        if self.bias is not None:
            out += self.bias.value
        return out


class ConvCenteringTranspose3D(ConvTranspose3D):
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
        assert (k > 1 if isinstance(k, int) else any((k_i > 1 for k_i in k))) or (
            nin > 1
        ), (
            "Normalizing weights of a conv layer with k=1 and nin=1 is not defined"
            "Use normal conv instead"
        )
        super().__init__(nin, nout, k, strides, dilations, padding, use_bias, w_init)

    def __call__(self, x: JaxArray) -> JaxArray:
        weight = self.weight.value
        weight -= weight.mean(axis=(0, 1, 2, 3), keepdims=True)
        out = lax.conv_transpose(
            x,
            weight,
            self.strides,
            self.padding,
            rhs_dilation=self.dilations,
            dimension_numbers=("NCHWD", "HWDIO", "NCHWD"),
            transpose_kernel=True,
        )
        if self.bias is not None:
            out += self.bias.value
        return out


class BatchNorm3D(nn.BatchNorm):
    """Applies a 3D batch normalization on a 5D-input batch of shape (N,C,H,W,D).

    The module follows the operation described in Algorithm 1 of
    `Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
    <https://arxiv.org/abs/1502.03167>`_.
    """

    def __init__(self, nin: int, momentum: float = 0.999, eps: float = 1e-6):
        """Creates a BatchNorm2D module instance.

        Args:
            nin: number of channels in the input example.
            momentum: value used to compute exponential moving average of batch statistics.
            eps: small value which is used for numerical stability.
        """
        super().__init__((1, nin, 1, 1, 1), (0, 2, 3, 4), momentum, eps)

    def __repr__(self):
        return (
            f"{util.class_name(self)}(nin={self.beta.value.shape[1]},"
            f" momentum={self.momentum}, eps={self.eps})"
        )


class GroupNorm3D(nn.GroupNorm):
    """Applies a 3D group normalization on a input batch of shape (N,C,H,W,D)."""

    def __init__(self, nin: int, groups: int = 32, eps: float = 1e-5):
        """Creates a GroupNorm3D module instance.

        Args:
            nin: number of input channels.
            groups: number of normalization groups.
            eps: small value which is used for numerical stability.
        """
        super().__init__(nin, rank=5, groups=groups, eps=eps)

    def __repr__(self):
        return f"{util.class_name(self)}(nin={self.nin}, groups={self.groups}, eps={self.eps})"


def max_pool_3d(
    inpt: JaxArray,
    size: Union[Tuple[int, int, int], int] = 2,
    strides: Optional[Union[Tuple[int, int, int], int]] = None,
    padding: Union[ConvPadding, str, ConvPaddingInt] = ConvPadding.VALID,
) -> JaxArray:
    """Applies max pooling using a square 2D filter.

    Args:
        x: input tensor of shape (N, C, H, W).
        size: size of pooling filter.
        strides: stride step, use size when stride is none (default).
        padding: padding of the input tensor, either Padding.SAME or Padding.VALID
        or numerical values.

    Returns:
        output tensor of shape (N, C, H, W).
    """
    size = util.to_tuple(size, 3)
    strides = util.to_tuple(strides, 3) if strides else size
    padding = util.to_padding(padding, 3)
    if isinstance(padding, tuple):
        padding = ((0, 0), (0, 0)) + padding
    return lax.reduce_window(
        inpt, -jn.inf, lax.max, (1, 1) + size, (1, 1) + strides, padding=padding
    )


def average_pool_3d(
    inpt: JaxArray,
    size: Union[Tuple[int, int, int], int] = 2,
    strides: Optional[Union[Tuple[int, int, int], int]] = None,
    padding: Union[ConvPadding, str, ConvPaddingInt] = ConvPadding.VALID,
) -> JaxArray:
    """Applies average pooling using a square 3D filter.

    Args:
        x: input tensor of shape (N, C, H, W, D).
        size: size of pooling filter.
        strides: stride step, use size when stride is none (default).
        padding: padding of the input tensor, either Padding.SAME or Padding.VALID
        or numerical values.

    Returns:
        output tensor of shape (N, C, H, W, D).
    """
    size = util.to_tuple(size, 3)
    strides = util.to_tuple(strides, 3) if strides else size
    padding = util.to_padding(padding, 3)
    if isinstance(padding, tuple):
        padding = ((0, 0), (0, 0)) + padding
    return lax.reduce_window(
        inpt, 0, lax.add, (1, 1) + size, (1, 1) + strides, padding=padding
    ) / prod(size)
