from typing import Union, Tuple, Callable
from jax import numpy as jn, lax

from objax import util
from objax.constants import ConvPadding
from objax.module import Module
from objax.nn.init import kaiming_normal, xavier_normal
from objax.typing import JaxArray, ConvPaddingInt
from objax.util import class_name
from objax.variable import TrainVar


class ComplexConv2D(Module):
    """Applies a 2D convolution on a 4D-input batch of shape (N,C,H,W)."""

    def __init__(  # pylint:disable=too-many-arguments
        self,
        nin: int,
        nout: int,
        k: Union[Tuple[int, int], int],
        strides: Union[Tuple[int, int], int] = 1,
        dilations: Union[Tuple[int, int], int] = 1,
        groups: int = 1,
        padding: Union[ConvPadding, str, ConvPaddingInt] = ConvPadding.SAME,
        use_bias: bool = True,
        w_init: Callable = kaiming_normal,
    ):
        """Creates a Conv2D module instance.
        Args:
            nin: number of channels of the input tensor.
            nout: number of channels of the output tensor.
            k: size of the convolution kernel, either tuple (height, width)
             or single number if they're the same.
            strides: convolution strides, either tuple (stride_y, stride_x)
             or single number if they're the same.
            dilations: spacing between kernel points (also known as astrous convolution),
                       either tuple (dilation_y, dilation_x) or single number
                        if they're the same.
            groups: number of input and output channels group. When groups > 1
             convolution operation is applied
                    individually for each group. nin and nout must both be
                    divisible by groups.
            padding: padding of the input tensor, either Padding.SAME,
            Padding.VALID or numerical values.
            use_bias: if True then convolution will have bias term.
            w_init: initializer for convolution kernel (a function that
            takes in a HWIO shape and returns a 4D matrix).
        """
        super().__init__()
        assert nin % groups == 0, "nin should be divisible by groups"
        assert nout % groups == 0, "nout should be divisible by groups"

        # complex zeros are fine
        self.bias = (
            TrainVar(jn.zeros((nout, 1, 1)).astype(jn.complex64)) if use_bias else None
        )

        # initialise real and complex weight parts separately
        w_real = w_init((*util.to_tuple(k, 2), nin // groups, nout))
        w_imag = w_init((*util.to_tuple(k, 2), nin // groups, nout))
        self.weights = TrainVar(w_real + 1j * w_imag)  # HWIO

        self.padding = util.to_padding(padding, 2)
        self.strides = util.to_tuple(strides, 2)
        self.dilations = util.to_tuple(dilations, 2)
        self.groups = groups
        self.w_init = w_init

    def __call__(self, x: JaxArray) -> JaxArray:
        """Returns the results of applying the convolution to input x."""
        nin = self.weights.value.shape[2] * self.groups
        assert x.shape[1] == nin, (
            f"Attempting to convolve an input with {x.shape[1]} input channels "
            f"when the convolution expects {nin} channels. For reference, "
            f"self.w.value.shape={self.weights.value.shape} and x.shape={x.shape}."
        )
        out = lax.conv_general_dilated(
            x,
            self.weights.value,
            self.strides,
            self.padding,
            rhs_dilation=self.dilations,
            feature_group_count=self.groups,
            dimension_numbers=("NCHW", "HWIO", "NCHW"),
        )
        if self.bias is not None:
            out += self.bias.value
        return out

    def __repr__(self):
        args = dict(
            nin=self.weights.value.shape[2] * self.groups,
            nout=self.weights.value.shape[3],
            k=self.weights.value.shape[:2],
            strides=self.strides,
            dilations=self.dilations,
            groups=self.groups,
            padding=self.padding,
            use_bias=self.bias is not None,
        )
        args = ", ".join(f"{k}={repr(v)}" for k, v in args.items())
        return f"{class_name(self)}({args}, w_init={util.repr_function(self.w_init)})"


class ComplexWSConv2D(ComplexConv2D):
    def __call__(self, x: JaxArray) -> JaxArray:
        """Returns the results of applying the convolution to input x."""
        nin = self.weights.value.shape[2] * self.groups
        assert x.shape[1] == nin, (
            f"Attempting to convolve an input with {x.shape[1]} input channels "
            f"when the convolution expects {nin} channels. For reference, "
            f"self.w.value.shape={self.weights.value.shape} and x.shape={x.shape}."
        )
        mean = jn.mean(self.weights.value, axis=(1, 2, 3), keepdims=True)
        std = jn.std(self.weights.value, axis=(1, 2, 3), keepdims=True)
        self.weights.assign((self.weights.value - mean) / (std + 1e-5))
        out = lax.conv_general_dilated(
            x,
            self.weights.value,
            self.strides,
            self.padding,
            rhs_dilation=self.dilations,
            feature_group_count=self.groups,
            dimension_numbers=("NCHW", "HWIO", "NCHW"),
        )
        if self.bias is not None:
            out += self.bias.value
        return out


class ComplexLinear(Module):
    """Applies a linear transformation on an input batch."""

    def __init__(
        self,
        nin: int,
        nout: int,
        use_bias: bool = True,
        w_init: Callable = xavier_normal,
    ):
        """Creates a Linear module instance.
        Args:
            nin: number of channels of the input tensor.
            nout: number of channels of the output tensor.
            use_bias: if True then linear layer will have bias term.
            w_init: weight initializer for linear layer (a function
            that takes in a IO shape and returns a 2D matrix).
        """
        super().__init__()
        self.w_init = w_init
        # complex zeros are fine
        self.bias = TrainVar(jn.zeros(nout) + 1j * jn.zeros(nout)) if use_bias else None
        # Initialise real and complex part separately
        w_real = w_init((nin, nout))
        w_imag = w_init((nin, nout))
        self.weights = TrainVar(w_real + 1j * w_imag)

    def __call__(self, x: JaxArray) -> JaxArray:
        """Returns the results of applying the linear transformation to input x."""
        out = jn.dot(x, self.weights.value)
        if self.bias is not None:
            out += self.bias.value
        return out

    def __repr__(self):
        weight_shape = self.weights.value.shape
        args = (
            f"nin={weight_shape[0]}, nout={weight_shape[1]}, use_bias={self.bias is not None},"
            f" w_init={util.repr_function(self.w_init)}"
        )
        return f"{class_name(self)}({args})"
