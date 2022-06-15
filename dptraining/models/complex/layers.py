from objax.nn import Conv2D, Linear
from objax import Module
from objax.typing import JaxArray, ConvPaddingInt
from objax.constants import ConvPadding
from objax.nn.init import kaiming_normal, xavier_normal
from typing import Callable
from jax import numpy as np
from jax import vmap
from functools import partial
from jax.lax import conv_general_dilated


# Helper
def conjugate_apply(f_real, f_imag, inp):
    return (f_real(inp.real) - f_imag(inp.imag)) + 1j * (
        f_real(inp.imag) + f_imag(inp.real)
    )


class ComplexConv2D(Module):
    def __init__(
        self,
        nin: int,
        nout: int,
        k: tuple[int, int] | int,
        strides: tuple[int, int] | int = 1,
        dilations: tuple[int, int] | int = 1,
        groups: int = 1,
        padding: ConvPaddingInt | ConvPadding | str = ConvPadding.SAME,
        w_init: Callable = kaiming_normal,
        use_bias=False,
    ) -> None:
        super().__init__()
        if use_bias:
            raise ValueError("Bias is not supported.")
        self.convr = Conv2D(
            nin=nin,
            nout=nout,
            k=k,
            strides=strides,
            dilations=dilations,
            groups=groups,
            padding=padding,
            use_bias=False,
            w_init=w_init,
        )
        self.convi = Conv2D(
            nin=nin,
            nout=nout,
            k=k,
            strides=strides,
            dilations=dilations,
            groups=groups,
            padding=padding,
            use_bias=False,
            w_init=w_init,
        )

    def __call__(self, x: JaxArray) -> JaxArray:
        return conjugate_apply(self.convr, self.convi, x)


def complex_ws(w_real: JaxArray, w_imag: JaxArray) -> tuple[JaxArray, ...]:
    # each weight has shape H,W,I,O
    # permute to O,I,H,W, then stack to O,2,I,H,W
    stacked = np.stack(
        [w_real.transpose(3, 2, 0, 1), w_imag.transpose(3, 2, 0, 1)], axis=1
    )
    # subtract mean for I,H,W axes
    centered = stacked - stacked.mean(axis=(2, 3, 4), keepdims=True)
    centered = centered.reshape(centered.shape[0], 2, -1)
    # vmap the following computations over the O axis:
    def whitening_matrix(centered):
        # calculate covariance between real and imag
        sigma = np.cov(centered)
        # Compute inverse square root of covariance matrix
        u_mat, lmbda, _ = np.linalg.svd(sigma, full_matrices=False)
        # compute whitening matrix
        w_mat = np.matmul(
            u_mat, np.matmul(np.diag(1.0 / np.sqrt(lmbda + 1e-5)), u_mat.T)
        )
        # multiply centered weights with whitening matrix
        return np.matmul(w_mat, centered)

    whitened = vmap(whitening_matrix)(centered)
    res_r, res_i = whitened[:, 0, :], whitened[:, 1, :]
    # reshape back to original shape
    return res_r.transpose(1, 0).reshape(w_real.shape), res_i.transpose(1, 0).reshape(
        w_imag.shape
    )


class ComplexWSConv2D(ComplexConv2D):
    def __call__(self, x: JaxArray) -> JaxArray:
        weight_r, weight_i = complex_ws(self.convr.w.value, self.convi.w.value)
        return conjugate_apply(
            partial(
                conv_general_dilated,
                rhs=weight_r,
                window_strides=self.convr.strides,
                padding=self.convr.padding,
                rhs_dilation=self.convr.dilations,
                feature_group_count=self.convr.groups,
                dimension_numbers=("NCHW", "HWIO", "NCHW"),
            ),
            partial(
                conv_general_dilated,
                rhs=weight_i,
                window_strides=self.convi.strides,
                padding=self.convi.padding,
                rhs_dilation=self.convi.dilations,
                feature_group_count=self.convi.groups,
                dimension_numbers=("NCHW", "HWIO", "NCHW"),
            ),
            x,
        )


class ComplexLinear(Module):
    def __init__(
        self,
        nin: int,
        nout: int,
        use_bias: bool = True,
        w_init: Callable = xavier_normal,
    ) -> None:
        super().__init__()
        self.linr = Linear(nin=nin, nout=nout, use_bias=use_bias, w_init=w_init)
        self.lini = Linear(nin=nin, nout=nout, use_bias=use_bias, w_init=w_init)

    def __call__(self, x: JaxArray) -> JaxArray:
        return conjugate_apply(self.linr, self.lini, x)
