from functools import partial
from typing import Callable
from warnings import warn

from jax import numpy as jnp
from jax import vmap
from jax.lax import conv_general_dilated
from objax import Module
from objax.constants import ConvPadding
from objax.nn import Conv2D, Linear
from objax.nn.init import kaiming_normal, xavier_normal
from objax.typing import ConvPaddingInt, JaxArray
from objax.util import class_name
from objax.functional import rsqrt

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
            warn("Bias is not supported.")
            use_bias = False
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

    def __repr__(self):
        args = dict(
            nin=self.convr.w.value.shape[2] * self.convr.groups,
            nout=self.convr.w.value.shape[3],
            k=self.convr.w.value.shape[:2],
            strides=self.convr.strides,
            dilations=self.convr.dilations,
            groups=self.convr.groups,
            padding=self.convr.padding,
            use_bias=self.convr.b is not None,
        )
        args = ", ".join(f"{k}={repr(v)}" for k, v in args.items())
        return f"{class_name(self)}({args})"


# def complex_ws(w_real: JaxArray, w_imag: JaxArray) -> tuple[JaxArray, ...]:
#     # each weight has shape H,W,I,O
#     # permute to O,I,H,W, then stack to O,2,I,H,W
#     stacked = jnp.stack(
#         [w_real.transpose(3, 2, 0, 1), w_imag.transpose(3, 2, 0, 1)], axis=1
#     )
#     # subtract mean for I,H,W axes
#     centered = stacked - stacked.mean(axis=(2, 3, 4), keepdims=True)
#     centered = centered.reshape(centered.shape[0], 2, -1)
#     # vmap the following computations over the O axis:
#     def whitening_matrix(centered):
#         # calculate covariance between real and imag
#         sigma = jnp.cov(centered)
#         # Compute inverse square root of covariance matrix
#         u_mat, lmbda, _ = jnp.linalg.svd(sigma, full_matrices=False)
#         # compute whitening matrix
#         w_mat = jnp.matmul(
#             u_mat, jnp.matmul(jnp.diag(1.0 / jnp.sqrt(lmbda + 1e-5)), u_mat.T)
#         )
#         # adjust whitening matrix so that outputs have unit variance
#         inv_sqrt = rsqrt(2.0) * jnp.eye(2)
#         w_mat = w_mat @ inv_sqrt
#         # multiply centered weights with whitening matrix
#         return jnp.matmul(w_mat, centered)

#     whitened = vmap(whitening_matrix)(centered)
#     res_r, res_i = whitened[:, 0, :], whitened[:, 1, :]
#     # reshape back to original shape
#     return (
#         res_r.transpose(1, 0).reshape(w_real.shape),
#         res_i.transpose(1, 0).reshape(w_imag.shape),
#     )


rsqrt2 = rsqrt(2.0)  # define to avoid two function evaluations later


def complex_ws(w_real: JaxArray, w_imag: JaxArray) -> tuple[JaxArray]:
    O = w_real.shape[-1]
    x = jnp.stack((w_real, w_imag))  # 2, H, W, I, O
    mean = x.mean(axis=(1, 2, 3), keepdims=True)  # 2, 1, 1, 1, O
    x = x - mean  # 2, H, W, I, O
    var = x.var(axis=(1, 2, 3)) + 1e-5  # 2, O
    cov_uu, cov_vv = var[0], var[1]  # O,
    cov_vu = cov_uv = (x[0] * x[1]).mean(axis=(0, 1, 2))  # O
    sqrdet = jnp.sqrt(cov_uu * cov_vv - cov_uv * cov_vu)
    denom = sqrdet * jnp.sqrt(cov_uu + 2 * sqrdet + cov_vv)
    p, q = (cov_vv + sqrdet) / denom, -cov_uv / denom  # O
    r, s = -cov_vu / denom, (cov_uu + sqrdet) / denom  # O
    tail = 1, 1, 1, O
    ret_r = x[0] * p.reshape(tail) + x[1] * r.reshape(tail)
    ret_i = x[0] * q.reshape(tail) + x[1] * s.reshape(tail)
    return ret_r * rsqrt2, ret_i * rsqrt2


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


class ComplexWSConv2DNative(ComplexConv2D):
    def __call__(self, x):
        weight_r, weight_i = complex_ws(self.convr.w.value, self.convi.w.value)
        weight = weight_r + 1j * weight_i
        return conv_general_dilated(
            lhs=x,
            rhs=weight,
            window_strides=self.convr.strides,
            padding=self.convr.padding,
            rhs_dilation=self.convr.dilations,
            feature_group_count=self.convr.groups,
            dimension_numbers=("NCHW", "HWIO", "NCHW"),
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


class ComplexToReal(Module):
    def __init__(self) -> None:
        pass

    def __call__(self, x):
        return jnp.abs(x)
        # return jnp.sqrt(jnp.square(x.real) + jnp.square(x.imag))


# fast implementations
# def fast_conv(inp, weight_r, weight_i, stride=(1,1), padding="same", dilation=(1,1)):
#     n_out = weight_r.shape[-1]
#     ww = jnp.concatenate((weight_r, weight_i), axis=-1)
#     wr = conv_general_dilated(inp.real, ww, window_strides=stride, padding=padding, rhs_dilation=dilation, dimension_numbers=("NCHW", "HWIO", "NCHW"))
#     wi = conv_general_dilated(inp.imag, ww, window_strides=stride, padding=padding, rhs_dilation=dilation, dimension_numbers=("NCHW", "HWIO", "NCHW"))
#     rwr, iwr = wr[:, :n_out], wr[:, n_out:]
#     rwi, iwi = wi[:, :n_out], wi[:, n_out:]
#     return (rwr-iwi) + 1j* (iwr+rwi)
