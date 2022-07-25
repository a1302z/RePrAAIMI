from functools import partial
from typing import Callable
from warnings import warn

from jax import numpy as jnp
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


rsqrt2 = rsqrt(2.0)  # define to avoid two function evaluations later


def complex_ws_whiten(w_real: JaxArray, w_imag: JaxArray) -> tuple[JaxArray]:
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


def complex_ws_nowhiten(w_real, w_imag):
    w_real_n = w_real - w_real.mean(axis=(0, 1, 2), keepdims=True)
    w_imag_n = w_imag - w_imag.mean(axis=(0, 1, 2), keepdims=True)
    return w_real_n, w_imag_n


class ComplexWSConv2D(ComplexConv2D):
    def __call__(self, x: JaxArray) -> JaxArray:
        weight_r, weight_i = complex_ws_whiten(self.convr.w.value, self.convi.w.value)
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


class ComplexWSConv2DNoWhiten(ComplexConv2D):
    def __call__(self, x: JaxArray) -> JaxArray:
        weight_r, weight_i = complex_ws_nowhiten(self.convr.w.value, self.convi.w.value)
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

# def linear_3m(inp, w_real, w_imag, bias=None):
#   K1 = (inp.real + inp.imag) @ w_real
#   K2 = inp.real @ (w_imag - w_real)
#   K3 = inp.imag @ (w_real + w_imag)
#   out = (K1-K3) + 1j * (K1 + K2)
#   if bias is not None:
#     out += bias
#   return out
