from functools import partial
from typing import Callable
from warnings import warn

from jax import numpy as jnp
from jax.lax import conv_general_dilated, conv_transpose
from objax import Module, TrainVar
from objax.constants import ConvPadding
from objax.nn import Conv2D, ConvTranspose2D
from objax.nn.init import kaiming_normal, xavier_normal
from objax.typing import ConvPaddingInt, JaxArray
from objax.util import class_name
from objax.functional import rsqrt

# Helper
def conjugate_apply(f_real, f_imag, inp):
    return (f_real(inp.real) - f_imag(inp.imag)) + 1j * (
        f_real(inp.imag) + f_imag(inp.real)
    )


# pylint:disable=duplicate-code
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


class ComplexConv2DTranspose(Module):
    def __init__(
        self,
        nin: int,
        nout: int,
        k: tuple[int, int] | int,
        strides: tuple[int, int] | int = 1,
        dilations: tuple[int, int] | int = 1,
        padding: ConvPaddingInt | ConvPadding | str = ConvPadding.SAME,
        w_init: Callable = kaiming_normal,
        use_bias=False,
    ) -> None:
        super().__init__()
        if use_bias:
            warn("Bias is not supported.")
            use_bias = False
        self.convr = ConvTranspose2D(
            nin=nin,
            nout=nout,
            k=k,
            strides=strides,
            dilations=dilations,
            padding=padding,
            use_bias=False,
            w_init=w_init,
        )
        self.convi = ConvTranspose2D(
            nin=nin,
            nout=nout,
            k=k,
            strides=strides,
            dilations=dilations,
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
    out_shape = w_real.shape[-1]
    stacked_weight = jnp.stack((w_real, w_imag))  # 2, H, W, I, O
    mean = stacked_weight.mean(axis=(1, 2, 3), keepdims=True)  # 2, 1, 1, 1, O
    stacked_weight = stacked_weight - mean  # 2, H, W, I, O
    var = stacked_weight.var(axis=(1, 2, 3)) + 1e-5  # 2, O
    cov_uu, cov_vv = var[0], var[1]  # O,
    cov_vu = cov_uv = (stacked_weight[0] * stacked_weight[1]).mean(axis=(0, 1, 2))  # O
    sqrdet = jnp.sqrt(cov_uu * cov_vv - cov_uv * cov_vu)
    denom = sqrdet * jnp.sqrt(cov_uu + 2 * sqrdet + cov_vv)
    p, q = (cov_vv + sqrdet) / denom, -cov_uv / denom  # O # pylint:disable=invalid-name
    r, s = -cov_vu / denom, (cov_uu + sqrdet) / denom  # O # pylint:disable=invalid-name
    tail = 1, 1, 1, out_shape
    ret_r = stacked_weight[0] * p.reshape(tail) + stacked_weight[1] * r.reshape(tail)
    ret_i = stacked_weight[0] * q.reshape(tail) + stacked_weight[1] * s.reshape(tail)
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
        return fast_conv(
            x,
            weight_r,
            weight_i,
            stride=self.convr.strides,
            padding=self.convr.padding,
            dilation=self.convr.dilations,
        )
        # return conjugate_apply(
        #     partial(
        #         conv_general_dilated,
        #         rhs=weight_r,
        #         window_strides=self.convr.strides,
        #         padding=self.convr.padding,
        #         rhs_dilation=self.convr.dilations,
        #         feature_group_count=self.convr.groups,
        #         dimension_numbers=("NCHW", "HWIO", "NCHW"),
        #     ),
        #     partial(
        #         conv_general_dilated,
        #         rhs=weight_i,
        #         window_strides=self.convi.strides,
        #         padding=self.convi.padding,
        #         rhs_dilation=self.convi.dilations,
        #         feature_group_count=self.convi.groups,
        #         dimension_numbers=("NCHW", "HWIO", "NCHW"),
        #     ),
        #     x,
        # )


class ComplexWSConv2DTranspose(ComplexConv2D):
    def __call__(self, x: JaxArray) -> JaxArray:
        weight_r, weight_i = complex_ws_whiten(self.convr.w.value, self.convi.w.value)
        return conjugate_apply(
            partial(
                conv_transpose,
                rhs=weight_r,
                strides=self.convr.strides,
                padding=self.convr.padding,
                rhs_dilation=self.convr.dilations,
                feature_group_count=self.convr.groups,
                dimension_numbers=("NCHW", "HWIO", "NCHW"),
            ),
            partial(
                conv_transpose,
                rhs=weight_i,
                strides=self.convi.strides,
                padding=self.convi.padding,
                rhs_dilation=self.convi.dilations,
                feature_group_count=self.convi.groups,
                dimension_numbers=("NCHW", "HWIO", "NCHW"),
            ),
            x,
        )


class ComplexWSConv2DNoWhitenTranspose(ComplexConv2D):
    def __call__(self, x: JaxArray) -> JaxArray:
        weight_r, weight_i = complex_ws_nowhiten(self.convr.w.value, self.convi.w.value)
        return fast_conv_transpose(
            x,
            weight_r,
            weight_i,
            stride=self.convr.strides,
            padding=self.convr.padding,
            dilation=self.convr.dilations,
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
        del use_bias
        # self.linr = Linear(nin=nin, nout=nout, use_bias=use_bias, w_init=w_init)
        # self.lini = Linear(nin=nin, nout=nout, use_bias=use_bias, w_init=w_init)
        self.linr = TrainVar(w_init((nin, nout)))
        self.lini = TrainVar(w_init((nin, nout)))

    def __call__(self, x: JaxArray) -> JaxArray:
        return linear_3m(x, self.linr, self.lini)
        # return conjugate_apply(self.linr, self.lini, x)


class ComplexToReal(Module):
    def __init__(self) -> None:
        pass

    def __call__(self, x):
        return jnp.abs(x)
        # return jnp.sqrt(jnp.square(x.real) + jnp.square(x.imag))


# fast implementations
def fast_conv(inp, weight_r, weight_i, stride=(1, 1), padding="same", dilation=(1, 1)):
    n_out = weight_r.shape[-1]
    concat_weights = jnp.concatenate((weight_r, weight_i), axis=-1)
    weight_real = conv_general_dilated(
        inp.real,
        concat_weights,
        window_strides=stride,
        padding=padding,
        rhs_dilation=dilation,
        dimension_numbers=("NCHW", "HWIO", "NCHW"),
    )
    weight_imag = conv_general_dilated(
        inp.imag,
        concat_weights,
        window_strides=stride,
        padding=padding,
        rhs_dilation=dilation,
        dimension_numbers=("NCHW", "HWIO", "NCHW"),
    )
    rwr, iwr = weight_real[:, :n_out], weight_real[:, n_out:]
    rwi, iwi = weight_imag[:, :n_out], weight_imag[:, n_out:]
    return (rwr - iwi) + 1j * (iwr + rwi)


def fast_conv_transpose(
    inp, weight_r, weight_i, stride=(1, 1), padding="same", dilation=(1, 1)
):
    n_out = weight_r.shape[-1]
    concat_weights = jnp.concatenate((weight_r, weight_i), axis=-1)
    weight_real = conv_transpose(
        inp.real,
        concat_weights,
        strides=stride,
        padding=padding,
        rhs_dilation=dilation,
        dimension_numbers=("NCHW", "HWIO", "NCHW"),
    )
    weight_imag = conv_transpose(
        inp.imag,
        concat_weights,
        strides=stride,
        padding=padding,
        rhs_dilation=dilation,
        dimension_numbers=("NCHW", "HWIO", "NCHW"),
    )
    rwr, iwr = weight_real[:, :n_out], weight_real[:, n_out:]
    rwi, iwi = weight_imag[:, :n_out], weight_imag[:, n_out:]
    return (rwr - iwi) + 1j * (iwr + rwi)


def linear_3m(inp, w_real, w_imag, bias=None):
    out1 = (inp.real + inp.imag) @ w_real
    out2 = inp.real @ (w_imag - w_real)
    out3 = inp.imag @ (w_real + w_imag)
    out = (out1 - out3) + 1j * (out1 + out2)
    if bias is not None:
        out += bias
    return out
