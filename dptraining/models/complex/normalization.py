from abc import abstractmethod
from jax import numpy as jn
from objax import functional
from objax.module import Module
from objax.typing import JaxArray
from objax.util import class_name
from objax.variable import TrainVar, StateVar
import numpy as onp
from typing import Any


class ComplexGroupNormWhitening(Module):
    """This exists for backwards compatibility reasons"""

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class ComplexGroupNorm2DWhitening(ComplexGroupNormWhitening):
    """Applies a 2D group normalization on a input batch of shape (N,C,H,W)."""

    def __init__(self, nin: int, groups: int = 32, eps: Any = 1e-5) -> None:
        """Creates a GroupNorm2D module instance.
        Args:
            nin: number of input channels.
            groups: number of normalization groups.
            eps: small value which is used for numerical stability. Ignored.
        """
        super().__init__()
        groups = min(groups, nin)
        assert nin % groups == 0, "nin should be divisible by groups"
        del eps
        self.nin = nin
        self.groups = groups
        self.bias = TrainVar(
            jn.zeros((1, nin, 2, 1)).reshape(2, 1, nin, 1, 1)
        )  # (2, 1, C, 1 ,1)
        weight = onp.zeros((2, 2, 1, nin, 1, 1))  # (2, 2, 1, C, 1, 1)
        weight[0, 0] = weight[1, 1] = functional.rsqrt(2.0)
        weight[1, 0] = weight[0, 1] = 0.0
        self.weight = TrainVar(jn.array(weight))

    def __call__(self, x: JaxArray, training: Any = None) -> JaxArray:
        del training
        N, C, H, W = x.shape  # pylint:disable=invalid-name
        groups = self.groups
        x = x.reshape(N, groups, C // groups, H, W)  # N, G, C', H, W
        x = jn.stack((x.real, x.imag), axis=0)  # 2, N, G, C', H, W
        mean = x.mean(axis=(3, 4, 5), keepdims=True)  # 2, N, G, 1, 1, 1
        x = x - mean  # 2, N, G, C', H, W
        var = x.var(axis=(3, 4, 5)) + 1e-5  # 2, N, G
        cov_uu, cov_vv = var[0], var[1]  # N, G
        cov_vu = cov_uv = (x[0] * x[1]).mean(axis=(2, 3, 4))  # N,G
        sqrdet = jn.sqrt(cov_uu * cov_vv - cov_uv * cov_vu)
        denom = sqrdet * jn.sqrt(cov_uu + 2 * sqrdet + cov_vv)
        p, q = (cov_vv + sqrdet) / denom, -cov_uv / denom  # pylint:disable=invalid-name
        r, s = -cov_vu / denom, (cov_uu + sqrdet) / denom  # pylint:disable=invalid-name
        tail = N, groups, 1, 1, 1
        output_array = jn.stack(
            [
                x[0] * p.reshape(tail) + x[1] * r.reshape(tail),
                x[0] * q.reshape(tail) + x[1] * s.reshape(tail),
            ],
            axis=0,
        )  # 2, N, G, C', H, W
        output_array = output_array.reshape(2, N, C, H, W)  # 2, N, C, H, W
        weight = self.weight.value  # 2, 2, 1, C, 1, 1
        output_array = (
            jn.stack(
                [
                    output_array[0] * weight[0, 0] + output_array[1] * weight[0, 1],
                    output_array[0] * weight[1, 0] + output_array[1] * weight[1, 1],
                ],
                axis=0,
            )
            + self.bias.value
        )  # 2, 1, C, 1, 1
        return output_array[0] + 1j * output_array[1]

    def __repr__(self) -> str:
        return f"{class_name(self)}(nin={self.nin}, groups={self.groups})"


class ComplexBatchNorm2D(Module):
    """Applies a batch normalization on different ranks of an input tensor.
    The module follows the operation described in Algorithm 1 of
    `Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
    <https://arxiv.org/abs/1502.03167>`_. However, it's corrected for complex inputs according to
    https://github.com/ivannz/cplxmodule/blob/master/cplxmodule/nn/modules/batchnorm.py
    """

    def __init__(self, nin: int, momentum: float = 0.999, eps: Any = 1e-6):
        """Creates a BatchNorm2D module instance.
        Args:
            nin: number of channels in the input example.
            momentum: value used to compute exponential moving average of batch statistics.
            eps: small value which is used for numerical stability. Ignored.
        """
        super().__init__()
        del eps
        self.nin = nin
        self.momentum = momentum

        self.running_mean = StateVar(jn.zeros((2, nin)))
        running_var = onp.zeros((2, 2, nin))
        running_var[0, 0] = running_var[1, 1] = functional.rsqrt(2.0)
        self.running_var = StateVar(jn.array(running_var))

        self.bias = TrainVar(jn.zeros((nin, 2, 1)))  # (n_channels, 2, 1)
        gamma_rr = gamma_ii = functional.rsqrt(2.0)
        gamma_ri = 0.0
        weight = jn.array([[gamma_rr, gamma_ri], [gamma_ri, gamma_ii]])
        weight = jn.tile(weight, (self.nin, 1, 1))
        self.weight = TrainVar(weight)  # (n_channels, 2, 2)

    def whiten2x2(self, inpt_array: JaxArray, training: bool | None) -> JaxArray:
        tail = 1, inpt_array.shape[2], *([1] * (inpt_array.ndim - 3))
        axes = 1, *range(3, inpt_array.ndim)
        if training:
            mean = inpt_array.mean(axis=axes)
            self.running_mean.value = (self.momentum * mean) + (
                1 - self.momentum
            ) * self.running_mean.value

        else:
            mean = self.running_mean.value
        inpt_array = inpt_array - mean.reshape(2, *tail)
        if training:
            var = inpt_array.var(axis=axes) + 1e-5
            cov_uu, cov_vv = var[0], var[1]
            cov_vu = cov_uv = (inpt_array[0] * inpt_array[1]).mean(
                [a - 1 for a in axes]
            )
            cov = jn.stack(
                [
                    cov_uu,
                    cov_uv,
                    cov_vu,
                    cov_vv,
                ],
                axis=0,
            ).reshape(2, 2, -1)
            self.running_var.value = (self.momentum * cov) + (
                1 - self.momentum
            ) * self.running_var.value
        else:
            cov_uu, cov_uv = self.running_var.value[0, 0], self.running_var.value[0, 1]
            cov_vu, cov_vv = self.running_var.value[1, 0], self.running_var.value[1, 1]
        sqrdet = jn.sqrt(
            cov_uu * cov_vv - cov_uv * cov_vu
        )  # no need to check for negativity since covariance matrix is PSD
        denom = sqrdet * jn.sqrt(cov_uu + 2 * sqrdet + cov_vv)
        p, q = (cov_vv + sqrdet) / denom, -cov_uv / denom  # pylint:disable=invalid-name
        r, s = -cov_vu / denom, (cov_uu + sqrdet) / denom  # pylint:disable=invalid-name
        out = jn.stack(
            [
                inpt_array[0] * p.reshape(tail) + inpt_array[1] * r.reshape(tail),
                inpt_array[0] * q.reshape(tail) + inpt_array[1] * s.reshape(tail),
            ],
            axis=0,
        )
        return out

    def batch_norm(self, inpt_array: JaxArray, training: bool | None) -> JaxArray:
        inpt_array = jn.stack([inpt_array.real, inpt_array.imag], axis=0)
        output_array = self.whiten2x2(inpt_array=inpt_array, training=training)
        shape = 1, inpt_array.shape[2], *([1] * (inpt_array.ndim - 3))
        weight = self.weight.value.reshape(2, 2, *shape)
        output_array = jn.stack(
            [
                output_array[0] * weight[0, 0] + output_array[1] * weight[0, 1],
                output_array[0] * weight[1, 0] + output_array[1] * weight[1, 1],
            ],
            axis=0,
        ) + self.bias.value.reshape(2, *shape)
        return output_array[0] + 1j * output_array[1]

    def __call__(self, x: JaxArray, training: bool | None = True) -> JaxArray:
        return self.batch_norm(x, training=training)

    def __repr__(self) -> str:
        return f"{class_name(self)}(nin={self.nin}, momentum={self.momentum})"
