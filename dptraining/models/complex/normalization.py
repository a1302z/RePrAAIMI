from jax import numpy as jn, vmap
from objax import functional
from objax.module import Module
from objax.variable import TrainVar
from objax.typing import JaxArray
from objax.util import class_name
from objax.variable import TrainVar, StateVar
import numpy as onp


def gn_single(vec: JaxArray) -> JaxArray:
    """Takes a complex vector with three dimensions and
    1. computes the mean
    2. computes the covariance matrix
    3. centers and whitens the vector
    This method is optimised for group norm where it is supposed to
    be vmapped (twice) over the batch and group dimensions. Will give
    back a split vector ready to apply affine parameters.
    """
    split = jn.stack([vec.real, vec.imag]).reshape(2, -1)
    mean = split.mean(-1, keepdims=True)
    centered = split - mean
    sigma = jn.cov(split[0], split[1])
    u_mat, lmbda, _ = jn.linalg.svd(sigma, full_matrices=False)
    w_mat = jn.dot(u_mat, jn.dot(jn.diag(1.0 / jn.sqrt(lmbda + 1e-5)), u_mat.T))
    result = w_mat @ centered
    return (result[0] + 1j * result[1]).reshape(vec.shape)


group_normalize = vmap(vmap(gn_single))


class ComplexGroupNormWhitening(Module):
    """This exists for backwards compatibility reasons"""

    pass


class ComplexGroupNorm2DWhitening(ComplexGroupNormWhitening):
    """Applies a 2D group normalization on a input batch of shape (N,C,H,W)."""

    def __init__(self, nin: int, groups: int = 32, eps: float = 1e-5) -> None:
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
        self.bias = TrainVar(jn.zeros((1, nin, 2, 1)))  # (1, C, 2 ,1)
        gamma_rr = gamma_ii = functional.rsqrt(2.0)
        gamma_ri = 0.0
        weight = jn.array([[gamma_rr, gamma_ri], [gamma_ri, gamma_ii]])
        weight = jn.tile(weight, (1, self.nin, 1, 1))
        self.weight = TrainVar(weight)  # (1, C, 2, 2)

    def __call__(self, x: JaxArray, training: bool | None = True) -> JaxArray:
        del training
        N, C, H, W = x.shape
        G = self.groups
        x = x.reshape(N, G, C // G, H, W)
        y = group_normalize(x)
        z = y.reshape(N, C, H, W)
        u = jn.stack((z.real, z.imag), 2).reshape(N, C, 2, -1)  # N, C, 2, ...
        # (1, C, 2, 2) @ (N, C, 2, ...) -> (N, C, 2, ...)
        # (N, C, 2, ...) + (1, C, 2, 1) -> (N, C, 2, ...)
        r = jn.matmul(self.weight.value, u) + self.bias.value
        return (r[:, :, 0, :] + 1j * r[:, :, 1, :]).reshape(N, C, H, W)

    def __repr__(self) -> str:
        return f"{class_name(self)}(nin={self.nin}, groups={self.groups})"


# Batch Norm


class ComplexBatchNorm2D(Module):
    """Applies a batch normalization on different ranks of an input tensor.
    The module follows the operation described in Algorithm 1 of
    `Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
    <https://arxiv.org/abs/1502.03167>`_. However, it's corrected for complex inputs according to
    https://github.com/ivannz/cplxmodule/blob/master/cplxmodule/nn/modules/batchnorm.py
    """

    def __init__(self, nin: int, momentum: float = 0.999, eps: float = 1e-6):
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
        running_var[0, 0] = 1.0
        running_var[1, 1] = 1.0
        self.running_var = StateVar(jn.array(running_var))

        self.bias = TrainVar(jn.zeros((nin, 2, 1)))  # (n_channels, 2, 1)
        gamma_rr = gamma_ii = functional.rsqrt(2.0)
        gamma_ri = 0.0
        weight = jn.array([[gamma_rr, gamma_ri], [gamma_ri, gamma_ii]])
        weight = jn.tile(weight, (self.nin, 1, 1))
        self.weight = TrainVar(weight)  # (n_channels, 2, 2)

    def whiten2x2(self, x, training):
        tail = 1, x.shape[2], *([1] * (x.ndim - 3))
        axes = 1, *range(3, x.ndim)
        if training:
            mean = x.mean(axis=axes)
            self.running_mean.value = (self.momentum * mean) + (
                1 - self.momentum
            ) * self.running_mean.value

        else:
            mean = self.running_mean.value
        x = x - mean.reshape(2, *tail)
        if training:
            var = x.var(axis=axes) + 1e-5
            cov_uu, cov_vv = var[0], var[1]
            cov_vu = cov_uv = (x[0] * x[1]).mean([a - 1 for a in axes])
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
        p, q = (cov_vv + sqrdet) / denom, -cov_uv / denom
        r, s = -cov_vu / denom, (cov_uu + sqrdet) / denom
        out = jn.stack(
            [
                x[0] * p.reshape(tail) + x[1] * r.reshape(tail),
                x[0] * q.reshape(tail) + x[1] * s.reshape(tail),
            ],
            axis=0,
        )
        return out

    def batch_norm(self, x, training):
        x = jn.stack([x.real, x.imag], axis=0)
        z = self.whiten2x2(x=x, training=training)
        shape = 1, x.shape[2], *([1] * (x.ndim - 3))
        weight = self.weight.value.reshape(2, 2, *shape)
        z = jn.stack(
            [
                z[0] * weight[0, 0] + z[1] * weight[0, 1],
                z[0] * weight[1, 0] + z[1] * weight[1, 1],
            ],
            axis=0,
        ) + self.bias.value.reshape(2, *shape)
        return z[0] + 1j * z[1]

    def __call__(self, x: JaxArray, training: bool | None = True) -> JaxArray:
        return self.batch_norm(x, training=training)

    def __repr__(self) -> str:
        return f"{class_name(self)}(nin={self.nin}, momentum={self.momentum})"
