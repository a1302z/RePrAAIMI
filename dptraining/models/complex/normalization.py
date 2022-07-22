from jax import numpy as jn, vmap
from objax import functional
from objax.module import Module
from objax.variable import TrainVar
from objax.typing import JaxArray
from objax.util import class_name
from objax.variable import TrainVar, StateVar


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


def _whiten2x2(
    tensor,
    training=True,
    running_mean=None,
    running_cov=None,
    momentum=0.1,
):
    tail = 1, tensor.shape[2], *([1] * (tensor.ndim - 3))
    axes = 1, *range(3, tensor.ndim)
    if training:
        mean = tensor.mean(axis=axes)
        if running_mean is not None:
            running_mean += momentum * (mean - running_mean)
    else:
        mean = running_mean
    tensor = tensor - mean.reshape(2, *tail)
    if training:
        var = tensor.var(axis=axes) + 1e-5
        cov_uu, cov_vv = var[0], var[1]
        cov_vu = cov_uv = (tensor[0] * tensor[1]).mean([a - 1 for a in axes])
        if running_cov is not None:
            cov = jn.stack(
                [
                    cov_uu,
                    cov_uv,
                    cov_vu,
                    cov_vv,
                ],
                axis=0,
            ).reshape(2, 2, -1)
            running_cov += momentum * (cov - running_cov)
    else:
        cov_uu, cov_uv = running_cov[0, 0], running_cov[0, 1]
        cov_vu, cov_vv = running_cov[1, 0], running_cov[1, 1]
    sqrdet = jn.sqrt(cov_uu * cov_vv - cov_uv * cov_vu)
    denom = sqrdet * jn.sqrt(cov_uu + 2 * sqrdet + cov_vv)
    p, q = (cov_vv + sqrdet) / denom, -cov_uv / denom
    r, s = -cov_vu / denom, (cov_uu + sqrdet) / denom
    out = jn.stack(
        [
            tensor[0] * p.reshape(tail) + tensor[1] * r.reshape(tail),
            tensor[0] * q.reshape(tail) + tensor[1] * s.reshape(tail),
        ],
        axis=0,
    )
    return out


def batch_norm(
    input,
    running_mean,
    running_var,
    weight=None,
    bias=None,
    training=False,
    momentum=0.1,
):
    x = jn.stack([input.real, input.imag], axis=0)
    z = _whiten2x2(
        x,
        training=training,
        running_mean=running_mean,
        running_cov=running_var,
        momentum=momentum,
    )

    if weight is not None and bias is not None:
        shape = 1, x.shape[2], *([1] * (x.ndim - 3))
        weight = weight.reshape(2, 2, *shape)
        z = jn.stack(
            [
                z[0] * weight[0, 0] + z[1] * weight[0, 1],
                z[0] * weight[1, 0] + z[1] * weight[1, 1],
            ],
            axis=0,
        ) + bias.reshape(2, *shape)

    return z[0] + 1j * z[1]


class ComplexBatchNorm2D(Module):
    """Applies a batch normalization on different ranks of an input tensor.
    The module follows the operation described in Algorithm 1 of
    `Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
    <https://arxiv.org/abs/1502.03167>`_. However, it's corrected for complex inputs according to
    https://github.com/ivannz/cplxmodule/blob/master/cplxmodule/nn/modules/batchnorm.py
    """

    def __init__(self, nin: int, momentum: float = 0.1, eps: float = 1e-6):
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
        self.running_var = StateVar(jn.tile(jn.eye(2), (nin, 1, 1)).reshape(2, 2, nin))

        self.bias = TrainVar(jn.zeros((nin, 2, 1)))  # (n_channels, 2, 1)
        gamma_rr = gamma_ii = functional.rsqrt(2.0)
        gamma_ri = 0.0
        weight = jn.array([[gamma_rr, gamma_ri], [gamma_ri, gamma_ii]])
        weight = jn.tile(weight, (self.nin, 1, 1))
        self.weight = TrainVar(weight)  # (n_channels, 2, 2)

    def __call__(self, x: JaxArray, training: bool | None = True) -> JaxArray:
        if training:
            return batch_norm(
                x,
                running_mean=self.running_mean.value,
                running_var=self.running_var.value,
                weight=self.weight.value,
                bias=self.bias.value,
                training=True,
                momentum=self.momentum,
            )
        else:
            return batch_norm(
                x,
                running_mean=self.running_mean.value,
                running_var=self.running_var.value,
                weight=self.weight.value,
                bias=self.bias.value,
                training=True,
                momentum=self.momentum,
            )

    def __repr__(self) -> str:
        return f"{class_name(self)}(nin={self.nin}, momentum={self.momentum})"
