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
        r = jn.matmul(self.weight, u) + self.bias
        return (r[:, :, 0, :] + 1j * r[:, :, 1, :]).reshape(N, C, H, W)

    def __repr__(self) -> str:
        return f"{class_name(self)}(nin={self.nin}, groups={self.groups})"


# Batch Norm


def bn_single(
    vec: JaxArray,
    weight: JaxArray,
    bias: JaxArray,
    mean: JaxArray | None = None,
    sigma: JaxArray | None = None,
) -> tuple[JaxArray, JaxArray, JaxArray]:
    """Takes a complex vector with three dimensions and
    1. computes the mean if it's not provided
    2. computes the covariance matrix if it's not provided
    3. centers and whitens the vector
    4. applies the affine parameters weight and bias which are
    assumed to be (2,2) and (2,) matrices, respectively
    5. returns the mean and covariance matrix

    This method is optimised for batch norm where it is supposed to
    be vmapped over the channel dimensions. For this, first move the
    channel axis to be first. In training mode, leave mean and sigma
    as None and use the return to update the running stats.
    In inference mode, send in the running mean and covariance.
    Then move the channel axis back to the original position.
    """
    split = jn.stack([vec.real, vec.imag]).reshape(2, -1)
    if mean is None:
        mean = split.mean(-1)
    centered = split - mean.reshape(2, 1)
    if sigma is None:
        sigma = jn.cov(split[0], split[1])
    u_mat, lmbda, _ = jn.linalg.svd(sigma, full_matrices=False)
    w_mat = jn.dot(u_mat, jn.dot(jn.diag(1.0 / jn.sqrt(lmbda + 1e-5)), u_mat.T))
    result = (weight @ w_mat) @ centered + bias.reshape(2, 1)
    return (result[0] + 1j * result[1]).reshape(vec.shape), mean, sigma


batch_normalize = vmap(bn_single)


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

        self.running_mean = StateVar(jn.zeros((nin, 2)))  # (n_channels, 2)
        self.running_var = StateVar(
            jn.tile(jn.eye(2), (nin, 1, 1))
        )  # (n_channels, 2, 2)

        self.bias = TrainVar(jn.zeros((nin, 2)))  # (n_channels, 2)
        gamma_rr = gamma_ii = functional.rsqrt(2.0)
        gamma_ri = 0.0
        weight = jn.array([[gamma_rr, gamma_ri], [gamma_ri, gamma_ii]])
        weight = jn.tile(weight, (self.nin, 1, 1))
        self.weight = TrainVar(weight)  # (n_channels, 2, 2)

    def __call__(self, x: JaxArray, training: bool | None = True) -> JaxArray:
        x = jn.swapaxes(x, 0, 1)  # channels first
        if training:
            y, mean, sigma = batch_normalize(
                x, weight=self.weight, bias=self.bias, mean=None, sigma=None
            )
            self.running_mean.value += (1 - self.momentum) * (
                mean - self.running_mean.value
            )
            self.running_var.value += (1 - self.momentum) * (
                sigma - self.running_var.value
            )
        else:
            mean, sigma = self.running_mean.value, self.running_var.value
            y, *_ = batch_normalize(
                x, weight=self.weight, bias=self.bias, mean=mean, sigma=sigma
            )
        return jn.swapaxes(y, 1, 0)

    def __repr__(self) -> str:
        return f"{class_name(self)}(nin={self.nin}, momentum={self.momentum})"
