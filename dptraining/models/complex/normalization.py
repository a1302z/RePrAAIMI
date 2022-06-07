from jax import numpy as jn
from objax import functional
from objax.module import Module
from objax.variable import TrainVar
from objax.typing import JaxArray
from objax.util import class_name
from jax import vmap


def _whiten_one(vec: JaxArray):
    """Perform zca whitening on an array. This function assumes that the array
    is of shape (C,H,W), with the leading G(roup) and N (batch) dimensions omitted.
    Modified from https://gist.github.com/joelouismarino/ce239b5601fff2698895f48003f7464b
    """
    fl = vec.flatten()
    # subtract mean to center the tensor
    centered = fl - fl.mean()
    # compute covariance between real and imaginary. Trabelsi call this V
    sigma = jn.cov(centered.real, centered.imag)
    # Compute inverse square root of covariance matrix.
    U, Lambda, _ = jn.linalg.svd(sigma, full_matrices=False)
    W = jn.dot(U, jn.dot(jn.diag(1.0 / jn.sqrt(Lambda + 1e-5)), U.T))
    # convert complex to 2D real for dot product
    two_channel = jn.stack([centered.real, centered.imag])
    result = jn.dot(W, two_channel)
    # convert back to complex and reshape to original shape
    return (result[0] + result[1] * 1j).reshape(vec.shape)


# Since in the GroupNorm function the input will be (N,G,C,H,W), we vmap once over G
# and once over N since we want to compute the whitening over every group and every
# sample individually
_whiten_batch = vmap(vmap(_whiten_one))


class ComplexGroupNormWhitening(Module):
    """Like regular GroupNorm but uses whitening."""

    def __init__(self, nin: int, rank: int, groups: int = 32, eps: float = 1e-5):
        """Creates a GroupNorm module instance.
        Args:
            nin: number of input channels.
            rank: rank of the input tensor.
            groups: number of normalization groups.
            eps: small value which is used for numerical stability.
        """
        groups = min(groups, nin)
        assert nin % groups == 0, "nin should be divisible by groups"

        super().__init__()
        self.nin = nin
        self.groups = groups
        self.eps = eps
        self.redux = tuple(range(2, rank + 1))
        var_shape = (1, nin) + (1,) * (rank - 2)

        # complex affine parameters
        # Trabelsi initialises the gamma param with 1/sqrt(2) to have norm 1
        self.gamma = TrainVar(
            jn.full(var_shape, functional.rsqrt(2.0))
            + 1j * jn.full(var_shape, functional.rsqrt(2.0))
        )
        # The beta is initialised as zero
        self.beta = TrainVar(jn.zeros(var_shape) + 1j * jn.zeros(var_shape))

    def __call__(self, x: JaxArray, training: bool = True) -> JaxArray:
        """Returns the results of applying group normalization to input x."""
        del training
        group_shape = (-1, self.groups, self.nin // self.groups) + x.shape[2:]
        x = x.reshape(group_shape)
        x = _whiten_batch(x)
        x = x.reshape(
            (
                -1,
                self.nin,
            )
            + group_shape[3:]
        )
        x = x * self.gamma.value + self.beta.value
        return x

    def __repr__(self):
        args = f"nin={self.nin}, rank={len(self.gamma.shape)}, groups={self.groups}, eps={self.eps}"
        return f"{class_name(self)}({args})"


class ComplexGroupNorm2DWhitening(ComplexGroupNormWhitening):
    """Applies a 2D group normalization on a input batch of shape (N,C,H,W)."""

    def __init__(self, nin: int, groups: int = 32, eps: float = 1e-5):
        """Creates a GroupNorm2D module instance.
        Args:
            nin: number of input channels.
            groups: number of normalization groups.
            eps: small value which is used for numerical stability.
        """
        super().__init__(nin, rank=4, groups=groups, eps=eps)

    def __repr__(self):
        return (
            f"{class_name(self)}(nin={self.nin}, groups={self.groups}, eps={self.eps})"
        )


class ComplexGroupNorm(Module):
    """Applies a group normalization to input tensor.
    The module follows the operation described the paper
    `Group Normalization <https://arxiv.org/abs/1803.08494>`_.
    """

    def __init__(self, nin: int, rank: int, groups: int = 32, eps: float = 1e-5):
        """Creates a GroupNorm module instance.
        Args:
            nin: number of input channels.
            rank: rank of the input tensor.
            groups: number of normalization groups.
            eps: small value which is used for numerical stability.
        """
        groups = min(groups, nin)
        assert nin % groups == 0, "nin should be divisible by groups"

        super().__init__()
        self.nin = nin
        self.groups = groups
        self.eps = eps
        self.redux = tuple(range(2, rank + 1))
        var_shape = (1, nin) + (1,) * (rank - 2)

        # complex affine parameters
        self.gamma = TrainVar(jn.ones(var_shape) + 1j * jn.ones(var_shape))
        self.beta = TrainVar(jn.zeros(var_shape) + 1j * jn.zeros(var_shape))

    def __call__(self, x: JaxArray, training: bool = True) -> JaxArray:
        """Returns the results of applying group normalization to input x."""
        del training
        group_shape = (-1, self.groups, self.nin // self.groups) + x.shape[2:]
        x = x.reshape(group_shape)
        mean = x.mean(axis=self.redux, keepdims=True)
        var = x.var(axis=self.redux, keepdims=True)

        # normalise with twice the variance to get total variance 1
        x = (x - mean) * functional.rsqrt(2 * var + self.eps)
        x = x.reshape(
            (
                -1,
                self.nin,
            )
            + group_shape[3:]
        )
        x = x * self.gamma.value + self.beta.value
        return x

    def __repr__(self):
        args = f"nin={self.nin}, rank={len(self.gamma.shape)}, groups={self.groups}, eps={self.eps}"
        return f"{class_name(self)}({args})"


class ComplexGroupNorm2D(ComplexGroupNorm):
    """Applies a 2D group normalization on a input batch of shape (N,C,H,W)."""

    def __init__(self, nin: int, groups: int = 32, eps: float = 1e-5):
        """Creates a GroupNorm2D module instance.
        Args:
            nin: number of input channels.
            groups: number of normalization groups.
            eps: small value which is used for numerical stability.
        """
        super().__init__(nin, rank=4, groups=groups, eps=eps)

    def __repr__(self):
        return (
            f"{class_name(self)}(nin={self.nin}, groups={self.groups}, eps={self.eps})"
        )
