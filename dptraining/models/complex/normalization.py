from jax import numpy as jn, vmap
from objax import functional
from objax.module import Module
from objax.variable import TrainVar
from objax.typing import JaxArray
from objax.util import class_name
from objax.variable import TrainVar, StateVar


# def _whiten_one(vec: JaxArray):
#     """Perform zca whitening on an array. This function assumes that the array
#     is of shape (C,H,W), with the leading G(roup) and N (batch) dimensions omitted.
#     Modified from https://gist.github.com/joelouismarino/ce239b5601fff2698895f48003f7464b
#     """
#     flat_vector = vec.flatten()
#     # subtract mean to center the tensor
#     centered = flat_vector - flat_vector.mean()
#     # compute covariance between real and imaginary. Trabelsi call this V
#     sigma = jn.cov(centered.real, centered.imag)
#     # Compute inverse square root of covariance matrix.
#     u_mat, lmbda, _ = jn.linalg.svd(sigma, full_matrices=False)
#     w_mat = jn.dot(u_mat, jn.dot(jn.diag(1.0 / jn.sqrt(lmbda + 1e-5)), u_mat.T))
#     # convert complex to 2D real for dot product
#     two_channel = jn.stack([centered.real, centered.imag])
#     result = jn.dot(w_mat, two_channel)
#     # convert back to complex and reshape to original shape
#     return (result[0] + result[1] * 1j).reshape(vec.shape)


# # Since in the GroupNorm function the input will be (N,G,C,H,W), we vmap once over G
# # and once over N since we want to compute the whitening over every group and every
# # sample individually
# _whiten_batch = vmap(vmap(_whiten_one))


# class ComplexGroupNormWhitening(Module):
#     """Like regular GroupNorm but uses whitening."""

#     def __init__(self, nin: int, rank: int, groups: int = 32, eps: float = 1e-5):
#         """Creates a GroupNorm module instance.
#         Args:
#             nin: number of input channels.
#             rank: rank of the input tensor.
#             groups: number of normalization groups.
#             eps: small value which is used for numerical stability.
#         """
#         groups = min(groups, nin)
#         assert nin % groups == 0, "nin should be divisible by groups"

#         super().__init__()
#         self.nin = nin
#         self.groups = groups
#         self.eps = eps
#         self.redux = tuple(range(2, rank + 1))
#         var_shape = (1, nin) + (1,) * (rank - 2)

#         # complex affine parameters
#         # Trabelsi initialises the gamma param with 1/sqrt(2) to have norm 1
#         self.gamma_r = TrainVar(jn.full(var_shape, functional.rsqrt(2.0)))
#         self.gamma_i = TrainVar(jn.full(var_shape, functional.rsqrt(2.0)))
#         # The beta is initialised as zero
#         self.beta_r = TrainVar(jn.zeros(var_shape))
#         self.beta_i = TrainVar(jn.zeros(var_shape))

#     def __call__(self, x: JaxArray, training: bool = True) -> JaxArray:
#         """Returns the results of applying group normalization to input x."""
#         del training
#         group_shape = (-1, self.groups, self.nin // self.groups) + x.shape[2:]
#         x = x.reshape(group_shape)
#         x = _whiten_batch(x)
#         x = x.reshape(
#             (
#                 -1,
#                 self.nin,
#             )
#             + group_shape[3:]
#         )
#         gamma = self.gamma_r.value + 1j * self.gamma_i.value
#         beta = self.beta_r.value + 1j * self.beta_i.value
#         x = x * gamma + beta
#         return x

#     def __repr__(self):
#         args = (
#             f"nin={self.nin}, rank={len(self.gamma_r.shape)}"
#             f", groups={self.groups}, eps={self.eps}"
#         )
#         return f"{class_name(self)}({args})"


# class ComplexGroupNorm2DWhitening(ComplexGroupNormWhitening):
#     """Applies a 2D group normalization on a input batch of shape (N,C,H,W)."""

#     def __init__(self, nin: int, groups: int = 32, eps: float = 1e-5):
#         """Creates a GroupNorm2D module instance.
#         Args:
#             nin: number of input channels.
#             groups: number of normalization groups.
#             eps: small value which is used for numerical stability.
#         """
#         super().__init__(nin, rank=4, groups=groups, eps=eps)

#     def __repr__(self):
#         return (
#             f"{class_name(self)}(nin={self.nin}, groups={self.groups}, eps={self.eps})"
#         )


# class ComplexBatchNorm2D(Module):
#     """Applies a batch normalization on different ranks of an input tensor.
#     The module follows the operation described in Algorithm 1 of
#     `Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
#     <https://arxiv.org/abs/1502.03167>`_. However, it's corrected for complex inputs according to
#     https://github.com/ivannz/cplxmodule/blob/master/cplxmodule/nn/modules/batchnorm.py
#     """

#     def __init__(self, nin: int, momentum: float = 0.999, eps: float = 1e-6):
#         """Creates a BatchNorm2D module instance.
#         Args:
#             nin: number of channels in the input example.
#             momentum: value used to compute exponential moving average of batch statistics.
#             eps: small value which is used for numerical stability.
#         """
#         super().__init__()
#         self.momentum = momentum
#         self.eps = eps
#         self.nin = nin

#         self.running_mean = StateVar(jn.zeros((2, 1, self.nin, 1, 1)))
#         self.running_var = StateVar(jn.ones((3, self.nin)))

#         self.gamma = TrainVar(jn.full((2, 1, self.nin, 1, 1), functional.rsqrt(2.0)))
#         self.beta = TrainVar(jn.zeros((2, 1, self.nin, 1, 1)))

#     def __call__(self, x: JaxArray, training: bool) -> JaxArray:
#         # Input is a complex array of shape N,C,H,W
#         # Split it to become 2,N,C,H,W where 2 are the real/complex components
#         x = jn.stack([x.real, x.imag], axis=0)

#         tail = 1, self.nin, 1, 1
#         axes = 1, 3, 4

#         if training:
#             # calculate mean across N, H, W but keep the channels untouched
#             mean = x.mean(axis=axes, keepdims=True)
#             # calculate variance across N, H, W, not across channels
#             var = x.var(axis=axes) + self.eps
#             # Calculate covariances (real/complex etc.)
#             cov_uu, cov_vv = var[0], var[1]
#             cov_vu = cov_uv = (x[0] * x[1]).mean([a - 1 for a in axes])
#             # for our running mean. Two are identical to each other so only 3 numbers
#             # are required
#             cov_vec = jn.array([cov_uu, cov_vv, cov_vu])
#             # update the running statistics
#             self.running_mean.value += (1 - self.momentum) * (
#                 mean - self.running_mean.value
#             )
#             self.running_var.value += (1 - self.momentum) * (
#                 cov_vec - self.running_var.value
#             )
#         else:
#             # not training, pull the stored mean and covariance stats
#             mean = self.running_mean.value
#             cov_uu = self.running_var.value[0]
#             cov_vv = self.running_var.value[1]
#             cov_vu = cov_uv = self.running_var.value[2]

#         # center with calculated or stored mean
#         x = x - mean
#         # compute inverse square root of covariance matrix
#         sqrdet = jn.sqrt(cov_uu * cov_vv - cov_uv * cov_vu)
#         denom = sqrdet * jn.sqrt(cov_uu + 2 * sqrdet + cov_vv)
#         p, q = (cov_vv + sqrdet) / (denom + self.eps), -cov_uv / (denom + self.eps)
#         r, s = -cov_vu / (denom + self.eps), (cov_uu + sqrdet) / (denom + self.eps)
#         # multiply with whitening matrix to obtain zero real/imag covariance
#         out = jn.stack(
#             [
#                 x[0] * p.reshape(tail) + x[1] * r.reshape(tail),
#                 x[0] * q.reshape(tail) + x[1] * s.reshape(tail),
#             ],
#             axis=0,
#         )
#         # multiply/scale by affine parameters
#         y = self.gamma.value * out + self.beta.value
#         return y[0] + 1j * y[1]

#     def __repr__(self):
#         return f"{class_name(self)}(nin={self.beta.value.shape[1]}, momentum={self.momentum}, eps={self.eps})"


class ComplexGroupNorm2DWhitening(Module):
    def __init__(self, nin: int, groups: int, eps: float = 1e-5) -> None:
        assert nin % groups == 0, "in_channels must be divisible by groups"
        self.in_channels = nin
        self.groups = min(groups, nin)
        gamma_rr = gamma_ii = functional.rsqrt(2.0)
        gamma_ri = 0.0
        weight = jn.array([[gamma_rr, gamma_ri], [gamma_ri, gamma_ii]])  # (2, 2)
        weight = jn.tile(weight, (self.in_channels, 1, 1))
        self.weight = TrainVar(weight)  # (in_channels, 2, 2)
        self.bias = TrainVar(
            jn.zeros((self.in_channels, 2))
        )  # (in_channels, 2), 2 being re(beta) and im(beta)
        self.eps = eps

    def __call__(self, x: JaxArray) -> JaxArray:
        # incoming array is NCHW. Reshape to NGCHW
        N, C, H, W = x.shape
        G = self.groups
        x = x.reshape(N, G, C // G, H, W)
        # subtract mean of each group
        mean = x.mean(axis=(2, 3, 4), keepdims=True)
        x = x - mean
        # flatten into NG(CHW)
        flattened = x.reshape(N, G, C // G * H * W)
        # split into real and imaginary
        split = jn.stack([flattened.real, flattened.imag], axis=0)
        # whiten
        var = split.var(axis=-1, keepdims=True) + self.eps
        cov_uu, cov_vv = var[0], var[1]
        cov_vu = cov_uv = (split[0] * split[1]).mean(axis=-1, keepdims=True)
        sqrdet = jn.sqrt(cov_uu * cov_vv - cov_uv * cov_vu)
        denom = sqrdet * jn.sqrt(cov_uu + 2 * sqrdet + cov_vv) + 1e-6
        p, q = (cov_vv + sqrdet) / denom, -cov_uv / denom
        r, s = -cov_vu / denom, (cov_uu + sqrdet) / denom
        whitened_re = (split[0] * p + split[1] * r).reshape(N, C, H, W)
        whitened_im = (split[0] * q + split[1] * s).reshape(N, C, H, W)
        # perform affine operation
        weight = self.weight.value.reshape((2, 2, 1, C, 1, 1))
        bias = self.bias.value.reshape((2, 1, C, 1, 1))
        out_re = (whitened_re * weight[0, 0] + whitened_im * weight[0, 1]) + bias[0]
        out_im = (whitened_re * weight[1, 0] + whitened_im * weight[1, 1]) + bias[1]
        return out_re + 1j * out_im


class ComplexBatchNorm2D(Module):
    def __init__(self, nin: int, momentum: float = 0.999, eps: float = 1e-6) -> None:
        self.nin = nin
        self.momentum = momentum

        self.running_mean = StateVar(jn.zeros((2, 1, self.nin, 1, 1)))
        self.running_var = StateVar(jn.ones((2, 2, self.nin)))

        gamma_rr = gamma_ii = functional.rsqrt(2.0)
        gamma_ri = 0.0
        weight = jn.array([[gamma_rr, gamma_ri], [gamma_ri, gamma_ii]])  # (2, 2)
        weight = jn.tile(weight, (self.nin, 1, 1))
        self.weight = TrainVar(weight)  # (in_channels, 2, 2)
        self.bias = TrainVar(
            jn.zeros((self.nin, 2))
        )  # (in_channels, 2), 2 being re(beta) and im(beta)
        self.eps = eps

    def __call__(self, x: JaxArray, training: bool = True) -> JaxArray:

        N, C, H, W = x.shape
        # split into real and imaginary
        split = jn.stack([x.real, x.imag], axis=0)  # now 2, N, C, H, W

        if training:
            mean = split.mean(axis=(1, 3, 4), keepdims=True)  # over N, H, W
            # update running means
            self.running_mean.value += (1 - self.momentum) * (
                mean - self.running_mean.value
            )

            # determine covariances
            var = split.var(axis=(1, 3, 4), keepdims=True) + self.eps  # over N, H, W
            cov_uu, cov_vv = var[0], var[1]
            cov_vu = cov_uv = (split[0] * split[1]).mean(axis=(0, 2, 3), keepdims=True)
            cov = jn.stack([cov_uu, cov_uv, cov_vu, cov_vv], axis=0).reshape(2, 2, -1)

            # update running covariances
            self.running_var.value += (1 - self.momentum) * (cov - self.running_var)

        else:
            mean = self.running_mean.value
            cov_uu = self.running_var.value[0, 0].reshape(1, self.nin, 1, 1)
            cov_uv = self.running_var.value[0, 1].reshape(1, self.nin, 1, 1)
            cov_vu = self.running_var.value[1, 0].reshape(1, self.nin, 1, 1)
            cov_vv = self.running_var.value[1, 1].reshape(1, self.nin, 1, 1)

        # subtract mean
        x = x - mean
        # whiten
        sqrdet = jn.sqrt(cov_uu * cov_vv - cov_uv * cov_vu)
        denom = sqrdet * jn.sqrt(cov_uu + 2 * sqrdet + cov_vv) + self.eps
        p, q = (cov_vv + sqrdet) / denom, -cov_uv / denom
        r, s = -cov_vu / denom, (cov_uu + sqrdet) / denom
        whitened_re = (split[0] * p + split[1] * r).reshape(N, C, H, W)
        whitened_im = (split[0] * q + split[1] * s).reshape(N, C, H, W)
        weight = self.weight.reshape((2, 2, 1, C, 1, 1))
        bias = self.bias.reshape((2, 1, C, 1, 1))
        out_re = (whitened_re * weight[0, 0] + whitened_im * weight[0, 1]) + bias[0]
        out_im = (whitened_re * weight[1, 0] + whitened_im * weight[1, 1]) + bias[1]
        return out_re + 1j * out_im
