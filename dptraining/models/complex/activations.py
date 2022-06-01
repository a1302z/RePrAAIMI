from objax.typing import JaxArray
from objax import Module
from jax import numpy as jn
from dptraining.models.activations import mish


class IGaussian(Module):
    def __init__(self, sigma: float = 0.4) -> None:
        super().__init__()
        self.sigma = sigma

    def __call__(self, x: JaxArray) -> JaxArray:
        norm_sq = x * jn.conj(x)
        grad = 1 - jn.exp(-norm_sq / (2 * self.sigma**2))
        arg = jn.angle(x)
        norm = jn.exp(1j * arg)
        return grad.real * norm


class ComplexMish(Module):
    """Apply Mish directly to the complex number. Not kind to phase."""

    def __call__(self, x: JaxArray) -> JaxArray:
        return mish(x)


class SeparableMish(Module):
    """Apply Mish separately to real and complex part"""

    def __call__(self, x: JaxArray) -> JaxArray:
        return mish(x.real) + 1j * mish(x.imag)


class ConjugateMish(Module):
    """Apply Mish in a conjugate way"""

    def __call__(self, x: JaxArray) -> JaxArray:
        real = mish(x.real) - mish(x.imag)
        imag = mish(x.real) + mish(x.imag)
        return real + 1j * imag


class Cardioid(Module):
    def __call__(self, x: JaxArray) -> JaxArray:
        return 0.5 * (1 + jn.cos(jn.angle(x))) * x
