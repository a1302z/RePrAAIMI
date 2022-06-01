from jax import numpy as jnp, nn as jnn


def mish(inpt):
    return inpt * jnp.tanh(jnn.softplus(inpt))
