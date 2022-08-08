import numpy as np
from jax import numpy as jnp
from objax import functional, random
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from dptraining.models.layers import AdaptivePooling, ConvWS2D


def test_adaptive_pooling():
    for res in [10, 224]:
        data = np.random.randn(10, 3, res, res)
        for i in range(1, 10):
            pool = AdaptivePooling(functional.average_pool_2d, i)
            pooled = pool(data)
            assert pooled.shape[2] == i
            assert pooled.shape[3] == i


def test_conv_ws():
    model = ConvWS2D(2, 10, 3, w_init=lambda x: jnp.ones(x) + random.normal(x))
    data = jnp.ones((10, 2, 32, 32))
    model(data)
    assert jnp.allclose(
        model.w.mean(axis=(0, 1, 2)), jnp.zeros((model.w.shape[-1], 1)), atol=1e-5
    ).item()
    assert jnp.allclose(
        model.w.std(axis=(0, 1, 2)), jnp.ones((model.w.shape[-1],)), atol=1e-5
    ).item()
