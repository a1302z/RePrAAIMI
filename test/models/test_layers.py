import numpy as np
from jax import numpy as jnp
from objax import functional
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from dptraining.models.layers import AdaptivePooling
from dptraining.models import ComplexWSConv2DNative, ComplexWSConv2D


def test_adaptive_pooling():
    for res in [10, 224]:
        data = np.random.randn(10, 3, res, res)
        for i in range(1, 10):
            pool = AdaptivePooling(functional.average_pool_2d, i)
            pooled = pool(data)
            assert pooled.shape[2] == i
            assert pooled.shape[3] == i


def test_equality_of_complex_ws_conv_implementation():
    conv1 = ComplexWSConv2D(3, 3, 1, w_init=jnp.ones)
    conv2 = ComplexWSConv2DNative(3, 3, 1, w_init=jnp.ones)

    fake_data = jnp.array(
        np.random.randn(1, 3, 224, 224) + 1j * np.random.randn(1, 3, 224, 224)
    )

    out1 = conv1(fake_data)
    out2 = conv2(fake_data)
    assert jnp.allclose(out1, out2)
