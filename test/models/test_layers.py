import numpy as np
from jax import numpy as jnp
from objax import functional
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from dptraining.models.layers import AdaptivePooling


def test_adaptive_pooling():
    for res in [10, 224]:
        data = np.random.randn(10, 3, res, res)
        for i in range(1, 10):
            pool = AdaptivePooling(functional.average_pool_2d, i)
            pooled = pool(data)
            assert pooled.shape[2] == i
            assert pooled.shape[3] == i
