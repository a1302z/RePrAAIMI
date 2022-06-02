import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from dptraining.datasets.utils import collate_np_arrays


def test_collate():
    batch = [
        (np.random.randn(3, 224, 224), np.random.randint(0, 10, size=(1,)).item())
        for _ in range(10)
    ]
    imgs, labels = collate_np_arrays(batch)
    assert imgs.shape == (10, 3, 224, 224)
    assert labels.shape == (10,)
