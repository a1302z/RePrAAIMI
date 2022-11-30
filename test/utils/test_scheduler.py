import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path.cwd()))
from dptraining.utils.scheduler import ConstantSchedule, CosineSchedule, ReduceOnPlateau


def test_constant_scheduler():
    sched = ConstantSchedule(1e-3, 100)
    for _ in range(100):
        new_lr = next(sched)
        assert abs(new_lr - 1e-3) < 1e-3


def test_constant_scheduler_overflow():
    sched = ConstantSchedule(1e-3, 100)
    with pytest.raises(StopIteration):
        for _ in range(101):
            new_lr = next(sched)
            assert abs(new_lr - 1e-3) < 1e-3


def test_cosine_scheduler():
    sched = CosineSchedule(1e-3, 1000)
    for _ in range(1000):
        new_lr = next(sched)
    assert new_lr < 1e-7


def test_reduce_on_plateau():
    sched = ReduceOnPlateau(base_lr=1.0, patience=5, factor=0.5, min_delta=0.01)
    scores = [0.1, 0.2, 0.3, 0.25, 0.28, 0.27, 0.29, 0.305]
    for s, lr in zip(scores, sched):
        sched.update_score(s)
        assert abs(lr - 1.0) < 1e-5
    scores = [0.29, 0.3, 0.1, 0.25, 0.299]
    for s, lr in zip(scores, sched):
        sched.update_score(s)
        assert abs(lr - 0.5) < 1e-5
    scores = [0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5]
    for s, lr in zip(scores, sched):
        sched.update_score(s)
        assert abs(lr - 0.25) < 1e-5
