import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path.cwd()))
from dptraining.utils.scheduler import ConstantSchedule, CosineSchedule


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
