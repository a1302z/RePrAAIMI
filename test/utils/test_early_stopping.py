import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from dptraining.utils import EarlyStopping


def test_earlystopping():
    stopper = EarlyStopping(patience=5, min_delta=0.01)
    scores = [0.8, 0.7, 0.6, 0.5, 0.85, 0.84, 0.83, 0.82, 0.81, 0.855]
    for s in scores[:-1]:
        assert not stopper(s)
    assert stopper(scores[-1])


def test_earlystopping_minimize():
    stopper = EarlyStopping(patience=5, min_delta=0.01, mode="minimize")
    scores = [0.3, 0.4, 0.5, 0.6, 0.25, 0.26, 0.27, 0.28, 0.29, 0.245]
    for s in scores[:-1]:
        assert not stopper(s)
    assert stopper(scores[-1])
