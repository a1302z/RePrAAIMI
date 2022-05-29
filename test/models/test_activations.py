import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from dptraining.models import make_activation_from_config


def test_relu():
    make_activation_from_config({"model": {"activation": "relu"}})


def test_selu():
    make_activation_from_config({"model": {"activation": "selu"}})


def test_leakyrelu():
    make_activation_from_config({"model": {"activation": "leakyrelu"}})
