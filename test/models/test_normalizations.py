import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from dptraining.models import make_normalization_from_config


def test_bn():
    make_normalization_from_config({"model": {"normalization": "bn"}})


def test_gn():
    make_normalization_from_config({"model": {"normalization": "gn"}})
