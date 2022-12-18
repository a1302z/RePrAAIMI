import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from dptraining.models import make_real_normalization2d_from_config


def test_bn(utils):
    config = utils.extend_base_config({"model": {"normalization": "bn"}})
    make_real_normalization2d_from_config(config)


def test_gn(utils):
    config = utils.extend_base_config({"model": {"normalization": "gn"}})
    make_real_normalization2d_from_config(config)
