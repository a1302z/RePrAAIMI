from contextlib import contextmanager
from omegaconf import OmegaConf
from pathlib import Path
import pytest
import sys

sys.path.insert(0, str(Path.cwd()))
from dptraining.config import Config
from dptraining.config.config_store import load_config_store

load_config_store()


class Utils:
    @staticmethod
    def extend_base_config(overrides: dict):
        base_conf = OmegaConf.structured(Config)
        merged_conf = OmegaConf.merge(base_conf, overrides)
        return merged_conf


@pytest.fixture
def utils():
    return Utils
