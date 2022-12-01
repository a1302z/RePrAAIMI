import sys
from pathlib import Path
from hydra.core.config_store import ConfigStore

sys.path.insert(0, str(Path.cwd()))

from dptraining.config import Config


def load_config_store():
    configstore = ConfigStore.instance()
    configstore.store(name="base_config", node=Config)
