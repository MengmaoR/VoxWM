"""Small YAML configuration helpers."""
from pathlib import Path

import yaml


def load_config(path):
    value = yaml.safe_load(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("VAE config must be a mapping")
    return value


def save_config(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(yaml.safe_dump(value, sort_keys=False))
    temporary.replace(path)
