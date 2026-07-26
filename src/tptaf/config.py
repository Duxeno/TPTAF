"""Configuration helpers."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml

def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("The configuration root must be a YAML mapping.")
    return config

def pair(value: Any, name: str) -> tuple[int, int]:
    if isinstance(value, int): return value, value
    if isinstance(value, (list, tuple)) and len(value) == 2: return int(value[0]), int(value[1])
    raise ValueError(f"{name} must be an integer or a two-element sequence.")
