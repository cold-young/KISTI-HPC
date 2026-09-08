"""Configuration loader for rcac-cli."""

import os
from pathlib import Path

import yaml


def get_config_dir() -> Path:
    """Return the configs directory path."""
    return Path(__file__).parent.parent / "configs"


def load_config(name: str = "rcac") -> dict:
    """Load a YAML config file from the configs directory."""
    config_path = get_config_dir() / f"{name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def get_rcac_config() -> dict:
    """Load rcac.yaml and resolve user placeholders."""
    cfg = load_config("rcac")
    user = cfg["ssh"].get("user") or os.environ.get("RCAC_USER", "")
    if user:
        for key, val in cfg.get("paths", {}).items():
            if isinstance(val, str):
                cfg["paths"][key] = val.format(user=user)
    cfg["ssh"]["user"] = user
    return cfg


def get_defaults() -> dict:
    """Load defaults.yaml."""
    return load_config("defaults")
