"""Configuration loader for gilbreth-cli.

Resolution order for the two values every command needs:
  user    : GILBRETH_USER env var  > configs/gilbreth.yaml ssh.user
  account : GILBRETH_ACCOUNT env   > configs/gilbreth.yaml slurm.account
"""

from __future__ import annotations

import copy
import os
import re
import sys
from pathlib import Path

import click
import yaml

ENV_USER = "GILBRETH_USER"
ENV_ACCOUNT = "GILBRETH_ACCOUNT"
ENV_CONFIG = "GILBRETH_CONFIG"


def get_config_dir() -> Path:
    return Path(__file__).parent.parent / "configs"


def get_config_path() -> Path:
    override = os.environ.get(ENV_CONFIG)
    if override:
        return Path(override).expanduser()
    return get_config_dir() / "gilbreth.yaml"


def _format_user(value, user: str):
    """Recursively substitute ``{user}`` in strings."""
    if isinstance(value, str):
        return value.replace("{user}", user)
    if isinstance(value, dict):
        return {k: _format_user(v, user) for k, v in value.items()}
    if isinstance(value, list):
        return [_format_user(v, user) for v in value]
    return value


def load_raw_config() -> dict:
    path = get_config_path()
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f) or {}


def get_config() -> dict:
    """Load gilbreth.yaml, apply env overrides and resolve ``{user}`` placeholders."""
    cfg = copy.deepcopy(load_raw_config())
    cfg.setdefault("ssh", {})
    cfg.setdefault("paths", {})
    cfg.setdefault("slurm", {})
    cfg.setdefault("environment", {})
    cfg.setdefault("sync", {})

    user = os.environ.get(ENV_USER) or cfg["ssh"].get("user") or ""
    account = os.environ.get(ENV_ACCOUNT) or cfg["slurm"].get("account") or ""
    cfg["ssh"]["user"] = user
    cfg["slurm"]["account"] = account

    if user:
        cfg["paths"] = _format_user(cfg["paths"], user)
        cfg["environment"] = _format_user(cfg["environment"], user)
    return cfg


def require_user(cfg: dict) -> str:
    """Exit with a helpful message when the Purdue username is not configured."""
    user = cfg["ssh"].get("user")
    if not user:
        click.echo(
            f"Error: Purdue username not set. Export {ENV_USER}=<career_account> "
            f"or set ssh.user in {get_config_path()}",
            err=True,
        )
        sys.exit(1)
    return user


def require_account(cfg: dict, override: str | None = None) -> str:
    """Return the SLURM account to charge, exiting if none is configured."""
    account = override or cfg["slurm"].get("account")
    if not account:
        click.echo(
            f"Error: SLURM account not set. Gilbreth requires -A <account>. "
            f"Run 'gilbreth-info accounts' to list yours, then export {ENV_ACCOUNT}=<account> "
            f"or set slurm.account in {get_config_path()}",
            err=True,
        )
        sys.exit(1)
    return account


_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def safe_name(name: str) -> str:
    """Reduce a project / job name to [A-Za-z0-9_.-] so it is safe in paths, -J and shell commands."""
    cleaned = _NAME_RE.sub("_", name.strip()).strip("_.")
    if not cleaned:
        raise click.BadParameter(f"invalid name: {name!r}")
    return cleaned
