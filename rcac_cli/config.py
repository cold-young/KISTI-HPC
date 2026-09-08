"""Configuration loader and credential resolution for rcac-cli.

Two properties of this module are load-bearing and easy to break:

**The loader never raises on an unresolved user (F-4 / R19).**
:func:`get_rcac_config` has 31 call sites across 11 modules, and several of
them -- ``rcac-partition`` above all -- are commands a brand-new user runs
*before* they know their Purdue id or their Slurm account.  Raising here would
make onboarding throw.  So the loader leaves ``cfg["ssh"]["user"] == ""``
observable and returns ``paths`` *unformatted*, and the error is raised at
point of use by :func:`require_user`, :func:`require_account`,
:func:`resolve_path` and :func:`scratch_root` (AC-2.3, AC-3.17).

**The environment wins over the file (B1).**
``configs/rcac.yaml`` ships with the placeholder ``user: "YOUR_PURDUE_ID"``,
which is *truthy*.  The pre-port loader wrote ``cfg.get("user") or
os.environ.get(...)``, so the placeholder silently shadowed the environment
variable and it was never consulted.  Placeholder membership is tested after
``.strip()`` because the shipped YAML carried trailing whitespace after the
value.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import click
import yaml

#: Values that mean "the user has not filled this in yet" (AC-2.2).  Compared
#: after ``.strip()``.  ``None`` is a member so a missing key resolves the same
#: way an empty one does.
PLACEHOLDERS = frozenset(
    {"", "YOUR_ID", "YOUR_PURDUE_ID", "YOUR_QUEUE", "YOUR_ACCOUNT", None}
)

#: Environment variable / config path pairs, quoted verbatim in the error
#: messages so the message is actionable without reading the source.
USER_ENV_VAR = "RCAC_USER"
USER_CONFIG_KEY = "ssh.user"
ACCOUNT_ENV_VAR = "RCAC_ACCOUNT"
ACCOUNT_CONFIG_KEY = "slurm.account"

#: Overrides the config search path entirely as its first tier (AC-2.8).
CONFIG_DIR_ENV_VAR = "RCAC_CONFIG_DIR"


class CredentialError(click.ClickException):
    """An unresolved credential, raised at point of use.

    Subclasses :class:`click.ClickException` so every one of the 12 console
    scripts exits 1 with the message on **stderr** (AC-2.3) without each
    command having to re-implement that.
    """

    exit_code = 1


def is_placeholder(value: object) -> bool:
    """True if *value* is unset or is still one of the shipped placeholders."""
    if value is None:
        return True
    return str(value).strip() in PLACEHOLDERS


def _resolve_credential(env_var: str, config_value: object) -> str:
    """Environment > non-placeholder config value > unresolved (AC-2.2)."""
    env_value = os.environ.get(env_var)
    if not is_placeholder(env_value):
        return str(env_value).strip()
    if not is_placeholder(config_value):
        return str(config_value).strip()
    return ""


# --------------------------------------------------------------------------
# AC-2.8 -- three-level config resolution
# --------------------------------------------------------------------------


def config_search_path() -> list[Path]:
    """The three config directories, in precedence order (AC-2.8).

    1. ``$RCAC_CONFIG_DIR``  -- explicit override, absent unless set.
    2. ``./configs/``        -- the working checkout.
    3. ``rcac_cli/configs/`` -- the copies shipped inside the wheel.  This tier
       is the only one that exists after ``pip install rcac-cli`` from a
       directory that is not a checkout, so it must be populated in the
       package or the package-data glob matches nothing (R2 / D-5).
    """
    search: list[Path] = []
    override = os.environ.get(CONFIG_DIR_ENV_VAR, "").strip()
    if override:
        search.append(Path(override).expanduser())
    search.append(Path.cwd() / "configs")
    search.append(Path(__file__).resolve().parent / "configs")
    return search


def resolve_config_path(name: str) -> Path:
    """First existing ``<name>.yaml`` across :func:`config_search_path`."""
    tried: list[str] = []
    for directory in config_search_path():
        candidate = directory / f"{name}.yaml"
        tried.append(str(candidate))
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Config {name!r} not found. Looked in:\n  " + "\n  ".join(tried)
    )


def load_config(name: str = "rcac") -> dict:
    """Load a YAML config file from the resolved config directory."""
    with open(resolve_config_path(name)) as f:
        return yaml.safe_load(f) or {}


def get_rcac_config() -> dict:
    """Load ``rcac.yaml`` and resolve credentials.

    **This function does not raise on an unresolved user or account.**  It
    normalises ``cfg["ssh"]["user"]`` and ``cfg["slurm"]["account"]`` to ``""``
    when they are unresolved and leaves ``cfg["paths"]`` in its *unformatted*
    ``{user}`` template form.  Use :func:`resolve_path` or :func:`scratch_root`
    to get a usable path; those raise (F-4).
    """
    cfg = load_config("rcac")
    ssh = cfg.setdefault("ssh", {})
    slurm = cfg.setdefault("slurm", {})
    ssh["user"] = _resolve_credential(USER_ENV_VAR, ssh.get("user"))
    slurm["account"] = _resolve_credential(ACCOUNT_ENV_VAR, slurm.get("account"))
    return cfg


def get_defaults() -> dict:
    """Load ``defaults.yaml``."""
    return load_config("defaults")


# --------------------------------------------------------------------------
# AC-2.3 -- point-of-use accessors.  These are the ones that raise.
# --------------------------------------------------------------------------


def require_user(cfg: dict) -> str:
    """The resolved Purdue id, or raise an actionable :class:`CredentialError`."""
    user = cfg.get("ssh", {}).get("user") or ""
    if not user:
        raise CredentialError(
            f"{USER_ENV_VAR} is not set. Either export it "
            f"(`export {USER_ENV_VAR}=<your Purdue career account id>`, the "
            f"recommended path -- it needs no file edit) or replace the "
            f"placeholder at `{USER_CONFIG_KEY}` in {resolve_config_path('rcac')}."
        )
    return user


def require_account(cfg: dict) -> str:
    """The resolved Slurm account, or raise an actionable :class:`CredentialError`."""
    account = cfg.get("slurm", {}).get("account") or ""
    if not account:
        raise CredentialError(
            f"{ACCOUNT_ENV_VAR} is not set. Gilbreth requires an account on "
            f"every submission. Run `slist` on the cluster to see yours, then "
            f"either export it (`export {ACCOUNT_ENV_VAR}=<account>`) or "
            f"replace the placeholder at `{ACCOUNT_CONFIG_KEY}` in "
            f"{resolve_config_path('rcac')}."
        )
    return account


def resolve_path(cfg: dict, key: str, default: Optional[str] = None) -> str:
    """Format ``cfg['paths'][key]`` with the resolved user (AC-3.17).

    Raises :class:`CredentialError` when the user is unresolved, and
    :class:`KeyError` when *key* is absent and no *default* is given -- so an
    optional key such as ``paths.depot`` is asked for with a default and its
    absence is tolerated (AC-2.13).
    """
    paths = cfg.get("paths") or {}
    template = paths.get(key, default)
    if template is None:
        raise KeyError(
            f"paths.{key} is not set in {resolve_config_path('rcac')} and no "
            f"default was supplied"
        )
    return str(template).format(user=require_user(cfg))


def scratch_root(cfg: dict) -> str:
    """The user's scratch root, formatted and validated (B10, AC-3.17).

    This is the accessor that closes B10: the loader deliberately does not
    format ``paths``, so a literal ``{user}`` or a ``//`` can only reach a
    rendered ``sbatch -o`` line by bypassing this function.
    """
    return resolve_path(cfg, "scratch", "/scratch/gilbreth/{user}").rstrip("/")
