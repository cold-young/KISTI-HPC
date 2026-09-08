"""``notify.yaml`` resolves through the three config tiers (C7 carried defect).

``rcac_cli.notify`` was the one module that never adopted the resolution C2
built.  It looked its config up as ``Path(__file__).parent.parent / "configs" /
"notify.yaml"``, which is the repo root only while running from a checkout:
after ``pip install`` the same expression points inside ``site-packages``, so a
user's ``./configs/notify.yaml`` was invisible and ``rcac-notify setup`` tried
to write into the installed package.

The tests below pin both halves.  ``test_load_*`` covers reading -- tier 1, tier
2 and the no-config case -- and ``test_setup_writes_*`` covers writing, which is
the half that would have failed hardest, because a wheel installed into a
system prefix is not writable at all.

``test_lookup_is_cwd_relative_not_module_relative`` is the regression proper.
The old expression is a constant: it answers the same path from every working
directory.  Asserting that two different working directories resolve to two
different files is therefore something the pre-fix loader cannot do, and it
needs no write into the installed package to say so.  ``test_load_tier_2``
fails against the old code for the same reason.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from rcac_cli import notify as notify_mod
from rcac_cli.config import CONFIG_DIR_ENV_VAR
from rcac_cli.notify import (
    _load_notify_config,
    cli,
    notify_config_path,
    notify_config_target,
)

WEBHOOK = "https://hooks.slack.invalid/services/T000/B000/xxxx"


def write_notify(directory: Path, webhook: str = WEBHOOK) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "notify.yaml"
    path.write_text(
        yaml.safe_dump({"slack": {"webhook_url": webhook}}), encoding="utf-8"
    )
    return path


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """A cwd with no ``configs/`` and no ``$RCAC_CONFIG_DIR``."""
    monkeypatch.delenv(CONFIG_DIR_ENV_VAR, raising=False)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    return work


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------


def test_load_tier_1_env_dir_wins(isolated, monkeypatch):
    """``$RCAC_CONFIG_DIR`` beats a ``./configs/`` that also exists."""
    write_notify(isolated / "configs", webhook="from-cwd")
    override = isolated / "elsewhere"
    write_notify(override, webhook="from-env-dir")
    monkeypatch.setenv(CONFIG_DIR_ENV_VAR, str(override))

    assert notify_config_path() == override / "notify.yaml"
    assert _load_notify_config()["slack"]["webhook_url"] == "from-env-dir"


def test_load_tier_2_cwd_configs(isolated):
    """``./configs/notify.yaml`` is found from the working directory."""
    expected = write_notify(isolated / "configs")

    assert notify_config_path() == expected
    assert _load_notify_config()["slack"]["webhook_url"] == WEBHOOK


def test_load_returns_empty_dict_when_no_tier_holds_one(isolated):
    """No config anywhere is not an error -- ``notify.yaml`` is optional.

    Nothing ships a packaged ``notify.yaml``, so this is also the state a fresh
    ``pip install`` is in, and it is the state ``rcac-notify test`` has to
    survive in order to print its "not configured" message.
    """
    assert notify_config_path() is None
    assert _load_notify_config() == {}


def test_lookup_is_cwd_relative_not_module_relative(isolated, monkeypatch):
    """Two working directories, two configs -- which the old constant cannot do.

    ``Path(__file__).parent.parent / "configs" / "notify.yaml"`` evaluates to
    one fixed path no matter where the user stands, so under the pre-fix loader
    both halves of this test would resolve to the same file (and, from anywhere
    outside a checkout, to a file inside the installed package).
    """
    first = write_notify(isolated / "one" / "configs", webhook="from-one")
    second = write_notify(isolated / "two" / "configs", webhook="from-two")

    monkeypatch.chdir(isolated / "one")
    assert notify_config_path() == first
    assert _load_notify_config()["slack"]["webhook_url"] == "from-one"

    monkeypatch.chdir(isolated / "two")
    assert notify_config_path() == second
    assert _load_notify_config()["slack"]["webhook_url"] == "from-two"


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------


def test_setup_writes_into_the_cwd_configs_directory(isolated):
    """``rcac-notify setup`` creates ``./configs/notify.yaml``, not a package file."""
    result = CliRunner().invoke(cli, ["setup"])

    assert result.exit_code == 0, result.output
    written = isolated / "configs" / "notify.yaml"
    assert written.is_file()
    assert notify_config_target() == written
    doc = yaml.safe_load(written.read_text(encoding="utf-8"))
    assert doc["slack"]["webhook_url"] == ""
    assert doc["email"]["to"] == ""

    package_dir = Path(notify_mod.__file__).resolve().parent
    assert not (package_dir / "configs" / "notify.yaml").exists(), (
        "setup wrote into the installed package; that directory is read-only "
        "in a real installation"
    )


def test_setup_honours_the_config_dir_override(isolated, monkeypatch):
    """``$RCAC_CONFIG_DIR`` is the write target too, and is created if missing."""
    override = isolated / "elsewhere"
    monkeypatch.setenv(CONFIG_DIR_ENV_VAR, str(override))

    result = CliRunner().invoke(cli, ["setup"])

    assert result.exit_code == 0, result.output
    assert (override / "notify.yaml").is_file()


def test_setup_is_idempotent_and_reports_the_resolved_path(isolated):
    """A second ``setup`` refuses, naming the file it found rather than a guess."""
    existing = write_notify(isolated / "configs")

    result = CliRunner().invoke(cli, ["setup"])

    assert result.exit_code == 0, result.output
    assert str(existing) in result.output
    # And it did not overwrite the user's webhook with the template's "".
    assert yaml.safe_load(existing.read_text(encoding="utf-8"))["slack"][
        "webhook_url"
    ] == WEBHOOK


def test_test_command_names_the_resolved_path_when_unconfigured(isolated):
    """The "not configured" message points at a path the user can act on."""
    result = CliRunner().invoke(cli, ["test", "--method", "slack"])

    assert result.exit_code == 0, result.output
    assert str(notify_config_target()) in result.output
    assert "rcac-notify setup" in result.output
