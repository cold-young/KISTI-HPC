"""Configuration loading, credential precedence, and point-of-use errors (AC-2).

Three things here are worth more than their line count.

``test_loader_never_raises_*`` pins **F-4/R19**: the loader has 31 call sites
across 11 modules and several of them are commands a brand-new user runs before
they know their Purdue id.  If ``get_rcac_config()`` ever starts raising, the
onboarding path dies and these are the tests that say so.

``test_credential_precedence`` pins **B1**: the shipped placeholder is truthy,
so the pre-port ``cfg.get("user") or os.environ[...]`` silently ignored the
environment variable.  The trailing-whitespace case is not hypothetical -- the
shipped YAML carried it.

``test_config_resolution_tier_*`` covers all three tiers of AC-2.8 in-process.
The third is also gated end-to-end by the clean-venv wheel install, which is
the only check that proves the packaged copies were really installed.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

import rcac_cli
from rcac_cli import config as config_mod
from rcac_cli.config import (
    ACCOUNT_ENV_VAR,
    CONFIG_DIR_ENV_VAR,
    PLACEHOLDERS,
    USER_ENV_VAR,
    CredentialError,
    get_rcac_config,
    require_account,
    require_user,
    resolve_config_path,
    resolve_path,
    scratch_root,
)

PKG_ROOT = Path(rcac_cli.__file__).resolve().parent
REPO_ROOT = PKG_ROOT.parent


def write_config(directory: Path, **overrides) -> Path:
    """Write a minimal but structurally real ``rcac.yaml`` into *directory*."""
    directory.mkdir(parents=True, exist_ok=True)
    doc = {
        "ssh": {
            "login_host": "gilbreth.rcac.purdue.edu",
            "user": "YOUR_PURDUE_ID",
            "port": 22,
            "control_persist": "4h",
            "control_path": "~/.ssh/sockets/%r@%h-%p",
            "server_alive_interval": 60,
            "server_alive_count_max": 10,
        },
        "slurm": {"account": "YOUR_QUEUE"},
        "paths": {"scratch": "/scratch/gilbreth/{user}"},
    }
    for dotted, value in overrides.items():
        section, _, key = dotted.partition("__")
        doc[section][key] = value
    path = directory / "rcac.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return path


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """A cwd with no ``configs/`` and no credential environment.

    Every test here starts from "nothing is set", so a passing assertion can
    never be an accident of the developer's own exported RCAC_USER.
    """
    monkeypatch.delenv(USER_ENV_VAR, raising=False)
    monkeypatch.delenv(ACCOUNT_ENV_VAR, raising=False)
    monkeypatch.delenv(CONFIG_DIR_ENV_VAR, raising=False)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    return work


# --------------------------------------------------------------------------
# AC-2.1 -- what ships
# --------------------------------------------------------------------------


def test_shipped_config_is_gilbreth_with_placeholders():
    """AC-2.1 -- the Gilbreth host, the Gilbreth scratch root, both placeholders."""
    doc = yaml.safe_load((REPO_ROOT / "configs" / "rcac.yaml").read_text())
    assert doc["ssh"]["login_host"] == "gilbreth.rcac.purdue.edu"
    assert doc["paths"]["scratch"] == "/scratch/gilbreth/{user}"
    assert doc["ssh"]["user"].strip() in PLACEHOLDERS
    assert doc["slurm"]["account"].strip() in PLACEHOLDERS


def test_shipped_config_declares_no_transfer_node():
    """Gilbreth documents no data-transfer node; Globus is the bulk path.

    Asserted on the parsed document rather than the text so it cannot be
    satisfied by commenting the key out.
    """
    doc = yaml.safe_load((REPO_ROOT / "configs" / "rcac.yaml").read_text())
    assert set(doc["ssh"]) == {
        "login_host",
        "user",
        "port",
        "control_persist",
        "control_path",
        "server_alive_interval",
        "server_alive_count_max",
    }


def test_packaged_configs_match_the_checkout_configs():
    """The two copies AC-2.8 requires must not drift.

    ``configs/`` is what a developer edits; ``rcac_cli/configs/`` is what a
    user installs.  Nothing but this assertion keeps them equal, and a silent
    divergence would mean the wheel behaves differently from the checkout.
    """
    for name in ("rcac.yaml", "defaults.yaml"):
        checkout = (REPO_ROOT / "configs" / name).read_bytes()
        packaged = (PKG_ROOT / "configs" / name).read_bytes()
        assert checkout == packaged, (
            f"configs/{name} and {PKG_ROOT.name}/configs/{name} have drifted. "
            f"Copy the checkout copy over the packaged one."
        )


# --------------------------------------------------------------------------
# AC-2.2 -- precedence (B1)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "env_value,config_value,expected,why",
    [
        (
            "envid",
            "YOUR_PURDUE_ID",
            "envid",
            "env over placeholder -- the case B1 broke: the placeholder is "
            "truthy, so `config or env` never consulted the environment",
        ),
        (
            "envid",
            "fileid",
            "envid",
            "env over a real config value (12-factor: env always wins)",
        ),
        (None, "fileid", "fileid", "config used when the env var is absent"),
        (None, "YOUR_PURDUE_ID", "", "both unset -> unresolved, not an error"),
        (
            None,
            "YOUR_ID  ",
            "",
            "placeholder membership is tested AFTER .strip() -- the shipped "
            "YAML really did carry trailing whitespace after the value",
        ),
        (
            "",
            "fileid",
            "fileid",
            "an empty env var is a placeholder member, not an override",
        ),
    ],
)
def test_credential_precedence(
    isolated, monkeypatch, env_value, config_value, expected, why
):
    """AC-2.2 -- env > non-placeholder config > unresolved."""
    write_config(isolated / "configs", ssh__user=config_value)
    if env_value is None:
        monkeypatch.delenv(USER_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(USER_ENV_VAR, env_value)

    assert get_rcac_config()["ssh"]["user"] == expected, why


@pytest.mark.parametrize(
    "env_value,config_value,expected",
    [
        ("envacct", "YOUR_QUEUE", "envacct"),
        (None, "fileacct", "fileacct"),
        (None, "YOUR_ACCOUNT", ""),
    ],
)
def test_account_precedence_is_the_same_rule(
    isolated, monkeypatch, env_value, config_value, expected
):
    """AC-2.2 -- ``RCAC_ACCOUNT`` resolves by the identical rule."""
    write_config(isolated / "configs", slurm__account=config_value)
    if env_value is None:
        monkeypatch.delenv(ACCOUNT_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(ACCOUNT_ENV_VAR, env_value)

    assert get_rcac_config()["slurm"]["account"] == expected


def test_placeholder_set_is_exactly_the_specified_one():
    """AC-2.2 names the set explicitly; drift here silently accepts a value."""
    assert PLACEHOLDERS == frozenset(
        {"", "YOUR_ID", "YOUR_PURDUE_ID", "YOUR_QUEUE", "YOUR_ACCOUNT", None}
    )


# --------------------------------------------------------------------------
# F-4 / R19 -- the loader must never raise
# --------------------------------------------------------------------------


def test_loader_never_raises_on_an_unresolved_user(isolated):
    """F-4 -- ``rcac-partition`` is run before the user knows their account.

    Raising here would make the first command a new user types throw, which is
    exactly the onboarding path AC-4.6 protects.  The loader instead leaves the
    unresolved state *observable*.
    """
    write_config(isolated / "configs")

    cfg = get_rcac_config()

    assert cfg["ssh"]["user"] == ""
    assert cfg["slurm"]["account"] == ""


def test_loader_leaves_paths_unformatted_when_the_user_is_unresolved(isolated):
    """B10 -- the loader must not emit a half-formatted path.

    The pre-port loader formatted ``paths`` only ``if user:``, so an
    unresolved user silently produced a literal ``{user}`` (or a ``//``) that
    travelled all the way into a rendered ``sbatch -o`` line.  Now the template
    form is preserved and only :func:`scratch_root` may resolve it.
    """
    write_config(isolated / "configs")

    cfg = get_rcac_config()

    assert cfg["paths"]["scratch"] == "/scratch/gilbreth/{user}"


# --------------------------------------------------------------------------
# AC-2.3 -- point of use is where it raises
# --------------------------------------------------------------------------


def test_require_user_names_both_the_env_var_and_the_config_path(isolated):
    """AC-2.3 -- the message must be actionable without reading the source."""
    write_config(isolated / "configs")
    cfg = get_rcac_config()

    with pytest.raises(CredentialError) as excinfo:
        require_user(cfg)

    message = str(excinfo.value)
    assert USER_ENV_VAR in message
    assert "ssh.user" in message
    assert str(isolated / "configs" / "rcac.yaml") in message
    assert excinfo.value.exit_code == 1


def test_require_account_names_both_the_env_var_and_the_config_path(isolated):
    """AC-2.3 -- same contract for the Slurm account."""
    write_config(isolated / "configs")
    cfg = get_rcac_config()

    with pytest.raises(CredentialError) as excinfo:
        require_account(cfg)

    message = str(excinfo.value)
    assert ACCOUNT_ENV_VAR in message
    assert "slurm.account" in message
    assert excinfo.value.exit_code == 1


def test_scratch_root_raises_at_point_of_use(isolated):
    """AC-3.17 -- the accessor, not the loader, is what refuses."""
    write_config(isolated / "configs")
    cfg = get_rcac_config()

    with pytest.raises(CredentialError):
        scratch_root(cfg)


def test_scratch_root_formats_and_has_no_leftover_template(isolated, monkeypatch):
    """AC-3.17 -- no ``{user}`` and no ``//`` survive a resolved path."""
    write_config(isolated / "configs")
    monkeypatch.setenv(USER_ENV_VAR, "testid")

    root = scratch_root(get_rcac_config())

    assert root == "/scratch/gilbreth/testid"
    assert "{user}" not in root
    assert "//" not in root


# --------------------------------------------------------------------------
# AC-2.13 -- paths.depot is optional
# --------------------------------------------------------------------------


def test_optional_depot_key_absence_is_tolerated(isolated, monkeypatch):
    """AC-2.13 -- ``/depot/<group>`` is undocumented for Gilbreth.

    Asking for it with a default must not raise; asking for it without one
    must raise a ``KeyError`` naming the key rather than producing ``None``.
    """
    write_config(isolated / "configs")
    monkeypatch.setenv(USER_ENV_VAR, "testid")
    cfg = get_rcac_config()

    assert "depot" not in cfg["paths"]
    assert resolve_path(cfg, "depot", "/depot/{user}") == "/depot/testid"
    with pytest.raises(KeyError, match="paths.depot"):
        resolve_path(cfg, "depot")


# --------------------------------------------------------------------------
# AC-2.8 -- three-level resolution, one test per level
# --------------------------------------------------------------------------


def test_config_resolution_tier_1_env_dir_wins(isolated, monkeypatch):
    """Tier 1: ``$RCAC_CONFIG_DIR`` beats a ``./configs/`` that also exists."""
    write_config(isolated / "configs", ssh__login_host="from-cwd.invalid")
    override = isolated / "elsewhere"
    write_config(override, ssh__login_host="from-env-dir.invalid")
    monkeypatch.setenv(CONFIG_DIR_ENV_VAR, str(override))

    assert resolve_config_path("rcac") == override / "rcac.yaml"
    assert get_rcac_config()["ssh"]["login_host"] == "from-env-dir.invalid"


def test_config_resolution_tier_2_cwd_configs(isolated):
    """Tier 2: ``./configs/`` beats the packaged copy."""
    write_config(isolated / "configs", ssh__login_host="from-cwd.invalid")

    assert resolve_config_path("rcac") == isolated / "configs" / "rcac.yaml"
    assert get_rcac_config()["ssh"]["login_host"] == "from-cwd.invalid"


def test_config_resolution_tier_3_packaged(isolated):
    """Tier 3: with no override and no ``./configs/``, the wheel's copy answers.

    This is the tier that was vacuous before ``rcac_cli/configs/`` existed --
    the package-data glob matched nothing, so an installed rcac-cli had no
    config at all and this assertion would have had nothing to find (D-5).
    """
    assert not (isolated / "configs").exists()

    resolved = resolve_config_path("rcac")

    assert resolved == PKG_ROOT / "configs" / "rcac.yaml"
    assert get_rcac_config()["ssh"]["login_host"] == "gilbreth.rcac.purdue.edu"


def test_config_resolution_reports_every_tier_it_tried(isolated, monkeypatch):
    """A miss must say where it looked, or the three tiers are undebuggable."""
    monkeypatch.setenv(CONFIG_DIR_ENV_VAR, str(isolated / "nope"))

    with pytest.raises(FileNotFoundError) as excinfo:
        resolve_config_path("no_such_config")

    message = str(excinfo.value)
    assert str(isolated / "nope") in message
    assert str(isolated / "configs") in message
    assert str(PKG_ROOT / "configs") in message


def test_config_search_path_order_is_the_documented_one(isolated, monkeypatch):
    """AC-2.8 names the order; this pins it against reordering."""
    monkeypatch.setenv(CONFIG_DIR_ENV_VAR, str(isolated / "override"))

    assert config_mod.config_search_path() == [
        isolated / "override",
        isolated / "configs",
        PKG_ROOT / "configs",
    ]


def test_config_search_path_omits_the_override_tier_when_unset(isolated):
    """Without the env var there are two tiers, not an empty first one."""
    assert config_mod.config_search_path() == [
        isolated / "configs",
        PKG_ROOT / "configs",
    ]


def test_partial_override_dir_falls_through_per_file(isolated, monkeypatch):
    """Resolution is per file, so an override holding only one config works."""
    override = isolated / "only-rcac"
    write_config(override, ssh__login_host="from-env-dir.invalid")
    monkeypatch.setenv(CONFIG_DIR_ENV_VAR, str(override))

    assert resolve_config_path("rcac") == override / "rcac.yaml"
    assert resolve_config_path("defaults") == PKG_ROOT / "configs" / "defaults.yaml"


def test_malformed_yaml_surfaces_as_a_yaml_error(isolated):
    """A broken config must not silently resolve to an empty dict."""
    (isolated / "configs").mkdir()
    (isolated / "configs" / "rcac.yaml").write_text(
        textwrap.dedent(
            """
            ssh:
              user: "x
            """
        ),
        encoding="utf-8",
    )
    with pytest.raises(yaml.YAMLError):
        get_rcac_config()


def test_dry_run_launch_succeeds_with_no_depot_key(isolated, monkeypatch):
    """AC-2.13, end to end: the optional key's absence must not reach the CLI.

    A unit test on :func:`resolve_path` proves the accessor tolerates it; this
    proves nothing between the loader and a rendered script quietly requires
    it.
    """
    from click.testing import CliRunner

    from rcac_cli.experiment import cli

    write_config(isolated / "configs")
    (isolated / "configs" / "defaults.yaml").write_text(
        (PKG_ROOT / "configs" / "defaults.yaml").read_text(), encoding="utf-8"
    )
    monkeypatch.setenv(USER_ENV_VAR, "testid")
    monkeypatch.setenv(ACCOUNT_ENV_VAR, "testqueue")
    # A resolved user is all `launch` needs to start probing the cluster (C5),
    # and this test is about path resolution, not about what Gilbreth answers.
    monkeypatch.setenv("RCAC_NO_DISCOVER", "1")
    experiment = isolated / "experiment_simple.yaml"
    experiment.write_text(
        (REPO_ROOT / "examples" / "experiment_simple.yaml").read_text(),
        encoding="utf-8",
    )

    assert "depot" not in yaml.safe_load(resolve_config_path("rcac").read_text())["paths"]

    result = CliRunner().invoke(cli, ["launch", str(experiment), "--dry-run"])

    assert result.exit_code == 0, result.output
