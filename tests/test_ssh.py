"""The single SSH option builder, and the two rsync calls that bypassed it.

**AC-2.11 (B12)** was five hand-rolled ControlMaster strings in three shapes
across twelve call sites.  The interesting assertion is not that
:mod:`rcac_cli.ssh` works -- it is the *text scan* proving no other module
builds an ssh option any more, since a sixth hand-rolled string would otherwise
pass every behavioural test in this file.

**AC-2.9 (B4)** is the pair of ``rsync`` invocations that carried no ``-e`` at
all.  rsync then opens its own SSH connection, which on an MFA cluster means a
second prompt in the middle of an upload.  Both are asserted by monkeypatching
``subprocess.run`` and reading the argv, because the bug is invisible in the
return value.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from click.testing import CliRunner

import rcac_cli
from rcac_cli import container as container_mod
from rcac_cli import results as results_mod
from rcac_cli.ssh import (
    GLOBUS_COLLECTION,
    GLOBUS_ENDPOINT,
    GLOBUS_THRESHOLD_BYTES,
    globus_guidance,
    ssh_opt_argv,
    ssh_opt_str,
    ssh_options,
    ssh_target,
    transfer_host,
)

PKG_ROOT = Path(rcac_cli.__file__).resolve().parent

CFG = {
    "ssh": {
        "login_host": "gilbreth.rcac.purdue.edu",
        "user": "testid",
        "port": 22,
        "control_persist": "4h",
        "control_path": "~/.ssh/sockets/%r@%h-%p",
        "server_alive_interval": 60,
        "server_alive_count_max": 10,
    },
    "slurm": {"account": "testqueue"},
    "paths": {"scratch": "/scratch/gilbreth/{user}"},
}


# --------------------------------------------------------------------------
# The two entry points
# --------------------------------------------------------------------------


def test_ssh_opt_argv_carries_the_full_option_set_and_the_port():
    argv = ssh_opt_argv(CFG)

    assert argv == [
        "-o", "ControlMaster=auto",
        "-o", "ControlPath=~/.ssh/sockets/%r@%h-%p",
        "-o", "ControlPersist=4h",
        "-o", "ServerAliveInterval=60",
        "-o", "ServerAliveCountMax=10",
        "-p", "22",
    ]


def test_ssh_opt_str_is_the_same_options_in_the_rsync_encoding():
    """The two encodings must not drift -- that drift was B12."""
    assert ssh_opt_str(CFG) == "ssh " + " ".join(ssh_opt_argv(CFG))


def test_ssh_opt_str_is_unquoted_because_rsync_does_not_unquote():
    """rsync splits ``-e`` on whitespace and does no quote processing.

    A ``shlex``-quoted string reaches ssh with literal quote characters still
    in the argument, which fails with an unhelpful error deep inside a
    transfer.  This pins the deliberate absence of quoting.
    """
    assert "'" not in ssh_opt_str(CFG)
    assert '"' not in ssh_opt_str(CFG)


def test_ssh_target_is_user_at_login_host():
    assert ssh_target(CFG) == "testid@gilbreth.rcac.purdue.edu"


def test_transfer_host_is_the_login_host():
    """Gilbreth documents no data-transfer node (R10)."""
    assert transfer_host(CFG) == CFG["ssh"]["login_host"]


def test_option_order_is_stable():
    """Stable order keeps the argv assertions in this suite meaningful."""
    assert [key for key, _ in ssh_options(CFG)] == [
        "ControlMaster",
        "ControlPath",
        "ControlPersist",
        "ServerAliveInterval",
        "ServerAliveCountMax",
    ]


# --------------------------------------------------------------------------
# AC-2.11 -- the text scan that closes B12 as a class
# --------------------------------------------------------------------------

#: The ``-o Key=Value`` command-line form.  Anywhere but ``ssh.py`` this is a
#: hand-rolled option and a future drift site.
OPTION_FORM = re.compile(r"ControlMaster=|ControlPath=|ControlPersist=|ServerAlive\w+=")

#: The SSH *config-file* form as it appears in source: a quoted literal whose
#: content is four spaces, a directive name, a space, then the value.  No
#: ``-o``, no ``=``.  A different grammar entirely, which is why converting it
#: was never possible and the conversion count is 12 rather than 14.
DIRECTIVE_FORM = re.compile(
    r"""["']    (ControlMaster|ControlPath|ControlPersist|ServerAlive\w+) """
)


def package_sources() -> list[Path]:
    return sorted(PKG_ROOT.glob("*.py"))


def test_no_module_outside_ssh_builds_an_ssh_option():
    """AC-2.11 -- one builder, twelve call sites, zero hand-rolled strings."""
    offenders = []
    for path in package_sources():
        if path.name == "ssh.py":
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            if OPTION_FORM.search(line):
                offenders.append(f"{path.name}:{lineno}: {line.strip()}")
    assert not offenders, (
        "these lines build an ssh -o option outside rcac_cli/ssh.py; route "
        "them through ssh_opt_argv() or ssh_opt_str():\n  "
        + "\n  ".join(offenders)
    )


def test_ssh_option_builder_really_produces_the_scanned_form():
    """Guard the premise: if ssh.py stopped emitting ``-o K=V``, the scan above
    would pass vacuously against a codebase that had gone back to hand-rolling.
    """
    assert OPTION_FORM.search(ssh_opt_str(CFG))


def test_ssh_config_file_directives_are_exempt_and_live_only_in_connect():
    """AC-2.11's exempt sites, pinned rather than left as prose.

    ``connect._generate_ssh_config`` writes ``    ControlPath <path>`` into
    ``~/.ssh/config``.  That is a config-file directive, not an ssh option: no
    ``-o``, space-separated, indented.  An option builder cannot produce it,
    which is why the conversion count is 12 and not 14.  The assertion is that
    they exist *only* there.
    """
    found: dict[str, int] = {}
    for path in package_sources():
        count = sum(
            1
            for line in path.read_text().splitlines()
            if DIRECTIVE_FORM.search(line)
        )
        if count:
            found[path.name] = count

    assert set(found) == {"connect.py"}, (
        f"SSH config-file directives appeared outside connect.py: {found}"
    )


def test_exactly_one_control_path_directive_survives():
    """Two config-file ControlPath lines existed; the second fed the removed
    data-transfer stanza and went with it (AC-2.12).
    """
    from rcac_cli.connect import _generate_ssh_config

    directives = [
        line
        for line in _generate_ssh_config(CFG).splitlines()
        if line.startswith("    ControlPath ")
    ]
    assert len(directives) == 1, directives


# --------------------------------------------------------------------------
# AC-2.5 -- Globus guidance
# --------------------------------------------------------------------------


def test_globus_guidance_is_silent_below_the_threshold():
    assert globus_guidance(GLOBUS_THRESHOLD_BYTES - 1) == ""


def test_globus_guidance_names_the_endpoint_and_the_collection():
    text = globus_guidance(GLOBUS_THRESHOLD_BYTES)
    assert GLOBUS_ENDPOINT in text
    assert GLOBUS_COLLECTION in text


# --------------------------------------------------------------------------
# AC-2.9 (B4) -- both rsync invocations must ride the ControlMaster
# --------------------------------------------------------------------------


class RecordingRun:
    """A ``subprocess.run`` stand-in that records argv and reports success."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def __call__(self, cmd, *args, **kwargs):
        self.calls.append(list(cmd))

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return Result()

    def rsync_calls(self) -> list[list[str]]:
        return [c for c in self.calls if c and c[0] == "rsync"]


def assert_rides_the_control_master(argv: list[str]) -> None:
    """AC-2.9 -- ``-e`` present, and its value carries a ControlPath."""
    assert "-e" in argv, (
        "this rsync has no -e, so it opens its own SSH connection and costs "
        f"the user a second MFA prompt: {argv}"
    )
    remote_shell = argv[argv.index("-e") + 1]
    assert "ControlPath" in remote_shell, (
        f"rsync -e does not reference a ControlPath: {remote_shell!r}"
    )


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("RCAC_USER", "testid")
    monkeypatch.setenv("RCAC_ACCOUNT", "testqueue")
    monkeypatch.delenv("RCAC_CONFIG_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_sync_wandb_rsync_rides_the_control_master(env, monkeypatch):
    """B4 site 1 -- ``rcac-results sync-wandb``."""
    recorder = RecordingRun()
    monkeypatch.setattr(results_mod.subprocess, "run", recorder)

    result = CliRunner().invoke(results_mod.cli, ["sync-wandb", "demo"])

    assert result.exit_code == 0, result.output
    calls = recorder.rsync_calls()
    assert len(calls) == 1, calls
    assert_rides_the_control_master(calls[0])


def test_container_push_rsync_rides_the_control_master(env, monkeypatch):
    """B4 site 2 -- ``rcac-container push``."""
    image = env / "demo.sif"
    image.write_bytes(b"not really an image")
    recorder = RecordingRun()
    monkeypatch.setattr(container_mod.subprocess, "run", recorder)

    result = CliRunner().invoke(container_mod.cli, ["push", str(image)])

    assert result.exit_code == 0, result.output
    calls = recorder.rsync_calls()
    assert len(calls) == 1, calls
    assert_rides_the_control_master(calls[0])


def test_container_push_targets_the_login_host_not_a_transfer_node(env, monkeypatch):
    """R10 -- the transfer-node key is gone; this must not have become a KeyError."""
    image = env / "demo.sif"
    image.write_bytes(b"x")
    recorder = RecordingRun()
    monkeypatch.setattr(container_mod.subprocess, "run", recorder)

    result = CliRunner().invoke(container_mod.cli, ["push", str(image)])

    assert result.exit_code == 0, result.output
    destination = recorder.rsync_calls()[0][-1]
    assert destination.startswith("testid@gilbreth.rcac.purdue.edu:")


# --------------------------------------------------------------------------
# AC-2.5 -- push-data survives the removal of the transfer node
# --------------------------------------------------------------------------


def test_push_data_works_without_a_transfer_node(env, monkeypatch):
    """AC-2.5 / R10 -- removing the config key must not leave a KeyError.

    ``push-data`` used to default to ``--via dm``, which read a key that no
    longer exists.  The flag is gone with the node: Gilbreth documents exactly
    one direct route.
    """
    from rcac_cli import sync as sync_mod

    payload = env / "dataset"
    payload.mkdir()
    (payload / "a.bin").write_bytes(b"0" * 1024)
    recorder = RecordingRun()
    monkeypatch.setattr(sync_mod.subprocess, "run", recorder)

    result = CliRunner().invoke(sync_mod.cli, ["push-data", str(payload)])

    assert result.exit_code == 0, result.output
    argv = recorder.rsync_calls()[0]
    assert_rides_the_control_master(argv)
    assert argv[-1] == "testid@gilbreth.rcac.purdue.edu:/scratch/gilbreth/testid/data/"


def test_push_data_stays_quiet_about_globus_for_a_small_transfer(env, monkeypatch):
    """Guidance on every push would be noise, and noise gets filtered out."""
    from rcac_cli import sync as sync_mod

    payload = env / "small"
    payload.mkdir()
    (payload / "a.bin").write_bytes(b"0" * 1024)
    monkeypatch.setattr(sync_mod.subprocess, "run", RecordingRun())

    result = CliRunner().invoke(sync_mod.cli, ["push-data", str(payload)])

    assert result.exit_code == 0, result.output
    assert GLOBUS_ENDPOINT not in result.output


def test_push_data_points_at_globus_above_the_threshold(env, monkeypatch):
    """AC-2.5 -- the bulk path is Globus, not rsync over a login node.

    The size is faked rather than written: the threshold is 10 GiB and the
    assertion is about the branch, not about the filesystem.
    """
    from rcac_cli import sync as sync_mod

    payload = env / "big"
    payload.mkdir()
    (payload / "a.bin").write_bytes(b"0")
    monkeypatch.setattr(sync_mod.subprocess, "run", RecordingRun())
    monkeypatch.setattr(sync_mod, "_tree_size", lambda _p: GLOBUS_THRESHOLD_BYTES)

    result = CliRunner().invoke(sync_mod.cli, ["push-data", str(payload)])

    assert result.exit_code == 0, result.output
    assert GLOBUS_ENDPOINT in result.output
    assert GLOBUS_COLLECTION in result.output


def test_push_data_has_no_transfer_node_flag(env):
    """The flag and the key had to die together, or one would outlive the other."""
    from rcac_cli import sync as sync_mod

    result = CliRunner().invoke(sync_mod.cli, ["push-data", "--help"])

    assert result.exit_code == 0
    assert "--via" not in result.output
