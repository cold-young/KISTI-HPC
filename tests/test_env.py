"""The `conda-env-mod` environment path (AC-5.1, AC-5.1b, AC-5.2, AC-5.8).

Three of these criteria are about *emitted text*, which is the whole product
here: ``rcac-env`` does not manage an environment so much as tell the user and
their job script exactly which module loads reproduce one.  A sequence that is
right in spirit and wrong in order is a job that dies at line one, and the
failure arrives minutes after submission with a message about a module name the
user never typed.

``rcac-env activate`` in particular had no output contract at all until
AC-5.1b: the plan said only what it *deletes*.  That left one of the twelve
console scripts free to print anything, including the explanatory chatter that
makes ``$(rcac-env activate NAME)`` produce a syntax error rather than an
environment.  Its contract is now two lines, and the test below asserts the
whole of stdout, not a substring of it.

Nothing here contacts a cluster: every remote call is captured.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from rcac_cli.env import (
    ANACONDA_MODULE,
    CONDA_ENV_NAMESPACE,
    GPU_MODULES,
    MODULE_QUERIES,
    USE_OWN_MODULE,
    _resolve_py_version,
    activate_commands,
    cli as env_cli,
    conda_env_module,
    setup_commands,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

CLI_ENV = {
    "RCAC_USER": "testid",
    "RCAC_ACCOUNT": "testqueue",
    "RCAC_CONFIG_DIR": str(REPO_ROOT / "configs"),
}

#: A whole `ssh:` block, because `_run_remote` builds a real ControlMaster
#: argv (AC-2.9) and a partial config would fail on a missing key rather than
#: on the thing under test.
CONFIG = {
    "ssh": {
        "user": "testid",
        "login_host": "gilbreth.rcac.purdue.edu",
        "port": 22,
        "control_persist": "4h",
        "control_path": "~/.ssh/sockets/%r@%h-%p",
        "server_alive_interval": 60,
        "server_alive_count_max": 10,
    },
    "paths": {"scratch": "/scratch/gilbreth/{user}"},
}


class FakeRemote:
    """Captures every remote command and replays a canned result."""

    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.sent: list[str] = []

    def __call__(self, cmd, *args, **kwargs):
        self.sent.append(cmd[-1])
        return subprocess.CompletedProcess(
            cmd, self.returncode, stdout=self.stdout, stderr=self.stderr
        )


# ==========================================================================
# AC-5.1 -- the documented creation sequence, in order
# ==========================================================================


def test_setup_sequence_is_exactly_the_documented_one() -> None:
    assert setup_commands("myenv", "3.12") == [
        "module purge",
        f"module load {ANACONDA_MODULE}",
        "conda-env-mod create -n myenv",
        f"module load {USE_OWN_MODULE}",
        f"module load {CONDA_ENV_NAMESPACE}/myenv-py3.12",
    ]


def test_gpu_modules_load_before_the_environment_is_created() -> None:
    """Order, not membership: `cuda`/`cudnn` after `anaconda`, before `create`.

    ``conda-env-mod`` bakes the loaded module set into the environment it
    generates, so a GPU module loaded *after* the create call is a module the
    environment does not know about.
    """
    commands = setup_commands("myenv", "3.12", with_gpu=True)
    load_gpu = commands.index("module load " + " ".join(GPU_MODULES))
    assert commands.index(f"module load {ANACONDA_MODULE}") < load_gpu
    assert load_gpu < commands.index("conda-env-mod create -n myenv")


def test_setup_prints_the_sequence_it_is_about_to_run(monkeypatch, tmp_path) -> None:
    """AC-5.1 on the real command, not merely on the helper."""
    remote = FakeRemote(stdout="done")
    monkeypatch.setattr(subprocess, "run", remote)
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(
        env_cli, ["setup", "--name", "myenv", "--py", "3.12"], env=CLI_ENV
    )
    assert result.exit_code == 0, result.output

    emitted = [
        line
        for line in result.output.splitlines()
        if line.startswith(("module ", "conda-env-mod "))
    ]
    assert emitted == setup_commands("myenv", "3.12"), result.output


def test_setup_sends_the_sequence_to_the_cluster(monkeypatch, tmp_path) -> None:
    """Printing the right thing while running a different one is the failure."""
    remote = FakeRemote(stdout="done")
    monkeypatch.setattr(subprocess, "run", remote)
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(
        env_cli, ["setup", "--name", "myenv", "--py", "3.12"], env=CLI_ENV
    )
    assert result.exit_code == 0, result.output
    assert " && ".join(setup_commands("myenv", "3.12")) in "\n".join(remote.sent)


# ==========================================================================
# AC-5.1b -- activate's output contract
# ==========================================================================


def test_activate_emits_exactly_two_module_loads() -> None:
    """The "and nothing else" half, asserted on the whole of stdout.

    `$(rcac-env activate NAME)` is a documented use, so a single stray line of
    explanation -- a banner, a blank line, a "# use in SLURM scripts" comment --
    is not cosmetic. It is a shell error.
    """
    result = CliRunner().invoke(
        env_cli, ["activate", "myenv", "--py", "3.12"], env=CLI_ENV
    )
    assert result.exit_code == 0, result.output
    assert result.output.splitlines() == [
        f"module load {USE_OWN_MODULE}",
        f"module load {CONDA_ENV_NAMESPACE}/myenv-py3.12",
    ]


def test_activate_is_the_tail_of_setup() -> None:
    """AC-5.1 and AC-5.1b describe one sequence, so they cannot disagree."""
    assert setup_commands("myenv", "3.12")[-2:] == activate_commands("myenv", "3.12")


def test_activate_with_an_explicit_py_contacts_nothing(monkeypatch) -> None:
    def explode(*args, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("activate --py opened a connection")

    monkeypatch.setattr(subprocess, "run", explode)
    result = CliRunner().invoke(
        env_cli, ["activate", "myenv", "--py", "3.12"], env=CLI_ENV
    )
    assert result.exit_code == 0, result.output


# ==========================================================================
# AC-5.2 -- pyX.Y is never guessed
# ==========================================================================


def test_py_version_comes_from_the_cluster_when_not_given(monkeypatch) -> None:
    remote = FakeRemote(stdout="Loading anaconda\n3.13\n")
    monkeypatch.setattr(subprocess, "run", remote)
    assert _resolve_py_version(CONFIG, None) == "3.13"


def test_a_patch_level_from_the_cluster_is_reduced_to_x_y(monkeypatch) -> None:
    remote = FakeRemote(stdout="3.12.7\n")
    monkeypatch.setattr(subprocess, "run", remote)
    assert _resolve_py_version(CONFIG, None) == "3.12"


def test_an_undiscoverable_py_version_fails_naming_the_flag(monkeypatch) -> None:
    """The "else actionable failure" branch -- and it must not fall back.

    The tempting fallback is this machine's own ``sys.version_info``. It would
    make the command succeed and compose a module name Gilbreth has never
    generated, which fails at ``module load`` with an error naming a version
    the user did not choose.
    """
    remote = FakeRemote(stdout="", stderr="module: command not found", returncode=127)
    monkeypatch.setattr(subprocess, "run", remote)

    with pytest.raises(click.ClickException) as excinfo:
        _resolve_py_version(CONFIG, None)
    message = str(excinfo.value)
    assert "--py" in message
    assert "module avail" in message


@pytest.mark.parametrize("version", ["3.9", "3.12", "3.13"])
def test_the_module_name_is_composed_from_the_resolved_version(version: str) -> None:
    assert conda_env_module("myenv", version) == f"conda-env/myenv-py{version}"


# ==========================================================================
# AC-5.8 -- `modules` asks the cluster; it ships no table
# ==========================================================================


def test_modules_queries_lmod_and_invents_no_name(monkeypatch) -> None:
    remote = FakeRemote(stdout="anaconda/2025.06-py313\n")
    monkeypatch.setattr(subprocess, "run", remote)

    result = CliRunner().invoke(env_cli, ["modules"], env=CLI_ENV)
    assert result.exit_code == 0, result.output

    sent = "\n".join(remote.sent)
    assert "module avail" in sent
    for query in MODULE_QUERIES:
        assert query in sent

    # The one name in the output came back from the (faked) cluster, not from
    # a constant in the source.
    assert "anaconda/2025.06-py313" in result.output


def test_modules_with_a_pattern_uses_spider(monkeypatch) -> None:
    remote = FakeRemote(stdout="")
    monkeypatch.setattr(subprocess, "run", remote)

    result = CliRunner().invoke(env_cli, ["modules", "pytorch"], env=CLI_ENV)
    assert result.exit_code == 0, result.output
    assert "module spider pytorch" in "\n".join(remote.sent)


def test_no_module_name_in_this_module_pins_a_version() -> None:
    """AC-5.8's ban, asserted on the values rather than on the source text.

    A version can only be right by accident: RCAC's instruction is not to
    invent module names *or versions*, and the module tree moves under us.
    Every name spelled here is a generic family -- `anaconda`, `cuda`, `cudnn`,
    `use.own`, the `conda-env` namespace -- that Lmod resolves to whatever the
    site default is on the day the job runs. The one versioned name this module
    ever produces, `conda-env/<name>-pyX.Y`, is composed from a version that
    came from `--py` or from the cluster (AC-5.2), never from a constant.
    """
    import re

    for name in (ANACONDA_MODULE, USE_OWN_MODULE, CONDA_ENV_NAMESPACE, *GPU_MODULES):
        assert not re.search(r"\d", name), f"{name!r} pins a version"

    targets = [
        target
        for query in MODULE_QUERIES
        for target in re.findall(r"module (?:load|avail|spider) ([\w.+-]+)", query)
    ]
    assert targets, "MODULE_QUERIES names no module at all; the scan is vacuous"
    pinned = [t for t in targets if re.search(r"\d", t)]
    assert not pinned, f"MODULE_QUERIES asks for pinned versions: {pinned}"


# ==========================================================================
# AC-5.11 -- nothing here touches a login start-up file
# ==========================================================================


def test_setup_writes_no_start_up_file(monkeypatch, tmp_path) -> None:
    """The behavioural half of the hygiene row.

    The hygiene scan bans the *literals*; this bans the *effect*, because a
    module loaded at login follows the user into every job they submit and the
    two guards fail in different ways.
    """
    remote = FakeRemote(stdout="done")
    monkeypatch.setattr(subprocess, "run", remote)
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(
        env_cli, ["setup", "--name", "myenv", "--py", "3.12"], env=CLI_ENV
    )
    assert result.exit_code == 0, result.output
    sent = "\n".join(remote.sent)
    assert ">>" not in sent, f"setup appended to a file on the cluster:\n{sent}"
