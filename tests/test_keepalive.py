"""Retention: the 60-day policy, the config scratch root, and where it runs.

Three properties, and each was a live defect before C6:

**AC-6.1 -- one retention number, and it is Gilbreth's.**  Every window this
module prints is derived from :data:`SCRATCH_RETENTION_DAYS`; the previous
cluster's much shorter figure is banned from the package by the ``AC-6.1``
hygiene row and survives only inside the README's history fence.

**AC-6.2 -- the scratch root comes from ``paths.scratch``.**  ``keepalive.py``
built it from an f-string at three sites, and what it built has no cluster
segment, so it names a directory that does not exist on Gilbreth.  The sharpest
form is the batch job: its ``-o``/``-e`` came from the config through
``scratch_root`` while its *body* came from the f-string, so one generated
script named two different roots and the half that silently matched nothing was
the half that did the work.

**AC-6.5 -- the sweep runs on a front-end node unless asked otherwise.**  The
batch form is not free on Gilbreth: ``--gpus-per-node`` has a floor of 1 and
every partition is a GPU partition, so a job that only touches atimes still
holds a GPU.

Nothing here reaches a cluster: ``subprocess.run`` is replaced, so what the
tests read is exactly the argv the CLI would have sent.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

from rcac_cli import discovery
from rcac_cli.keepalive import (
    BATCH_TRADEOFF_HELP,
    DEFAULT_WARN_DAYS,
    SCRATCH_RETENTION_DAYS,
    cli as keepalive_cli,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = REPO_ROOT / "tests" / "fixtures"

#: The root ``configs/rcac.yaml`` yields for this user.  Written out rather
#: than composed so the test states the answer independently of the code that
#: produces it.
SCRATCH = "/scratch/gilbreth/testid"

BASE_ENV = {
    "RCAC_USER": "testid",
    "RCAC_ACCOUNT": "testqueue",
    "RCAC_CONFIG_DIR": str(REPO_ROOT / "configs"),
}

#: Discovery off: the commands that do not assert on a probe must not ssh to a
#: Purdue login node from the suite.
OFFLINE_ENV = {**BASE_ENV, "RCAC_NO_DISCOVER": "1"}

#: Discovery on, but with the transport replaced -- see :func:`fake_cluster`.
PROBING_ENV = {**BASE_ENV, "RCAC_NO_DISCOVERY_CACHE": "1"}


def load_fixture(name: str) -> str:
    """A doc-derived fixture with its ``# PROVENANCE:`` block stripped."""
    lines = (FIXTURES / name).read_text(encoding="utf-8").splitlines()
    return "\n".join(line for line in lines if not line.startswith("#")) + "\n"


@pytest.fixture(autouse=True)
def _no_probe_cache():
    """A session cache shared between tests would leak one probe into the next."""
    discovery.clear_cache()
    yield
    discovery.clear_cache()


def fake_cluster(monkeypatch, replies: dict | None = None) -> list[str]:
    """Capture every remote command; answer the named ones from *replies*."""
    sent: list[str] = []
    replies = replies or {}

    def fake_run(cmd, *_args, **_kwargs):
        command = cmd[-1]
        sent.append(command)
        stdout = ""
        for needle, reply in replies.items():
            if needle in command:
                stdout = reply
                break
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    return sent


def invoke(args, env) -> tuple:
    result = CliRunner().invoke(keepalive_cli, args, env=env)
    assert result.exit_code == 0, result.output
    return result


def scratch_roots(text: str) -> set:
    """Every ``/scratch/<segment>/<segment>`` root named anywhere in *text*."""
    return set(re.findall(r"/scratch/[\w.-]+/[\w.-]+", text))


# --------------------------------------------------------------------------
# AC-6.1 -- one retention number
# --------------------------------------------------------------------------


def test_the_retention_window_is_gilbreths():
    assert SCRATCH_RETENTION_DAYS == 60


def test_the_warning_window_is_derived_and_not_written_down_twice():
    """Raising the policy must move the warning, not leave it behind.

    The pre-port default was a fixed number chosen against a different
    policy; expressing it as a fraction of the window is what stops the two
    from drifting apart the next time RCAC changes one of them.
    """
    assert DEFAULT_WARN_DAYS == 45
    assert SCRATCH_RETENTION_DAYS - DEFAULT_WARN_DAYS == SCRATCH_RETENTION_DAYS // 4


def test_check_counts_down_from_the_policy_not_from_a_constant(monkeypatch):
    """The "deleted in N days" arithmetic reads the policy at both ends."""
    listing = "2026-01-01+00:00:00 1048576 /scratch/gilbreth/testid/old.pt\n"
    fake_cluster(monkeypatch, {"-atime": listing})

    result = invoke(["check", "--no-discover", "--days", "50"], OFFLINE_ENV)

    assert f"deleted in {SCRATCH_RETENTION_DAYS - 50} days or less" in result.output


# --------------------------------------------------------------------------
# AC-6.2 -- the scratch root comes from the config
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "args",
    [["touch"], ["touch", "--dry-run"], ["check", "--no-discover"], ["setup-cron"]],
    ids=["touch", "touch-dry-run", "check", "setup-cron"],
)
def test_every_command_addresses_the_configured_scratch_root(monkeypatch, args):
    """No command may build ``/scratch/<user>``; the root has a cluster in it."""
    sent = fake_cluster(monkeypatch)
    invoke(args, OFFLINE_ENV)

    written = "\n".join(sent)
    assert SCRATCH in written, f"{args} never named the configured root:\n{written}"
    assert scratch_roots(written) == {SCRATCH}, (
        f"{args} named a root other than the configured one: "
        f"{sorted(scratch_roots(written))}"
    )


def test_the_generated_batch_job_names_one_root_in_its_header_and_its_body(
    monkeypatch,
):
    """The seam C3c left open, closed.

    ``-o``/``-e`` rendered from ``paths.scratch`` while the ``find`` lines came
    from an f-string, so the job wrote its logs to a directory that exists and
    swept a directory that does not.  It would have exited 0 either way: every
    ``find`` in the body is ``2>/dev/null`` and a ``find`` over a missing path
    simply matches nothing.
    """
    sent = fake_cluster(monkeypatch)
    invoke(["setup-cron", "--batch"], OFFLINE_ENV)

    written = "\n".join(sent)
    header = [line for line in written.splitlines() if line.startswith("#SBATCH -")]
    output_directives = [ln for ln in header if ln.startswith(("#SBATCH -o", "#SBATCH -e"))]
    sweeps = [ln for ln in written.splitlines() if "-exec touch -a" in ln]

    assert len(output_directives) == 2, header
    assert sweeps, "the generated job had no sweep in its body"
    assert scratch_roots("\n".join(output_directives)) == {SCRATCH}
    assert scratch_roots("\n".join(sweeps)) == {SCRATCH}
    assert scratch_roots(written) == {SCRATCH}


# --------------------------------------------------------------------------
# AC-6.3 -- `check` asks the cluster what is actually staged
# --------------------------------------------------------------------------


@pytest.mark.unverified_fixture
def test_check_invokes_purgelist_and_shows_what_it_named(monkeypatch):
    """The authoritative answer, not an inference from file metadata.

    A ``find -atime`` scan cannot see the purge scheduler's state; ``purgelist``
    is the cluster's own list, so it leads the report.
    """
    sent = fake_cluster(
        monkeypatch,
        {
            discovery.PURGE_COMMAND: load_fixture("purgelist.txt"),
            discovery.QUOTA_COMMAND: load_fixture("myquota.txt"),
        },
    )

    result = invoke(["check"], PROBING_ENV)

    assert any(cmd == discovery.PURGE_COMMAND for cmd in sent), sent
    assert "/scratch/gilbreth/myusername/logs/train.log" in result.output
    assert "and are eligible for purge" not in result.output, (
        "the preamble reached the report as if it were a file"
    )


@pytest.mark.unverified_fixture
def test_check_reports_the_byte_quota_and_not_the_inode_column(monkeypatch):
    """The `parse_table` fix, seen from the command that renders it.

    ``myquota`` heads two ``Used`` and two ``Use%`` columns.  While a row was a
    dict the home line reported ``-`` for both; the two numbers below are the
    ones the report exists to show.
    """
    fake_cluster(
        monkeypatch,
        {
            discovery.PURGE_COMMAND: "",
            discovery.QUOTA_COMMAND: load_fixture("myquota.txt"),
        },
    )

    result = invoke(["check"], PROBING_ENV)

    assert "16.3GB" in result.output
    assert "65%" in result.output
    assert "1.2TB" in result.output


def test_check_degrades_to_a_warning_when_the_probes_cannot_run(monkeypatch):
    """AC-4.2 -- a probe that did not answer must not abort the command."""
    fake_cluster(monkeypatch)

    result = invoke(["check", "--no-discover"], OFFLINE_ENV)

    assert "purge discovery did not run" in result.output
    assert "quota discovery did not run" in result.output


# --------------------------------------------------------------------------
# AC-6.5 -- front-end by default, batch on request
# --------------------------------------------------------------------------


def test_the_batch_flag_documents_both_halves_of_the_tradeoff():
    """A flag whose help says only what it does leaves the choice unmade."""
    result = CliRunner().invoke(keepalive_cli, ["setup-cron", "--help"])
    assert result.exit_code == 0, result.output
    rendered = " ".join(result.output.split())

    assert "--batch" in result.output
    # Cheap on a front-end node ...
    assert "metadata-only" in rendered
    assert "CPU limit" in rendered
    # ... and disproportionate as a job.
    assert "GPU it never uses" in rendered
    assert "disproportionate" in rendered
    assert "metadata-only" in " ".join(BATCH_TRADEOFF_HELP.split())
