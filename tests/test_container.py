"""The Apptainer runtime, in the CLI and in the rendered job scripts (AC-5.4-5.9).

Gilbreth runs **Apptainer**, and three of the facts that follow from that are
absences rather than presences -- no ``module load`` for the runtime, no
``--bind`` for a path it already mounts, no trace of the previous cluster's
runtime name.  Absences are what a test suite is worst at noticing, so they are
asserted here directly on rendered output rather than left to the reader of a
template diff.

The two halves matter separately:

* ``rcac-container`` is what a user types.  Its ``run`` builds an ``exec``
  argv, and if that argv carries a redundant ``--bind`` it shadows a mount the
  runtime had already made.
* The four job templates are what actually runs on a compute node, hours later
  and unattended.  ``AC-5.5``/``5.6``/``5.7`` are scoped to *rendered* scripts
  for that reason, so they are asserted against a real render.

Nothing here contacts a cluster.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from rcac_cli._legacy import LEGACY_ENV_TYPES
from rcac_cli.container import (
    ACCOUNT_PLACEHOLDER,
    BUILD_FALLBACK_PARTITION,
    BUILD_FALLBACK_QOS,
    cli as container_cli,
    interactive_build_hint,
)
from rcac_cli.experiment import resolve_env_type
from rcac_cli.gilbreth import PARTITION_HINTS, QOS_HINTS
from rcac_cli.scheduler import render_slurm_script
from tests.test_repo_hygiene import HYGIENE_SCOPE

REPO_ROOT = Path(__file__).resolve().parent.parent


def hygiene_pattern(ac_id: str) -> str:
    """The ban regex of one :data:`HYGIENE_SCOPE` row.

    Reused rather than re-typed: the hygiene module scans *files*, this one
    scans *rendered output*, and a criterion enforced by two independently
    maintained copies of the same regex is enforced by neither.
    """
    return next(row for row in HYGIENE_SCOPE if row.ac_id == ac_id).pattern


CLI_ENV = {
    "RCAC_USER": "testid",
    "RCAC_ACCOUNT": "testqueue",
    "RCAC_CONFIG_DIR": str(REPO_ROOT / "configs"),
}

CONFIG = {
    "ssh": {"user": "testid"},
    "paths": {"scratch": "/scratch/gilbreth/{user}"},
}

#: A context sufficient to render any of the four job templates.
RENDER_CONTEXT = {
    "job_name": "proj_exp",
    "account": "testqueue",
    "partition": "a100-40gb",
    "qos": "normal",
    "nodes": 1,
    "ntasks_per_node": 1,
    "gpus_per_node": 1,
    "cpus_per_gpu": 8,
    "mem": "32G",
    "time": "04:00:00",
    "constraint": None,
    "project": "proj",
    "exp_id": "exp_001",
    "stage_name": "train",
    "script": "train.py",
    "args": "--lr 1e-4",
    "config_path": "configs/exp_001.yaml",
    "modules": [],
    "conda_env": "train_env-py3.12",
    "dependency_job_id": None,
}

CONTAINER_TEMPLATES = ("single_gpu.sh.j2", "multi_gpu.sh.j2", "pipeline.sh.j2")


def render(template: str, **overrides) -> str:
    context = {**RENDER_CONTEXT, **overrides}
    return render_slurm_script(template, config=CONFIG, **context)


class FakeRun:
    """Captures every subprocess argv and replays a canned result."""

    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.calls: list[list[str]] = []

    def __call__(self, cmd, *args, **kwargs):
        self.calls.append(list(cmd))
        return subprocess.CompletedProcess(
            cmd, self.returncode, stdout=self.stdout, stderr=self.stderr
        )


# ==========================================================================
# AC-5.5 / 5.6 / 5.7 -- the rendered container branch
# ==========================================================================


@pytest.mark.parametrize("template", CONTAINER_TEMPLATES)
def test_the_container_branch_calls_apptainer_with_nv(template: str) -> None:
    out = render(template, env_type="apptainer", env_container="/img/x.sif")
    assert "apptainer exec --nv" in out, out


@pytest.mark.parametrize("template", CONTAINER_TEMPLATES)
def test_no_rendered_script_binds_a_path_apptainer_auto_binds(template: str) -> None:
    """AC-5.6.  Apptainer auto-binds /home, /apps, /scratch and /depot.

    The pre-port template bound the scratch root onto itself. That is not
    merely redundant: an explicit bind of an already-mounted path is a second
    mount whose failure mode is silent, and the citation for why it is
    unnecessary belongs next to the runtime facts in `container.py`, not
    duplicated into four templates that no one reads.
    """
    out = render(template, env_type="apptainer", env_container="/img/x.sif")
    assert "--bind" not in out, out


@pytest.mark.parametrize("template", CONTAINER_TEMPLATES)
def test_no_rendered_script_loads_a_module_for_the_runtime(template: str) -> None:
    """AC-5.5.  Apptainer is system-installed; there is no module for it.

    A `module load` of a module that does not exist fails the job at its first
    line, and it does so hours after submission.
    """
    out = render(template, env_type="apptainer", env_container="/img/x.sif")
    assert "module load apptainer" not in out, out


@pytest.mark.parametrize("template", CONTAINER_TEMPLATES)
def test_the_conda_branch_loads_the_generated_environment_module(
    template: str,
) -> None:
    """The other half of C4's template rewrite (AC-5.3).

    The pre-port conda branch sourced a hand-rolled conda install on the
    previous cluster's filesystem. Gilbreth's documented replacement is not a
    different path to source -- it is two module loads, the same pair
    `rcac-env activate` prints.
    """
    out = render(template, env_type="conda")
    assert "module load use.own" in out, out
    assert "module load conda-env/train_env-py3.12" in out, out
    assert "conda activate" not in out, out


@pytest.mark.parametrize("template", CONTAINER_TEMPLATES + ("job_array.sh.j2",))
def test_no_rendered_script_mentions_a_pre_port_conda_install(template: str) -> None:
    """AC-5.3 on rendered output, which is the half the hygiene scan cannot see.

    The banned literals are not spelled here. They are read off the AC-5.3 row
    of :data:`~tests.test_repo_hygiene.HYGIENE_SCOPE`, which is both P6's
    single source of truth and the only way this file can assert them without
    becoming a violation of the row it is checking.
    """
    banned = re.compile(hygiene_pattern("AC-5.3"))
    for env_type in (None, "conda", "apptainer"):
        out = render(
            template,
            env_type=env_type,
            env_container="/img/x.sif",
            config_dir="configs/sweep",
            array_max=3,
            array_simultaneous=None,
        )
        match = banned.search(out)
        assert match is None, (
            f"{template} ({env_type}) rendered {match.group(0)!r}:\n{out}"
        )


# ==========================================================================
# `rcac-container run` -- the argv a user gets
# ==========================================================================


def test_run_builds_an_apptainer_exec_with_nv_and_no_bind(monkeypatch) -> None:
    fake = FakeRun()
    monkeypatch.setattr(subprocess, "run", fake)

    result = CliRunner().invoke(
        container_cli, ["run", "/img/x.sif", "nvidia-smi"], env=CLI_ENV
    )
    assert result.exit_code == 0, result.output
    assert "apptainer exec --nv /img/x.sif nvidia-smi" in result.output
    assert "--bind" not in result.output


def test_bind_still_threads_through_when_explicitly_asked(monkeypatch) -> None:
    """The flag survives for paths *outside* the auto-bound set.

    Deleting it would have been the tidier-looking change and the wrong one:
    auto-binding covers /home, /apps, /scratch and /depot, and says nothing
    about anywhere else.
    """
    fake = FakeRun()
    monkeypatch.setattr(subprocess, "run", fake)

    result = CliRunner().invoke(
        container_cli,
        ["run", "-B", "/tmp/work:/work", "/img/x.sif", "ls"],
        env=CLI_ENV,
    )
    assert result.exit_code == 0, result.output
    assert "--bind /tmp/work:/work" in result.output


def test_list_images_names_the_runtime_gilbreth_actually_has(monkeypatch) -> None:
    fake = FakeRun(stdout="")
    monkeypatch.setattr(subprocess, "run", fake)

    result = CliRunner().invoke(container_cli, ["list"], env=CLI_ENV)
    assert result.exit_code == 0, result.output
    assert result.output.strip() == "No Apptainer images found."


# ==========================================================================
# `--remote-build` -- reconciled, not assumed (plan row L17)
# ==========================================================================


def test_remote_build_tries_the_direct_rootless_build_first(
    monkeypatch, tmp_path
) -> None:
    """RCAC documents Apptainer as rootless, so no allocation is requested.

    The hint is *not* printed on the happy path: printing an
    interactive-allocation recipe next to a build that just succeeded teaches
    the user to ask for a GPU they do not need.
    """
    def_file = tmp_path / "img.def"
    def_file.write_text("Bootstrap: docker\nFrom: alpine\n")

    fake = FakeRun(returncode=0, stdout="built")
    monkeypatch.setattr(subprocess, "run", fake)

    result = CliRunner().invoke(
        container_cli, ["build", str(def_file), "--remote-build"], env=CLI_ENV
    )
    assert result.exit_code == 0, result.output
    sent = "\n".join(" ".join(c) for c in fake.calls)
    assert "apptainer build" in sent
    assert "srun" not in result.output, (
        "an allocation was suggested for a build that succeeded without one"
    )


def test_a_failed_remote_build_prints_the_fallback(monkeypatch, tmp_path) -> None:
    def_file = tmp_path / "img.def"
    def_file.write_text("Bootstrap: docker\nFrom: alpine\n")

    fake = FakeRun(returncode=1, stderr="boom")
    monkeypatch.setattr(subprocess, "run", fake)

    result = CliRunner().invoke(
        container_cli, ["build", str(def_file), "--remote-build"], env=CLI_ENV
    )
    assert result.exit_code == 1
    assert "srun" in result.output


def test_the_fallback_hint_uses_valid_gilbreth_directives() -> None:
    """The assertion the plan asks for by name.

    The pre-port hint named a partition from the previous cluster and a
    ``--comment`` job class Gilbreth has no flag for -- copy-pasteable, and
    rejected. The replacement is composed from a validated ``SubmissionPlan``,
    so this asserts the triple is present *and* that the two names in it are
    ones RCAC documents.
    """
    hint = "\n".join(interactive_build_hint(CONFIG, "/s/x.sif", "/s/x.def"))

    assert "srun " in hint
    for flag in ("-A ", "-p ", "-q ", "--gpus-per-node=", "--mem=", "--time="):
        assert flag in hint, f"{flag} missing from the hint:\n{hint}"

    assert "--comment" not in hint, "Gilbreth has no --comment job class"
    assert "--gres" not in hint, "the GPU request form is --gpus-per-node"
    assert BUILD_FALLBACK_PARTITION in PARTITION_HINTS
    assert BUILD_FALLBACK_QOS in QOS_HINTS
    assert f"-p {BUILD_FALLBACK_PARTITION}" in hint
    assert f"-q {BUILD_FALLBACK_QOS}" in hint
    assert "apptainer build /s/x.sif /s/x.def" in hint


def test_the_hint_survives_an_unresolved_account() -> None:
    """P4: a hint is not a submission.

    A user whose build just failed and who has not set ``RCAC_ACCOUNT`` needs
    the recipe more than anyone, and turning their build failure into a
    credential error would hide it.
    """
    hint = "\n".join(interactive_build_hint({}, "/s/x.sif", "/s/x.def"))
    assert f"-A {ACCOUNT_PLACEHOLDER}" in hint
    assert "slist" in hint


def test_the_hint_uses_a_resolved_account_when_there_is_one() -> None:
    hint = "\n".join(
        interactive_build_hint(
            {"slurm": {"account": "testqueue"}}, "/s/x.sif", "/s/x.def"
        )
    )
    assert "-A testqueue" in hint
    assert ACCOUNT_PLACEHOLDER not in hint


# ==========================================================================
# AC-5.9 -- the migration error
# ==========================================================================


def test_a_pre_port_environment_type_fails_naming_its_replacement() -> None:
    """A hard error, not an alias.

    Aliasing would work exactly once and leave the manifest wrong for the next
    reader; ignoring it would select the template's host-python branch and fail
    minutes later, inside the user's imports, with no mention of the manifest.
    """
    legacy, replacement = next(iter(LEGACY_ENV_TYPES.items()))

    with pytest.raises(click.ClickException) as excinfo:
        resolve_env_type(legacy)
    message = str(excinfo.value)
    assert replacement in message
    assert "environment.type" in message


@pytest.mark.parametrize("spelling_case", [str.upper, str.title])
def test_the_migration_error_is_case_insensitive(spelling_case) -> None:
    legacy = next(iter(LEGACY_ENV_TYPES))
    with pytest.raises(click.ClickException):
        resolve_env_type(spelling_case(legacy))


@pytest.mark.parametrize("value", ["apptainer", "conda", None])
def test_current_environment_types_pass_through(value) -> None:
    assert resolve_env_type(value) == value


def test_the_shipped_examples_declare_a_type_this_release_accepts() -> None:
    """AC-8.7's half of AC-5.9: the examples cannot be the thing that breaks."""
    import yaml

    for path in sorted((REPO_ROOT / "examples").glob("*.yaml")):
        manifest = yaml.safe_load(path.read_text())
        declared = (manifest.get("environment") or {}).get("type")
        assert resolve_env_type(declared) == declared, path
