"""SLURM script generation for Gilbreth (AC-3.1-3.4, 3.12-3.18).

``tests/test_gilbreth.py`` proves the *rules*.  This file proves the **wiring**:
that the four templates and the four generators actually emit what those rules
decided, and that nothing can reach a rendered ``#SBATCH`` line without having
gone through :func:`rcac_cli.gilbreth.build_directives` first.

Everything here asserts on **rendered output** or on template *source*, never on
a generator's internals.  That is deliberate: the generated script is the
product, it is the artefact a live cluster sees, and it is the only thing a
reviewer can check offline.

The pre-port assertions this file replaced are worth naming, because each was
true of the previous cluster and is now actively wrong:

* ``#SBATCH -p amd_a100nv_8`` -- not a Gilbreth partition, and ``-p`` now
  selects a GPU *type*.
* ``#SBATCH --comment pytorch`` -- Gilbreth has no ``--comment`` job class.
* ``module load cuda/12.4.1`` -- an invented module name for this cluster; the
  default module list is now empty.

Rendering here goes through the injected ``config`` parameter rather than the
on-disk YAML, so these tests do not depend on whether ``RCAC_USER`` happens to
be exported in the shell running them.  The CLI tests at the bottom deliberately
do the opposite -- they go through the real loader, because that is the path a
user is on.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
from click.testing import CliRunner
from jinja2 import UndefinedError

from rcac_cli.experiment import cli as exp_cli
from rcac_cli.gilbreth import DEFAULT_QOS, SubmissionError
from rcac_cli.scheduler import (
    generate_job_array_script,
    generate_multi_gpu_script,
    generate_pipeline_script,
    generate_single_gpu_script,
    get_template_dir,
    render_slurm_script,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = get_template_dir()

#: The shared ``#SBATCH`` partial. AC-3.1 is parametrised over all four
#: templates, but this file is what makes the property structural rather than
#: four copies agreeing by luck.
SHARED_HEADER = "_sbatch_header.j2"

#: Injected config. ``paths.scratch`` keeps its ``{user}`` template form: the
#: loader deliberately does not format it, and ``scratch_root()`` is what turns
#: it into a path (B10/AC-3.17).
CONFIG = {
    "ssh": {"user": "testid"},
    "paths": {"scratch": "/scratch/gilbreth/{user}"},
}
SCRATCH = "/scratch/gilbreth/testid"

#: A submission valid in every respect, so a test that varies one field knows
#: what it is observing.
VALID = {
    "account": "ml-lab",
    "partition": "a10",
    "time": "01:00:00",
    "config": CONFIG,
}


def render_all(**overrides) -> dict[str, str]:
    """Render every template through its own generator."""
    kwargs = {**VALID, **overrides}
    return {
        "single_gpu": generate_single_gpu_script(
            project="proj", exp_id="exp_001", script="train.py", **kwargs
        ),
        "multi_gpu": generate_multi_gpu_script(
            project="proj", exp_id="exp_001", script="train.py", **kwargs
        ),
        "job_array": generate_job_array_script(
            project="proj", script="train.py", config_dir="cfg", array_max=3,
            **kwargs
        ),
        "pipeline": generate_pipeline_script(
            project="proj", exp_id="exp_001", stage_name="train",
            script="train.py", **kwargs
        ),
    }


RENDERED = render_all()
TEMPLATE_NAMES = sorted(RENDERED)
TEMPLATE_FILES = sorted(p.name for p in TEMPLATE_DIR.glob("*.sh.j2"))


def directives(script: str) -> list[str]:
    """Every ``#SBATCH`` line of *script*, in order."""
    return [
        line for line in script.splitlines() if line.startswith("#SBATCH")
    ]


# ==========================================================================
# AC-3.1 -- every template emits the Gilbreth triple plus the resource trio
# ==========================================================================

#: ``-A`` account, ``-p`` GPU type, ``-q`` QOS, and the three resource
#: directives Gilbreth requires. ``--gres`` is absent by design (AC-3.2).
REQUIRED_DIRECTIVES = ("-A ", "-p ", "-q ", "--gpus-per-node=", "--mem=", "--time=")


@pytest.mark.parametrize("name", TEMPLATE_NAMES)
@pytest.mark.parametrize("flag", REQUIRED_DIRECTIVES)
def test_every_template_emits_the_required_directive(name: str, flag: str) -> None:
    lines = directives(RENDERED[name])
    assert any(line.startswith(f"#SBATCH {flag}") for line in lines), (
        f"{name}: no `#SBATCH {flag}` in the rendered header:\n  "
        + "\n  ".join(lines)
    )


@pytest.mark.parametrize("name", TEMPLATE_NAMES)
def test_required_directives_carry_the_effective_values(name: str) -> None:
    """The directives are not merely present -- they carry what was asked for."""
    lines = directives(RENDERED[name])
    assert "#SBATCH -A ml-lab" in lines
    assert "#SBATCH -p a10" in lines
    assert f"#SBATCH -q {DEFAULT_QOS}" in lines
    assert "#SBATCH --time=01:00:00" in lines


@pytest.mark.parametrize("filename", TEMPLATE_FILES)
def test_every_template_includes_the_shared_header(filename: str) -> None:
    """AC-3.1, structurally.

    A parametrised assertion proves the four templates agree *today*.  The
    shared partial is what stops them drifting apart tomorrow, so its use is
    asserted directly on the template source.
    """
    source = (TEMPLATE_DIR / filename).read_text()
    assert f"include '{SHARED_HEADER}'" in source, (
        f"{filename} builds its own #SBATCH header instead of including "
        f"{SHARED_HEADER}"
    )


def test_the_template_directory_holds_exactly_the_known_templates() -> None:
    """Four experiment templates, plus the keepalive job C3c moved in here.

    ``keepalive.sh.j2`` is not an experiment template and is not rendered by
    ``rcac-exp``; it is here because a template is the only place a ``#SBATCH``
    literal may live (AC-3.19(a)), and ``keepalive.py`` used to build one from
    an f-string.  It earns its place in the two parametrised assertions above
    the same way the other four do -- it includes the shared header and
    composes no job name of its own.
    """
    assert TEMPLATE_FILES == [
        "job_array.sh.j2",
        "keepalive.sh.j2",
        "multi_gpu.sh.j2",
        "pipeline.sh.j2",
        "single_gpu.sh.j2",
    ]
    assert (TEMPLATE_DIR / SHARED_HEADER).is_file()


def test_per_template_directives_stay_per_template() -> None:
    """The shared partial must not absorb the three genuinely local directives.

    ``--array=`` belongs to the sweep, ``--dependency=`` to the pipeline, and
    ``-o``/``-e`` differ in their filename pattern (``%x_%A_%a`` for an array
    task, ``%x_%j`` otherwise).  Folding any of them into the shared header
    would emit them where they are meaningless.
    """
    emitted = [
        line
        for line in (TEMPLATE_DIR / SHARED_HEADER).read_text().splitlines()
        if line.startswith("#SBATCH")
    ]
    for local in ("--array=", "--dependency=", "-o ", "-e "):
        assert not any(local in line for line in emitted), (
            f"{SHARED_HEADER} carries {local!r}, which is per-template:\n  "
            + "\n  ".join(emitted)
        )

    array = RENDERED["job_array"]
    assert "#SBATCH --array=0-3" in array
    assert "%x_%A_%a" in array
    for other in ("single_gpu", "multi_gpu", "pipeline"):
        assert "--array=" not in RENDERED[other]
        assert "%x_%j" in RENDERED[other]


def test_pipeline_dependency_is_emitted_only_when_asked_for() -> None:
    with_dep = generate_pipeline_script(
        project="proj", exp_id="exp_001", stage_name="evaluate",
        script="evaluate.py", dependency_job_id="12345", **VALID
    )
    assert "#SBATCH --dependency=afterok:12345" in with_dep
    assert "--dependency" not in RENDERED["pipeline"]


def test_constraint_is_emitted_only_when_the_plan_keeps_it() -> None:
    """AC-3.9 seen from the template side.

    ``nvlink`` on ``a100-40gb`` with 2 GPUs is the documented case and renders.
    The same request on ``a10`` is suppressed by ``build_directives``, and the
    template renders from ``plan.constraint``, so it cannot re-emit it.
    """
    kept = generate_single_gpu_script(
        project="proj", exp_id="exp_001", script="train.py",
        account="ml-lab", partition="a100-40gb", time="01:00:00",
        gpus_per_node=2, constraint="nvlink", config=CONFIG,
    )
    assert "#SBATCH --constraint=nvlink" in kept

    dropped = generate_single_gpu_script(
        project="proj", exp_id="exp_001", script="train.py",
        account="ml-lab", partition="a10", time="01:00:00",
        gpus_per_node=2, constraint="nvlink", config=CONFIG,
    )
    assert "--constraint" not in dropped
    assert "#SBATCH --constraint" not in RENDERED["single_gpu"]


# ==========================================================================
# AC-3.2 -- the two directives that must never appear again
# ==========================================================================


@pytest.mark.parametrize("name", TEMPLATE_NAMES)
@pytest.mark.parametrize("banned", ("--comment", "--gres"))
def test_banned_directives_are_absent_from_rendered_output(
    name: str, banned: str
) -> None:
    script = RENDERED[name]
    assert banned not in script, (
        f"{name}: {banned!r} survives in rendered output at line "
        + str(
            next(
                i
                for i, line in enumerate(script.splitlines(), start=1)
                if banned in line
            )
        )
    )


@pytest.mark.parametrize("name", TEMPLATE_NAMES)
def test_the_gpu_request_is_gpus_per_node(name: str) -> None:
    """The positive half of AC-3.2: something must request the GPUs."""
    assert "#SBATCH --gpus-per-node=1" in directives(RENDERED[name]) or (
        "#SBATCH --gpus-per-node=2" in directives(RENDERED[name])
    )


def test_generators_reject_the_removed_comment_parameter() -> None:
    """``comment=`` is gone from the signature, not merely ignored.

    Ignoring it would let an un-migrated caller keep passing a job class that
    silently does nothing.
    """
    with pytest.raises(TypeError):
        generate_single_gpu_script(
            project="proj", exp_id="exp_001", script="train.py",
            comment="pytorch", **VALID
        )


# ==========================================================================
# AC-3.4 -- nothing unrendered survives
# ==========================================================================


@pytest.mark.parametrize("name", TEMPLATE_NAMES)
def test_no_leftover_jinja_delimiters(name: str) -> None:
    for delim in ("{{", "{%"):
        assert delim not in RENDERED[name], f"{name}: {delim!r} survived"


# ==========================================================================
# AC-3.12 / B9 -- scratch comes from the config, not from a template literal
# ==========================================================================


@pytest.mark.parametrize(
    "filename", sorted(p.name for p in TEMPLATE_DIR.glob("*.j2"))
)
def test_no_template_writes_a_scratch_path_literal(filename: str) -> None:
    """B9: 23 lines / 26 occurrences of ``/scratch/{{ user }}`` are gone.

    The ban is on the literal path prefix, not just on the old spelling: a
    template that wrote ``/scratch/gilbreth/{{ user }}`` instead would pass a
    naive check and still ignore ``paths.scratch``.
    """
    source = (TEMPLATE_DIR / filename).read_text()
    offenders = [
        (i, line)
        for i, line in enumerate(source.splitlines(), start=1)
        if "/scratch/" in line
    ]
    assert not offenders, (
        f"{filename} hardcodes a scratch path; use {{{{ scratch_root }}}}:\n  "
        + "\n  ".join(f"line {i}: {line.strip()}" for i, line in offenders)
    )


@pytest.mark.parametrize("name", TEMPLATE_NAMES)
def test_changing_paths_scratch_changes_every_rendered_path(name: str) -> None:
    """The half of AC-3.12 that a literal-ban alone cannot prove."""
    alternate = {
        "ssh": {"user": "testid"},
        "paths": {"scratch": "/depot/somegroup/{user}/scratch"},
    }
    base = RENDERED[name]
    moved = render_all(config=alternate)[name]

    occurrences = base.count(SCRATCH)
    assert occurrences >= 4, (
        f"{name}: only {occurrences} scratch path(s) rendered; this test would "
        "be near-vacuous"
    )
    assert SCRATCH not in moved, (
        f"{name}: a path survived the config change and is therefore not "
        "coming from paths.scratch"
    )
    assert moved.count("/depot/somegroup/testid/scratch") == occurrences


@pytest.mark.parametrize("name", TEMPLATE_NAMES)
def test_rendered_paths_are_well_formed(name: str) -> None:
    """AC-3.17 as a property of output: no ``//`` and no literal placeholder."""
    for line in RENDERED[name].splitlines():
        assert "{user}" not in line, f"{name}: unformatted placeholder in {line!r}"
        body = line.split("://", 1)[0]
        assert "//" not in body, f"{name}: doubled separator in {line!r}"


def test_scratch_root_is_required_and_raises_rather_than_rendering_blank() -> None:
    """An unresolved user must not reach an ``#SBATCH -o`` line as ``//``."""
    from rcac_cli.config import CredentialError

    with pytest.raises(CredentialError):
        generate_single_gpu_script(
            project="proj", exp_id="exp_001", script="train.py",
            account="ml-lab", partition="a10", time="01:00:00",
            config={"ssh": {"user": ""}, "paths": {"scratch": "/scratch/gilbreth/{user}"}},
        )


# ==========================================================================
# AC-3.13 / B2 -- StrictUndefined, the reason it was installed at C0
# ==========================================================================

#: Everything ``single_gpu.sh.j2`` and the shared header consume.
FULL_CONTEXT = {
    "job_name": "proj_exp_001",
    "account": "ml-lab",
    "partition": "a10",
    "qos": "normal",
    "nodes": 1,
    "ntasks_per_node": 1,
    "gpus_per_node": 1,
    "cpus_per_gpu": 8,
    "mem": "32G",
    "time": "01:00:00",
    "constraint": None,
    "scratch_root": SCRATCH,
    "project": "proj",
    "exp_id": "exp_001",
    "script": "train.py",
    "args": "",
    "modules": [],
    "conda_env": "base",
    "env_type": None,
    "env_container": None,
    "config_path": "",
}


def test_the_full_context_renders() -> None:
    """The control: without this, every case below could pass for the wrong reason."""
    out = render_slurm_script("single_gpu.sh.j2", config=CONFIG, **FULL_CONTEXT)
    assert "#SBATCH -A ml-lab" in out


#: The variables ``single_gpu.sh.j2`` evaluates on *every* render.  The three
#: it does not are excluded and covered separately below, because a test that
#: expected them to raise here would be asserting that a branch it never takes
#: is nonetheless evaluated.
ALWAYS_EVALUATED = sorted(
    set(FULL_CONTEXT) - {"scratch_root", "conda_env", "env_container"}
)


@pytest.mark.parametrize("missing", ALWAYS_EVALUATED)
def test_a_missing_variable_raises_undefined_error(missing: str) -> None:
    """R1: a dropped variable must not render as an empty, plausible directive."""
    context = {k: v for k, v in FULL_CONTEXT.items() if k != missing}
    with pytest.raises(UndefinedError) as excinfo:
        render_slurm_script("single_gpu.sh.j2", config=CONFIG, **context)
    assert missing in str(excinfo.value)


@pytest.mark.parametrize(
    "env_type,missing",
    [("conda", "conda_env"), ("singularity", "env_container")],
)
def test_branch_only_variables_also_raise_when_their_branch_is_taken(
    env_type: str, missing: str
) -> None:
    context = {k: v for k, v in FULL_CONTEXT.items() if k != missing}
    context["env_type"] = env_type
    with pytest.raises(UndefinedError) as excinfo:
        render_slurm_script("single_gpu.sh.j2", config=CONFIG, **context)
    assert missing in str(excinfo.value)


def test_scratch_root_cannot_be_undefined_because_it_is_injected() -> None:
    """The one header variable StrictUndefined will never see.

    ``render_slurm_script`` fills ``scratch_root`` from the config when the
    caller omits it, and ``config.scratch_root()`` raises on an unresolved
    user.  So the failure mode AC-3.13 guards against is closed here by
    construction rather than by the undefined check -- which is why it is
    stated rather than left looking like an untested gap.
    """
    context = {k: v for k, v in FULL_CONTEXT.items() if k != "scratch_root"}
    out = render_slurm_script("single_gpu.sh.j2", config=CONFIG, **context)
    assert f"#SBATCH -o {SCRATCH}/logs/%x_%j.out" in out


# ==========================================================================
# AC-3.18 -- three job-name compositions, composed in Python
# ==========================================================================


def test_job_name_compositions_are_distinct() -> None:
    assert "#SBATCH -J proj_exp_001" in RENDERED["single_gpu"]
    assert "#SBATCH -J proj_exp_001" in RENDERED["multi_gpu"]
    assert "#SBATCH -J proj_sweep" in RENDERED["job_array"]
    assert "#SBATCH -J proj_train_exp_001" in RENDERED["pipeline"]


@pytest.mark.parametrize("name", TEMPLATE_NAMES)
def test_job_name_is_a_parameter_on_every_generator(name: str) -> None:
    overridden = render_all(job_name="explicit_name")[name]
    assert "#SBATCH -J explicit_name" in overridden


@pytest.mark.parametrize(
    "filename", sorted(p.name for p in TEMPLATE_DIR.glob("*.j2"))
)
def test_no_template_composes_the_job_name_itself(filename: str) -> None:
    """AC-3.18: composition happens in Python, so the three cannot be flattened.

    ``{{ project }}_{{ exp_id }}`` in Jinja is one edit away from being copied
    into the sweep template, where it is wrong.
    """
    source = (TEMPLATE_DIR / filename).read_text()
    for line in source.splitlines():
        if line.startswith("#SBATCH -J"):
            assert line.strip() == "#SBATCH -J {{ job_name }}", (
                f"{filename} composes its own job name: {line!r}"
            )


# ==========================================================================
# AC-3.16 / B6 -- no generator is imported without a call site
# ==========================================================================

GENERATOR_NAME = re.compile(r"generate_\w+_script")


def _package_modules() -> list[Path]:
    return sorted((REPO_ROOT / "rcac_cli").glob("*.py"))


@pytest.mark.parametrize(
    "module", _package_modules(), ids=lambda p: p.name
)
def test_no_generator_is_imported_without_a_call_site(module: Path) -> None:
    """The narrowed form of AC-3.16.

    A naive "imported but unreferenced" walk false-positives across most of the
    package, so this is scoped to names matching ``generate_*_script`` -- which
    is exactly the class B6 belonged to.
    """
    tree = ast.parse(module.read_text())
    imported = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
        if GENERATOR_NAME.fullmatch(alias.asname or alias.name)
    }
    if not imported:
        return
    referenced = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    }
    unused = sorted(imported - referenced)
    assert not unused, (
        f"{module.name} imports {unused} and never uses them (B6/AC-3.16)"
    )


def test_the_generators_module_still_defines_all_four() -> None:
    """Deleting the dead *import* must not be confused with deleting the code."""
    import rcac_cli.scheduler as scheduler

    for name in (
        "generate_single_gpu_script",
        "generate_multi_gpu_script",
        "generate_job_array_script",
        "generate_pipeline_script",
    ):
        assert callable(getattr(scheduler, name))


# ==========================================================================
# The choke point -- callers cannot bypass validation
# ==========================================================================


@pytest.mark.parametrize(
    "generator,extra",
    [
        (generate_single_gpu_script, {"project": "p", "exp_id": "e", "script": "t.py"}),
        (generate_multi_gpu_script, {"project": "p", "exp_id": "e", "script": "t.py"}),
        (
            generate_job_array_script,
            {"project": "p", "script": "t.py", "config_dir": "c", "array_max": 1},
        ),
        (
            generate_pipeline_script,
            {"project": "p", "exp_id": "e", "stage_name": "s", "script": "t.py"},
        ),
    ],
    ids=["single_gpu", "multi_gpu", "job_array", "pipeline"],
)
def test_every_generator_goes_through_build_directives(generator, extra) -> None:
    """The abolished ``-A standby`` form must fail in all four, not just one."""
    with pytest.raises(SubmissionError) as excinfo:
        generator(account="standby", partition="a10", time="01:00:00",
                  config=CONFIG, **extra)
    assert "-q standby" in str(excinfo.value)


@pytest.mark.parametrize(
    "field", ["account", "partition", "time"]
)
def test_the_three_required_fields_have_no_default(field: str) -> None:
    """A default here would silently submit a job the user did not describe."""
    kwargs = {k: v for k, v in VALID.items() if k != field}
    with pytest.raises(TypeError):
        generate_single_gpu_script(
            project="proj", exp_id="exp_001", script="train.py", **kwargs
        )


def test_modules_default_to_empty_not_to_an_invented_module_name() -> None:
    """RCAC's guidance is explicit: do not invent module names or versions."""
    assert "module load" not in RENDERED["single_gpu"]
    with_modules = render_all(modules=["anaconda"])["single_gpu"]
    assert "module load anaconda" in with_modules


def test_mem_is_sized_from_the_gpu_count_when_not_given() -> None:
    """AC-3.11 seen through the template."""
    assert "#SBATCH --mem=32G" in RENDERED["single_gpu"]
    assert "#SBATCH --mem=128G" in render_all(gpus_per_node=4)["single_gpu"]
    assert "#SBATCH --mem=8G" in render_all(mem="8G")["single_gpu"]


# ==========================================================================
# The CLI: AC-3.14, AC-3.15 and the four guardrails of plan section 6.1
# ==========================================================================

SIMPLE_EXAMPLE = REPO_ROOT / "examples" / "experiment_simple.yaml"

#: The credentials the guardrails need in order to reach ``build_directives``
#: at all.  Without them the command exits from the *credential* check and
#: every guardrail passes for the wrong reason -- the defect that made
#: iteration 1's version of these commands vacuous.
CLI_ENV = {
    "RCAC_USER": "testid",
    "RCAC_ACCOUNT": "testqueue",
    "RCAC_CONFIG_DIR": str(REPO_ROOT / "configs"),
}


def run_launch(*args, env=None):
    """Invoke ``rcac-exp launch`` on the shipped simple example."""
    runner = CliRunner()
    return runner.invoke(
        exp_cli,
        ["launch", str(SIMPLE_EXAMPLE), "--dry-run", *args],
        env={**CLI_ENV, **(env or {})},
    )


def test_dry_run_prints_the_sbatch_command_and_the_whole_script() -> None:
    """AC-3.15.  A dry run whose only output is a path is not a dry run.

    No cluster contact: ``launch`` no longer runs its ``mkdir -p`` over SSH
    before checking ``--dry-run``, which it used to do unconditionally.
    """
    result = run_launch("-n", "1")
    assert result.exit_code == 0, result.output
    assert re.search(r"Would run: sbatch \S+/job_000\.sh", result.stdout)
    assert "#SBATCH -A testqueue" in result.stdout
    assert "#SBATCH -p a100-40gb" in result.stdout
    assert "#SBATCH --gpus-per-node=2" in result.stdout
    assert "the cluster was not contacted" in result.stdout


def test_dry_run_contacts_nothing(monkeypatch) -> None:
    """The other half of AC-3.15, and a real regression.

    ``launch`` used to run its ``mkdir -p`` over SSH *before* looking at
    ``--dry-run``, so "generate scripts without submitting" still opened a
    connection -- and on a cluster behind MFA that means a prompt. Any
    subprocess at all now fails this test.
    """
    import rcac_cli.experiment as experiment

    def explode(*args, **kwargs):  # pragma: no cover - only runs on failure
        raise AssertionError(f"--dry-run spawned a subprocess: {args!r}")

    monkeypatch.setattr(experiment.subprocess, "run", explode)
    result = run_launch("-n", "1")
    assert result.exit_code == 0, result.output
    assert "#SBATCH -A testqueue" in result.stdout


def test_dry_run_renders_a_complete_header() -> None:
    result = run_launch("-n", "1")
    emitted = [l for l in result.stdout.splitlines() if l.startswith("#SBATCH")]
    for flag in REQUIRED_DIRECTIVES:
        assert any(l.startswith(f"#SBATCH {flag}") for l in emitted), (
            f"--dry-run output has no `#SBATCH {flag}`:\n  " + "\n  ".join(emitted)
        )


def test_legacy_slurm_comment_warns_and_is_never_silently_dropped(tmp_path) -> None:
    """AC-3.14.  A silently ignored key looks exactly like an honoured one."""
    manifest = tmp_path / "legacy.yaml"
    manifest.write_text(
        "project: legacy\n"
        "slurm:\n"
        "  partition: a10\n"
        "  time: '01:00:00'\n"
        "  comment: pytorch\n"
    )
    runner = CliRunner()
    result = runner.invoke(
        exp_cli, ["launch", str(manifest), "--dry-run", "-n", "1"], env=CLI_ENV
    )
    assert result.exit_code == 0, result.output
    assert "slurm.comment is ignored" in result.stderr
    assert "--comment" not in result.stdout


# -- Guardrail 1 (plan section 6.1): the abolished account form --------------


def test_guardrail_abolished_account_is_rejected_naming_the_qos() -> None:
    result = run_launch(env={"RCAC_ACCOUNT": "standby"})
    assert result.exit_code != 0
    assert "-q standby" in result.stderr


# -- Guardrail 2: the standby walltime cap ----------------------------------


def test_guardrail_standby_over_four_hours_is_rejected_naming_the_cap() -> None:
    result = run_launch("--qos", "standby", "--time", "08:00:00")
    assert result.exit_code != 0
    assert re.search(r"4 ?h|4-hour|04:00:00 cap", result.stderr, re.I), result.stderr


def test_guardrail_standby_inside_the_cap_is_accepted() -> None:
    """The control for guardrail 2: it must not reject every standby job."""
    result = run_launch("--qos", "standby", "--time", "03:59:59", "-n", "1")
    assert result.exit_code == 0, result.output
    assert "#SBATCH -q standby" in result.stdout


# -- Guardrail 3: the training partition's mandatory QOS ---------------------


def test_guardrail_training_partition_with_wrong_qos_is_rejected() -> None:
    result = run_launch("--partition", "training", "--qos", "normal")
    assert result.exit_code != 0
    assert "-q training" in result.stderr


# -- Guardrail 4: an unrecognised partition warns; it does not block ---------


def test_guardrail_unknown_partition_warns_and_still_exits_zero() -> None:
    """AC-3.8's tri-state at the CLI boundary.

    ``h100`` is the partition RCAC's own pages disagree about.  With no
    discovery to consult, blocking it would refuse a job that may well run;
    ``sbatch`` rejects a wrong partition in under a second and names the valid
    ones.
    """
    result = run_launch("--partition", "h100", "--no-discover", "-n", "1")
    assert result.exit_code == 0, result.output
    assert re.search(r"unverified|unknown partition", result.stderr, re.I), (
        result.stderr
    )


def test_warnings_are_repeated_at_the_end_of_the_run() -> None:
    """One warning above a 40-job sweep has scrolled off by the time it matters."""
    result = run_launch("--partition", "h100", "--no-discover", "-n", "2")
    assert result.stderr.count("h100") >= 2, result.stderr
    assert "Warnings raised during this run" in result.stderr
