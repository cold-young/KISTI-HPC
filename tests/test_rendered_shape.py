"""Shape assertions on rendered SLURM scripts (AC-3.3, AC-3.4).

These tests deliberately say nothing about *which* directives a template emits
or what values they carry. They assert only structural properties of rendered
output, so they must survive a full rewrite of the templates unchanged.
"""

from __future__ import annotations

import re

import pytest

from rcac_cli.scheduler import (
    generate_job_array_script,
    generate_multi_gpu_script,
    generate_pipeline_script,
    generate_single_gpu_script,
    render_slurm_script,
)

# Rendered through the injected config rather than the on-disk YAML, so these
# shape assertions do not depend on whether RCAC_USER is exported.
CONFIG = {
    "ssh": {"user": "testid"},
    "paths": {"scratch": "/scratch/gilbreth/{user}"},
}

# A submission valid in every respect. From C3b the generators validate through
# `gilbreth.build_directives`, so `-A`, `-p` and `--time` are required and have
# no defaults -- there is nothing plausible left to fall back on.
VALID = {
    "account": "ml-lab",
    "partition": "a10",
    "time": "01:00:00",
    "config": CONFIG,
}

# AC-3.3. Matches an ``#SBATCH`` line whose directive carries no value:
# both the trailing-space form (``#SBATCH -p ``) and the trailing-equals
# form (``#SBATCH --mem=``), plus the bare form with nothing after it.
EMPTY_DIRECTIVE_RE = re.compile(r"^#SBATCH\s+-{1,2}[\w-]+[=\s]*$")

# AC-3.4. Any surviving Jinja delimiter means the template did not render.
LEFTOVER_DELIMITERS = ("{{", "{%")

# Lines the guard MUST flag.
EMPTY_DIRECTIVE_CATCH = [
    "#SBATCH -p ",
    "#SBATCH -p",
    "#SBATCH --mem=",
    "#SBATCH --time=",
    "#SBATCH -A  ",
]

# Lines the guard MUST NOT flag.
EMPTY_DIRECTIVE_NO_CATCH = [
    "#SBATCH -p a10",
    "#SBATCH --gpus-per-node=1",
]


def find_empty_directives(script: str) -> list[tuple[int, str]]:
    """Return ``(lineno, line)`` for every valueless ``#SBATCH`` directive."""
    return [
        (i, line)
        for i, line in enumerate(script.splitlines(), start=1)
        if EMPTY_DIRECTIVE_RE.match(line)
    ]


def _render_all(**overrides) -> dict[str, str]:
    """Render every template through its generator."""
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


# The directive values every template's header consumes, plus the body
# variables. Used only by the blank-value control below, which has to reach the
# renderer *underneath* the generators: from C3b a generator cannot produce a
# blank `-p` or `--time`, because `build_directives` raises first. The guard
# still has to be proven against real rendered output, so the control renders
# the templates directly.
_BLANKABLE_CONTEXT = {
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
    "project": "proj",
    "exp_id": "exp_001",
    "stage_name": "train",
    "script": "train.py",
    "args": "",
    "modules": [],
    "conda_env": "base",
    "env_type": None,
    "env_container": None,
    "config_path": "",
    "config_dir": "cfg",
    "array_max": 3,
    "array_simultaneous": None,
    "dependency_job_id": None,
}

_TEMPLATE_FOR = {
    "single_gpu": "single_gpu.sh.j2",
    "multi_gpu": "multi_gpu.sh.j2",
    "job_array": "job_array.sh.j2",
    "pipeline": "pipeline.sh.j2",
}


def _render_raw(name: str, **overrides) -> str:
    """Render one template directly, bypassing validation."""
    context = {**_BLANKABLE_CONTEXT, **overrides}
    return render_slurm_script(_TEMPLATE_FOR[name], config=CONFIG, **context)


RENDERED = _render_all()
TEMPLATE_NAMES = sorted(RENDERED)


# --- The guard itself, proven against the catch / no-catch list -------------


@pytest.mark.parametrize("line", EMPTY_DIRECTIVE_CATCH)
def test_empty_directive_regex_catches(line: str) -> None:
    assert find_empty_directives(line) == [(1, line)], (
        f"guard failed to flag valueless directive: {line!r}"
    )


@pytest.mark.parametrize("line", EMPTY_DIRECTIVE_NO_CATCH)
def test_empty_directive_regex_does_not_catch(line: str) -> None:
    assert find_empty_directives(line) == [], (
        f"guard wrongly flagged a directive that has a value: {line!r}"
    )


# --- AC-3.3: applied to real rendered output -------------------------------


@pytest.mark.parametrize("name", TEMPLATE_NAMES)
def test_no_directive_renders_empty(name: str) -> None:
    hits = find_empty_directives(RENDERED[name])
    assert not hits, (
        f"{name}: #SBATCH directive(s) rendered with no value: "
        + "; ".join(f"line {n}: {line!r}" for n, line in hits)
    )


@pytest.mark.parametrize("name", TEMPLATE_NAMES)
def test_guard_fires_when_values_render_empty(name: str) -> None:
    """The guard is wired to rendered output, not just to string literals.

    This renders the template directly rather than through its generator: from
    C3b the generators validate first, so a blank ``-p`` or ``--time`` can no
    longer be produced that way. That is a stronger guarantee, not a weaker
    one -- but it would silently turn this control into a vacuous test, so the
    control now reaches the renderer underneath.
    """
    blanked = _render_raw(name, partition="", time="")
    assert find_empty_directives(blanked), (
        f"{name}: guard did not fire even though directive values were blank"
    )


# --- AC-3.4 ----------------------------------------------------------------


@pytest.mark.parametrize("name", TEMPLATE_NAMES)
def test_no_leftover_jinja_delimiters(name: str) -> None:
    script = RENDERED[name]
    for delim in LEFTOVER_DELIMITERS:
        assert delim not in script, (
            f"{name}: unrendered Jinja delimiter {delim!r} survived at line "
            + str(
                next(
                    i
                    for i, line in enumerate(script.splitlines(), start=1)
                    if delim in line
                )
            )
        )
