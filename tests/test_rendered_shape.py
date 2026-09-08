"""Shape assertions on rendered SLURM scripts (AC-3.3, AC-3.4).

These tests deliberately say nothing about *which* directives a template emits
or what values they carry. They assert only structural properties of rendered
output, so they must survive a full rewrite of the templates unchanged.
"""

from __future__ import annotations

import re

import pytest

from neuron_cli.scheduler import (
    generate_job_array_script,
    generate_multi_gpu_script,
    generate_pipeline_script,
    generate_single_gpu_script,
)

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
    return {
        "single_gpu": generate_single_gpu_script(
            project="proj", exp_id="exp_001", script="train.py", **overrides
        ),
        "multi_gpu": generate_multi_gpu_script(
            project="proj", exp_id="exp_001", script="train.py", **overrides
        ),
        "job_array": generate_job_array_script(
            project="proj", script="train.py", config_dir="cfg", array_max=3,
            **overrides
        ),
        "pipeline": generate_pipeline_script(
            project="proj", exp_id="exp_001", stage_name="train",
            script="train.py", **overrides
        ),
    }


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
    """The guard is wired to rendered output, not just to string literals."""
    blanked = _render_all(partition="", time="")[name]
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
