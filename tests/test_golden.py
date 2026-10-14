"""Golden snapshots of rendered SLURM scripts.

``tests/golden/injected/``
    Rendered through the injection points of :func:`render_slurm_script`
    (explicit ``config``, explicit ``template_dir``). Nothing on disk outside
    ``templates/`` influences the result, so this set is a pure template
    regression guard.

A second set, ``tests/golden/realconfig/``, was captured at C0 and rendered
through the default path -- the real config loader plus the on-disk ``configs/``
YAML. It was the gate for the C1 mechanical rename, which renames the loader
function, the ``load_config`` default name, *and* the config file: renaming one
but not the others would have broken it. It rendered byte-identical across C1
and was retired at the end of that commit (plan section 8.3), because C2 kills
it twice over -- the placeholder becomes ``YOUR_PURDUE_ID`` and the new
``scratch_root`` accessor changes how paths resolve.

Regenerate deliberately (never to make a red test go green) with::

    .venv/bin/python -m tests.test_golden
"""

from __future__ import annotations

import difflib
from pathlib import Path

import pytest

from rcac_cli.scheduler import (
    generate_job_array_script,
    generate_multi_gpu_script,
    generate_pipeline_script,
    generate_single_gpu_script,
    get_template_dir,
)

GOLDEN_DIR = Path(__file__).parent / "golden"

# The user value used by the injected set. Deliberately different from the
# on-disk config value so the two sets cannot silently collapse into one.
# `paths.scratch` keeps its `{user}` template form -- the loader does not format
# it, `config.scratch_root()` does (B10/AC-3.17).
INJECTED_CONFIG = {
    "ssh": {"user": "golden_user"},
    "paths": {"scratch": "/scratch/gilbreth/{user}"},
}

# From C3b the generators render from a `gilbreth.SubmissionPlan`, so `-A`,
# `-p` and `--time` are required and carry no defaults. These values are
# fixtures, not recommendations: `a100-40gb` with 2 GPUs is chosen because it
# is the one combination for which `--constraint=nvlink` is documented, so the
# snapshots cover the constraint line as well as the plain header.
SUBMISSION = {
    "account": "golden_account",
    "partition": "a100-40gb",
    "qos": "normal",
    "time": "04:00:00",
}

# name -> callable(**injection_kwargs) -> rendered script
CASES = {
    "single_gpu": lambda **inj: generate_single_gpu_script(
        project="golden_project",
        exp_id="exp_001",
        script="train.py",
        args="--lr 1e-4",
        conda_env="golden_env",
        config_path="configs/exp_001.yaml",
        **SUBMISSION,
        **inj,
    ),
    "multi_gpu": lambda **inj: generate_multi_gpu_script(
        project="golden_project",
        exp_id="exp_001",
        script="train.py",
        args="--lr 1e-4",
        nodes=1,
        gpus_per_node=2,
        constraint="nvlink",
        conda_env="golden_env",
        config_path="configs/exp_001.yaml",
        **SUBMISSION,
        **inj,
    ),
    "job_array": lambda **inj: generate_job_array_script(
        project="golden_project",
        script="train.py",
        config_dir="configs/sweep",
        array_max=7,
        args="--epochs 10",
        array_simultaneous=2,
        conda_env="golden_env",
        **SUBMISSION,
        **inj,
    ),
    "pipeline_with_dependency": lambda **inj: generate_pipeline_script(
        project="golden_project",
        exp_id="exp_001",
        stage_name="evaluate",
        script="evaluate.py",
        dependency_job_id="12345",
        conda_env="golden_env",
        config_path="configs/exp_001.yaml",
        **SUBMISSION,
        **inj,
    ),
    "pipeline_without_dependency": lambda **inj: generate_pipeline_script(
        project="golden_project",
        exp_id="exp_001",
        stage_name="train",
        script="train.py",
        conda_env="golden_env",
        config_path="configs/exp_001.yaml",
        **SUBMISSION,
        **inj,
    ),
}

# set name -> injection kwargs handed to every generator in that set
SETS = {
    "injected": lambda: {
        "template_dir": get_template_dir(),
        "config": INJECTED_CONFIG,
    },
}


def render(set_name: str, case_name: str) -> str:
    """Render one case for one snapshot set."""
    return CASES[case_name](**SETS[set_name]())


def golden_path(set_name: str, case_name: str) -> Path:
    return GOLDEN_DIR / set_name / f"{case_name}.sh"


ALL_CASES = [(s, c) for s in SETS for c in CASES]


@pytest.mark.parametrize("set_name,case_name", ALL_CASES)
def test_golden_snapshot_matches(set_name: str, case_name: str) -> None:
    path = golden_path(set_name, case_name)
    assert path.exists(), (
        f"Missing golden snapshot {path}. "
        f"Regenerate with: python -m tests.test_golden"
    )
    expected = path.read_bytes()
    actual = render(set_name, case_name).encode()
    if actual != expected:
        diff = "\n".join(
            difflib.unified_diff(
                expected.decode(errors="replace").splitlines(),
                actual.decode(errors="replace").splitlines(),
                fromfile=f"golden/{set_name}/{case_name}.sh",
                tofile=f"rendered/{set_name}/{case_name}.sh",
                lineterm="",
            )
        )
        pytest.fail(
            f"Rendered output drifted from {path}:\n{diff}\n\n"
            f"If this change is intended, regenerate with: "
            f"python -m tests.test_golden"
        )


def test_template_dir_injection_is_honoured(tmp_path: Path) -> None:
    """``template_dir`` must select the loader root, not be ignored."""
    (tmp_path / "single_gpu.sh.j2").write_text(
        "sentinel {{ project }} {{ user }} {{ scratch_root }}\n"
    )
    from rcac_cli.scheduler import render_slurm_script

    out = render_slurm_script(
        "single_gpu.sh.j2",
        template_dir=tmp_path,
        config=INJECTED_CONFIG,
        project="golden_project",
    )
    assert out == "sentinel golden_project golden_user /scratch/gilbreth/golden_user\n"


def _regenerate() -> None:
    for set_name in SETS:
        for case_name in CASES:
            path = golden_path(set_name, case_name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(render(set_name, case_name))
            print(f"wrote {path}")


if __name__ == "__main__":
    _regenerate()
