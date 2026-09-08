"""Golden snapshots of rendered SLURM scripts.

Two sets are captured, and they are not redundant:

``tests/golden/injected/``
    Rendered through the injection points of :func:`render_slurm_script`
    (explicit ``config``, explicit ``template_dir``). Nothing on disk outside
    ``templates/`` influences the result, so this set is a pure template
    regression guard.

``tests/golden/realconfig/``
    Rendered through the default path: the real config loader plus the on-disk
    ``configs/`` YAML. This set is the meaningful gate for the mechanical
    rename commit, because that commit renames the loader function, the
    ``load_config`` default name, *and* the config file. Renaming one but not
    the others breaks these snapshots.

Regenerate deliberately (never to make a red test go green) with::

    .venv/bin/python -m tests.test_golden
"""

from __future__ import annotations

import difflib
from pathlib import Path

import pytest

from neuron_cli.scheduler import (
    generate_job_array_script,
    generate_multi_gpu_script,
    generate_pipeline_script,
    generate_single_gpu_script,
    get_template_dir,
)

GOLDEN_DIR = Path(__file__).parent / "golden"

# The user value used by the injected set. Deliberately different from the
# on-disk config value so the two sets cannot silently collapse into one.
INJECTED_CONFIG = {"ssh": {"user": "golden_user"}}

# name -> callable(**injection_kwargs) -> rendered script
CASES = {
    "single_gpu": lambda **inj: generate_single_gpu_script(
        project="golden_project",
        exp_id="exp_001",
        script="train.py",
        args="--lr 1e-4",
        conda_env="golden_env",
        config_path="configs/exp_001.yaml",
        **inj,
    ),
    "multi_gpu": lambda **inj: generate_multi_gpu_script(
        project="golden_project",
        exp_id="exp_001",
        script="train.py",
        args="--lr 1e-4",
        nodes=2,
        gpus=4,
        conda_env="golden_env",
        config_path="configs/exp_001.yaml",
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
        **inj,
    ),
    "pipeline_without_dependency": lambda **inj: generate_pipeline_script(
        project="golden_project",
        exp_id="exp_001",
        stage_name="train",
        script="train.py",
        conda_env="golden_env",
        config_path="configs/exp_001.yaml",
        **inj,
    ),
}

# set name -> injection kwargs handed to every generator in that set
SETS = {
    "injected": lambda: {
        "template_dir": get_template_dir(),
        "config": INJECTED_CONFIG,
    },
    "realconfig": lambda: {},
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


@pytest.mark.parametrize("case_name", list(CASES))
def test_injected_and_realconfig_sets_differ(case_name: str) -> None:
    """The injected config must actually reach the template.

    If these two sets were byte-identical the injected set would be dead
    weight and a broken ``config`` parameter would go unnoticed.
    """
    assert golden_path("injected", case_name).read_bytes() != golden_path(
        "realconfig", case_name
    ).read_bytes()


def test_template_dir_injection_is_honoured(tmp_path: Path) -> None:
    """``template_dir`` must select the loader root, not be ignored."""
    (tmp_path / "single_gpu.sh.j2").write_text("sentinel {{ project }} {{ user }}\n")
    from neuron_cli.scheduler import render_slurm_script

    out = render_slurm_script(
        "single_gpu.sh.j2",
        template_dir=tmp_path,
        config=INJECTED_CONFIG,
        project="golden_project",
    )
    assert out == "sentinel golden_project golden_user\n"


def _regenerate() -> None:
    for set_name in SETS:
        for case_name in CASES:
            path = golden_path(set_name, case_name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(render(set_name, case_name))
            print(f"wrote {path}")


if __name__ == "__main__":
    _regenerate()
