"""AC-3.19 and AC-6.4 -- one choke point, two distinct bypasses.

``tests/test_gilbreth.py`` proves the submission *rules*.
``tests/test_scheduler.py`` proves the primary path renders from them.  This
file proves the property those two cannot: that **nothing else** in the package
produces SLURM directives on its own.

The two bypasses are genuinely different, which is why one assertion cannot
cover both:

(a) **Header construction.**  ``keepalive.py`` built ``#SBATCH`` lines from an
    f-string.  Caught by a *text scan* for the token, which lives with the rest
    of the content bans in :mod:`tests.test_repo_hygiene` (row ``AC-3.19a``) --
    the token only ever occurs inside string literals, so a text scan answers
    the question exactly and an AST walk would be a fragile way to answer it no
    better.

(b) **Directive values.**  ``simulator.py`` builds no header at all.  It writes
    a ``slurm:`` mapping into generated YAML, where an invalid partition/QOS
    pair survives generation, survives the file being committed and shared, and
    only surfaces at ``rcac-exp launch`` -- possibly on another machine, days
    later, to someone who assumed a generated file was a valid one.  No scan
    for a directive token can see that.  Caught here, structurally, by
    :func:`test_no_module_builds_a_slurm_dict_literal`, and behaviourally by
    the ``init-pipeline`` tests below.

(b) is the generalisation of a specific miss, and it is deliberately not
written as "``simulator.py`` must call ``build_directives``".  The failure
class is *a producer nobody remembered*, so the assertion is phrased about
every module rather than the one that was caught.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from rcac_cli.experiment import cli as exp_cli
from rcac_cli.gilbreth import QUEUES_DOC, SubmissionError
from rcac_cli.keepalive import (
    KEEPALIVE_FALLBACK_PARTITION,
    KEEPALIVE_QOS,
    KEEPALIVE_WALLTIME,
)
from rcac_cli.keepalive import cli as keepalive_cli
from rcac_cli.scheduler import generate_keepalive_script
from rcac_cli.simulator import cli as sim_cli

REPO_ROOT = Path(__file__).resolve().parent.parent
PKG_ROOT = REPO_ROOT / "rcac_cli"

#: The module allowed to build a directive mapping: it is the choke point, and
#: ``SubmissionPlan.to_slurm_dict`` is the mapping every other module must get
#: its values from.
CHOKE_POINT = "gilbreth.py"

#: The two keys that make a mapping a *submission* rather than a report: ``-p``
#: chooses the GPU type and ``-q`` chooses the walltime cap and the charging.
#: A mapping carrying either one decides where a job runs and who pays for it.
DIRECTIVE_KEYS = frozenset({"partition", "qos"})

CONFIG = {
    "ssh": {"user": "testid"},
    "paths": {"scratch": "/scratch/gilbreth/{user}"},
}

CLI_ENV = {
    "RCAC_USER": "testid",
    "RCAC_ACCOUNT": "testqueue",
    "RCAC_CONFIG_DIR": str(REPO_ROOT / "configs"),
    # C5 wired real discovery into `rcac-exp launch`, so a run without this
    # would ssh to a Purdue login node from the test suite. The suite asserts
    # on what the CLI renders, not on what the cluster says; the discovery
    # path itself is covered by tests/test_discovery.py and by the AC-4.6
    # onboarding test, both of which set this deliberately.
    "RCAC_NO_DISCOVER": "1",
}


# ==========================================================================
# AC-3.19(b) -- the structural half
# ==========================================================================


def find_directive_dicts(source: str) -> list[tuple[int, list[str]]]:
    """Every dict *literal* in *source* carrying a partition/QOS key.

    Two forms are looked for, because both construct a mapping with a literal
    key spelled at the construction site: ``{"partition": ...}`` and
    ``dict(partition=...)``.  Neither is reachable from a
    :class:`~rcac_cli.gilbreth.SubmissionPlan`, which hands out its mapping
    through ``to_slurm_dict()`` -- a call, not a literal.

    What this cannot see is a mapping assembled key by key
    (``d["partition"] = x``).  That is a real limit and it is why the
    behavioural tests below exist: the two halves are complements, not
    duplicates.
    """
    tree = ast.parse(source)
    found: list[tuple[int, list[str]]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            keys = {
                k.value
                for k in node.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)
            }
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "dict"
        ):
            keys = {kw.arg for kw in node.keywords if kw.arg}
        else:
            continue
        hit = sorted(keys & DIRECTIVE_KEYS)
        if hit:
            found.append((node.lineno, hit))
    return found


PACKAGE_MODULES = sorted(PKG_ROOT.rglob("*.py"))


@pytest.mark.parametrize("module", PACKAGE_MODULES, ids=lambda p: p.name)
def test_no_module_builds_a_slurm_dict_literal(module: Path) -> None:
    """AC-3.19(b).  Directive values come from ``to_slurm_dict()``, or nowhere.

    The complement of the text scan.  ``simulator.py`` was the caught case, but
    the assertion is about every module because the failure class is a producer
    nobody thought of -- a fourth one writing its own ``slurm:`` block would
    reintroduce exactly the defect this criterion exists for, and no scan for
    ``#SBATCH`` would notice.

    There are no exemptions.  ``scheduler._plan_context`` used to be one and is
    not any more: it builds the header's template variables from
    ``plan.to_slurm_dict()``, so the mapping the templates render and the
    mapping ``simulator.py`` writes into YAML are now literally the same
    mapping.
    """
    if module.name == CHOKE_POINT:
        return
    offenders = find_directive_dicts(module.read_text(encoding="utf-8"))
    assert not offenders, (
        f"{module.name} constructs a dict carrying "
        + ", ".join(f"{keys} at line {line}" for line, keys in offenders)
        + f".\n  A mapping with a {sorted(DIRECTIVE_KEYS)} key is a submission: "
        "it decides which GPU the job lands on, what it is charged and how "
        "long it may run.\n  Build it with "
        "`rcac_cli.gilbreth.build_directives(...).to_slurm_dict()` so it "
        "cannot carry a pair the cluster will reject."
    )


def test_the_choke_point_is_the_one_module_that_may_build_one() -> None:
    """The control.  Without it the parametrised test could pass by finding nothing."""
    assert find_directive_dicts((PKG_ROOT / CHOKE_POINT).read_text()), (
        f"{CHOKE_POINT} no longer builds the directive mapping; if "
        "to_slurm_dict() moved, this file's exemption must move with it"
    )
    assert len(PACKAGE_MODULES) >= 10, "the module glob has gone vacuous"


@pytest.mark.parametrize(
    "source",
    [
        'PLAN = {"partition": p, "time": t}',
        'def f(): return dict(qos="standby")',
        'CFG = {"slurm": {"nodes": 1, "qos": q}}',
    ],
    ids=["dict-literal", "dict-call", "nested"],
)
def test_the_dict_scanner_fires_on_a_seeded_bypass(source: str) -> None:
    """AC-0.9's argument applied here: a scanner that cannot fail proves nothing."""
    assert find_directive_dicts(source)


@pytest.mark.parametrize(
    "source",
    [
        'CTX = {"nodes": 1, "mem": "8G"}',
        "CTX = plan.to_slurm_dict()",
        'REPORT = {"compatible": True, "gpu": "A100"}',
    ],
    ids=["no-directive-keys", "through-the-choke-point", "compat-report"],
)
def test_the_dict_scanner_does_not_fire_on_innocent_mappings(source: str) -> None:
    assert not find_directive_dicts(source)


# ==========================================================================
# AC-3.19(b) -- the behavioural half: generation time, not launch time
# ==========================================================================


def run_init_pipeline(tmp_path: Path, *args) -> tuple:
    """``rcac-sim init-pipeline`` into *tmp_path*, returning ``(result, out)``."""
    out = tmp_path / "experiment_sim.yaml"
    runner = CliRunner()
    result = runner.invoke(
        sim_cli,
        ["init-pipeline", "genesis", "-o", str(out), *args],
        env=CLI_ENV,
    )
    return result, out


@pytest.mark.parametrize(
    "partition,expected",
    [("standby", "-q standby"), ("training", "-q training")],
)
def test_init_pipeline_rejects_an_illegal_submission_at_generation(
    tmp_path: Path, partition: str, expected: str
) -> None:
    """The gate: the file must not exist if it could not be launched.

    Both cases fail *inside* ``build_directives``, which is the point --
    before C3c this command reached neither, and answered ``-p standby`` with
    "Unknown partition", a message that sends the user looking for a partition
    that was never the problem.
    """
    result, out = run_init_pipeline(tmp_path, "-p", partition)
    assert result.exit_code != 0
    assert expected in result.output
    assert QUEUES_DOC in result.output, (
        "the rejection did not come from the choke point: it cites no RCAC "
        f"page.\n{result.output}"
    )
    assert not out.exists(), (
        "init-pipeline wrote a manifest describing a submission it had already "
        "rejected"
    )


def test_init_pipeline_still_writes_a_manifest_it_can_validate(
    tmp_path: Path,
) -> None:
    """The control for the two above: it must not reject everything."""
    result, out = run_init_pipeline(tmp_path)
    assert result.exit_code == 0, result.output
    assert out.is_file()


def test_the_generated_slurm_block_is_a_submission_plan(tmp_path: Path) -> None:
    """AC-3.19(b) on the artefact.

    ``qos`` and ``mem`` are present because they came off the plan, not because
    this command remembered them; ``comment`` and ``gpus`` are absent because
    the plan has no such fields.  Both used to be written here, and
    ``rcac-exp launch`` answered the freshly generated file with two
    deprecation warnings.
    """
    result, out = run_init_pipeline(tmp_path)
    assert result.exit_code == 0, result.output
    slurm = yaml.safe_load(out.read_text())["slurm"]

    for key in ("partition", "qos", "time", "nodes", "gpus_per_node", "mem"):
        assert key in slurm, f"the generated slurm block has no {key!r}: {slurm}"
    for legacy in ("comment", "gpus", "gres", "queue"):
        assert legacy not in slurm, (
            f"the generated slurm block still carries the pre-port {legacy!r} "
            f"key, which launch answers with a deprecation warning: {slurm}"
        )
    assert "account" not in slurm, (
        "the account is a credential: launch resolves it from RCAC_ACCOUNT "
        "ahead of the manifest so nobody is nudged into committing it (P4)"
    )


def test_the_generated_slurm_block_survives_a_launch(tmp_path: Path) -> None:
    """The positive statement of AC-3.19(b): validated at generation *means* accepted.

    Otherwise "cannot write a YAML that only fails later" would be tested only
    in the direction that is easy to test.  The generated block must reach a
    rendered header carrying its own values and raise no deprecation warning
    on a file that was written seconds earlier.

    The **exit code is deliberately not asserted**.  ``init-pipeline`` also
    scaffolds stage arguments containing cross-stage placeholders
    (``{generate_data.output_path}``, ``{train.best_model}``) that ``launch``
    does not implement, so the second stage aborts.  That is a real defect and
    it is not this one: it lives in the pipeline-argument scaffold, has nothing
    to do with submission directives, and C3c would be over-reaching to fix it
    here.  Asserting on the header keeps this test about the thing it is named
    for; the placeholder defect is recorded in the C3c commit message.
    """
    result, out = run_init_pipeline(tmp_path)
    assert result.exit_code == 0, result.output
    slurm = yaml.safe_load(out.read_text())["slurm"]

    runner = CliRunner()
    launched = runner.invoke(
        exp_cli, ["launch", str(out), "--dry-run", "-n", "1"], env=CLI_ENV
    )
    for directive, value in (
        ("-p", slurm["partition"]),
        ("-q", slurm["qos"]),
        ("--time=", slurm["time"]),
        ("--mem=", slurm["mem"]),
    ):
        joiner = "" if directive.endswith("=") else " "
        assert f"{directive}{joiner}{value}" in launched.stdout, (
            f"the generated {directive} did not reach the rendered header:\n"
            f"{launched.stdout}"
        )
    assert "is ignored" not in launched.stderr, launched.stderr
    assert "pre-port spelling" not in launched.stderr, launched.stderr


# ==========================================================================
# AC-6.4 -- the keepalive header
# ==========================================================================

KEEPALIVE_ARGS = {
    "account": "ml-lab",
    "partition": KEEPALIVE_FALLBACK_PARTITION,
    "qos": KEEPALIVE_QOS,
    "time": KEEPALIVE_WALLTIME,
    "config": CONFIG,
}


def keepalive_directives(**overrides) -> list[str]:
    script = generate_keepalive_script(
        "/scratch/gilbreth/testid/.rcac-tools/keepalive.sh",
        **{**KEEPALIVE_ARGS, **overrides},
    )
    return [line for line in script.splitlines() if line.startswith("#SBATCH")]


@pytest.mark.parametrize(
    "flag", ("-A ", "-p ", "-q ", "--gpus-per-node=", "--mem=", "--time=")
)
def test_the_keepalive_header_carries_every_required_directive(flag: str) -> None:
    """AC-6.4 -- the same six AC-3.1 asserts of the four job templates.

    Not a coincidence and not a copy: the keepalive template includes the same
    ``_sbatch_header.j2``, so the property is inherited rather than re-tested.
    """
    lines = keepalive_directives()
    assert any(line.startswith(f"#SBATCH {flag}") for line in lines), (
        f"no `#SBATCH {flag}` in the keepalive header:\n  " + "\n  ".join(lines)
    )


def test_the_keepalive_header_no_longer_names_the_previous_cluster() -> None:
    """The three things the hand-built header got wrong, stated as a test.

    ``-p cas_v100_2`` is not a Gilbreth partition, ``--comment`` is not a
    Gilbreth flag, and there was no ``-A`` at all -- which alone makes the job
    unsubmittable, since Gilbreth requires an account on every submission.
    """
    script = "\n".join(keepalive_directives())
    assert "cas_v100_2" not in script
    assert "--comment" not in script
    assert "--gres" not in script
    assert "#SBATCH -A ml-lab" in script
    assert f"#SBATCH -q {KEEPALIVE_QOS}" in script


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({"account": "standby"}, "-q standby"),
        ({"time": "08:00:00"}, "cap"),
        ({"partition": "training"}, "-q training"),
    ],
    ids=["abolished-account", "over-the-standby-cap", "partition-qos-mismatch"],
)
def test_the_keepalive_job_inherits_the_ac3_guarantees(
    overrides: dict, expected: str
) -> None:
    """AC-6.4's "inherits every AC-3 guarantee", made checkable.

    The 8-hour case is the sharpest: it is rejected only because the keepalive
    job runs under ``-q standby``, whose 4-hour cap this generator never
    mentions.  The rule is enforced somewhere else entirely, which is what
    inheritance means.
    """
    with pytest.raises(SubmissionError) as excinfo:
        keepalive_directives(**overrides)
    assert expected in str(excinfo.value)


def test_setup_cron_writes_a_validated_header_to_the_cluster(monkeypatch) -> None:
    """AC-6.4 on the real path, not merely on the generator.

    Asserting on ``generate_keepalive_script`` alone would leave the thing that
    actually reaches a cluster untested: ``setup-cron`` could still be shipping
    its old f-string.  Every remote command is captured instead of run.
    """
    sent: list[str] = []

    def fake_run(cmd, *args, **kwargs):
        sent.append(cmd[-1])
        return subprocess.CompletedProcess(cmd, 0, stdout="Submitted 1", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = CliRunner().invoke(keepalive_cli, ["setup-cron"], env=CLI_ENV)
    assert result.exit_code == 0, result.output

    written = "\n".join(sent)
    assert "#SBATCH -A testqueue" in written, (
        "setup-cron sent no account-bearing header to the cluster:\n" + written
    )
    assert f"#SBATCH -p {KEEPALIVE_FALLBACK_PARTITION}" in written
    assert f"#SBATCH -q {KEEPALIVE_QOS}" in written
    assert "cas_v100_2" not in written
    assert "--comment" not in written
