"""AC-4 at the command boundary -- ``rcac-partition`` and the tri-state it feeds.

``tests/test_discovery.py`` proves the probes and the parsers.  This file proves
what the *commands* do with them, which is a different property and the one that
actually reaches a user:

* **AC-4.6 -- the onboarding path.**  ``rcac-partition --no-discover`` with no
  ``RCAC_USER`` at all must exit **0** and print the hint table under its
  banner.  This is the first command a new user runs, before they know their
  account name, and it is the whole reason
  :func:`rcac_cli.config.get_rcac_config` was made not to raise (P4/R19/F-4).
  The pre-port command called ``require_user`` on line 3 and exited 1 telling
  the user to set a variable they would only understand after reading the
  output they were being denied.

* **AC-3.8 end to end.**  The tri-state rule lives in
  :func:`rcac_cli.gilbreth.build_directives` and is unit-tested there.  What is
  tested here is the *wiring*: that ``rcac-exp launch`` really supplies a
  discovered partition set, that a partition missing from it is a hard error,
  and that a discovery which came back empty folds to "not discovered" instead
  of rejecting every submission on the strength of an answer nobody gave.

The ``slist`` and ``sinfo`` payloads below are inline rather than fixtures on
purpose: they are the smallest input that drives *this file's* assertions, and
the doc-derived fixtures under ``tests/fixtures`` belong to the parser tests.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

from rcac_cli import discovery, partition_selector
from rcac_cli.discovery import UNVERIFIED_BANNER
from rcac_cli.experiment import cli as exp_cli
from rcac_cli.gilbreth import PARTITION_GPU, PARTITION_HINTS
from rcac_cli.partition_selector import cost_label, select_partition
from rcac_cli.simulator import cli as sim_cli

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Enough of a config for `get_rcac_config` to resolve, with no credentials in
#: it.  Individual tests add `RCAC_USER` when they want a probe to be *tried*.
BASE_ENV = {"RCAC_CONFIG_DIR": str(REPO_ROOT / "configs")}

#: ``sinfo -o '%P %a %D %T %N %G' --noheader``.  Three partitions, one of them
#: (`a30`) not in the `slist` payload, so the union in
#: `ClusterView.known_partitions` is observable.
SINFO_OUTPUT = """\
a10 up 5 idle gilbreth-a[000-004] gpu:a10:3
a30 up 2 mix gilbreth-b[000-001] gpu:a30:3
a100-80gb up 3 alloc gilbreth-c[000-002] gpu:a100:2
"""

#: ``slist``, wide shape: one Total/Queued/Running/Free group per GPU type.
SLIST_OUTPUT = """\
Current Number of GPUs
                       a10                          a100-80gb
Account   Total Queued Running Free      Total Queued Running Free
========= ===== ====== ======= ====      ===== ====== ======= ====
debug         6      0       1    5          4      0       0    4
mylab        12      2       3    9          8      0       2    6
"""


@pytest.fixture(autouse=True)
def _isolated_session_cache():
    """A probe cached by one test must not answer another one's question."""
    discovery.clear_cache()
    yield
    discovery.clear_cache()


def fake_cluster(monkeypatch, *, slist=SLIST_OUTPUT, sinfo=SINFO_OUTPUT, rc=0):
    """Answer every probe from a string, at the transport boundary.

    Patched at :func:`rcac_cli.discovery._run_remote` rather than at
    :func:`~rcac_cli.discovery.survey` so the real command strings, the real
    parsers and the real :class:`~rcac_cli.discovery.ClusterView` all run.  A
    test that stubs the view proves the view's own logic, not the wiring.
    """
    calls: list[str] = []

    def _fake(cfg, command, timeout=None):
        calls.append(command)
        if command.startswith("slist"):
            payload = slist
        elif command.startswith("sinfo"):
            payload = sinfo
        else:
            payload = ""
        if payload is None:
            return None
        return subprocess.CompletedProcess(["ssh"], rc, stdout=payload, stderr="")

    monkeypatch.setattr(discovery, "_run_remote", _fake)
    return calls


# ==========================================================================
# AC-4.6 -- the onboarding regression test
# ==========================================================================


def test_partition_with_no_user_and_no_discovery_exits_zero_with_the_banner():
    """The command a brand-new user runs first must not throw.

    ``RCAC_USER`` unset, ``--no-discover``, no network.  Exit 0, the hint table,
    and a banner that names the fix.  A non-zero exit here is the R19 failure:
    the user cannot set the variable the error demands until they have read the
    table the error is withholding.
    """
    result = CliRunner().invoke(
        select_partition,
        ["--no-discover"],
        env={**BASE_ENV, "RCAC_USER": None, "RCAC_ACCOUNT": None},
    )

    assert result.exit_code == 0, result.output
    assert UNVERIFIED_BANNER in result.output
    # The hint table really is there -- a banner over an empty table would pass
    # a naive assertion and help nobody.
    for name in ("a10", "a30", "a100-40gb", "a100-80gb"):
        assert name in result.output, result.output
    assert "Recommended: --partition=" in result.output


def test_the_onboarding_path_makes_no_ssh_attempt(monkeypatch):
    """`--no-discover` must skip the probes, not run and discard them.

    Without this the AC-4.6 test above would pass for the wrong reason on a
    laptop with no network: the probes would run, fail, and be reported as
    warnings, and the user would wait out two SSH timeouts to be told nothing.
    """
    calls = fake_cluster(monkeypatch)
    result = CliRunner().invoke(
        select_partition, ["--no-discover"], env={**BASE_ENV, "RCAC_USER": "testid"}
    )
    assert result.exit_code == 0, result.output
    assert calls == []
    # And it does not warn about the flag the user just passed.
    assert "did not run" not in result.output


def test_no_discover_environment_variable_is_honoured(monkeypatch):
    """AC-4.7 -- the env var is the flag for commands run from a script."""
    calls = fake_cluster(monkeypatch)
    result = CliRunner().invoke(
        select_partition,
        [],
        env={**BASE_ENV, "RCAC_USER": "testid", discovery.NO_DISCOVER_ENV: "1"},
    )
    assert result.exit_code == 0, result.output
    assert calls == []
    assert UNVERIFIED_BANNER in result.output


# ==========================================================================
# AC-4.1 / AC-4.2 -- slist is the headline, sinfo enriches
# ==========================================================================


def test_slist_matrix_is_printed_above_the_ranking(monkeypatch):
    """AC-4.1 -- "which accounts can I charge?" is answered first.

    A ranked partition list the user holds no allocation on is advice they
    cannot act on, so the accounts come above the ranking, not below it.
    """
    fake_cluster(monkeypatch)
    result = CliRunner().invoke(
        select_partition, [], env={**BASE_ENV, "RCAC_USER": "testid"}
    )
    assert result.exit_code == 0, result.output
    assert "debug" in result.output and "mylab" in result.output
    assert result.output.index("Accounts (slist)") < result.output.index("Partitions")
    # free/total, read out of the matrix rather than echoed from the input.
    assert "5/6" in result.output and "6/8" in result.output
    # Discovery succeeded, so the table is not a hint table any more.
    assert UNVERIFIED_BANNER not in result.output


def test_a_failed_sinfo_is_a_warning_not_an_abort(monkeypatch):
    """AC-4.2 -- each probe is optional and fails on its own."""
    fake_cluster(monkeypatch, sinfo=None)
    result = CliRunner().invoke(
        select_partition, [], env={**BASE_ENV, "RCAC_USER": "testid"}
    )
    assert result.exit_code == 0, result.output
    assert "sinfo discovery did not run" in result.output
    # slist still answered, so the headline is real.
    assert "mylab" in result.output


def test_a_failed_slist_still_ranks_partitions(monkeypatch):
    """The other half of AC-4.2: the primary probe may fail too."""
    fake_cluster(monkeypatch, slist=None)
    result = CliRunner().invoke(
        select_partition, [], env={**BASE_ENV, "RCAC_USER": "testid"}
    )
    assert result.exit_code == 0, result.output
    assert "slist discovery did not run" in result.output
    assert "not discovered" in result.output
    assert "Recommended: --partition=" in result.output


def test_discovered_idle_nodes_change_the_recommendation(monkeypatch):
    """`sinfo` is an enrichment that must actually enrich something.

    Offline, `a100-80gb` wins on published GPU memory.  With `a10` reporting
    five idle nodes and `a100-80gb` fully allocated, availability wins -- which
    is the entire point of asking the cluster.
    """
    offline = CliRunner().invoke(
        select_partition, ["--no-discover"], env=BASE_ENV
    )
    assert "Recommended: --partition=a100-80gb" in offline.output

    fake_cluster(monkeypatch)
    online = CliRunner().invoke(
        select_partition, ["-g", "1"], env={**BASE_ENV, "RCAC_USER": "testid"}
    )
    assert online.exit_code == 0, online.output
    assert "Recommended: --partition=a10" in online.output


def test_the_session_cache_asks_once_and_no_cache_asks_again(monkeypatch):
    """AC-4.7 -- one process, one probe; `--no-cache` is the escape."""
    calls = fake_cluster(monkeypatch)
    env = {**BASE_ENV, "RCAC_USER": "testid"}

    CliRunner().invoke(select_partition, [], env=env)
    first = len(calls)
    assert first >= 2, calls

    CliRunner().invoke(select_partition, [], env=env)
    assert len(calls) == first, "the second run re-probed a cached cluster"

    CliRunner().invoke(select_partition, ["--no-cache"], env=env)
    assert len(calls) > first


# ==========================================================================
# AC-4.3 / AC-4.5 / AC-4.8 -- the module's own shape
# ==========================================================================


def test_partition_info_is_gone_and_the_hints_are_used():
    """AC-4.3 -- one partition table, and it is the one carrying sources."""
    source = (REPO_ROOT / "rcac_cli" / "partition_selector.py").read_text(
        encoding="utf-8"
    )
    assert "PARTITION_INFO" not in source
    assert "PARTITION_HINTS" in source
    for hint in PARTITION_HINTS.values():
        assert hint.get("source"), "every hint must name where it came from"


def test_partition_selector_holds_no_parser():
    """AC-4.8 -- every probe and every parser lives in `discovery.py`.

    Asserted on the source text because the property is about *where the code
    is*, not about what it computes: a parser that had merely been renamed
    would still be the accidental seam this criterion exists to close.
    """
    source = (REPO_ROOT / "rcac_cli" / "partition_selector.py").read_text(
        encoding="utf-8"
    )
    for token in ("def _parse", "def parse_", "sinfo -o", "subprocess", "splitlines"):
        assert token not in source, f"{token!r} is back in partition_selector.py"


@pytest.mark.parametrize("tier", [0, 1, 2, 3, 4, 5, 7, 99, -1, None, "x"])
def test_cost_label_never_raises_and_never_renders_empty(tier):
    """AC-4.5 / B5 -- the `IndexError` and the blank column, both closed.

    The pre-port code indexed the list ``["", "Low", "Med", "", "High"]``: tier
    3 rendered an empty cell and tier 5 or above raised `IndexError` from a
    display column.  Tier 7 is the named regression.
    """
    label = cost_label(tier)
    assert isinstance(label, str) and label.strip()


def test_cost_tier_seven_survives_the_whole_render():
    """The crash was in the table, so the table is what the test drives."""
    ranked = [("weird", 1.0, {
        "gpu": "?", "gpu_memory_gb": None, "gpus_per_node": None,
        "idle_nodes": 0, "mix_nodes": 0, "cost_tier": 7, "verified": True,
    })]
    assert any("t7" in line for line in partition_selector.render_ranking(ranked))


# ==========================================================================
# AC-4.4 / B8 -- one partition -> GPU map
# ==========================================================================


def test_simulator_uses_the_shared_partition_map():
    """B8 -- and the map must key on partitions that exist on this cluster."""
    from rcac_cli import simulator

    assert simulator.PARTITION_GPU is PARTITION_GPU
    assert set(PARTITION_GPU) == set(PARTITION_HINTS)


def test_init_pipeline_accepts_every_documented_partition(tmp_path, monkeypatch):
    """The regression B8 actually caused: every valid partition was rejected.

    `simulator.py` carried the previous cluster's partition names, so
    `rcac-sim init-pipeline -p a10` exited 1 with "Unknown partition: a10" --
    and would have accepted `amd_a100nv_8`, which does not exist here.
    """
    env = {
        **BASE_ENV,
        "RCAC_USER": "testid",
        "RCAC_ACCOUNT": "testqueue",
        discovery.NO_DISCOVER_ENV: "1",
    }
    # `training` needs `-q training`, which this command does not pass; that is
    # AC-3.6's rule and it is tested in test_choke_point.py.
    for name in ("a10", "a30", "a100-40gb", "a100-80gb"):
        out = tmp_path / f"{name}.yaml"
        result = CliRunner().invoke(
            sim_cli,
            ["init-pipeline", "genesis", "-p", name, "-o", str(out)],
            env=env,
        )
        assert result.exit_code == 0, (name, result.output)
        assert out.is_file()

    stale = CliRunner().invoke(
        sim_cli,
        ["init-pipeline", "genesis", "-p", "amd_a100nv_8",
         "-o", str(tmp_path / "stale.yaml")],
        env=env,
    )
    assert stale.exit_code != 0
    assert "Unknown partition" in stale.output


# ==========================================================================
# AC-3.8 -- the tri-state, wired end to end
# ==========================================================================

SIMPLE_EXAMPLE = REPO_ROOT / "examples" / "experiment_simple.yaml"

LAUNCH_ENV = {
    **BASE_ENV,
    "RCAC_USER": "testid",
    "RCAC_ACCOUNT": "testqueue",
}


def run_launch(*args, env=None):
    return CliRunner().invoke(
        exp_cli,
        ["launch", str(SIMPLE_EXAMPLE), "--dry-run", "-n", "1", *args],
        env={**LAUNCH_ENV, **(env or {})},
    )


def test_a_discovered_partition_list_rejects_a_partition_that_is_missing(monkeypatch):
    """State 2 of the tri-state: the cluster said, so absence is a hard error.

    `h100` is the partition RCAC's pages disagree about. Offline that is a
    warning; once the cluster has enumerated its partitions and h100 is not
    among them, it is an error -- there is nothing left to be uncertain about.
    """
    fake_cluster(monkeypatch)
    result = run_launch("--partition", "h100")
    assert result.exit_code != 0
    assert "does not exist on this cluster" in result.output + result.stderr


def test_a_discovered_partition_list_admits_a_partition_that_is_present(monkeypatch):
    """The same run, with a partition the fake cluster does report."""
    fake_cluster(monkeypatch)
    result = run_launch("--partition", "a30")
    assert result.exit_code == 0, result.output
    assert "#SBATCH -p a30" in result.output


def test_no_discover_returns_the_tri_state_to_warn_and_render(monkeypatch):
    """State 1: nothing was asked, so nothing may be concluded.

    Same partition, same cluster, opposite outcome -- which is what makes this
    a tri-state rather than an allowlist with a flag on it.
    """
    fake_cluster(monkeypatch)
    result = run_launch("--partition", "h100", "--no-discover")
    assert result.exit_code == 0, result.output
    assert "#SBATCH -p h100" in result.output


def test_an_empty_discovery_folds_back_to_not_discovered(monkeypatch):
    """The load-bearing fold.

    Both probes succeed and both report nothing.  Honouring that literally
    would produce an empty `known_partitions`, and every submission on the
    cluster would then be rejected for naming a partition that "does not
    exist".  A discovery that returned zero partitions is a discovery that
    failed without raising.
    """
    fake_cluster(monkeypatch, slist="", sinfo="")
    result = run_launch("--partition", "h100")
    assert result.exit_code == 0, result.output
    assert "#SBATCH -p h100" in result.output


def test_a_failed_probe_warns_the_launch_run_rather_than_aborting_it(monkeypatch):
    """AC-4.2 on the submission path, where an abort would cost the most."""
    fake_cluster(monkeypatch, slist=None, sinfo=None)
    result = run_launch()
    assert result.exit_code == 0, result.output
    assert "discovery did not run" in result.stderr
