"""Gilbreth's submission rules, tested against nothing else (AC-3.5-3.11, AC-4.9).

``rcac_cli/gilbreth.py`` is wired to no caller at C3a, and this file is the
reason that is a feature rather than dead code: the rules can be reviewed and
proved against RCAC's documentation before a single template or producer moves.
Every assertion here is about the *model* -- what Gilbreth accepts, what it
rejects, and what the CLI is merely unsure about -- and none of it touches a
template, ``scheduler.py``, ``experiment.py``, ``keepalive.py`` or
``simulator.py``.

The three-way split of outcomes is the thing to review:

* **hard error** -- RCAC documents the form as invalid (``-A standby``, standby
  over 4h, ``-p training`` under the wrong QOS, a partition the cluster itself
  denies having);
* **warn and render** -- we cannot tell, and ``sbatch`` can, in under a second;
* **silently fine** -- the common case, which must produce an *empty*
  ``warnings`` list, or the warnings channel becomes noise nobody reads.

That third property is why so many tests assert ``plan.warnings == []``.
"""

from __future__ import annotations

import pytest

from rcac_cli.gilbreth import (
    ABOLISHED_STANDBY_FORMS,
    ABSENT_QOS,
    DEFAULT_MEM_PER_GPU_GB,
    DEFAULT_QOS,
    NVLINK_CONSTRAINT,
    PARTITION_HINTS,
    QOS_HINTS,
    SubmissionError,
    SubmissionPlan,
    WalltimeError,
    build_directives,
    emit_warnings,
    parse_walltime,
)

#: A submission that is valid in every respect, so any single test can vary one
#: field and know that whatever it observes came from that field.
VALID = {
    "account": "ml-lab",
    "partition": "a10",
    "qos": "normal",
    "time": "01:00:00",
    "gpus_per_node": 1,
}


def plan_for(**overrides) -> SubmissionPlan:
    """Build a plan from :data:`VALID` with *overrides* applied."""
    kwargs = dict(VALID)
    kwargs.update(overrides)
    return build_directives(**kwargs)


# --------------------------------------------------------------------------
# AC-3.10 -- the walltime parser
# --------------------------------------------------------------------------

#: The twelve accepted cases AC-3.10 asks for, two per documented format.
#:
#: The two rows that matter most are ``("30", 1800)`` and ``("2-00", 172800)``.
#: SLURM's grammar is ambiguous to the eye in exactly those two places -- a bare
#: integer is *minutes*, and the ``D-HH`` hyphen separates days from hours
#: rather than denoting a range -- and reading either one backwards turns
#: AC-3.5's four-hour standby cap into a check that waves through a job sixty
#: times over the limit.
WALLTIME_CASES = [
    # MM -- a bare integer is MINUTES.
    ("30", 30 * 60),
    ("240", 4 * 3600),
    # MM:SS
    ("5:30", 5 * 60 + 30),
    ("00:45", 45),
    # HH:MM:SS
    ("01:00:00", 3600),
    ("03:59:59", 3 * 3600 + 59 * 60 + 59),
    # HH:MM:SS with unbounded hours -- SLURM accepts more than two digits.
    ("100:00:00", 100 * 3600),
    # D-HH -- days first.
    ("2-00", 2 * 86400),
    ("1-12", 86400 + 12 * 3600),
    # D-HH:MM
    ("1-00:30", 86400 + 30 * 60),
    # D-HH:MM:SS
    ("14-00:00:00", 14 * 86400),
    ("0-00:00:01", 1),
]


def test_walltime_case_table_has_twelve_rows():
    """AC-3.10 names a count; a table that quietly shrinks should fail loudly."""
    assert len(WALLTIME_CASES) == 12


@pytest.mark.parametrize("text,seconds", WALLTIME_CASES)
def test_parse_walltime_accepts(text, seconds):
    assert parse_walltime(text) == seconds


def test_bare_integer_is_minutes_not_hours_or_seconds():
    """The single most consequential reading in the whole parser."""
    assert parse_walltime("30") == 1800
    assert parse_walltime("30") != 30
    assert parse_walltime("30") != 30 * 3600


def test_day_hour_form_is_days_not_minutes():
    """``2-00`` is two days.  Read as ``2`` minutes it is 1440x too small."""
    assert parse_walltime("2-00") == 172800
    assert parse_walltime("2-00") > parse_walltime("23:59:59")


@pytest.mark.parametrize(
    "garbage",
    [
        "",
        "   ",
        "abc",
        "1:2:3:4",
        "-1",
        "1-",
        "-12",
        "1.5",
        "2h",
        "01:00:00:00",
        "1--12",
        "00:00:00:",
        None,
    ],
)
def test_parse_walltime_rejects_garbage(garbage):
    with pytest.raises(WalltimeError):
        parse_walltime(garbage)


def test_walltime_error_is_a_submission_error():
    """So a caller can catch one exception type for the whole choke point."""
    assert issubclass(WalltimeError, SubmissionError)


def test_every_hint_walltime_parses():
    """The hint tables are written in the syntax the parser accepts.

    Without this, a typo in a ``max_walltime`` string would surface as a
    ``WalltimeError`` raised from deep inside :func:`build_directives` on a
    submission that was perfectly valid.
    """
    for name, hint in list(PARTITION_HINTS.items()) + list(QOS_HINTS.items()):
        raw = hint.get("max_walltime")
        if raw is not None:
            assert parse_walltime(raw) > 0, name


# --------------------------------------------------------------------------
# AC-4.3 / AC-4.9 -- the hint tables carry their evidence
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(PARTITION_HINTS))
def test_partition_hint_carries_source_and_verified(name):
    hint = PARTITION_HINTS[name]
    assert isinstance(hint["verified"], bool)
    assert hint["source"], f"{name} has no source URL"
    for url in hint["source"]:
        assert url.startswith("https://docs.rcac.purdue.edu/"), url


@pytest.mark.parametrize("name", sorted(QOS_HINTS))
def test_qos_hint_carries_source_and_verified(name):
    hint = QOS_HINTS[name]
    assert isinstance(hint["verified"], bool)
    assert hint["source"]
    for url in hint["source"]:
        assert url.startswith("https://docs.rcac.purdue.edu/"), url


def test_documented_partitions_are_present():
    """The five partitions the queues page lists, by name."""
    for name in ("a10", "a30", "a100-40gb", "a100-80gb", "training"):
        assert name in PARTITION_HINTS


def test_documented_qos_names_are_exactly_three():
    assert set(QOS_HINTS) == {"normal", "standby", "training"}
    assert DEFAULT_QOS == "normal"


def test_absent_qos_names_are_recorded_as_absent():
    """``preemptible`` and ``debug`` exist elsewhere at RCAC, not on Gilbreth."""
    assert ABSENT_QOS == frozenset({"preemptible", "debug"})
    assert not (ABSENT_QOS & set(QOS_HINTS))


def test_h100_records_the_documentation_conflict():
    """AC-4.9 / R5.

    ``h100`` may be absent, or present with ``verified: False`` and two source
    URLs.  It is present, because the contradiction is itself a fact: the FAQ's
    sample ``slist`` output shows an H100 column that the queues page does not
    list.  An entry that recorded only one side would be indistinguishable from
    an entry written by someone who never noticed.
    """
    hint = PARTITION_HINTS["h100"]
    assert hint["verified"] is False
    assert len(hint["source"]) == 2
    assert len(set(hint["source"])) == 2


def test_only_h100_is_unverified():
    """Everything else is corroborated by the queues page without contradiction."""
    unverified = {n for n, h in PARTITION_HINTS.items() if not h["verified"]}
    assert unverified == {"h100"}


def test_standby_is_a_qos_and_not_a_partition_or_account():
    assert "standby" in QOS_HINTS
    assert "standby" not in PARTITION_HINTS
    assert "standby" in ABOLISHED_STANDBY_FORMS


def test_standby_qos_is_uncharged_and_capped_at_four_hours():
    assert QOS_HINTS["standby"]["charged"] is False
    assert parse_walltime(QOS_HINTS["standby"]["max_walltime"]) == 4 * 3600


# --------------------------------------------------------------------------
# The both-flags-required model
# --------------------------------------------------------------------------


def test_happy_path_is_warning_free():
    plan = plan_for()
    assert plan.warnings == []
    assert plan.account == VALID["account"]
    assert plan.partition == "a10"
    assert plan.qos == "normal"
    assert plan.constraint is None


def test_account_is_required():
    with pytest.raises(SubmissionError, match="account"):
        plan_for(account="")


def test_partition_is_required():
    with pytest.raises(SubmissionError, match="partition"):
        plan_for(partition="")


def test_time_is_required_and_says_why():
    """The default is 30 minutes, which is the reason omitting it is an error."""
    with pytest.raises(SubmissionError, match="30 minutes"):
        plan_for(time="")


def test_qos_defaults_to_normal():
    assert plan_for(qos="").qos == "normal"


# --------------------------------------------------------------------------
# AC-3.5 -- the standby four-hour cap, at the boundary
# --------------------------------------------------------------------------


@pytest.mark.parametrize("time", ["03:59:59", "04:00:00"])
def test_standby_accepts_up_to_four_hours(time):
    plan = plan_for(qos="standby", time=time)
    assert plan.qos == "standby"
    assert plan.warnings == []


def test_standby_rejects_one_second_over_four_hours():
    with pytest.raises(SubmissionError, match="04:00:00"):
        plan_for(qos="standby", time="04:00:01")


def test_standby_rejects_a_long_job_in_day_hour_form():
    """The cap is enforced on parsed seconds, not on the string's shape."""
    with pytest.raises(SubmissionError):
        plan_for(qos="standby", time="1-00")


def test_standby_cap_is_a_hard_error_not_a_warning():
    """Silently clamping to 4h would submit a job that dies at hour four."""
    with pytest.raises(SubmissionError):
        plan_for(qos="standby", time="08:00:00")


# --------------------------------------------------------------------------
# AC-3.6 -- the training partition and its mandatory QOS
# --------------------------------------------------------------------------


def test_training_partition_with_training_qos_is_accepted():
    plan = plan_for(partition="training", qos="training", time="12:00:00")
    assert plan.partition == "training"
    assert plan.qos == "training"
    assert plan.warnings == []


@pytest.mark.parametrize("qos", ["normal", "standby"])
def test_training_partition_with_wrong_qos_is_rejected(qos):
    with pytest.raises(SubmissionError, match="-q training"):
        plan_for(partition="training", qos=qos, time="01:00:00")


def test_training_partition_qos_is_not_silently_forced():
    """AC-3.6 says *rejected*, not corrected.

    Forcing ``-q training`` would change both the walltime cap and the charging
    behind the user's back, so the mismatch has to surface as an error rather
    than as a quietly different job.
    """
    with pytest.raises(SubmissionError):
        plan_for(partition="training", qos=DEFAULT_QOS, time="01:00:00")


def test_training_rejects_over_twenty_four_hours():
    with pytest.raises(SubmissionError, match="24:00:00"):
        plan_for(partition="training", qos="training", time="24:00:01")


def test_training_accepts_exactly_twenty_four_hours():
    plan = plan_for(partition="training", qos="training", time="24:00:00")
    assert plan.time_seconds == 24 * 3600


def test_training_qos_on_another_partition_warns():
    """Undocumented rather than documented-invalid, so it warns and renders."""
    plan = plan_for(partition="a10", qos="training", time="01:00:00")
    assert any("training" in w for w in plan.warnings)


def test_normal_qos_rejects_over_two_weeks():
    with pytest.raises(SubmissionError):
        plan_for(time="14-00:00:01")


def test_normal_qos_accepts_exactly_two_weeks():
    assert plan_for(time="14-00:00:00").time_seconds == 14 * 86400


# --------------------------------------------------------------------------
# AC-3.7 -- the abolished `-A standby` / `-p standby` forms
# --------------------------------------------------------------------------


def test_account_standby_is_rejected_naming_the_replacement():
    with pytest.raises(SubmissionError, match=r"-q standby"):
        plan_for(account="standby")


def test_partition_standby_is_rejected_naming_the_replacement():
    with pytest.raises(SubmissionError, match=r"-q standby"):
        plan_for(partition="standby")


@pytest.mark.parametrize("value", ["standby", "STANDBY", " Standby "])
def test_abolished_account_form_is_caught_regardless_of_spelling(value):
    with pytest.raises(SubmissionError, match=r"-q standby"):
        plan_for(account=value)


def test_abolished_form_is_not_silently_rewritten_to_the_qos():
    """A job charged to the wrong account is worse than a job that never ran."""
    with pytest.raises(SubmissionError):
        plan_for(account="standby", qos="standby", time="01:00:00")


def test_abolished_check_is_exact_and_not_a_substring_match():
    """A real allocation may legitimately have ``standby`` in its name."""
    plan = plan_for(account="standby-lab", partition="a10")
    assert plan.account == "standby-lab"


def test_abolished_form_error_beats_the_missing_partition_error():
    """The specific message wins, so the user learns the replacement flag."""
    with pytest.raises(SubmissionError, match=r"-q standby"):
        build_directives(account="standby", partition="", time="01:00:00")


# --------------------------------------------------------------------------
# AC-3.8 -- the tri-state `known_partitions`
# --------------------------------------------------------------------------


def test_unknown_partition_warns_when_discovery_was_not_attempted():
    """``None`` means we do not know.  Absence of evidence is not evidence."""
    plan = plan_for(partition="a100-13gb", known_partitions=None)
    assert plan.partition == "a100-13gb"
    assert any("a100-13gb" in w for w in plan.warnings)


def test_unknown_partition_is_a_hard_error_when_discovery_succeeded():
    """A populated set is the cluster's own answer, and it is authoritative."""
    with pytest.raises(SubmissionError, match="does not exist"):
        plan_for(
            partition="a100-13gb",
            known_partitions=frozenset({"a10", "a30", "a100-40gb"}),
        )


def test_known_partition_present_in_discovery_is_silent():
    plan = plan_for(partition="a10", known_partitions=frozenset({"a10", "a30"}))
    assert plan.warnings == []


def test_discovered_partition_absent_from_hints_is_accepted():
    """The cluster outranks the offline table in both directions."""
    plan = plan_for(partition="h200", known_partitions=frozenset({"h200"}))
    assert plan.partition == "h200"
    assert plan.warnings == []


def test_empty_discovery_result_is_treated_as_no_discovery():
    """Judgment call, documented in ``_known_partition_set``.

    A discovery that succeeded and found zero partitions is a discovery that
    failed without raising; honouring it would hard-error every submission,
    which is the exact outcome AC-3.8's tri-state exists to prevent.  Asserted
    with an *unknown* partition, because a known one would pass either way.
    """
    plan = plan_for(partition="a100-13gb", known_partitions=frozenset())
    assert plan.partition == "a100-13gb"
    assert any("a100-13gb" in w for w in plan.warnings)


def test_unverified_partition_warns_about_the_doc_conflict():
    """``h100`` renders, and says why it might not (R5)."""
    plan = plan_for(partition="h100", known_partitions=None)
    assert plan.partition == "h100"
    assert any("unverified" in w for w in plan.warnings)


def test_unverified_partition_confirmed_by_discovery_is_silent():
    """One ``slist`` settles what two RCAC pages could not (plan 6.2 L1)."""
    plan = plan_for(partition="h100", known_partitions=frozenset({"h100", "a10"}))
    assert plan.warnings == []


def test_absent_qos_warns_rather_than_erroring():
    plan = plan_for(qos="preemptible")
    assert plan.qos == "preemptible"
    assert any("preemptible" in w for w in plan.warnings)


def test_unknown_qos_warns():
    plan = plan_for(qos="turbo")
    assert any("turbo" in w for w in plan.warnings)


# --------------------------------------------------------------------------
# AC-3.9 -- the nvlink truth table
# --------------------------------------------------------------------------

#: ``(partition, gpus_per_node, opted_in) -> (effective constraint, warns)``.
#:
#: ``--constraint=nvlink`` is documented for one situation only: several GPUs on
#: ``a100-40gb`` serving a **single task**, in order to land on the NVLinked N
#: nodes.  Misuse sets ``plan.constraint = None`` *and* warns -- dropping it
#: silently would leave the user believing they got NVLink, and erroring would
#: fail a job that runs perfectly well without the feature.
NVLINK_TRUTH_TABLE = [
    ("a100-40gb", 2, True, NVLINK_CONSTRAINT, False),
    ("a100-40gb", 2, False, None, False),
    ("a100-40gb", 1, True, None, True),
    ("a100-40gb", 1, False, None, False),
    ("a100-80gb", 2, True, None, True),
    ("a100-80gb", 2, False, None, False),
    ("a100-80gb", 1, True, None, True),
    ("a100-80gb", 1, False, None, False),
]


@pytest.mark.parametrize(
    "partition,gpus,opted_in,expected,warns", NVLINK_TRUTH_TABLE
)
def test_nvlink_truth_table(partition, gpus, opted_in, expected, warns):
    plan = plan_for(
        partition=partition,
        gpus_per_node=gpus,
        constraint=NVLINK_CONSTRAINT if opted_in else None,
    )
    assert plan.constraint == expected
    assert bool(plan.warnings) is warns


def test_nvlink_misuse_warns_as_well_as_suppressing():
    """AC-3.9 requires both halves; either alone is a silent failure mode."""
    plan = plan_for(partition="a100-80gb", gpus_per_node=2, constraint="nvlink")
    assert plan.constraint is None
    assert any(NVLINK_CONSTRAINT in w for w in plan.warnings)


@pytest.mark.parametrize("nodes,ntasks_per_node", [(2, 1), (1, 2), (2, 4)])
def test_nvlink_requires_a_single_task(nodes, ntasks_per_node):
    """The precondition is asserted, not inherited from the templates.

    All four templates hardcode ``--ntasks-per-node=1`` today, which would make
    this check vacuous -- but ``nodes`` is a caller parameter and nothing stops
    a future template from parametrising the task count.  A precondition that
    holds by coincidence is not being checked.
    """
    plan = plan_for(
        partition="a100-40gb",
        gpus_per_node=2,
        nodes=nodes,
        ntasks_per_node=ntasks_per_node,
        constraint=NVLINK_CONSTRAINT,
    )
    assert plan.constraint is None
    assert any("single task" in w for w in plan.warnings)


def test_nvlink_single_task_multi_gpu_is_the_documented_case():
    plan = plan_for(
        partition="a100-40gb",
        gpus_per_node=4,
        nodes=1,
        ntasks_per_node=1,
        constraint=NVLINK_CONSTRAINT,
    )
    assert plan.constraint == NVLINK_CONSTRAINT
    assert plan.warnings == []


def test_unknown_constraint_warns_but_is_passed_through():
    """``nvlink`` is the only documented feature; the rest is the user's risk."""
    plan = plan_for(constraint="gpu_gen:volta")
    assert plan.constraint == "gpu_gen:volta"
    assert any("gpu_gen:volta" in w for w in plan.warnings)


def test_blank_constraint_is_no_constraint():
    assert plan_for(constraint="   ").constraint is None


# --------------------------------------------------------------------------
# AC-3.11 -- memory defaulting
# --------------------------------------------------------------------------


def test_mem_defaults_to_gpus_times_mem_per_gpu():
    plan = plan_for(gpus_per_node=2)
    assert plan.mem == f"{2 * DEFAULT_MEM_PER_GPU_GB}G"


def test_mem_default_scales_with_gpu_count():
    assert plan_for(gpus_per_node=1).mem == f"{DEFAULT_MEM_PER_GPU_GB}G"
    assert plan_for(gpus_per_node=4).mem == f"{4 * DEFAULT_MEM_PER_GPU_GB}G"


def test_explicit_mem_wins():
    assert plan_for(gpus_per_node=4, mem="120G").mem == "120G"


def test_mem_per_gpu_is_overridable():
    """It is a guess, so it must be a parameter and not a constant in a branch."""
    assert plan_for(gpus_per_node=2, mem_per_gpu_gb=64).mem == "128G"


def test_zero_gpus_is_rejected():
    with pytest.raises(SubmissionError, match="gpus-per-node"):
        plan_for(gpus_per_node=0)


# --------------------------------------------------------------------------
# AC-3.19(b) -- `to_slurm_dict`
# --------------------------------------------------------------------------


def test_to_slurm_dict_carries_the_effective_values():
    plan = plan_for(gpus_per_node=2, time="02:00:00")
    payload = plan.to_slurm_dict()
    assert payload["account"] == VALID["account"]
    assert payload["partition"] == "a10"
    assert payload["qos"] == "normal"
    assert payload["time"] == "02:00:00"
    assert payload["gpus_per_node"] == 2
    assert payload["mem"] == f"{2 * DEFAULT_MEM_PER_GPU_GB}G"


def test_to_slurm_dict_omits_a_suppressed_constraint():
    """A dropped nvlink must not come back as ``constraint: null`` in YAML."""
    plan = plan_for(partition="a100-80gb", gpus_per_node=2, constraint="nvlink")
    assert "constraint" not in plan.to_slurm_dict()


def test_to_slurm_dict_keeps_a_valid_constraint():
    plan = plan_for(partition="a100-40gb", gpus_per_node=2, constraint="nvlink")
    assert plan.to_slurm_dict()["constraint"] == NVLINK_CONSTRAINT


def test_no_gres_gpu_type_is_ever_emitted():
    """The GPU type is chosen by the partition; a --gres type contradicts it."""
    for partition in ("a10", "a30", "a100-40gb", "a100-80gb"):
        payload = plan_for(partition=partition).to_slurm_dict()
        assert "gres" not in payload
        for value in payload.values():
            assert "gpu:" not in str(value)


def test_gres_free_across_the_whole_hint_table():
    for name, hint in PARTITION_HINTS.items():
        assert "gres" not in hint, name


# --------------------------------------------------------------------------
# Warning transport
# --------------------------------------------------------------------------


def test_warnings_go_to_stderr(capsys):
    emit_warnings(["something is off"])
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "something is off" in captured.err
    assert captured.err.startswith("warning: ")


def test_warnings_can_be_repeated_under_a_header(capsys):
    """The end-of-run summary: one warning above a 40-job sweep scrolls off."""
    emit_warnings(["a", "b"], header="warnings from this run:")
    err = capsys.readouterr().err
    assert err.splitlines() == [
        "warnings from this run:",
        "warning: a",
        "warning: b",
    ]


def test_emit_warnings_prints_nothing_for_an_empty_list(capsys):
    emit_warnings([])
    assert capsys.readouterr().err == ""


def test_plan_carries_warnings_as_data_not_as_output(capsys):
    """The plan is the contract; printing is the caller's decision.

    Building a plan must not print, or a sweep that builds forty plans before
    submitting any of them emits its warnings in the wrong place.
    """
    plan = plan_for(qos="turbo")
    assert plan.warnings
    assert capsys.readouterr().err == ""
