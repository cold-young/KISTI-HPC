"""The discovery parsers, tested against doc-derived fixtures (AC-4.1, AC-7.6).

**These tests prove nothing about Gilbreth.**  Every fixture under
``tests/fixtures`` was reconstructed from RCAC's published sample output; not
one of them has been captured from a live login node.  Each carries a
``# PROVENANCE:`` block saying so, and every test that reads one carries
``@pytest.mark.unverified_fixture``.

The marker is a **provenance label, not a skip condition** (AC-7.6).  These
tests run and must pass.  What they establish is the half of the problem that
*can* be established offline: that :mod:`rcac_cli.discovery`'s parsers turn the
documented shapes into the right structures, drop what they cannot read instead
of raising, and never invent a name.  Plan section 6.2 rows L5 and L5b replace
the fixtures with real captures; the marker is how you find every assertion
that has to be re-read on that day.

Assertions are on the returned structures -- tuples, dicts, frozensets,
:class:`SlistMatrix` -- never on a substring of a repr, because the repr is not
the contract and a parser that returned the right characters in the wrong shape
would pass such a test.

Three defects in ``discovery.py`` were found by writing this file and are fixed
in the same commit; the tests that pin them say so at their definition:
:func:`test_slist_parses_the_published_gpu_matrix`,
:func:`test_slist_partition_names_are_the_names_sbatch_takes` and
:func:`test_modules_keeps_versioned_names`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rcac_cli.discovery import (
    ClusterView,
    ProbeResult,
    SlistMatrix,
    parse_modules,
    parse_purgelist,
    parse_sfeatures,
    parse_sinfo,
    parse_slist,
    parse_table,
    probe,
)
from rcac_cli.gilbreth import PARTITION_HINTS

FIXTURES = Path(__file__).resolve().parent / "fixtures"

#: Every fixture, and the only ones.  A new file under ``tests/fixtures`` has to
#: be added here, which is what forces it through the provenance test below.
FIXTURE_NAMES = (
    "slist.txt",
    "sinfo.txt",
    "sfeatures.txt",
    "modules.txt",
    "myquota.txt",
    "purgelist.txt",
)

PROVENANCE_PREFIX = "# PROVENANCE:"


def provenance_of(name: str) -> str:
    """The leading ``#`` block of a fixture: where the sample came from."""
    text = (FIXTURES / name).read_text(encoding="utf-8")
    block = []
    for line in text.splitlines():
        if not line.startswith("#"):
            break
        block.append(line)
    return "\n".join(block)


def load_fixture(name: str) -> str:
    """A fixture's payload -- the sample output, provenance block removed.

    The provenance is stripped rather than fed to the parser on purpose: the
    parsers must be tested against exactly what the command prints, or a passing
    test would be partly a test of comment tolerance.  Comment tolerance is
    tested separately, on strings written for it.

    It is also what keeps the fixtures replaceable.  ``slist | tee
    tests/fixtures/slist.txt`` (plan row L5b) writes a file with no provenance
    block; this loader still returns it, and
    :func:`test_every_fixture_declares_its_provenance` then fails until whoever
    ran the capture records where it came from.  That failure is the feature.
    """
    text = (FIXTURES / name).read_text(encoding="utf-8")
    payload = [line for line in text.splitlines() if not line.startswith("#")]
    return "\n".join(payload).strip("\n") + "\n"


# --------------------------------------------------------------------------
# AC-7.6 -- provenance is machine-checked, not promised in a review comment
# --------------------------------------------------------------------------


@pytest.mark.unverified_fixture
@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_every_fixture_declares_its_provenance(name):
    """Each fixture says, in the file, that it is doc-derived and unverified."""
    block = provenance_of(name)
    assert block.startswith(PROVENANCE_PREFIX), (
        f"{name} must open with a {PROVENANCE_PREFIX!r} block: a fixture whose "
        "origin is not written down becomes indistinguishable from a real "
        "capture the moment the commit message scrolls out of view"
    )
    lowered = block.lower()
    assert "doc-derived" in lowered
    assert "not captured from a live" in lowered
    assert "docs.rcac.purdue.edu" in block, (
        f"{name} must cite the page it was reconstructed from"
    )
    assert "unverified_fixture" in block, (
        f"{name} must name the marker that labels its tests"
    )


@pytest.mark.unverified_fixture
def test_no_fixture_escapes_the_provenance_check():
    """Every file under ``tests/fixtures`` is declared in FIXTURE_NAMES."""
    present = sorted(p.name for p in FIXTURES.iterdir() if p.is_file())
    assert present == sorted(FIXTURE_NAMES)


@pytest.mark.unverified_fixture
@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_loader_hands_the_parser_the_raw_sample(name):
    """The payload carries no provenance comment and is not empty."""
    payload = load_fixture(name)
    assert PROVENANCE_PREFIX not in payload
    assert payload.strip()


# --------------------------------------------------------------------------
# AC-4.1 -- slist, the primary probe, in the shape RCAC publishes
# --------------------------------------------------------------------------


@pytest.mark.unverified_fixture
def test_slist_parses_the_published_gpu_matrix():
    """The account x GPU-type matrix, cell for cell.

    **Defect fixed in this commit.**  Before it, ``parse_slist`` returned
    ``SlistMatrix()`` for this input -- an empty matrix, indistinguishable from
    a cluster that answered nothing.  Two reasons: the header line reads
    ``Accounts`` (plural) and the header lookup matched only the singular, and
    the four counters arrive packed into one column per GPU type
    (``11/0/2/11``) rather than spread across four, which is neither of the two
    shapes the parser knew.  The primary probe could not read the primary
    probe's own documented output.
    """
    matrix = parse_slist(load_fixture("slist.txt"))

    assert matrix.partitions == ("a10", "a30", "a100-40gb", "a100-80gb", "h100")
    assert matrix.counters == ("total", "queued", "running", "free")
    assert matrix.account_names == ("account1", "account2", "account3")
    assert bool(matrix) is True

    by_account = {row.account: row.counts for row in matrix.accounts}
    assert by_account["account1"] == {
        "a10": {"total": 1, "queued": 0, "running": 0, "free": 1},
    }
    assert by_account["account2"] == {
        "a100-40gb": {"total": 3, "queued": 0, "running": 3, "free": 3},
        "a100-80gb": {"total": 11, "queued": 0, "running": 2, "free": 11},
    }
    assert by_account["account3"] == {
        "a10": {"total": 1, "queued": 0, "running": 0, "free": 1},
        "a30": {"total": 1, "queued": 0, "running": 0, "free": 1},
        "h100": {"total": 11, "queued": 0, "running": 2, "free": 11},
    }


@pytest.mark.unverified_fixture
def test_slist_na_cells_are_unreported_not_zero():
    """``N/A`` means *no allocation*, which is not the same as *zero free*."""
    matrix = parse_slist(load_fixture("slist.txt"))
    account1 = matrix.accounts[0]

    assert "a30" not in account1.counts
    assert account1.free("a30") is None
    assert account1.total("a30") is None
    # ...and the difference survives into the aggregate: an account with no
    # allocation contributes nothing rather than a zero that looks like a
    # measurement.
    assert matrix.accounts_for("a30") == ("account3",)


@pytest.mark.unverified_fixture
def test_slist_carries_the_h100_column_through():
    """AC-4.9 / R5: the column the queues page does not list is still parsed.

    RCAC's pages contradict each other about ``h100``.  The parser's job is not
    to adjudicate that -- it is to report what the cluster said, so the
    contradiction reaches ``known_partitions`` as evidence instead of being
    silently resolved in favour of the offline table.
    """
    matrix = parse_slist(load_fixture("slist.txt"))

    assert "h100" in matrix.partitions
    assert matrix.accounts_for("h100") == ("account3",)
    assert matrix.free("h100") == 11
    assert PARTITION_HINTS["h100"]["verified"] is False


@pytest.mark.unverified_fixture
def test_slist_free_sums_over_every_chargeable_account():
    matrix = parse_slist(load_fixture("slist.txt"))

    assert matrix.free("a10") == 2  # account1 and account3, one each
    assert matrix.free("a100-80gb") == 11  # account2 only
    assert matrix.free("training") == 0  # not a column here at all
    assert matrix.accounts_for("a10") == ("account1", "account3")


@pytest.mark.unverified_fixture
def test_slist_partition_names_are_the_names_sbatch_takes():
    """The header prints ``A100-40gb``; ``sbatch -p`` wants ``a100-40gb``.

    **Defect fixed in this commit.**  The GPU-type labels were carried through
    with the header's own capitalisation, and ``build_directives`` compares
    ``-p`` against ``known_partitions`` exactly.  A discovered ``A10`` would
    therefore have made ``-p a10`` a *hard* SubmissionError -- discovery
    rejecting the very partition it had just discovered, and doing it only for
    users whose discovery succeeded.
    """
    matrix = parse_slist(load_fixture("slist.txt"))

    assert all(name == name.lower() for name in matrix.partitions)
    assert set(matrix.partitions) <= set(PARTITION_HINTS)


# -- slist edge cases, on strings written for them --------------------------

MATRIX_HEADER = """\
              |           Number of GPUs (Total/Queued/Running/Free)
Accounts      |   A10        A30       H100
==============|========== ========== ==========
"""


def test_slist_blank_input_is_an_empty_matrix():
    assert parse_slist("") == SlistMatrix()
    assert bool(parse_slist("")) is False


def test_slist_whitespace_only_input_is_an_empty_matrix():
    assert parse_slist("   \n\n\t\n") == SlistMatrix()


def test_slist_rules_only_file_is_an_empty_matrix():
    """A file that is nothing but table furniture has no header and no rows."""
    text = "=========================\n-------------------------\n"
    assert parse_slist(text) == SlistMatrix()


def test_slist_header_with_no_rows_is_an_empty_matrix():
    """Partitions without accounts are folded away, not reported.

    A matrix that named partitions but held no rows would satisfy
    ``known_partitions`` and hard-error every submission outside it, on the
    strength of an answer nobody gave.
    """
    assert parse_slist(MATRIX_HEADER) == SlistMatrix()


def test_slist_short_row_reports_only_the_cells_it_has():
    """A ragged row is truncated, not padded and not dropped."""
    matrix = parse_slist(MATRIX_HEADER + "account1      | 1/0/0/1\n")

    assert matrix.account_names == ("account1",)
    assert matrix.accounts[0].counts == {
        "a10": {"total": 1, "queued": 0, "running": 0, "free": 1}
    }
    assert matrix.accounts[0].free("h100") is None
    # The header still names all three partitions: the columns exist, this
    # account simply has nothing in two of them.
    assert matrix.partitions == ("a10", "a30", "h100")


def test_slist_surplus_cells_are_dropped_not_misattributed():
    """More cells than headers: the extras have no partition to belong to."""
    matrix = parse_slist(
        MATRIX_HEADER + "account1      | 1/0/0/1  2/0/0/2  3/0/0/3  9/9/9/9\n"
    )

    assert set(matrix.accounts[0].counts) == {"a10", "a30", "h100"}
    assert matrix.accounts[0].total("h100") == 3


def test_slist_all_na_row_is_kept_as_a_chargeable_account():
    """An account with no GPUs anywhere is still an account.

    Half of what AC-4.1 asks ``slist`` is *which accounts may I charge?*, and
    the answer does not depend on today's free-GPU count.
    """
    matrix = parse_slist(MATRIX_HEADER + "account9      |   N/A   N/A   N/A\n")

    assert matrix.account_names == ("account9",)
    assert matrix.accounts[0].counts == {}
    assert matrix.free("a10") == 0


def test_slist_unreadable_cell_costs_its_row_and_nothing_else():
    text = MATRIX_HEADER + (
        "account1      | 1/0/0/1   N/A   N/A\n"
        "broken        | ?/?/?/?   N/A   N/A\n"
        "account3      | 2/0/0/2   N/A   N/A\n"
    )
    matrix = parse_slist(text)

    assert matrix.account_names == ("account1", "account3")


def test_slist_ignores_prose_and_warning_lines():
    """A banner above, a note between rows and a footer below all survive."""
    text = (
        "slist: reading allocation database, this may take a moment\n"
        "\n"
        + MATRIX_HEADER
        + "account1      | 1/0/0/1   N/A   N/A\n"
        "WARNING: allocation data is refreshed nightly and may be stale\n"
        "account3      | 2/0/0/2   N/A   N/A\n"
        "\n"
        "Contact the help desk to request an allocation.\n"
    )
    matrix = parse_slist(text)

    assert matrix.account_names == ("account1", "account3")
    assert matrix.partitions == ("a10", "a30", "h100")


def test_slist_counter_names_are_numbered_when_the_prose_line_is_missing():
    """Unnamed beats invented: a wrong ``free`` is worse than an unnamed one."""
    text = (
        "Accounts      |   A10        A30       H100\n"
        "==============|========== ========== ==========\n"
        "account1      | 1/0/0/1      N/A        N/A\n"
    )
    matrix = parse_slist(text)

    assert matrix.counters == ()
    assert matrix.accounts[0].counts == {
        "a10": {"count0": 1, "count1": 0, "count2": 0, "count3": 1}
    }
    assert matrix.accounts[0].free("a10") is None


def test_slist_reads_the_counter_names_rather_than_assuming_them():
    """A site that renames or reorders the counters still parses."""
    text = (
        "              |     Number of GPUs (Free/Total)\n"
        "Accounts      |   A10\n"
        "==============|==========\n"
        "account1      | 3/4\n"
    )
    matrix = parse_slist(text)

    assert matrix.counters == ("free", "total")
    assert matrix.accounts[0].free("a10") == 3
    assert matrix.accounts[0].total("a10") == 4


def test_slist_matrix_header_row_is_not_read_as_an_account():
    text = MATRIX_HEADER + "Accounts      |  N/A  N/A  N/A\naccount1 | 1/0/0/1 N/A N/A\n"
    matrix = parse_slist(text)

    assert matrix.account_names == ("account1",)


def test_slist_wide_shape_still_parses():
    """The CPU-style report: one Total/Queue/Run/Free group, prose title above.

    Kept because ``slist`` on a non-GPU cluster prints this, and because the
    matrix branch added in this commit must not have taken the general path
    with it.
    """
    text = (
        "                     Current Number of Cores           Node\n"
        "Account          Total    Queue      Run     Free   Max Walltime\n"
        "============= ========= ======== ======== ======== ==============\n"
        "debug                32        0        0       32       00:30:00\n"
        "ml-lab             8000        0      100     7900    14-00:00:00\n"
    )
    matrix = parse_slist(text)

    assert matrix.account_names == ("debug", "ml-lab")
    assert matrix.counters == ("total", "queue", "run", "free")
    # No GPU-type header, so the single column is a placeholder -- and a
    # placeholder must never reach `known_partitions`.
    assert matrix.partitions == ()
    assert matrix.accounts[1].counts["column0"]["free"] == 7900


def test_slist_long_shape_still_parses():
    """One row per (account, partition) pair, partition in its own column."""
    text = (
        "Account   Queue       Total  Queued  Running  Free\n"
        "========= =========== ====== ======= ======== =====\n"
        "ml-lab    a10              3       0        1     2\n"
        "ml-lab    a100-80gb        2       0        0     2\n"
    )
    matrix = parse_slist(text)

    assert matrix.partitions == ("a10", "a100-80gb")
    assert matrix.accounts[0].free("a10") == 2
    assert matrix.accounts[1].total("a100-80gb") == 2


def test_slist_placeholder_columns_never_reach_known_partitions():
    """The ``columnN`` fallback is a placeholder and is filtered downstream."""
    text = (
        "                     Current Number of Cores\n"
        "Account          Total    Queue      Run     Free\n"
        "============= ========= ======== ======== ========\n"
        "debug                32        0        0       32\n"
    )
    view = ClusterView(
        slist=ProbeResult("slist", "slist", True, parse_slist(text)),
        sinfo=ProbeResult("sinfo", "sinfo", False, {}, reason="no"),
    )

    assert view.known_partitions is None


def test_slist_never_raises_on_junk():
    """AC-4.2: an unparseable payload is a warning, so it must not throw here."""
    for junk in ("\x00\x01\x02", "account", "Accounts |", "|||", "a" * 5000):
        assert isinstance(parse_slist(junk), SlistMatrix)


# --------------------------------------------------------------------------
# AC-4.8 -- sinfo, parsed here and nowhere else
# --------------------------------------------------------------------------


@pytest.mark.unverified_fixture
def test_sinfo_aggregates_node_states_per_partition():
    """``-o '%P %a %D %T %N %G'``: partition, avail, nodes, state, list, gres."""
    state = parse_sinfo(load_fixture("sinfo.txt"))

    assert set(state) == {"a10", "a30", "a100-40gb", "a100-80gb", "training"}
    assert state["a10"] == {
        "idle": 12,
        "mix": 4,
        "alloc": 0,
        "total": 16,
        "avail": "up",
    }
    assert state["a100-40gb"] == {
        "idle": 6,
        "mix": 4,
        "alloc": 0,
        "total": 10,
        "avail": "up",
    }


@pytest.mark.unverified_fixture
def test_sinfo_counts_an_unknown_state_in_the_total_only():
    """``down`` nodes exist but are not available; both facts are reported."""
    state = parse_sinfo(load_fixture("sinfo.txt"))

    assert state["a30"]["total"] == 10
    assert state["a30"]["alloc"] == 8
    assert state["a30"]["idle"] == 0
    assert state["a30"]["mix"] == 0


@pytest.mark.unverified_fixture
def test_sinfo_omits_h100_which_is_the_other_half_of_the_contradiction():
    state = parse_sinfo(load_fixture("sinfo.txt"))
    assert "h100" not in state


def test_sinfo_strips_the_default_partition_marker():
    """``%P`` marks the default partition with ``*``; the name does not."""
    state = parse_sinfo("a10* up 4 idle gilbreth-a[000-003] gpu:A10:3\n")

    assert set(state) == {"a10"}
    assert state["a10"]["idle"] == 4


def test_sinfo_accepts_the_tilde_and_long_state_spellings():
    text = (
        "a10 up 2 idle~ node[0-1] gpu:A10:3\n"
        "a10 up 3 mixed node[2-4] gpu:A10:3\n"
        "a10 up 4 allocated node[5-8] gpu:A10:3\n"
    )
    state = parse_sinfo(text)

    assert state["a10"] == {
        "idle": 2,
        "mix": 3,
        "alloc": 4,
        "total": 9,
        "avail": "up",
    }


def test_sinfo_drops_short_lines():
    """Fewer than six fields is not this format; the line is not guessed at."""
    text = "a10 up 4 idle\na30 up 2 idle node[0-1] gpu:A30:3\n"
    state = parse_sinfo(text)

    assert set(state) == {"a30"}


def test_sinfo_drops_a_line_whose_node_count_is_not_a_number():
    text = "a10 up N/A idle node[0-3] gpu:A10:3\na30 up 2 idle node[0-1] gpu:A30:3\n"
    state = parse_sinfo(text)

    assert set(state) == {"a30"}


def test_sinfo_blank_and_whitespace_input_is_an_empty_map():
    assert parse_sinfo("") == {}
    assert parse_sinfo("\n\n   \n") == {}


def test_sinfo_keeps_the_first_avail_it_saw():
    """``avail`` is set once per partition; the later line does not overwrite."""
    text = "a10 up 2 idle node[0-1] gpu:A10:3\na10 down 1 idle node2 gpu:A10:3\n"
    state = parse_sinfo(text)

    assert state["a10"]["avail"] == "up"
    assert state["a10"]["total"] == 3


# --------------------------------------------------------------------------
# sfeatures -- the most tolerant parser of the three
# --------------------------------------------------------------------------


@pytest.mark.unverified_fixture
def test_sfeatures_reads_the_feature_column():
    """One question is asked of this probe today: is ``nvlink`` real here?

    The exact set is pinned rather than a membership check, and it records a
    known looseness: the parser reads *every* column from the feature column
    rightwards, so the GRES strings come back as "features" too.  That is
    deliberate tolerance (the layout is unconfirmed -- plan row L5) and it is
    harmless for the one consumer, but it is not obvious, so it is written down
    here rather than left for the next reader to rediscover.
    """
    features = parse_sfeatures(load_fixture("sfeatures.txt"))

    assert "nvlink" in features
    assert features == frozenset(
        {
            "A",
            "B",
            "G",
            "I",
            "N",
            "a10",
            "a30",
            "a100-40gb",
            "a100-80gb",
            "nvlink",
            "gpu:A10:3",
            "gpu:A30:3",
            "gpu:A100:2",
            "gpu:A100:4",
        }
    )


def test_sfeatures_splits_comma_separated_features():
    text = "NODELIST FEATURES\nnode1 a,b,c\n"
    assert parse_sfeatures(text) == frozenset({"a", "b", "c"})


def test_sfeatures_falls_back_to_the_last_column_without_a_header():
    """No column named "feature": read the last column and say so by doing it."""
    text = "node1 128 512000 a10,nvlink\nnode2 128 512000 a30\n"
    assert parse_sfeatures(text) == frozenset({"a10", "nvlink", "a30"})


def test_sfeatures_drops_the_null_placeholders():
    text = "NODELIST AVAIL_FEATURES\nnode1 (null)\nnode2 none\nnode3 a10\n"
    assert parse_sfeatures(text) == frozenset({"a10"})


def test_sfeatures_drops_rule_lines_and_blank_input():
    assert parse_sfeatures("") == frozenset()
    assert parse_sfeatures("\n  \n") == frozenset()
    assert parse_sfeatures("FEATURES\n=======\n----\na10\n") == frozenset({"a10"})


def test_sfeatures_rejects_a_token_that_cannot_be_a_feature_name():
    """A number or a path is not a SLURM feature name, however loose we are."""
    text = "AVAIL_FEATURES\n128\n/apps/modulefiles\n-\na10\n"
    assert parse_sfeatures(text) == frozenset({"a10"})


# --------------------------------------------------------------------------
# module avail
# --------------------------------------------------------------------------


@pytest.mark.unverified_fixture
def test_modules_keeps_versioned_names():
    """AC-5.8: the CLI may only name a module the cluster listed.

    **Defect fixed in this commit.**  ``parse_modules`` validated names with the
    *feature-name* pattern, which has no ``/`` in its character class, so every
    ``name/version`` token failed it.  A real ``module avail`` parsed down to
    the handful of unversioned modules and dropped the rest -- silently, and in
    the direction that matters: the version is exactly the part AC-5.2 forbids
    guessing.
    """
    names = parse_modules(load_fixture("modules.txt"))

    assert names == (
        "anaconda/2024.02-py311",
        "cudnn/cuda-12.6_9.5",
        "openmpi/4.1.6",
        "apptainer/1.3.4",
        "gcc/12.3.0",
        "use.own",
        "cuda/12.6.0",
        "gcc/11.2.0",
        "conda-env/rl-py3.11",
    )


@pytest.mark.unverified_fixture
def test_modules_returns_nothing_that_is_not_a_module():
    """Directory banners, the ``Where:`` legend and the footers are not modules.

    **Defect fixed in this commit** for the legend body: the guard covered the
    ``Where:`` line but not the ``D:  Default Module`` entries beneath it, so
    ``Default`` and ``Module`` were returned as loadable module names -- the
    precise failure AC-5.8 exists to prevent, since the CLI would then print
    ``module load Default``.
    """
    names = parse_modules(load_fixture("modules.txt"))

    assert not any(name.startswith("/") for name in names)
    for junk in ("Where:", "D:", "L:", "Default", "Module", "is", "loaded", "any"):
        assert junk not in names


def test_modules_strips_lmod_decorations_without_duplicating():
    text = "   cuda/12.6.0 (D)   cuda/12.4.0   gcc/12.3.0(L)   gcc/12.3.0\n"
    assert parse_modules(text) == ("cuda/12.6.0", "cuda/12.4.0", "gcc/12.3.0")


def test_modules_blank_input_is_an_empty_tuple():
    assert parse_modules("") == ()
    assert parse_modules("\n----------\n\n") == ()


def test_modules_ignores_the_directory_banner():
    text = "------- /apps/spack/modules -------\n   gcc/12.3.0\n"
    assert parse_modules(text) == ("gcc/12.3.0",)


# --------------------------------------------------------------------------
# parse_table -- myquota and anything else shown to a human
# --------------------------------------------------------------------------


@pytest.mark.unverified_fixture
def test_table_keeps_every_cell_under_its_own_column():
    """C5 pinned this as broken; C6 fixed it, and this is the difference.

    ``myquota`` heads two columns ``Used`` and two ``Use%`` -- bytes on the
    left, inodes on the right.  While a row was a ``{header: cell}`` dict the
    second pair overwrote the first, so the home row reported ``-`` for both
    of the numbers a quota report exists to show.  A row is positional now,
    so nothing collapses: ten headers, ten cells, in order.
    """
    rows = parse_table(load_fixture("myquota.txt"))

    assert len(rows) == 2
    home = rows[0]
    assert tuple(home.cells) == (
        "home",
        "myusername",
        "25GB",
        "16.3GB",
        "8.7GB",
        "65%",
        "-",
        "-",
        "-",
        "-",
    )
    assert len(home.headers) == len(home.cells) == 10
    assert home[0] == "home"
    assert home.label == "home"


@pytest.mark.unverified_fixture
def test_table_name_lookup_defaults_to_the_first_of_a_duplicated_header():
    """The regression itself, stated as the two numbers that were being lost."""
    home, scratch = parse_table(load_fixture("myquota.txt"))

    assert home.get("used") == "16.3GB"
    assert home.get("use%") == "65%"
    assert scratch.get("used") == "1.2TB"
    assert scratch.get("use%") == "1%"


@pytest.mark.unverified_fixture
def test_table_reaches_the_shadowed_column_by_occurrence():
    """The inode pair is not discarded either -- it is simply the second one."""
    home, scratch = parse_table(load_fixture("myquota.txt"))

    assert home.indexes("used") == (3, 7)
    assert home.indexes("use%") == (5, 9)
    assert scratch.get("used", 1) == "47,113"
    assert scratch.get("use%", 1) == "2%"
    assert home.get("used", 1) == "-"


def test_table_get_returns_the_default_for_what_is_not_there():
    (row,) = parse_table("Type Size\nhome 25GB\n")

    assert row.get("nosuchcolumn") == ""
    assert row.get("nosuchcolumn", default="n/a") == "n/a"
    assert row.get("size", 1) == ""


def test_table_keeps_a_ragged_row_rather_than_truncating_it():
    """Surplus cells are cells with no header -- a ragged report beats a wrong one.

    The old ``extra`` key is gone with the dict: joining the surplus back into
    one string was a workaround for a shape that could not hold it.
    """
    (row,) = parse_table("Type Size\nhome 25GB and then some\n")

    assert tuple(row.cells) == ("home", "25GB", "and", "then", "some")
    assert row.get("size") == "25GB"
    assert row[4] == "some"


def test_table_drops_rule_lines_and_empty_input():
    assert parse_table("") == ()
    assert parse_table("\n  \n") == ()
    (row,) = parse_table("Type Size\n==== ====\nhome 25GB\n")
    assert tuple(row.cells) == ("home", "25GB")


def test_table_short_row_reports_only_the_cells_present():
    (row,) = parse_table("Type Size Used\nhome 25GB\n")

    assert tuple(row.cells) == ("home", "25GB")
    assert row.get("used") == ""


# --------------------------------------------------------------------------
# purgelist
# --------------------------------------------------------------------------


@pytest.mark.unverified_fixture
def test_purgelist_keeps_paths_and_drops_the_prose():
    """The caller acts on this list, so a sentence must not reach it as a file."""
    paths = parse_purgelist(load_fixture("purgelist.txt"))

    assert paths == (
        "/scratch/gilbreth/myusername/old_run/checkpoint_0001.pt",
        "/scratch/gilbreth/myusername/old_run/checkpoint_0002.pt",
        "/scratch/gilbreth/myusername/logs/train.log",
    )


def test_purgelist_drops_a_relative_path_and_a_line_with_a_space():
    """Both rules are deliberate; a path with a space in it is lost with them."""
    text = (
        "old_run/checkpoint.pt\n"
        "/scratch/gilbreth/me/a file with spaces.pt\n"
        "/scratch/gilbreth/me/keeper.pt\n"
    )
    assert parse_purgelist(text) == ("/scratch/gilbreth/me/keeper.pt",)


def test_purgelist_blank_input_is_an_empty_tuple():
    assert parse_purgelist("") == ()
    assert parse_purgelist("\n   \n") == ()


# --------------------------------------------------------------------------
# AC-4.2 -- no parser may raise, whatever it is handed
# --------------------------------------------------------------------------

PARSERS = (
    parse_slist,
    parse_sinfo,
    parse_sfeatures,
    parse_modules,
    parse_table,
    parse_purgelist,
)

HOSTILE_INPUTS = (
    "",
    "\n",
    "   ",
    "\t\t\n\t\n",
    "=" * 200,
    "|" * 200,
    "\x00\x01\x02\x03",
    "Account\n" * 50,
    "a b c d e f g h i j k\n" * 50,
    "N/A N/A N/A\n",
    "-1 -2 -3\n",
    "1,000 2,000 3,000\n",
)


@pytest.mark.parametrize("parser", PARSERS, ids=lambda p: p.__name__)
@pytest.mark.parametrize("text", HOSTILE_INPUTS, ids=repr)
def test_no_parser_raises(parser, text):
    """A discovery layer that can abort a command is worse than none at all."""
    parser(text)


# --------------------------------------------------------------------------
# What the parsed values become: the tri-state, assembled offline
# --------------------------------------------------------------------------


def cluster_view(slist_text=None, sinfo_text=None) -> ClusterView:
    """A :class:`ClusterView` over parsed text, with no cluster involved."""
    return ClusterView(
        slist=(
            ProbeResult("slist", "slist", True, parse_slist(slist_text))
            if slist_text is not None
            else ProbeResult("slist", "slist", False, SlistMatrix(), reason="disabled")
        ),
        sinfo=(
            ProbeResult("sinfo", "sinfo", True, parse_sinfo(sinfo_text))
            if sinfo_text is not None
            else ProbeResult("sinfo", "sinfo", False, {}, reason="disabled")
        ),
    )


@pytest.mark.unverified_fixture
def test_known_partitions_comes_from_sinfo_alone_not_from_slist():
    """AC-3.8's tri-state is fed by "what exists", not "what may I charge".

    ``slist`` is the primary probe everywhere else, but RCAC's FAQ says it
    prints the access to each GPU type *for each account you are a member of*.
    The only use of ``known_partitions`` is a hard error, so sourcing it from
    ``slist`` would turn "you have no allocation here" into ``partition 'X'
    does not exist on this cluster`` -- false, and unactionable.

    The fixtures make the difference observable: ``slist`` carries an ``h100``
    column that ``sinfo`` does not list, and ``sinfo`` lists ``training``,
    which has no ``slist`` column.  Only the second set may veto a submission.
    """
    view = cluster_view(load_fixture("slist.txt"), load_fixture("sinfo.txt"))

    assert view.known_partitions == frozenset(view.queue_state)
    assert "training" in view.known_partitions
    # Present in slist, absent from sinfo: it must NOT become submittable-by-veto.
    assert "h100" in view.matrix.partitions
    assert "h100" not in view.known_partitions
    assert view.discovered is True
    assert view.warnings == []


@pytest.mark.unverified_fixture
def test_one_failed_probe_does_not_cost_the_other(caplog):
    """AC-4.2: each probe fails on its own and degrades to a warning."""
    view = cluster_view(slist_text=load_fixture("slist.txt"))

    assert view.discovered is True
    # slist answered, so the matrix is shown to the user...
    assert view.matrix.partitions
    # ...but with sinfo absent nothing enumerated what EXISTS, so the tri-state
    # stays at "not attempted": an unrecognised -p warns and renders rather
    # than hard-erroring on the strength of an account-scoped answer.
    assert view.known_partitions is None
    assert view.warnings == ["sinfo discovery did not run: disabled"]
    assert view.queue_state == {}
    assert view.features == frozenset()


def test_known_partitions_folds_an_empty_answer_back_to_none():
    """A probe that succeeded and found nothing is a probe that failed quietly."""
    view = cluster_view("", "")

    assert view.discovered is True
    assert view.known_partitions is None


def test_known_partitions_is_none_when_nothing_was_asked():
    view = cluster_view()

    assert view.discovered is False
    assert view.known_partitions is None
    assert len(view.warnings) == 2


def test_probe_turns_a_parser_failure_into_a_warning(monkeypatch):
    """AC-4.2's last failure mode: the cluster answered, the parser could not.

    Exercised with the transport stubbed out -- no ssh, no network, no cluster.
    """

    class Completed:
        returncode = 0
        stdout = "whatever"
        stderr = ""

    monkeypatch.delenv("RCAC_NO_DISCOVER", raising=False)
    monkeypatch.delenv("RCAC_NO_DISCOVERY_CACHE", raising=False)
    monkeypatch.setattr(
        "rcac_cli.discovery._run_remote", lambda cfg, command, timeout=0: Completed()
    )

    def explode(_text):
        raise ValueError("boom")

    result = probe(
        {"ssh": {"user": "someone", "login_host": "gilbreth.example"}},
        name="slist",
        command="slist",
        parser=explode,
        empty=SlistMatrix(),
        no_cache=True,
    )

    assert result.ok is False
    assert result.value == SlistMatrix()
    assert "could not parse" in result.reason
    assert result.warning.startswith("slist discovery did not run:")
