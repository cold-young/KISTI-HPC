"""``rcac-partition`` -- rank Gilbreth's GPU partitions, ``slist`` first (AC-4).

**This module ranks and prints.  It does not parse and it does not probe.**

Before C5 it did all four: it invoked ``sinfo`` in one function, parsed the
output two functions below the invocation, and carried a private partition
table of its own, keyed by the *previous* cluster's partition names.  Nothing outside the
module could reach the parser, which made the invoke-here / parse-there split an
accident rather than a design.  AC-4.8 turns it into a designed seam: every
probe and every parser now lives in :mod:`rcac_cli.discovery`, every documented
partition fact lives in :data:`rcac_cli.gilbreth.PARTITION_HINTS`, and what is
left here is the one thing neither of those can supply -- an opinion about which
partition to ask for, and a table a human can read.

**``slist`` is the headline (AC-4.1).**  It is printed first and in full,
because it is the only probe that answers *which accounts may I charge?*  A
ranked partition list that the user has no allocation on is a recommendation
they cannot act on, so the accounts come above the ranking, not below it.

**No probe is required (AC-4.2).**  ``slist``, ``sinfo`` and ``sfeatures`` each
fail independently and each failure is a warning.  With no network, no
``RCAC_USER`` and ``--no-discover``, this command still prints the hint table
under :data:`~rcac_cli.discovery.UNVERIFIED_BANNER` and **exits 0** -- AC-4.6,
and the reason :func:`rcac_cli.config.get_rcac_config` was made not to raise.
It is the first command a new user runs, before they know their account name.
"""

from __future__ import annotations

import sys
from typing import Optional

import click

from . import discovery
from .config import get_rcac_config
from .discovery import UNVERIFIED_BANNER, ClusterView, SlistMatrix
from .gilbreth import PARTITION_HINTS, emit_warnings

# --------------------------------------------------------------------------
# AC-4.5 / B5 -- the cost tier
#
# The pre-port code indexed a *list*, `["", "Low", "Med", "", "High"]`, by a
# tier number.  Tier 3 rendered an empty label and tier 5 or above raised
# `IndexError` -- a crash in a display column, from a table whose values the
# module itself chose.  A dict with a default cannot do either.
# --------------------------------------------------------------------------

#: Tier -> label.  ``0`` is "we do not know", which is a real answer here:
#: ``PARTITION_HINTS`` records ``gpu_memory_gb: None`` for the partitions RCAC
#: does not publish a figure for.
COST_TIER_LABELS: dict[int, str] = {
    0: "?",
    1: "Low",
    2: "Med",
    3: "High",
    4: "Highest",
}

#: The label for a tier :data:`COST_TIER_LABELS` has no entry for.  Returning
#: the number is more useful than an empty string and, unlike a list index, it
#: is defined for every integer.
def cost_label(tier) -> str:
    """The display label for a cost tier.  Never raises, for any input."""
    try:
        key = int(tier)
    except (TypeError, ValueError):
        return COST_TIER_LABELS[0]
    return COST_TIER_LABELS.get(key, f"t{key}")


def cost_tier(hint: dict) -> int:
    """A coarse cost tier for a partition, derived from its documented GPU memory.

    RCAC does not publish a per-partition SU rate, so this is explicitly a
    *heuristic for ordering*, not a cluster fact: bigger GPUs are scarcer and
    cost more of an allocation.  It is derived from ``gpu_memory_gb`` rather
    than written out per partition so that it cannot drift from the hint table,
    and an unpublished memory figure yields tier 0 rather than a guess.
    """
    memory = hint.get("gpu_memory_gb")
    if not isinstance(memory, int):
        return 0
    if memory <= 24:
        return 1
    if memory <= 40:
        return 2
    if memory <= 80:
        return 3
    return 4


# --------------------------------------------------------------------------
# Ranking
# --------------------------------------------------------------------------

#: Subtracted from a partition that neither the cluster confirmed nor the hint
#: table marks ``verified``.  `h100` is the live case (AC-4.9): RCAC's own pages
#: disagree about whether it exists, so it stays in the list -- dropping it
#: would be indistinguishable from never having looked -- but it must not be
#: the recommendation when a documented partition can serve the same request.
UNVERIFIED_PENALTY = 20.0

#: The GPU memory, in GB, that scores full marks on the capability term.  Set
#: to the largest figure `PARTITION_HINTS` publishes so the term is a ratio
#: against something real rather than an invented ceiling.
CAPABILITY_REFERENCE_GB = 80.0


def score_partition(
    name: str,
    hint: dict,
    queue_state: dict,
    gpus_needed: int,
    prefer_cost: bool = False,
    known: Optional[frozenset] = None,
) -> Optional[float]:
    """Score one partition for one request, or ``None`` if it cannot serve it.

    Every term is either a documented fact from ``PARTITION_HINTS`` or a number
    the cluster reported through ``sinfo``.  The pre-port ``perf`` column is
    gone: it held invented relative-performance figures for GPUs this cluster
    does not have, and there is no RCAC page to replace them from (P1).  What it
    was standing in for -- "bigger GPU, better" -- is now read off the published
    ``gpu_memory_gb``.
    """
    capacity = hint.get("gpus_per_node")
    # An unpublished capacity is *not* a rejection.  `h100` has no published
    # per-node GPU count, and refusing to rank it would silently answer the
    # question AC-4.9 says we cannot answer offline.
    if isinstance(capacity, int) and gpus_needed > capacity:
        return None

    state = queue_state.get(name, {})
    if state.get("avail") == "down":
        return None

    # Availability (unbounded, and deliberately dominant): a partition with
    # idle nodes now beats a better GPU with a full queue.  Zero when `sinfo`
    # did not answer, which is why the ranking degrades to hint order offline.
    idle = state.get("idle", 0) or 0
    mix = state.get("mix", 0) or 0
    score = idle * 10.0 + mix * 3.0

    # Capability (0-30), from the published GPU memory.  Unknown scores the
    # midpoint rather than zero: absence of a figure is not a small figure.
    memory = hint.get("gpu_memory_gb")
    if isinstance(memory, int):
        score += min(memory / CAPABILITY_REFERENCE_GB, 1.0) * 30.0
    else:
        score += 15.0

    # Cost (0-20), or a flat neutral term when the user did not ask for it.
    if prefer_cost:
        score += (5 - cost_tier(hint)) * 5.0
    else:
        score += 10.0

    if not _is_verified(name, hint, known):
        score -= UNVERIFIED_PENALTY
    return score


def _is_verified(name: str, hint: dict, known: Optional[frozenset]) -> bool:
    """Whether this partition is confirmed, by the cluster or by RCAC's docs.

    The cluster outranks the table: a partition the cluster enumerated is real
    whatever ``PARTITION_HINTS`` says about it, which is the whole point of
    discovery being primary (D2/P1).
    """
    if known is not None and name in known:
        return True
    return bool(hint.get("verified", False))


def rank_partitions(
    view: ClusterView,
    gpus_needed: int = 2,
    prefer_cost: bool = False,
) -> list[tuple[str, float, dict]]:
    """Rank every partition we know of into ``(name, score, details)``, best first.

    The candidate set is the *union* of the hint table and whatever the cluster
    reported, so a partition RCAC has not documented still appears -- with an
    unknown GPU type -- rather than being dropped for disagreeing with a table.
    """
    queue_state = view.queue_state
    known = view.known_partitions
    candidates = list(PARTITION_HINTS)
    for name in sorted(queue_state):
        if name not in candidates:
            candidates.append(name)

    scored: list[tuple[str, float, dict]] = []
    for name in candidates:
        hint = PARTITION_HINTS.get(name, {})
        score = score_partition(
            name, hint, queue_state, gpus_needed, prefer_cost, known
        )
        if score is None:
            continue
        state = queue_state.get(name, {})
        scored.append(
            (
                name,
                score,
                {
                    "gpu": hint.get("gpu") or "?",
                    "gpu_memory_gb": hint.get("gpu_memory_gb"),
                    "gpus_per_node": hint.get("gpus_per_node"),
                    "idle_nodes": state.get("idle", 0) or 0,
                    "mix_nodes": state.get("mix", 0) or 0,
                    "cost_tier": cost_tier(hint),
                    "verified": _is_verified(name, hint, known),
                },
            )
        )

    scored.sort(key=lambda row: (-row[1], row[0]))
    return scored


# --------------------------------------------------------------------------
# Presentation
# --------------------------------------------------------------------------


def _cell(value) -> str:
    return "?" if value is None else str(value)


def render_slist_matrix(matrix: SlistMatrix) -> list[str]:
    """The account x GPU-type headline (AC-4.1), as display lines.

    ``free/total`` per cell, because those are the two numbers that decide
    whether to submit now: how much of the allocation exists and how much of it
    is idle.  Queued and running are parsed and available on
    :class:`~rcac_cli.discovery.AccountRow`; they are not shown because a matrix
    with four numbers per cell stops being readable at three GPU types.
    """
    if not matrix or not matrix.partitions:
        return []
    width = max([len("Account")] + [len(row.account) for row in matrix.accounts])
    columns = [max(len(name), 7) for name in matrix.partitions]
    header = "Account".ljust(width) + "  " + "  ".join(
        name.rjust(size) for name, size in zip(matrix.partitions, columns)
    )
    lines = [header, "-" * len(header)]
    for row in matrix.accounts:
        cells = []
        for name, size in zip(matrix.partitions, columns):
            free, total = row.free(name), row.total(name)
            cells.append(f"{_cell(free)}/{_cell(total)}".rjust(size))
        lines.append(row.account.ljust(width) + "  " + "  ".join(cells))
    lines.append("")
    lines.append("free/total GPUs per account.  Charge a job with `-A <account>`.")
    return lines


def render_ranking(ranked: list[tuple[str, float, dict]]) -> list[str]:
    """The ranked partition table, as display lines."""
    header = (
        f"{'Rank':<5} {'Partition':<12} {'GPU':<12} {'Mem':<6} {'/node':<6} "
        f"{'Idle':<6} {'Mix':<5} {'Cost':<8} {'Score':<8}"
    )
    lines = [header, "-" * len(header)]
    for position, (name, score, detail) in enumerate(ranked, 1):
        memory = detail["gpu_memory_gb"]
        marker = " <-- recommended" if position == 1 else ""
        if not detail["verified"]:
            marker += " (unverified)" if position == 1 else "  (unverified)"
        lines.append(
            f"{position:<5} {name:<12} {detail['gpu']:<12} "
            f"{_cell(memory):<6} {_cell(detail['gpus_per_node']):<6} "
            f"{detail['idle_nodes']:<6} {detail['mix_nodes']:<5} "
            f"{cost_label(detail['cost_tier']):<8} {score:<8.1f}{marker}"
        )
    return lines


# --------------------------------------------------------------------------
# The command
# --------------------------------------------------------------------------


@click.command("partition")
@click.option("--gpus", "-g", default=2, help="Number of GPUs needed per node.")
@click.option("--prefer-cost", is_flag=True,
              help="Rank cheaper GPUs above larger ones.")
@click.option("--no-discover", is_flag=True,
              help="Do not ask the cluster anything; print the offline hint "
                   f"table under its banner. Also settable as "
                   f"{discovery.NO_DISCOVER_ENV}=1.")
@click.option("--no-cache", is_flag=True,
              help="Re-run every probe instead of reusing this process's "
                   f"answers. Also settable as {discovery.NO_CACHE_ENV}=1.")
def select_partition(
    gpus: int, prefer_cost: bool, no_discover: bool, no_cache: bool
) -> None:
    """Show which accounts you can charge, and rank the GPU partitions.

    Exits 0 whether or not the cluster answered.  A partition recommendation is
    advice, and advice that aborts is worse than advice with a caveat on it.
    """
    # Deliberately no `require_user` call.  The pre-port command required a
    # resolved user before it did anything, so `rcac-partition` -- the first
    # command a new user runs -- exited 1 telling them to set a variable they
    # would only understand *after* reading this command's output (AC-4.6/P4).
    # An unset user is now just another reason a probe did not run.
    cfg = get_rcac_config()
    disabled = discovery.discovery_disabled(no_discover)
    view = discovery.survey(cfg, no_discover=no_discover, no_cache=no_cache)

    click.echo(f"Analyzing partitions for {gpus} GPU(s)...\n")

    # AC-4.2: a probe that failed is a warning, never an abort.  A probe that
    # was *switched off* is neither -- warning about the flag the user just
    # passed is noise, and noise is what trains people to ignore warnings.
    if not disabled:
        emit_warnings(view.warnings)

    click.echo("Accounts (slist)")
    matrix_lines = render_slist_matrix(view.matrix)
    if matrix_lines:
        for line in matrix_lines:
            click.echo(line)
    else:
        click.echo(
            "  not discovered -- run `slist` on a Gilbreth login node to see "
            "which accounts you can charge."
        )
    click.echo("")

    ranked = rank_partitions(view, gpus_needed=gpus, prefer_cost=prefer_cost)
    if not ranked:
        # The one real failure: no partition anywhere offers this many GPUs per
        # node.  That is the user's request being impossible, not the cluster
        # being unreachable, so it exits non-zero.
        click.echo(
            f"No partition offers {gpus} GPU(s) per node. Documented maxima: "
            + ", ".join(
                f"{name}={_cell(hint.get('gpus_per_node'))}"
                for name, hint in sorted(PARTITION_HINTS.items())
            )
        )
        sys.exit(1)

    click.echo("Partitions")
    for line in render_ranking(ranked):
        click.echo(line)

    best = ranked[0][0]
    click.echo(f"\nRecommended: --partition={best}")
    click.echo(f"  Use: rcac-exp launch experiment.yaml --partition {best}")

    # The banner is tied to the same tri-state that gates submission
    # (AC-3.8): if the cluster did not enumerate its partitions, the table
    # above is RCAC's documentation and nothing more.
    if view.known_partitions is None:
        click.echo("")
        click.echo(UNVERIFIED_BANNER)
