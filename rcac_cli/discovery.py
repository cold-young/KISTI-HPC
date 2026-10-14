"""Ask Gilbreth what Gilbreth is (AC-4).

RCAC's own agent-facing guidance is blunt about this: *"Do NOT invent module
names or versions... Do NOT invent partition or QOS names."*  This module is
that instruction turned into code.  Everything :mod:`rcac_cli.gilbreth` knows
statically is a ``*_HINTS`` table transcribed from a documentation page that
carries a date we cannot see; everything here comes from the cluster itself and
supersedes the tables at runtime.

**``slist`` is the primary probe, and it is primary for a structural reason.**
It prints an account x GPU-type matrix -- Total / Queued / Running / Free per
partition, one row per account -- so a single call answers both of the two
questions a new user actually has: *which accounts may I charge?* and *which
partitions do they reach?*  ``sinfo`` and ``sfeatures`` only enrich that; they
cannot replace it, because neither knows anything about allocations.

The published sample ``slist`` output is also one half of the documented
``h100`` contradiction (AC-4.9 / R5): it shows an ``h100`` column that the
queues page does not list.  That is precisely why discovery is primary and the
tables are labelled fallback.

**No probe is ever required (AC-4.2).**  Every function here returns a
:class:`ProbeResult` and none of them raise for a cluster that did not answer:
no network, no ``RCAC_USER``, a non-zero exit, a timeout and an unparseable
payload all produce ``ok=False`` with a one-line reason, and the caller degrades
to a warning.  A discovery layer that can abort a command is worse than no
discovery layer, because it converts an optional enrichment into a new way for
the tool to fail.

**Every parser lives here (AC-4.8).**  Before C5, ``sinfo`` was *invoked* in
``partition_selector.py`` and *parsed* two functions below the invocation in the
same file, while nothing else could reach either -- an accidental seam rather
than a designed one.  ``partition_selector.py`` now has no parser at all.

**The parsers are tolerant on purpose (R13).**  The exact column layout of
``slist`` and ``sfeatures`` is unconfirmed offline -- plan section 6.2 rows L5
and L5b are the live checks that confirm it.  So the parsers key off the header
row rather than off fixed offsets, accept either the wide (one column group per
GPU type) or the long (one row per account/partition pair) shape, and drop a
line they cannot read instead of raising.  The fixtures under ``tests/fixtures``
are doc-derived and carry :mod:`pytest` marker ``unverified_fixture`` saying so:
they test *the parser*, not the cluster.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from .ssh import ssh_cmd

# --------------------------------------------------------------------------
# The probe commands
# --------------------------------------------------------------------------

#: The primary probe.  RCAC's account/allocation reporter; on a GPU cluster its
#: matrix is indexed by GPU type, which is the same thing as ``sbatch -p``.
SLIST_COMMAND = "slist"

#: Partition state.  The format string is SLURM's own and is stable across
#: sites, which is why this is the one probe whose output shape is not in
#: question.  ``--noheader`` keeps :func:`parse_sinfo` from having to.
#:
#: ``-a`` is load-bearing, not tidiness.  SLURM defines it as showing partitions
#: "configured as hidden and partitions that are unavailable to the user's
#: group" -- so *without* it ``sinfo`` is access-scoped, exactly like ``slist``.
#: This output is the sole feeder of :attr:`ClusterView.known_partitions`, whose
#: only use is a hard error, so an access-scoped answer here reports a partition
#: you merely cannot use as one that "does not exist on this cluster".  That is
#: the same false, unactionable error that removing ``slist`` from this set was
#: meant to eliminate; dropping ``-a`` would just move it one probe to the left.
SINFO_COMMAND = "sinfo -a -o '%P %a %D %T %N %G' --noheader"

#: RCAC's node-feature reporter.  Consulted for exactly one thing today --
#: whether ``nvlink`` is a real feature name on this cluster -- so its parser is
#: the most tolerant of the three.
SFEATURES_COMMAND = "sfeatures"

#: RCAC's quota reporter.
QUOTA_COMMAND = "myquota"

#: RCAC's scratch-purge reporter.  Read by C6's retention work; defined here so
#: every cluster probe is in one place.
PURGE_COMMAND = "purgelist"

#: Long enough for a login node under load, short enough that a hung SSH does
#: not look like a wedged CLI.  A probe that times out is a probe that failed,
#: which is a warning (AC-4.2).
DEFAULT_TIMEOUT_SECONDS = 20

#: Set to a non-empty value to suppress every probe process-wide (AC-4.7).  The
#: flag equivalents are ``--no-discover`` on the commands that take one.
NO_DISCOVER_ENV = "RCAC_NO_DISCOVER"

#: Set to a non-empty value to bypass the session cache (AC-4.7).
NO_CACHE_ENV = "RCAC_NO_DISCOVERY_CACHE"

#: Printed by any command that is showing the offline hint tables rather than
#: what the cluster said.  It names the fix, because a banner that only says
#: "unverified" tells a new user nothing they can act on.
UNVERIFIED_BANNER = (
    "UNVERIFIED: this table is transcribed from RCAC's documentation, not read "
    "from the cluster.\n"
    "RCAC's pages already disagree with each other about at least one "
    "partition (h100),\n"
    "so treat it as a starting point and confirm with `slist` on a login node.\n"
    "To let this command ask the cluster itself: export RCAC_USER=<your Purdue "
    "career account id>\n"
    "and run without --no-discover."
)


# --------------------------------------------------------------------------
# The result type
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbeResult:
    """The outcome of one probe.

    ``ok`` is the only thing a caller must branch on.  ``value`` always holds a
    payload of the probe's declared type -- an empty one when ``ok`` is false --
    so a caller that forgets to branch degrades to "found nothing" rather than
    to ``AttributeError`` on ``None``.
    """

    name: str
    command: str
    ok: bool
    value: object
    reason: str = ""
    cached: bool = False

    @property
    def warning(self) -> str:
        """The one-line warning to show for a failed probe, or ``""``."""
        if self.ok:
            return ""
        return f"{self.name} discovery did not run: {self.reason}"


def _failed(name: str, command: str, empty: object, reason: str) -> ProbeResult:
    return ProbeResult(name=name, command=command, ok=False, value=empty, reason=reason)


# --------------------------------------------------------------------------
# Session cache (AC-4.7)
# --------------------------------------------------------------------------

#: Probe name + ssh target -> result.  A "session" is one process: a sweep that
#: renders 40 scripts asks the cluster for its partition list once, not 40
#: times, and a user who changes ``RCAC_USER`` mid-process gets a different key
#: rather than a stale answer.
_SESSION_CACHE: dict = {}


def clear_cache() -> None:
    """Drop every cached probe.  The programmatic escape from the cache."""
    _SESSION_CACHE.clear()


def _env_flag(name: str) -> bool:
    return bool(os.environ.get(name, "").strip())


def discovery_disabled(no_discover: bool = False) -> bool:
    """Whether discovery is switched off, by flag or by environment (AC-4.7).

    Public because the *callers* need the distinction :func:`probe` does not:
    a probe that failed is a warning, but a probe the user switched off is not
    an event at all.  Warning someone about the flag they just passed is noise,
    and noise is what teaches people to stop reading warnings.
    """
    return bool(no_discover) or _env_flag(NO_DISCOVER_ENV)


# --------------------------------------------------------------------------
# Transport
# --------------------------------------------------------------------------


def _remote_user(cfg: dict) -> str:
    return str((cfg.get("ssh") or {}).get("user") or "").strip()


def _run_remote(
    cfg: dict, command: str, timeout: int = DEFAULT_TIMEOUT_SECONDS
) -> Optional[subprocess.CompletedProcess]:
    """Run *command* on the login node, or return ``None`` if ssh never ran.

    ``None`` and a non-zero ``returncode`` are deliberately different: the first
    means the transport itself failed (no ssh binary, a timeout, an OS error),
    the second means the cluster answered and said no.  Both end up as a
    warning, but the reason strings differ and the reason is the whole value of
    a failed probe.
    """
    argv = [*ssh_cmd(cfg), command]
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None


def probe(
    cfg: dict,
    *,
    name: str,
    command: str,
    parser: Callable[[str], object],
    empty: object,
    no_discover: bool = False,
    no_cache: bool = False,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> ProbeResult:
    """Run one probe and parse it, never raising (AC-4.2).

    The five failure modes are kept distinct in the reason text because they
    have five different fixes: not attempted, no credential, transport failure,
    non-zero exit, unparseable output.
    """
    if no_discover or _env_flag(NO_DISCOVER_ENV):
        return _failed(name, command, empty, "discovery was disabled (--no-discover)")

    user = _remote_user(cfg)
    if not user:
        return _failed(
            name,
            command,
            empty,
            "RCAC_USER is not set, so there is no login to run it as",
        )

    key = (name, f"{user}@{(cfg.get('ssh') or {}).get('login_host', '')}")
    skip_cache = no_cache or _env_flag(NO_CACHE_ENV)
    if not skip_cache and key in _SESSION_CACHE:
        cached = _SESSION_CACHE[key]
        return ProbeResult(
            name=cached.name,
            command=cached.command,
            ok=cached.ok,
            value=cached.value,
            reason=cached.reason,
            cached=True,
        )

    completed = _run_remote(cfg, command, timeout=timeout)
    if completed is None:
        result = _failed(name, command, empty, f"could not run `{command}` over ssh")
    elif completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip().splitlines()
        result = _failed(
            name,
            command,
            empty,
            f"`{command}` exited {completed.returncode}"
            + (f": {detail[0]}" if detail else ""),
        )
    else:
        try:
            parsed = parser(completed.stdout or "")
        except Exception as exc:  # tolerant by contract: a parser bug is a warning
            result = _failed(
                name, command, empty, f"could not parse `{command}` output: {exc}"
            )
        else:
            result = ProbeResult(
                name=name, command=command, ok=True, value=parsed, reason=""
            )

    if not skip_cache:
        _SESSION_CACHE[key] = result
    return result


# --------------------------------------------------------------------------
# Table primitives shared by the parsers
# --------------------------------------------------------------------------

#: A line made only of the characters tables draw rules with.  ``|`` is in the
#: class because ``slist``'s GPU matrix draws its rule *through* the column
#: separator (``==============|==========``); without it that line reaches the
#: row loop as data.
_RULE_RE = re.compile(r"^[=\-_+|\s]+$")

_INT_RE = re.compile(r"^[+-]?[\d,]+$")


def _is_rule(line: str) -> bool:
    return bool(line.strip()) and bool(_RULE_RE.match(line))


def _as_int(token: str) -> Optional[int]:
    if not _INT_RE.match(token):
        return None
    try:
        return int(token.replace(",", ""))
    except ValueError:  # pragma: no cover - guarded by the regex
        return None


def _header_index(lines: Sequence[str], want: str) -> Optional[int]:
    """Index of the first line whose *first token* is *want*, case-insensitively.

    The plural is accepted because the two published ``slist`` samples disagree:
    the CPU-style report heads its rows ``Account`` and the GPU matrix heads
    them ``Accounts``.  Matching only the singular found no header at all in the
    matrix shape, which made :func:`parse_slist` return an empty matrix for the
    primary probe's own documented output.
    """
    for i, line in enumerate(lines):
        tokens = line.split()
        if not tokens:
            continue
        first = tokens[0].strip(":|").lower()
        if first == want or first == want + "s":
            return i
    return None


# --------------------------------------------------------------------------
# AC-4.1 -- the slist matrix
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AccountRow:
    """One account's row of the ``slist`` matrix."""

    account: str
    #: partition (GPU type) -> counter name -> count.  Counter names are the
    #: header's own, lowercased, so a site that renames "Running" to "Active"
    #: still parses.
    counts: dict = field(default_factory=dict)

    def free(self, partition: str) -> Optional[int]:
        """Free GPUs in *partition* for this account, or ``None`` if unreported."""
        return self.counts.get(partition, {}).get("free")

    def total(self, partition: str) -> Optional[int]:
        return self.counts.get(partition, {}).get("total")


@dataclass(frozen=True)
class SlistMatrix:
    """The parsed ``slist`` output: accounts down, GPU types across."""

    accounts: tuple = ()
    partitions: tuple = ()
    counters: tuple = ()

    def __bool__(self) -> bool:
        return bool(self.accounts)

    @property
    def account_names(self) -> tuple:
        return tuple(row.account for row in self.accounts)

    def free(self, partition: str) -> int:
        """Free GPUs in *partition* summed over every account we can charge."""
        return sum((row.free(partition) or 0) for row in self.accounts)


#: Header tokens that name the row, not a counter.  ``slist`` on a non-GPU
#: cluster prints a ``Queue`` column instead of GPU-type groups, which is the
#: long shape :func:`parse_slist` also accepts.
_SLIST_ROW_LABELS = frozenset({"account", "queue", "partition", "qos"})

#: Trailing columns that are not per-partition counters.  They are dropped
#: rather than mis-attributed to a GPU type.
_SLIST_TRAILING = frozenset({"count", "walltime", "max", "type", "node", "nodes"})

#: Words that appear in ``slist``'s title line ("Current Number of GPUs") and
#: never in a partition name.  Without this guard a four-word title sitting
#: directly above the header would be read as four GPU-type labels whenever the
#: group count happened to be four -- a silent mislabelling, which is worse than
#: the numbered fallback.
_SLIST_PROSE = frozenset(
    {
        "current",
        "number",
        "of",
        "the",
        "and",
        "per",
        "for",
        "in",
        "gpu",
        "gpus",
        "core",
        "cores",
        "node",
        "nodes",
    }
)


def _slist_group_labels(lines: Sequence[str], header_index: int) -> tuple:
    """GPU-type labels from the group header above the counter header, if any."""
    for i in range(header_index - 1, -1, -1):
        if not lines[i].strip():
            continue
        if _is_rule(lines[i]):
            continue
        tokens = lines[i].split()
        # A group header names GPU types; a title line ("Current Number of
        # GPUs") is prose.  The discriminators are that a group header holds no
        # numbers and no prose words.
        if any(_as_int(t) is not None for t in tokens):
            return ()
        if any(t.lower() in _SLIST_PROSE for t in tokens):
            return ()
        # Lowercased so a heading lifted out of a `slist` group header is
        # spelled the way `sinfo` reports the partition and the way `-p` takes
        # it -- `A100-40GB` and `a100-40gb` naming one column would otherwise
        # be a difference the reader has to resolve.
        #
        # This is a *display* rule and nothing more. These labels head the
        # `rcac-partition` matrix and key its per-account counters; they do not
        # reach `ClusterView.known_partitions`, whose sole feeder is `sinfo`
        # (see SINFO_COMMAND). `parse_sinfo` deliberately does NOT lowercase
        # in the same way: those names come from the scheduler itself, so they
        # are authoritative and case-exact, and folding their case would be
        # this module editing the one answer it is entitled to quote.
        return tuple(t.lower() for t in tokens)
    return ()


#: One cell of the matrix shape: the counters packed into a single column with
#: slashes, e.g. ``11/0/2/11``.  A bare integer is accepted so a site printing
#: one counter per GPU type still parses.
_SLIST_CELL_RE = re.compile(r"^\d+(?:/\d+)*$")

#: A matrix cell that reports nothing.  ``N/A`` is what the published sample
#: prints for an account with no allocation on that GPU type; the rest are the
#: spellings other reporters use for the same idea.  These are *skipped*, not
#: zeroed: "no allocation" and "zero free" are different answers.
_SLIST_UNREPORTED = frozenset({"n/a", "na", "-", "--", "none", "null", "."})

#: The counter names, which the matrix shape prints once, in prose, above the
#: header: ``Number of GPUs (Total/Queued/Running/Free)``.
_SLIST_COUNTER_RE = re.compile(r"\(([A-Za-z]+(?:/[A-Za-z]+)+)\)")


def _is_row_label(token: str) -> bool:
    """True for a token that names the row column rather than an account."""
    text = token.strip(":|").lower()
    if text in _SLIST_ROW_LABELS:
        return True
    # Only the plural, never an arbitrary suffix: `account1` is an account.
    return text.endswith("s") and text[:-1] in _SLIST_ROW_LABELS


def _parse_slist_matrix(lines: Sequence[str], header_index: int) -> SlistMatrix:
    """Parse the pipe-delimited account x GPU-type matrix (AC-4.1).

    The third shape, and the one RCAC actually publishes for a GPU cluster::

                      |           Number of GPUs (Total/Queued/Running/Free)
        Accounts      |   A10        A30     A100-40gb  A100-80gb     H100
        ==============|========== ========== ========== ========== ==========
        account1      | 1/0/0/1      N/A        N/A        N/A        N/A

    It is neither the wide shape (the counters are packed into one column per
    GPU type with slashes, not spread across four) nor the long shape (the
    partition is a *column header*, not a row field), so the general parser
    could not read a single row of it.

    The GPU types are the header's own tokens after the ``|``, lowercased to the
    partition names ``sbatch -p`` takes.  The counter names come from the prose
    line above -- read, never assumed, so a site that renames or reorders them
    still parses; when that line is missing the counters are numbered rather
    than invented, because a wrong ``free`` is worse than an unnamed one.

    An unreadable cell costs its row and nothing else, and a row whose cells are
    all ``N/A`` is *kept* with empty counts: the account still exists and is
    still chargeable, which is half of what AC-4.1 asks this probe.
    """
    _, _, header_tail = lines[header_index].partition("|")
    labels = tuple(token.lower() for token in header_tail.split())
    if not labels:
        return SlistMatrix()

    counters: tuple = ()
    for i in range(header_index - 1, -1, -1):
        found = _SLIST_COUNTER_RE.search(lines[i])
        if found:
            counters = tuple(part.lower() for part in found.group(1).split("/"))
            break

    rows: list = []
    for line in lines[header_index + 1 :]:
        if not line.strip() or _is_rule(line):
            continue
        account_text, separator, cell_text = line.partition("|")
        if not separator:
            # A footer, a note or a warning line: no column separator, no row.
            continue
        account_tokens = account_text.split()
        if len(account_tokens) != 1 or _is_row_label(account_tokens[0]):
            continue

        counts: dict = {}
        readable = True
        for index, cell in enumerate(cell_text.split()):
            if index >= len(labels):
                break  # more cells than headers: the surplus is unattributable
            if cell.lower() in _SLIST_UNREPORTED:
                continue
            if not _SLIST_CELL_RE.match(cell):
                readable = False
                break
            numbers = [int(part) for part in cell.split("/")]
            names = counters or tuple(f"count{n}" for n in range(len(numbers)))
            counts[labels[index]] = dict(zip(names, numbers))
        if readable:
            rows.append(AccountRow(account=account_tokens[0], counts=counts))

    if not rows:
        return SlistMatrix()
    return SlistMatrix(accounts=tuple(rows), partitions=labels, counters=counters)


def parse_slist(text: str) -> SlistMatrix:
    """Parse ``slist`` into an account x GPU-type matrix (AC-4.1).

    Three shapes are accepted, because which one Gilbreth prints is exactly what
    plan section 6.2 row L5b exists to confirm:

    *Matrix* -- accounts down, GPU types across, a ``|`` separating the account
    from its cells and the four counters packed into each cell with slashes
    (``11/0/2/11``).  This is the shape RCAC's published GPU sample shows,
    ``h100`` column and all; see :func:`_parse_slist_matrix`.

    *Wide* -- one group of ``Total/Queued/Running/Free`` columns per GPU type,
    with the GPU types on a header line above.

    *Long* -- one row per ``(account, partition)`` pair, with the partition in
    its own column.  Detected by the row's second token being non-numeric.

    Anything unreadable is dropped, never raised on: a trailing note, a
    site-local footer or a column we did not expect costs its own line and
    nothing else.
    """
    lines = text.splitlines()
    header_index = _header_index(lines, "account")
    if header_index is None:
        return SlistMatrix()

    # The matrix shape is discriminated by the column separator in its header,
    # which neither of the other two shapes has.
    if "|" in lines[header_index]:
        return _parse_slist_matrix(lines, header_index)

    header = lines[header_index].split()
    counter_tokens = [t.lower() for t in header[1:]]
    # A leading `Queue` / `Partition` column names the row, not a counter: it is
    # the long shape's discriminator and counting it into the group width is
    # what would make every long-shape row one column too wide to parse.
    while counter_tokens and counter_tokens[0] in _SLIST_ROW_LABELS:
        counter_tokens.pop(0)
    while counter_tokens and counter_tokens[-1] in _SLIST_TRAILING:
        counter_tokens.pop()
    if not counter_tokens:
        return SlistMatrix()

    # Group width: the counter names repeat once per GPU type, so the second
    # occurrence of the first counter marks the group boundary.
    width = len(counter_tokens)
    for i in range(1, len(counter_tokens)):
        if counter_tokens[i] == counter_tokens[0]:
            width = i
            break
    counters = tuple(counter_tokens[:width])
    group_count = len(counter_tokens) // width

    labels = _slist_group_labels(lines, header_index)
    labels_are_real = bool(labels) and len(labels) == group_count
    if not labels_are_real:
        # Either no group header, or one we cannot line up.  Numbered columns
        # are honest; invented GPU-type names would not be.
        labels = tuple(f"column{i}" for i in range(group_count))

    # Only *named* columns seed the partition list.  A generated ``columnN`` is
    # positional bookkeeping, not something the cluster said, and printing it
    # in the GPU-type row of the matrix would be inventing a partition name --
    # the one thing P1 forbids outright.  It cannot do worse than mislead a
    # reader: `known_partitions` is fed by `sinfo` alone, so no label from here
    # can hard-error a submission.  That is why the columns are still numbered
    # and shown rather than dropped -- the counters under them are real.
    partitions: list = list(labels) if labels_are_real else []
    rows: list = []
    for line in lines[header_index + 1 :]:
        if not line.strip() or _is_rule(line):
            continue
        tokens = line.split()
        if len(tokens) < 2:
            continue
        account = tokens[0]
        if account.lower() in _SLIST_ROW_LABELS:
            continue

        rest = tokens[1:]
        row_partitions = labels
        if _as_int(rest[0]) is None:
            # Long shape: this row names its own partition.
            row_partitions = (rest[0],)
            rest = rest[1:]
            if row_partitions[0] not in partitions:
                partitions.append(row_partitions[0])

        numbers = [_as_int(t) for t in rest]
        # Trailing non-numeric columns (a walltime, a node count with a unit)
        # are dropped rather than turned into a partition.
        while numbers and numbers[-1] is None:
            numbers.pop()
        if not numbers or any(n is None for n in numbers):
            continue

        counts: dict = {}
        for group, partition in enumerate(row_partitions):
            chunk = numbers[group * width : (group + 1) * width]
            if len(chunk) != width:
                break
            counts[partition] = dict(zip(counters, chunk))
        if counts:
            rows.append(AccountRow(account=account, counts=counts))

    if not rows:
        return SlistMatrix()
    # Preserve first-seen order without duplicates.
    seen: list = []
    for name in partitions:
        if name not in seen:
            seen.append(name)
    return SlistMatrix(
        accounts=tuple(rows), partitions=tuple(seen), counters=counters
    )


def discover_accounts(
    cfg: dict, *, no_discover: bool = False, no_cache: bool = False
) -> ProbeResult:
    """The primary probe: ``slist`` (AC-4.1).  ``value`` is a :class:`SlistMatrix`."""
    return probe(
        cfg,
        name="slist",
        command=SLIST_COMMAND,
        parser=parse_slist,
        empty=SlistMatrix(),
        no_discover=no_discover,
        no_cache=no_cache,
    )


# --------------------------------------------------------------------------
# AC-4.8 -- sinfo, parsed here and nowhere else
# --------------------------------------------------------------------------


def parse_sinfo(output: str) -> dict:
    """Parse ``sinfo -o '%P %a %D %T %N %G' --noheader`` into per-partition state.

    Moved verbatim in behaviour from ``partition_selector._parse_sinfo``
    (AC-4.8).  It was the only parser in that module and nothing else could
    reach it, which made the invoke-here / parse-there split an accident rather
    than a design.
    """
    partitions: dict = {}
    for line in output.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        # PARTITION AVAIL NODES STATE NODELIST GRES
        partition = parts[0].rstrip("*")
        avail = parts[1]
        node_count = _as_int(parts[2])
        if node_count is None:
            continue
        state = parts[3]

        entry = partitions.setdefault(
            partition,
            {"idle": 0, "mix": 0, "alloc": 0, "total": 0, "avail": avail},
        )
        entry["total"] += node_count
        if state in ("idle", "idle~"):
            entry["idle"] += node_count
        elif state in ("mix", "mixed"):
            entry["mix"] += node_count
        elif state in ("alloc", "allocated"):
            entry["alloc"] += node_count

    return partitions


def discover_partitions(
    cfg: dict, *, no_discover: bool = False, no_cache: bool = False
) -> ProbeResult:
    """``sinfo``.  ``value`` is ``{partition: {idle, mix, alloc, total, avail}}``."""
    return probe(
        cfg,
        name="sinfo",
        command=SINFO_COMMAND,
        parser=parse_sinfo,
        empty={},
        no_discover=no_discover,
        no_cache=no_cache,
    )


# --------------------------------------------------------------------------
# sfeatures -- the most tolerant parser of the three
# --------------------------------------------------------------------------

#: A plausible SLURM feature name.  Deliberately loose: the point of the probe
#: is to learn feature names we do not already know, so a pattern tight enough
#: to only admit the ones we do know would be pointless.
_FEATURE_RE = re.compile(r"^[A-Za-z][\w.:+-]*$")


def parse_sfeatures(text: str) -> frozenset:
    """Parse ``sfeatures`` into the set of node-feature names it reported.

    RCAC does not publish this command's column layout, so the parser looks for
    a header column whose name contains "feature" and reads from there; failing
    that it reads the last column of each line.  Both are guesses -- plan
    section 6.2 row L5 is the live check -- so the return type is deliberately
    the weakest thing that is still useful: a flat set of names, used today for
    exactly one question, *is ``nvlink`` real on this cluster?*
    """
    lines = [line for line in text.splitlines() if line.strip()]
    start = 0
    field_index: Optional[int] = None
    for i, line in enumerate(lines):
        tokens = line.split()
        for j, token in enumerate(tokens):
            if "feature" in token.lower():
                field_index, start = j, i + 1
                break
        if field_index is not None:
            break

    features: set = set()
    for line in lines[start:]:
        if _is_rule(line):
            continue
        tokens = line.split()
        if not tokens:
            continue
        if field_index is not None and len(tokens) > field_index:
            candidates = tokens[field_index:]
        else:
            candidates = tokens[-1:]
        for candidate in candidates:
            for name in candidate.split(","):
                name = name.strip()
                if name and name not in ("(null)", "none") and _FEATURE_RE.match(name):
                    features.add(name)
    return frozenset(features)


def discover_features(
    cfg: dict, *, no_discover: bool = False, no_cache: bool = False
) -> ProbeResult:
    """``sfeatures``.  ``value`` is a ``frozenset`` of feature names."""
    return probe(
        cfg,
        name="sfeatures",
        command=SFEATURES_COMMAND,
        parser=parse_sfeatures,
        empty=frozenset(),
        no_discover=no_discover,
        no_cache=no_cache,
    )


# --------------------------------------------------------------------------
# The remaining probes: quota, purge
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class TableRow:
    """One row of a whitespace-column report, kept aligned to its headers.

    **This type exists because a dict was the wrong contract, and ``myquota``
    is the report that proves it.**  ``myquota`` heads *two* columns ``Used``
    and *two* ``Use%`` -- bytes on the left, inode counts on the right -- so the
    previous ``{headers[i]: cell}`` comprehension had the second pair overwrite
    the first.  For the ``home`` row that meant ``row["used"]`` returned the
    file-count column and ``row["use%"]`` returned ``-``, instead of ``16.3GB``
    and ``65%``: the parser silently dropped the two numbers a quota report
    exists to show, while its own docstring promised to "keep the columns
    associated with their headers".

    A table is **positional** -- ``cells[i]`` belongs under ``headers[i]`` --
    so that is what is stored, and nothing is lost.  Name lookup is offered on
    top of it as :meth:`get`, which resolves a duplicated header by
    *occurrence* and defaults to the first, so ``get("used")`` is the byte
    column and ``get("used", 1)`` is the inode column.  Disambiguated keys
    (``used``/``used_2``) were rejected as the alternative: they encode
    position in a name, so a report that reorders its columns keeps the key and
    changes its meaning, which is the same silent-wrong-answer failure in a new
    place.

    A ragged row is kept rather than dropped -- truncating someone's quota
    report would be worse than showing a ragged one -- and falls out of the
    representation instead of needing the old ``extra`` key: extra cells are
    simply cells with no header, and missing cells are headers with no cell.
    """

    headers: tuple
    cells: tuple

    def __len__(self) -> int:
        return len(self.cells)

    def __getitem__(self, index: int) -> str:
        """Positional access -- the contract this type is built on."""
        return self.cells[index]

    def __iter__(self):
        return iter(self.cells)

    @property
    def label(self) -> str:
        """The first cell: what the row is *about* (``home``, ``scratch``)."""
        return self.cells[0] if self.cells else ""

    def indexes(self, header: str) -> tuple:
        """Every column index headed *header*, in order (case-insensitive)."""
        want = header.strip().lower()
        return tuple(i for i, h in enumerate(self.headers) if h == want)

    def get(self, header: str, occurrence: int = 0, default: str = "") -> str:
        """The cell under the *occurrence*-th column headed *header*.

        Returns *default* when the header is absent, when the row has fewer
        occurrences than asked for, or when the row is short of that column.
        """
        found = self.indexes(header)
        if occurrence >= len(found):
            return default
        index = found[occurrence]
        return self.cells[index] if index < len(self.cells) else default


def parse_table(text: str) -> tuple:
    """A generic whitespace-column table -> a tuple of :class:`TableRow`.

    Used by the one probe whose output we consume as a table rather than as a
    list: ``myquota``.  (``purgelist`` has its own parser -- it reports paths,
    not columns.)  The report is shown to a human, so the parser's job is to
    keep every cell associated with its header and stay out of the way; see
    :class:`TableRow` for why "associated with its header" cannot be a dict.
    """
    lines = [line for line in text.splitlines() if line.strip() and not _is_rule(line)]
    if not lines:
        return ()
    headers = tuple(h.lower() for h in lines[0].split())
    rows: list = []
    for line in lines[1:]:
        cells = tuple(line.split())
        if cells:
            rows.append(TableRow(headers=headers, cells=cells))
    return tuple(rows)


def discover_quota(
    cfg: dict, *, no_discover: bool = False, no_cache: bool = False
) -> ProbeResult:
    """``myquota``.  ``value`` is a tuple of :class:`TableRow`."""
    return probe(
        cfg,
        name="quota",
        command=QUOTA_COMMAND,
        parser=parse_table,
        empty=(),
        no_discover=no_discover,
        no_cache=no_cache,
    )


def parse_purgelist(text: str) -> tuple:
    """Parse ``purgelist`` into the absolute paths it named.

    Only absolute paths are kept.  ``purgelist`` prints a preamble about the
    retention window, and a "file" that is really a sentence is worse than no
    answer for the caller that will act on this list.
    """
    return tuple(
        line.strip()
        for line in text.splitlines()
        if line.strip().startswith("/") and " " not in line.strip()
    )


def discover_purge(
    cfg: dict, *, no_discover: bool = False, no_cache: bool = False
) -> ProbeResult:
    """``purgelist``.  ``value`` is a tuple of absolute paths."""
    return probe(
        cfg,
        name="purge",
        command=PURGE_COMMAND,
        parser=parse_purgelist,
        empty=(),
        no_discover=no_discover,
        no_cache=no_cache,
    )


# --------------------------------------------------------------------------
# The composed view -- what the CLI actually consumes
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ClusterView:
    """What one command learned about the cluster, and what it could not.

    Exists so a caller makes *one* call and gets both the data and the reasons
    for whatever is missing.  Assembling that per command is how a probe
    failure quietly becomes an abort in one of them.
    """

    slist: ProbeResult
    sinfo: ProbeResult
    sfeatures: Optional[ProbeResult] = None

    @property
    def matrix(self) -> SlistMatrix:
        """The ``slist`` matrix, empty when the probe did not succeed."""
        value = self.slist.value
        return value if isinstance(value, SlistMatrix) else SlistMatrix()

    @property
    def queue_state(self) -> dict:
        """The ``sinfo`` state map, empty when the probe did not succeed."""
        value = self.sinfo.value
        return value if isinstance(value, dict) else {}

    @property
    def features(self) -> frozenset:
        value = self.sfeatures.value if self.sfeatures is not None else None
        return value if isinstance(value, frozenset) else frozenset()

    @property
    def discovered(self) -> bool:
        """True when at least one probe came back from the cluster.

        This is what decides whether the unverified banner is printed, and it
        is deliberately an *any* and not an *all*: a ``slist`` that answered
        tells the user something real even if ``sfeatures`` did not exist.
        """
        return any(p.ok for p in self._probes)

    @property
    def warnings(self) -> list:
        return [p.warning for p in self._probes if p.warning]

    @property
    def _probes(self) -> tuple:
        probes = (self.slist, self.sinfo)
        return probes if self.sfeatures is None else probes + (self.sfeatures,)

    @property
    def known_partitions(self) -> Optional[frozenset]:
        """AC-3.8's tri-state input, assembled from what the cluster said.

        ``None`` means *not attempted, or attempted and failed* -- an
        unrecognised partition then warns and renders.  A populated frozenset
        means the cluster positively enumerated its partitions and an absent one
        is a hard error.

        **An empty result folds back to ``None``, and that is the load-bearing
        line.**  A probe that succeeded and returned zero partitions is a probe
        that failed without saying so -- a real cluster has partitions.
        Honouring it literally would reject every submission on the strength of
        an answer nobody gave.  :func:`rcac_cli.gilbreth._known_partition_set`
        repeats the fold for callers that bypass this property; the two are
        belt and braces on the same rule.

        **``slist`` is deliberately not a source here, though it is the primary
        probe everywhere else.**  RCAC's FAQ says it prints the access to each
        GPU type *for each account you are a member of* -- it answers "what may
        I charge", not "what exists".  Feeding it into a set whose only use is a
        hard error turns "you have no allocation on this partition" into
        ``partition 'a30' does not exist on this cluster``, which is both false
        and unactionable.  ``sinfo`` asks the scheduler what exists, which is
        the question being answered.  The matrix is still shown to the user by
        ``rcac-partition``; it just does not get to veto a submission.
        """
        if not self.sinfo.ok:
            return None
        return frozenset(self.queue_state) or None


def survey(
    cfg: dict,
    *,
    no_discover: bool = False,
    no_cache: bool = False,
    with_features: bool = False,
) -> ClusterView:
    """Run the partition-facing probes and return what they found.

    ``slist`` first because it is the primary source; ``sinfo`` and
    ``sfeatures`` enrich it (AC-4.2) and each may fail on its own without
    affecting the others.  Nothing here raises.
    """
    slist = discover_accounts(cfg, no_discover=no_discover, no_cache=no_cache)
    sinfo = discover_partitions(cfg, no_discover=no_discover, no_cache=no_cache)
    features = (
        discover_features(cfg, no_discover=no_discover, no_cache=no_cache)
        if with_features
        else None
    )
    return ClusterView(slist=slist, sinfo=sinfo, sfeatures=features)
