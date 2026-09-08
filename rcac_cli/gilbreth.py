"""Gilbreth's submission rules, expressed once (AC-3.5-3.11, AC-4.9).

This module is deliberately **wired to nothing** at C3a.  :func:`build_directives`
is defined and tested here and called from nowhere; C3b routes the three script
producers through it and C3c turns on the AC-3.19 choke-point assertions.  The
split exists so this file can be reviewed as what it is -- a transcription of
RCAC's documentation into executable rules -- without template or wiring noise
in the diff.

**Gilbreth's submission model is not the model the pre-port CLI assumed.**
The pre-port code carried the previous cluster's single-token partition names
and ``sbatch --comment``.  On Gilbreth, ``-A <account>`` and ``-p <partition>`` are
*both* required and they mean different things: the account is the **billing
entity**, the partition selects the **GPU type**.  Consequently the CLI must
never emit a ``--gres`` GPU *type* string -- the GPU type is chosen by the
partition, and a ``--gres=gpu:a100`` alongside ``-p a10`` is a contradiction the
scheduler resolves in whichever way the user did not intend.
Source: https://docs.rcac.purdue.edu/userguides/gilbreth/run_jobs/queues/
and https://docs.rcac.purdue.edu/userguides/gilbreth/run_jobs/submit_script/

**Hints, not info.**  The tables below are named ``*_HINTS`` rather than
``*_INFO`` because C5's ``discovery.py`` supersedes them at runtime from the
cluster's own ``slist`` / ``sinfo`` output.  A static table that contradicts what
the user can read in their own ``slist`` output is worse than no table, so every
entry carries the ``source`` URL that justifies it and a ``verified`` flag
(AC-4.3).

``verified`` means *"corroborated by the cited RCAC page(s), with no other RCAC
page contradicting it"*.  It does **not** mean "confirmed against a live
cluster" -- no live cluster was available while this was written.  The live
confirmations are section 6.2 rows L1 (``h100``) and L16 (A10 RT cores) of the
plan; running them flips flags here, and flips nothing else.

**Warnings go to stderr and are meant to be repeated.**  A single warning
printed above a 40-job sweep scrolls off the top of the terminal before the
sweep finishes, so warnings are *data* on the returned :class:`SubmissionPlan`
rather than a side effect.  The caller accumulates them and calls
:func:`emit_warnings` a second time with a ``header`` at the end of the run.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from typing import IO, Optional, Sequence

import click

# --------------------------------------------------------------------------
# Sources.  Every fact in this module traces to one of these.
# --------------------------------------------------------------------------

#: The queues page: partitions, their GPU types and counts, QOS names and caps,
#: and the "do not write ``-A standby``" instruction.
QUEUES_DOC = "https://docs.rcac.purdue.edu/userguides/gilbreth/run_jobs/queues/"

#: The submit-script page: ``-A``/``-p``/``-q`` are all submission-time flags,
#: ``--gpus-per-node`` is the documented GPU request form, and the default
#: walltime is 30 minutes.
SUBMIT_DOC = (
    "https://docs.rcac.purdue.edu/userguides/gilbreth/run_jobs/submit_script/"
)

#: The FAQ page.  Its sample ``slist`` output shows an ``h100`` column that the
#: queues page does not list.  Cited only on the ``h100`` entry, where it is one
#: half of a documented contradiction (AC-4.9 / R5).
FAQ_DOC = "https://docs.rcac.purdue.edu/userguides/gilbreth/faqs/"


# --------------------------------------------------------------------------
# AC-4.3 / AC-4.9 -- partition hints
# --------------------------------------------------------------------------

#: Partition -> hint.  Keys are exactly the partition names ``sbatch -p`` takes.
#:
#: ``max_walltime`` is the cap the *partition* imposes under the default QOS;
#: the QOS may cap it lower (see :data:`QOS_HINTS`).  It is written in SLURM
#: walltime syntax and parsed by :func:`parse_walltime`, so the table cannot
#: drift into a format the parser rejects -- ``test_every_hint_walltime_parses``
#: pins that.
#:
#: ``gpus_per_node`` is the *maximum* a node in the partition offers.  It is a
#: hint for humans and for C5's cross-check; nothing in :func:`build_directives`
#: clamps against it, because the authoritative answer comes from the cluster.
PARTITION_HINTS: dict[str, dict] = {
    "a10": {
        "gpu": "A10",
        "gpu_memory_gb": 24,
        "gpus_per_node": 3,
        "max_walltime": "14-00:00:00",
        "nvlink": False,
        "verified": True,
        "source": (QUEUES_DOC,),
        "notes": (
            "3x A10 24GB per node.  Whether these expose RT cores for Isaac Sim "
            "is unconfirmed offline -- plan section 6.2 row L16."
        ),
    },
    "a30": {
        "gpu": "A30",
        "gpu_memory_gb": 24,
        "gpus_per_node": 3,
        "max_walltime": "14-00:00:00",
        "nvlink": False,
        "verified": True,
        "source": (QUEUES_DOC,),
        "notes": "3x A30 24GB per node.",
    },
    "a100-40gb": {
        "gpu": "A100",
        "gpu_memory_gb": 40,
        # Heterogeneous partition: the G nodes carry 2 GPUs / 510GB RAM, the N
        # nodes carry 4 GPUs / 1TB RAM *and* NVLink.  The number recorded is the
        # maximum, i.e. the N nodes; `--constraint=nvlink` is what lands a job
        # on them (see NVLINK_CONSTRAINT).
        "gpus_per_node": 4,
        "max_walltime": "14-00:00:00",
        "nvlink": True,
        "verified": True,
        "source": (QUEUES_DOC,),
        "notes": (
            "G nodes: 2 GPU / 510GB.  N nodes: 4 GPU / 1TB with NVLink.  The "
            "only partition for which --constraint=nvlink is documented."
        ),
    },
    "a100-80gb": {
        "gpu": "A100",
        "gpu_memory_gb": 80,
        "gpus_per_node": 2,
        "max_walltime": "14-00:00:00",
        "nvlink": False,
        "verified": True,
        "source": (QUEUES_DOC,),
        "notes": "2x A100 80GB per node.",
    },
    "training": {
        "gpu": "unspecified",
        "gpu_memory_gb": None,
        "gpus_per_node": 4,
        # 24h, not two weeks: `training` is capped by both the partition and its
        # mandatory QOS of the same name.
        "max_walltime": "24:00:00",
        "nvlink": False,
        "verified": True,
        "source": (QUEUES_DOC,),
        "notes": (
            "4 GPUs per node.  Requires -q training; see REQUIRED_QOS and "
            "AC-3.6 -- the mismatch is an error, not a silent rewrite."
        ),
    },
    # AC-4.9 / R5.  RCAC's own pages contradict each other here: the FAQ's
    # sample `slist` output shows an h100 column, while the queues page does not
    # list the partition at all and the agent-facing context file warns "Do NOT
    # invent partition or QOS names."  The entry is kept with `verified: False`
    # and *both* URLs, because the contradiction is itself a documented fact --
    # dropping the entry would be indistinguishable from never having looked.
    # Plan section 6.2 row L1 resolves it with one `slist` on the cluster; it
    # never flips AC-3.8's tri-state.
    "h100": {
        "gpu": "H100",
        "gpu_memory_gb": None,
        "gpus_per_node": None,
        "max_walltime": None,
        "nvlink": False,
        "verified": False,
        "source": (FAQ_DOC, QUEUES_DOC),
        "notes": (
            "CONTRADICTED: listed in the FAQ's sample slist output, absent from "
            "the queues page.  Unresolvable offline (plan 6.2 L1)."
        ),
    },
}


# --------------------------------------------------------------------------
# AC-4.3 -- QOS hints
# --------------------------------------------------------------------------

#: The QOS name Gilbreth uses when ``-q`` is omitted.
DEFAULT_QOS = "normal"

#: QOS -> hint.  Same shape as :data:`PARTITION_HINTS`.
QOS_HINTS: dict[str, dict] = {
    "normal": {
        "max_walltime": "14-00:00:00",
        "charged": True,
        "default": True,
        "verified": True,
        "source": (QUEUES_DOC,),
        "notes": "The default QOS.  Charged against the account's allocation.",
    },
    "standby": {
        # The 4h cap is the whole point of AC-3.5.
        "max_walltime": "04:00:00",
        "charged": False,
        "default": False,
        "verified": True,
        "source": (QUEUES_DOC,),
        "notes": (
            "Uncharged, low priority, 4 hour cap.  This is a QOS and NOT an "
            "account -- see ABOLISHED_STANDBY_FORMS."
        ),
    },
    "training": {
        "max_walltime": "24:00:00",
        "charged": True,
        "default": False,
        "verified": True,
        "source": (QUEUES_DOC,),
        "notes": "Required for -p training; 24 hour cap.",
    },
}

#: Partition -> the QOS that partition *requires*.  AC-3.6: a mismatch is a hard
#: error, not a silent rewrite.  Silently forcing ``-q training`` would submit a
#: job the user did not ask for, under a 24h cap they did not budget for.
REQUIRED_QOS: dict[str, str] = {"training": "training"}

#: QOS names that exist on other RCAC clusters but *not* on Gilbreth.  Named
#: here so the warning can say why rather than just "unknown".
#: Source: QUEUES_DOC (Gilbreth lists exactly normal / standby / training).
ABSENT_QOS: frozenset = frozenset({"preemptible", "debug"})


# --------------------------------------------------------------------------
# Other documented facts
# --------------------------------------------------------------------------

#: RCAC abolished the older ``-A standby`` form -- the queues page says, in as
#: many words, "Do NOT write ``-A standby``".  ``-p standby`` is the same
#: mistake wearing the other flag.  Both are hard errors naming ``-q standby``
#: (AC-3.7); neither is silently rewritten, because a job charged to the wrong
#: account is worse than a job that did not start.
ABOLISHED_STANDBY_FORMS: frozenset = frozenset({"standby"})

#: The only constraint RCAC documents for Gilbreth.  It applies to
#: ``a100-40gb`` and lands the job on the NVLinked N nodes.
NVLINK_CONSTRAINT = "nvlink"

#: Partitions on which :data:`NVLINK_CONSTRAINT` is documented.  Derived from
#: :data:`PARTITION_HINTS` so the two cannot drift.
NVLINK_PARTITIONS: frozenset = frozenset(
    name for name, hint in PARTITION_HINTS.items() if hint.get("nvlink")
)

#: Gilbreth's default walltime when ``--time`` is omitted: 30 minutes.  That is
#: short enough that omitting ``--time`` is effectively always a mistake, so
#: :func:`build_directives` requires it and quotes this number in the error.
#: Source: SUBMIT_DOC.
DEFAULT_WALLTIME_SECONDS = 30 * 60

#: **Not an RCAC-documented fact.**  A conservative guess used only to size the
#: default ``--mem`` when the caller does not pass one (AC-3.11 / R8).  It is a
#: *resource request*, not a cluster fact: an over-request queues longer and an
#: under-request is OOM-killed, and either way ``sbatch`` or the job fails
#: loudly and immediately.  That is categorically different from an invented
#: partition name, which fails silently by running somewhere unintended.  If a
#: real per-GPU memory figure ever lands, it belongs in PARTITION_HINTS with a
#: source, not here.
DEFAULT_MEM_PER_GPU_GB = 32


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


class SubmissionError(click.ClickException):
    """A submission that Gilbreth is documented to reject.

    Subclasses :class:`click.ClickException` for the same reason
    :class:`rcac_cli.config.CredentialError` does: every one of the 12 console
    scripts then exits 1 with the message on **stderr** without re-implementing
    that at each call site.
    """

    exit_code = 1


class WalltimeError(SubmissionError):
    """A walltime string SLURM would not accept."""


# --------------------------------------------------------------------------
# AC-3.10 -- the walltime parser
# --------------------------------------------------------------------------

#: SLURM's ``--time`` grammar, restricted to the six forms Gilbreth's docs use.
#:
#: The parser exists because two of those forms are genuinely ambiguous to read
#: and easy to get backwards:
#:
#: * a bare ``30`` is **thirty minutes**, not thirty seconds and not thirty
#:   hours;
#: * ``2-00`` is **two days**, not two minutes -- the ``D-HH`` form's hyphen is
#:   a separator, not a range or a sign.
#:
#: Getting either backwards turns AC-3.5's 4-hour standby cap into a check that
#: passes for a job 60x over the limit.
_WALLTIME_RE = re.compile(
    r"""
    ^
    (?:
        # D-HH[:MM[:SS]] -- days first.
        (?P<days>\d+)-(?P<dhours>\d{1,2})
            (?::(?P<dminutes>\d{1,2})(?::(?P<dseconds>\d{1,2}))?)?
      |
        # HH:MM:SS -- hours are unbounded, SLURM accepts e.g. 100:00:00.
        (?P<hours>\d+):(?P<hminutes>\d{1,2}):(?P<hseconds>\d{1,2})
      |
        # MM[:SS] -- a bare integer is MINUTES.
        (?P<minutes>\d+)(?::(?P<seconds>\d{1,2}))?
    )
    $
    """,
    re.VERBOSE,
)


def parse_walltime(value: object) -> int:
    """Parse a SLURM ``--time`` string into seconds (AC-3.10).

    Accepts ``MM``, ``MM:SS``, ``HH:MM:SS``, ``D-HH``, ``D-HH:MM`` and
    ``D-HH:MM:SS``.  Raises :class:`WalltimeError` on anything else.

    >>> parse_walltime("30")
    1800
    >>> parse_walltime("2-00")
    172800
    """
    if value is None:
        raise WalltimeError("--time is required; got nothing")
    text = str(value).strip()
    if not text:
        raise WalltimeError("--time is required; got an empty string")

    match = _WALLTIME_RE.match(text)
    if match is None:
        raise WalltimeError(
            f"{text!r} is not a SLURM walltime.  Use MM, MM:SS, HH:MM:SS, "
            "D-HH, D-HH:MM or D-HH:MM:SS (a bare number is MINUTES; "
            "'2-00' is two days)."
        )

    g = match.groupdict()
    if g["days"] is not None:
        return (
            int(g["days"]) * 86400
            + int(g["dhours"]) * 3600
            + int(g["dminutes"] or 0) * 60
            + int(g["dseconds"] or 0)
        )
    if g["hours"] is not None:
        return (
            int(g["hours"]) * 3600
            + int(g["hminutes"]) * 60
            + int(g["hseconds"])
        )
    return int(g["minutes"]) * 60 + int(g["seconds"] or 0)


# --------------------------------------------------------------------------
# AC-3.9 -- the return-value contract
# --------------------------------------------------------------------------


@dataclass
class SubmissionPlan:
    """The *effective* directives for one job, plus why they are what they are.

    This is the whole reason :func:`build_directives` returns an object instead
    of a bare ``list[str]`` of warnings: a suppressed directive has to be
    *expressible*.  When ``--constraint=nvlink`` is misused, the plan must be
    able to say both "here is your job" and "the constraint you asked for is
    gone" -- and a caller that renders from ``plan.constraint`` then cannot
    accidentally re-emit it.  A list of warnings can only complain; it cannot
    suppress.
    """

    account: str
    partition: str
    qos: str
    time: str
    time_seconds: int
    gpus_per_node: int
    mem: str
    nodes: int = 1
    ntasks_per_node: int = 1
    constraint: Optional[str] = None
    warnings: list = field(default_factory=list)

    def to_slurm_dict(self) -> dict:
        """The plan as a ``slurm:`` config mapping (AC-3.19(b)).

        C3c asserts that **no module outside this one** constructs a dict
        literal carrying a ``"partition"`` or ``"qos"`` key.  ``simulator.py``
        emits exactly such a mapping into generated YAML, where an invalid
        partition/QOS pair survives until launch; routing it through here makes
        it fail at generation time instead.

        ``constraint`` is omitted when ``None`` so a suppressed nvlink request
        cannot reappear as ``constraint: null`` in someone's YAML.  There is no
        ``gres`` key and there never will be -- the GPU type is chosen by the
        partition.
        """
        payload = {
            "account": self.account,
            "partition": self.partition,
            "qos": self.qos,
            "time": self.time,
            "nodes": self.nodes,
            "ntasks_per_node": self.ntasks_per_node,
            "gpus_per_node": self.gpus_per_node,
            "mem": self.mem,
        }
        if self.constraint:
            payload["constraint"] = self.constraint
        return payload


# --------------------------------------------------------------------------
# Warning transport
# --------------------------------------------------------------------------

WARNING_PREFIX = "warning: "


def emit_warnings(
    warnings: Sequence,
    *,
    stream: Optional[IO] = None,
    header: Optional[str] = None,
) -> None:
    """Print *warnings* to **stderr**, one per line.

    Called twice in a normal run: once per job, and once more at the end with a
    ``header`` and the accumulated list.  The repeat is not redundancy -- in a
    40-job sweep the first copy has scrolled off the terminal long before the
    user reads the last ``sbatch`` id.

    ``stream`` is resolved at call time rather than defaulted in the signature
    so that ``capsys`` and ``click``'s runner both see the output.
    """
    if not warnings:
        return
    out = sys.stderr if stream is None else stream
    if header:
        print(header, file=out)
    for warning in warnings:
        print(f"{WARNING_PREFIX}{warning}", file=out)


# --------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------


def _hint_walltime_seconds(hint: dict) -> Optional[int]:
    """The ``max_walltime`` of a hint row in seconds, or ``None`` if unknown."""
    raw = hint.get("max_walltime")
    if raw is None:
        return None
    return parse_walltime(raw)


def _known_partition_set(known_partitions) -> Optional[frozenset]:
    """Normalise AC-3.8's tri-state input.

    **Judgment call, not a documented rule.**  An *empty* collection is folded
    back into ``None``.  A discovery that succeeded and reported zero partitions
    is a discovery that failed in a way that did not raise -- a real cluster has
    partitions.  Treating it as authoritative would hard-error every single
    submission, which is precisely the failure mode AC-3.8's tri-state exists to
    avoid.
    """
    if known_partitions is None:
        return None
    resolved = frozenset(known_partitions)
    return resolved or None


# --------------------------------------------------------------------------
# The choke point
# --------------------------------------------------------------------------


def build_directives(
    *,
    account: str,
    partition: str,
    time: str,
    qos: str = DEFAULT_QOS,
    gpus_per_node: int = 1,
    nodes: int = 1,
    ntasks_per_node: int = 1,
    mem: Optional[str] = None,
    mem_per_gpu_gb: int = DEFAULT_MEM_PER_GPU_GB,
    constraint: Optional[str] = None,
    known_partitions=None,
) -> SubmissionPlan:
    """Validate one submission and return its effective directives.

    The single choke point every script producer goes through from C3b onward.
    Raises :class:`SubmissionError` for anything RCAC documents as invalid;
    attaches a warning to the returned plan for anything merely *suspicious*.

    The line between the two is deliberate.  A hard error is reserved for a form
    RCAC's documentation positively rules out -- ``-A standby``, a standby job
    over four hours, ``-p training`` under the wrong QOS, a partition the
    cluster itself has told us does not exist.  Everything else warns and
    renders, because ``sbatch`` rejects a bad partition in well under a second
    with a message naming the valid ones, whereas a static allowlist that
    contradicts what the user can see in their own ``slist`` output blocks a job
    that would have run.

    ``known_partitions`` carries AC-3.8's tri-state: ``None`` means discovery
    was not attempted or failed, and an unrecognised partition only warns; a
    populated set means the cluster positively reported its partitions, and an
    absent one is a hard error.  Absence of evidence is not evidence of absence.
    """
    account_text = str(account or "").strip()
    partition_text = str(partition or "").strip()
    qos_text = str(qos or "").strip() or DEFAULT_QOS
    warnings: list = []

    # -- AC-3.7: the abolished `-A standby` / `-p standby` forms. ------------
    # Checked first, and before the "required" checks, so the user gets the
    # specific replacement rather than a generic complaint.
    if account_text.lower() in ABOLISHED_STANDBY_FORMS:
        raise SubmissionError(
            "-A standby is not a thing on Gilbreth: standby is a QOS, not an "
            "account.  Pass your real allocation account with -A and add "
            "-q standby.  See " + QUEUES_DOC
        )
    if partition_text.lower() in ABOLISHED_STANDBY_FORMS:
        raise SubmissionError(
            "-p standby is not a thing on Gilbreth: standby is a QOS, not a "
            "partition.  Select a GPU type with -p (one of "
            f"{', '.join(sorted(PARTITION_HINTS))}) and add -q standby.  "
            "See " + QUEUES_DOC
        )

    # -- Both -A and -p are required. ---------------------------------------
    if not account_text:
        raise SubmissionError(
            "-A/--account is required on Gilbreth; it names the allocation the "
            "job is billed to.  Set RCAC_ACCOUNT or slurm.account.  See "
            + SUBMIT_DOC
        )
    if not partition_text:
        raise SubmissionError(
            "-p/--partition is required on Gilbreth; it selects the GPU type.  "
            f"Documented partitions: {', '.join(sorted(PARTITION_HINTS))}.  See "
            + QUEUES_DOC
        )

    # -- --time is mandatory in practice. -----------------------------------
    if time is None or not str(time).strip():
        raise SubmissionError(
            "--time is required: Gilbreth's default walltime is only "
            f"{DEFAULT_WALLTIME_SECONDS // 60} minutes, which will kill "
            "essentially any real job.  See " + SUBMIT_DOC
        )
    time_text = str(time).strip()
    time_seconds = parse_walltime(time_text)

    # -- AC-3.8: the tri-state partition check. -----------------------------
    known = _known_partition_set(known_partitions)
    if known is not None:
        if partition_text not in known:
            raise SubmissionError(
                f"partition {partition_text!r} does not exist on this cluster.  "
                f"Available: {', '.join(sorted(known))}"
            )
    elif partition_text not in PARTITION_HINTS:
        warnings.append(
            f"partition {partition_text!r} is not in the offline hint table and "
            "no partition list was discovered, so it could not be checked.  "
            "sbatch will reject it immediately if it is wrong.  Documented "
            f"partitions: {', '.join(sorted(PARTITION_HINTS))}"
        )
    elif not PARTITION_HINTS[partition_text].get("verified", False):
        hint = PARTITION_HINTS[partition_text]
        warnings.append(
            f"partition {partition_text!r} is unverified: RCAC's own pages "
            "disagree about it ("
            + " vs ".join(hint["source"])
            + ").  Proceeding; sbatch is the authority."
        )

    # -- QOS existence. -----------------------------------------------------
    if qos_text in ABSENT_QOS:
        warnings.append(
            f"-q {qos_text} does not exist on Gilbreth (it exists on other RCAC "
            f"clusters).  Valid: {', '.join(sorted(QOS_HINTS))}"
        )
    elif qos_text not in QOS_HINTS:
        warnings.append(
            f"-q {qos_text} is not a documented Gilbreth QOS.  Documented: "
            f"{', '.join(sorted(QOS_HINTS))}"
        )

    # -- AC-3.6: partition/QOS pairing, never silently rewritten. -----------
    required = REQUIRED_QOS.get(partition_text)
    if required is not None and qos_text != required:
        raise SubmissionError(
            f"-p {partition_text} requires -q {required}, but -q {qos_text} was "
            f"given.  Pass -q {required} explicitly -- this is not corrected "
            "for you, because the QOS also changes the walltime cap and the "
            "charging.  See " + QUEUES_DOC
        )
    if (
        qos_text in REQUIRED_QOS.values()
        and REQUIRED_QOS.get(partition_text) != qos_text
    ):
        warnings.append(
            f"-q {qos_text} is documented only in combination with "
            f"-p {qos_text}; pairing it with -p {partition_text} is undocumented"
        )

    # -- AC-3.5 / AC-3.6: walltime caps, driven by the hint tables. ---------
    qos_cap = _hint_walltime_seconds(QOS_HINTS.get(qos_text, {}))
    if qos_cap is not None and time_seconds > qos_cap:
        raise SubmissionError(
            f"--time {time_text} exceeds the {QOS_HINTS[qos_text]['max_walltime']} "
            f"cap on -q {qos_text}.  See " + QUEUES_DOC
        )
    partition_cap = _hint_walltime_seconds(PARTITION_HINTS.get(partition_text, {}))
    if partition_cap is not None and time_seconds > partition_cap:
        raise SubmissionError(
            f"--time {time_text} exceeds the "
            f"{PARTITION_HINTS[partition_text]['max_walltime']} cap on "
            f"-p {partition_text}.  See " + QUEUES_DOC
        )

    # -- GPUs and memory (AC-3.11). -----------------------------------------
    gpus = int(gpus_per_node)
    if gpus < 1:
        raise SubmissionError(
            f"--gpus-per-node must be at least 1 on a GPU cluster; got {gpus}"
        )
    node_count = int(nodes)
    tasks_per_node = int(ntasks_per_node)
    if node_count < 1:
        raise SubmissionError(f"--nodes must be at least 1; got {node_count}")
    if tasks_per_node < 1:
        raise SubmissionError(
            f"--ntasks-per-node must be at least 1; got {tasks_per_node}"
        )

    mem_text = str(mem).strip() if mem is not None and str(mem).strip() else ""
    if not mem_text:
        mem_text = f"{gpus * int(mem_per_gpu_gb)}G"

    # -- AC-3.9: the nvlink truth table. ------------------------------------
    effective_constraint = _resolve_constraint(
        constraint=constraint,
        partition=partition_text,
        gpus_per_node=gpus,
        nodes=node_count,
        ntasks_per_node=tasks_per_node,
        warnings=warnings,
    )

    return SubmissionPlan(
        account=account_text,
        partition=partition_text,
        qos=qos_text,
        time=time_text,
        time_seconds=time_seconds,
        gpus_per_node=gpus,
        mem=mem_text,
        nodes=node_count,
        ntasks_per_node=tasks_per_node,
        constraint=effective_constraint,
        warnings=warnings,
    )


def _resolve_constraint(
    *,
    constraint: Optional[str],
    partition: str,
    gpus_per_node: int,
    nodes: int,
    ntasks_per_node: int,
    warnings: list,
) -> Optional[str]:
    """AC-3.9.  Return the constraint to actually emit, warning on misuse.

    ``--constraint=nvlink`` is documented for exactly one situation: multiple
    GPUs on ``a100-40gb`` serving a **single task**, where the point is to land
    on the NVLinked N nodes.  Every other use is dropped -- set to ``None`` --
    *and* warned about, so the caller renders a job that runs rather than one
    the scheduler cannot place.

    The single-task precondition is asserted here rather than assumed.  All four
    templates hardcode ``--ntasks-per-node=1`` today, which would make the check
    vacuous, but ``nodes`` is a caller parameter and nothing stops a future
    template from parametrising ``--ntasks-per-node``.  A precondition that is
    true by coincidence is not checked at all.
    """
    if constraint is None:
        return None
    text = str(constraint).strip()
    if not text:
        return None

    if text.lower() != NVLINK_CONSTRAINT:
        warnings.append(
            f"--constraint={text} is not documented for Gilbreth; "
            f"{NVLINK_CONSTRAINT!r} is the only documented feature.  Passing it "
            "through unchanged.  See " + QUEUES_DOC
        )
        return text

    if partition not in NVLINK_PARTITIONS:
        warnings.append(
            f"--constraint={NVLINK_CONSTRAINT} is documented only for "
            f"{', '.join(sorted(NVLINK_PARTITIONS))}, not for -p {partition}; "
            "dropping it"
        )
        return None
    if gpus_per_node < 2:
        warnings.append(
            f"--constraint={NVLINK_CONSTRAINT} needs more than one GPU per node "
            f"to mean anything (got --gpus-per-node={gpus_per_node}); "
            "dropping it"
        )
        return None
    if nodes != 1 or ntasks_per_node != 1:
        warnings.append(
            f"--constraint={NVLINK_CONSTRAINT} is documented for multiple GPUs "
            f"serving a single task, but this job requests nodes={nodes} and "
            f"--ntasks-per-node={ntasks_per_node}; dropping it"
        )
        return None
    return NVLINK_CONSTRAINT
