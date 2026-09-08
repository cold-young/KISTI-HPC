"""File preservation for Gilbreth's 60-day /scratch purge policy (AC-6)."""

from __future__ import annotations

import subprocess

import click

from . import discovery
from .config import get_rcac_config, require_account, require_user, scratch_root
from .gilbreth import emit_warnings
from .scheduler import generate_keepalive_script
from .ssh import ssh_cmd

# --------------------------------------------------------------------------
# The retention policy this module exists to work against (AC-6.1)
#
# Gilbreth purges a file from /scratch after 60 days without access OR
# modification, and that one number is the source of every other number here:
# the default warning window, the "deleted in N days" arithmetic, and the
# prose in --help are all derived from it rather than written down twice.
#
# The pre-port CLI derived them all from the previous cluster's much shorter
# window instead -- a quarter of this one -- so a sweep ran four times more
# often than it needed to and `check` told users a file had days left when it
# had weeks.  AC-6.1 bans the old figure from the package outright, including
# from comments like this one, because a retention number a reader finds in
# the source is a number they will repeat.  It survives in exactly one place,
# where it is history rather than a claim: the README's `history` fence
# (AC-8.4, C7).
# Source: https://docs.rcac.purdue.edu/userguides/gilbreth/storage/
# --------------------------------------------------------------------------

#: Days of no access *and* no modification after which /scratch purges a file.
SCRATCH_RETENTION_DAYS = 60

#: How stale a file must be before `check` calls it at risk.  Written as the
#: retention window less a quarter of it, so raising
#: :data:`SCRATCH_RETENTION_DAYS` moves the warning with it: a quarter of the
#: window is long enough to act on, and short enough that the list is not
#: simply every file the user has ever written.
DEFAULT_WARN_DAYS = SCRATCH_RETENTION_DAYS - SCRATCH_RETENTION_DAYS // 4

#: How many `purgelist` entries to show before summarising.  The list is the
#: cluster's own answer and can be long; the count is what matters once it is.
PURGE_PREVIEW_LIMIT = 25

# --------------------------------------------------------------------------
# The keepalive job's submission policy (AC-6.4)
#
# This module decides *what kind of job* a `touch -a` sweep deserves;
# `scheduler.generate_keepalive_script` decides what the header looks like and
# `gilbreth.build_directives` decides whether the combination is legal.  The
# pre-port header hardcoded a partition from the previous cluster and a job
# class Gilbreth has no flag for, and nothing checked either.
#
# None of it runs unless the user asks for `--batch` (AC-6.5).
# --------------------------------------------------------------------------

#: Uncharged, low priority, capped at 4 hours -- three properties that all fit
#: a job whose entire body is `find ... -exec touch -a`.  Waiting in the
#: standby queue costs the user nothing; running in `normal` would bill the
#: allocation for file maintenance.
KEEPALIVE_QOS = "standby"

#: Ten minutes, unchanged from the pre-port script and far inside the standby
#: cap.  Named rather than inlined so the 4-hour cap has something to be
#: compared against when someone raises it.
KEEPALIVE_WALLTIME = "00:10:00"

#: Deliberately small: the job runs `find`, not a model.  Without this the
#: choke point would size memory from the GPU count and ask for 32G.
KEEPALIVE_MEM = "4G"

#: The fallback GPU type, used only when the config names no partition.  Every
#: Gilbreth partition is a GPU partition and `--gpus-per-node` has a floor of
#: 1, so a keepalive job cannot avoid holding a GPU -- it can only hold the
#: least contended one.  `a10` is the smallest documented card (24GB, 3/node).
#: AC-6.5 makes the login-node form the default for exactly this reason.
KEEPALIVE_FALLBACK_PARTITION = "a10"

#: AC-6.5, stated once and shown wherever the choice is offered.  Both halves
#: are load-bearing: the walk is cheap, and the batch form is expensive in a
#: currency (an allocation) that the walk does not need to spend.
BATCH_TRADEOFF_HELP = (
    "Submit the sweep as a SLURM job instead of running it on a front-end "
    "node. Off by default: the sweep is a metadata-only `find ... -exec "
    "touch -a`, which reads directory entries and writes atimes without "
    "touching file contents, so it stays far below any reasonable front-end "
    "CPU limit -- while a batch job on Gilbreth must hold a GPU it never uses "
    "(--gpus-per-node has a floor of 1 and every partition is a GPU "
    "partition), which is a disproportionate price for updating access times. "
    "Use --batch when a sweep is large enough that a front-end node is the "
    "wrong place for it, or when local policy forbids running it there."
)

# Patterns for important files to keep alive
KEEPALIVE_PATTERNS = [
    # Data
    ("data", "-type f"),
    # Model checkpoints
    ("projects", "-name '*.pth' -o -name '*.pt' -o -name '*.ckpt'"),
    # Configs
    ("projects", "-name '*.yaml' -o -name '*.yml' -o -name '*.json'"),
    # Apptainer images
    ("", "-name '*.sif'"),
    # Source code
    ("projects", "-name '*.py'"),
]


def _run_remote(cfg: dict, command: str) -> subprocess.CompletedProcess:
    cmd = [*ssh_cmd(cfg), command]
    return subprocess.run(cmd, capture_output=True, text=True)


@click.group()
def cli() -> None:
    """Protect important files from Gilbreth's 60-day /scratch purge."""


@cli.command()
@click.option("--dry-run", is_flag=True, help="Show what would be touched without actually doing it")
@click.option("--verbose", "-v", is_flag=True, help="Show each file being touched")
def touch(dry_run: bool, verbose: bool) -> None:
    """Update access time of important files to prevent purging.

    Runs on a Gilbreth front-end node over the existing SSH session -- see
    `rcac-keepalive setup-cron --help` for why that is the default rather than
    a batch job.
    """
    cfg = get_rcac_config()

    require_user(cfg)

    scratch = scratch_root(cfg)

    total_files = 0

    for subdir, find_expr in KEEPALIVE_PATTERNS:
        target = f"{scratch}/{subdir}" if subdir else scratch

        if dry_run:
            cmd = f"find {target} \\( {find_expr} \\) -print 2>/dev/null | wc -l"
            result = _run_remote(cfg, cmd)
            count = result.stdout.strip()
            click.echo(f"  {target}: {count} files would be touched")
            total_files += int(count) if count.isdigit() else 0
        else:
            if verbose:
                cmd = f"find {target} \\( {find_expr} \\) -exec touch -a {{}} + -print 2>/dev/null | wc -l"
            else:
                cmd = f"find {target} \\( {find_expr} \\) -exec touch -a {{}} + 2>/dev/null && echo 'done'"
            result = _run_remote(cfg, cmd)
            if verbose and result.stdout.strip():
                click.echo(f"  {target}: {result.stdout.strip()} files touched")

    if dry_run:
        click.echo(f"\nTotal: {total_files} files would be touched")
    else:
        click.echo("Access times updated.")


# --------------------------------------------------------------------------
# `check` -- what the cluster says, then what we can infer (AC-6.3)
#
# `purgelist` is RCAC's own answer to "what am I about to lose", so it leads.
# The `find -atime` scan below it is an *early warning* and nothing more: it
# cannot see the purge scheduler's state, only file metadata, so presenting it
# as the answer would put a guess where a fact was available.
# `myquota` is here because the two questions arrive together -- a full
# filesystem and an expiring file are both "scratch is about to stop holding
# my data" -- and because pairing them is what the plan's live checklist row
# L9 does (`myquota ; purgelist | head`).
# --------------------------------------------------------------------------


def _report_purgelist(cfg: dict, no_discover: bool) -> None:
    """AC-6.3 -- the authoritative half of `check`."""
    result = discovery.discover_purge(cfg, no_discover=no_discover)
    click.echo(f"Staged for purge (`{discovery.PURGE_COMMAND}`):")
    if not result.ok:
        click.echo(f"  {result.warning}")
        return
    paths = list(result.value or ())
    if not paths:
        click.echo("  Nothing staged for purge.")
        return
    for path in paths[:PURGE_PREVIEW_LIMIT]:
        click.echo(f"  {path}")
    if len(paths) > PURGE_PREVIEW_LIMIT:
        click.echo(f"  ... and {len(paths) - PURGE_PREVIEW_LIMIT} more")


def _quota_line(row) -> str:
    """One `myquota` row, or its raw cells when the headers are unfamiliar.

    ``used`` and ``use%`` are asked for by *first* occurrence deliberately:
    ``myquota`` heads two columns of each -- bytes then inodes -- and the byte
    pair is the one a retention report is about.  See
    :class:`rcac_cli.discovery.TableRow` for why that is a positional question
    and not a dictionary lookup.
    """
    used = row.get("used")
    percent = row.get("use%")
    if not used and not percent:
        return "  " + " ".join(row.cells)
    limit = row.get("limit") or row.get("size") or row.get("quota") or "?"
    return f"  {row.label:<10} {used or '-':>10} / {limit:<10} {percent or '-':>6}"


def _report_quota(cfg: dict, no_discover: bool) -> None:
    result = discovery.discover_quota(cfg, no_discover=no_discover)
    click.echo(f"\nQuota (`{discovery.QUOTA_COMMAND}`):")
    if not result.ok:
        click.echo(f"  {result.warning}")
        return
    rows = list(result.value or ())
    if not rows:
        click.echo("  Reported nothing.")
        return
    for row in rows:
        click.echo(_quota_line(row))


def _report_at_risk(cfg: dict, scratch: str, days: int) -> None:
    click.echo(f"\nUntouched for {days}+ days (early warning, from `find`):")

    important_exts = "*.pth *.pt *.ckpt *.sif *.yaml *.yml *.py *.json"
    name_filters = " -o ".join(f"-name '{ext}'" for ext in important_exts.split())

    # Both times, because the policy is 60 days without access *or*
    # modification: a file written yesterday and never read since is not at
    # risk, and `-atime` alone would have listed it.
    cmd = (
        f"find {scratch} \\( {name_filters} \\) -atime +{days} -mtime +{days} "
        f"-printf '%A+ %s %p\\n' 2>/dev/null | sort | head -50"
    )

    result = _run_remote(cfg, cmd)

    if result.returncode != 0 or not result.stdout.strip():
        click.echo("  No at-risk files found. Everything looks safe.")
        return

    click.echo(f"  {'Last Access':<22} {'Size':>10}  Path")
    click.echo("  " + "-" * 68)

    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split(None, 2)
        if len(parts) == 3:
            atime, size_bytes, path = parts
            # Format date nicely
            date_part = atime.split("+")[0] if "+" in atime else atime
            try:
                size_mb = int(size_bytes) / (1024**2)
                size_str = f"{size_mb:.1f}MB"
            except ValueError:
                size_str = size_bytes
            click.echo(f"  {date_part:<22} {size_str:>10}  {path}")

    remaining = SCRATCH_RETENTION_DAYS - days
    click.echo(f"\n  These files may be deleted in {remaining} days or less.")
    click.echo("  Run 'rcac-keepalive touch' to update access times.")


@cli.command("check")
@click.option(
    "--days",
    "-d",
    default=DEFAULT_WARN_DAYS,
    show_default=True,
    help=(
        f"Warn for files not accessed in N days. The default is "
        f"{DEFAULT_WARN_DAYS}, which leaves a quarter of Gilbreth's "
        f"{SCRATCH_RETENTION_DAYS}-day purge window in hand."
    ),
)
@click.option(
    "--no-discover",
    is_flag=True,
    help=(
        f"Skip the `{discovery.PURGE_COMMAND}` and `{discovery.QUOTA_COMMAND}` "
        f"probes and show only the `find` scan. Also settable as "
        f"{discovery.NO_DISCOVER_ENV}=1."
    ),
)
def check_at_risk(days: int, no_discover: bool) -> None:
    """Show what /scratch is about to lose.

    Three answers, most authoritative first: what `purgelist` says is already
    staged for deletion, what `myquota` says about the space itself, and which
    files a `find -atime` scan puts close to the window.
    """
    cfg = get_rcac_config()

    require_user(cfg)

    scratch = scratch_root(cfg)

    _report_purgelist(cfg, no_discover)
    _report_quota(cfg, no_discover)
    _report_at_risk(cfg, scratch, days)


@cli.command("setup-cron")
@click.option("--interval", "-i", default=7, help="Run every N days (default: 7)")
@click.option("--batch", "as_batch", is_flag=True, help=BATCH_TRADEOFF_HELP)
def setup_cron(interval: int, as_batch: bool) -> None:
    """Install a recurring keepalive sweep.

    The sweep runs on a Gilbreth front-end node by default and as a SLURM job
    under --batch (AC-6.5); see that flag's help for the tradeoff.
    """
    cfg = get_rcac_config()

    require_user(cfg)

    # One root for the whole command.  Until C6 this was an f-string built
    # from `ssh.user` alone, so it named a scratch path with no cluster
    # segment -- which is not a directory that exists on Gilbreth -- while the
    # generated job's -o/-e came from `paths.scratch` through `scratch_root`.
    # One script, two roots, and the half that matched nothing was the half
    # doing the work: every `find` in the body redirects stderr, so the job
    # would have reported success while sweeping nothing (AC-6.2).
    scratch = scratch_root(cfg)

    # Create the keepalive script on RCAC
    script_content = f"""#!/bin/bash
# Auto-generated by rcac-keepalive setup-cron
# Touches important files to prevent /scratch purge

echo "$(date): Starting keepalive run"

find {scratch}/data -type f -exec touch -a {{}} + 2>/dev/null
find {scratch}/projects \\( -name '*.pth' -o -name '*.pt' -o -name '*.ckpt' \\) -exec touch -a {{}} + 2>/dev/null
find {scratch}/projects \\( -name '*.yaml' -o -name '*.yml' -o -name '*.json' \\) -exec touch -a {{}} + 2>/dev/null
find {scratch}/projects -name '*.py' -exec touch -a {{}} + 2>/dev/null
find {scratch} -name '*.sif' -exec touch -a {{}} + 2>/dev/null

echo "$(date): Keepalive complete"
"""

    remote_script = f"{scratch}/.rcac-tools/keepalive.sh"
    _run_remote(cfg, f"mkdir -p {scratch}/.rcac-tools")

    # Write script remotely
    escaped = script_content.replace("'", "'\\''")
    _run_remote(cfg, f"echo '{escaped}' > {remote_script} && chmod +x {remote_script}")

    click.echo("Keepalive script created on RCAC:")
    click.echo(f"  Script: {remote_script}")

    if as_batch:
        cron_command = _install_batch_job(cfg, scratch, remote_script)
    else:
        cron_command = f"bash {remote_script}"

    click.echo()
    click.echo("To run manually:")
    click.echo(f"  ssh gilbreth '{cron_command}'")
    click.echo()
    click.echo("To set up periodic execution, add to crontab on RCAC:")
    click.echo(
        f"  ssh gilbreth 'crontab -l 2>/dev/null; echo \"0 0 */{interval} * * "
        f"{cron_command}\"' | ssh gilbreth 'crontab -'"
    )
    click.echo()
    click.echo("Or run it now:")

    result = _run_remote(cfg, cron_command)
    if result.returncode == 0:
        click.echo(f"  Done: {result.stdout.strip()}")
    else:
        click.echo(f"  Failed: {result.stderr}")


def _install_batch_job(cfg: dict, scratch: str, remote_script: str) -> str:
    """Render and upload the batch job, and return the command that runs it.

    Rendered from a SubmissionPlan through the same shared header as every
    other job, so it inherits every AC-3 guarantee -- including the ones this
    command used to get wrong: a partition that does not exist on Gilbreth, no
    account at all, and a job class the scheduler has no flag for.

    `-p` prefers whatever the user already put in their config over the
    fallback: someone who knows which GPU type their account can reach knows
    better than a constant here.  The `slurm.partition` key is optional and
    `configs/rcac.yaml` ships without it.
    """
    partition = (cfg.get("slurm") or {}).get("partition") or (
        KEEPALIVE_FALLBACK_PARTITION
    )
    plan_out: list = []
    slurm_script = generate_keepalive_script(
        remote_script,
        account=require_account(cfg),
        partition=partition,
        qos=KEEPALIVE_QOS,
        time=KEEPALIVE_WALLTIME,
        mem=KEEPALIVE_MEM,
        config=cfg,
        plan_out=plan_out,
    )
    for plan in plan_out:
        emit_warnings(plan.warnings)

    remote_slurm = f"{scratch}/.rcac-tools/keepalive_job.sh"
    escaped_slurm = slurm_script.replace("'", "'\\''")
    _run_remote(cfg, f"echo '{escaped_slurm}' > {remote_slurm}")
    click.echo(f"  SLURM:  {remote_slurm}")
    return f"sbatch {remote_slurm}"
