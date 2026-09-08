"""File preservation automation for /scratch 15-day purge policy."""

from __future__ import annotations

import subprocess

import click

from .config import get_rcac_config, require_account, require_user
from .gilbreth import emit_warnings
from .scheduler import generate_keepalive_script
from .ssh import ssh_opt_argv, ssh_target

# --------------------------------------------------------------------------
# The keepalive job's submission policy (AC-6.4)
#
# This module decides *what kind of job* a `touch -a` sweep deserves;
# `scheduler.generate_keepalive_script` decides what the header looks like and
# `gilbreth.build_directives` decides whether the combination is legal.  The
# pre-port header hardcoded a partition from the previous cluster and a job
# class Gilbreth has no flag for, and nothing checked either.
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
#: C6's AC-6.5 makes the login-node form the default for exactly this reason.
KEEPALIVE_FALLBACK_PARTITION = "a10"

# Patterns for important files to keep alive
KEEPALIVE_PATTERNS = [
    # Data
    ("data", "-type f"),
    # Model checkpoints
    ("projects", "-name '*.pth' -o -name '*.pt' -o -name '*.ckpt'"),
    # Configs
    ("projects", "-name '*.yaml' -o -name '*.yml' -o -name '*.json'"),
    # Singularity images
    ("", "-name '*.sif'"),
    # Source code
    ("projects", "-name '*.py'"),
]


def _run_remote(cfg: dict, command: str) -> subprocess.CompletedProcess:
    cmd = ["ssh", *ssh_opt_argv(cfg), ssh_target(cfg), command]
    return subprocess.run(cmd, capture_output=True, text=True)


@click.group()
def cli() -> None:
    """Protect important files from /scratch 15-day auto-deletion."""


@cli.command()
@click.option("--dry-run", is_flag=True, help="Show what would be touched without actually doing it")
@click.option("--verbose", "-v", is_flag=True, help="Show each file being touched")
def touch(dry_run: bool, verbose: bool) -> None:
    """Update access time of important files to prevent purging."""
    cfg = get_rcac_config()

    require_user(cfg)

    user = cfg["ssh"]["user"]
    scratch = f"/scratch/{user}"

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


@cli.command("check")
@click.option("--days", "-d", default=12, help="Warn for files not accessed in N days (default: 12)")
def check_at_risk(days: int) -> None:
    """Show files at risk of being purged (not accessed recently)."""
    cfg = get_rcac_config()

    require_user(cfg)

    user = cfg["ssh"]["user"]
    scratch = f"/scratch/{user}"

    click.echo(f"Scanning for files not accessed in {days}+ days...\n")

    # Find important files not accessed in N days
    important_exts = "*.pth *.pt *.ckpt *.sif *.yaml *.yml *.py *.json"
    name_filters = " -o ".join(f"-name '{ext}'" for ext in important_exts.split())

    cmd = (
        f"find {scratch} \\( {name_filters} \\) -atime +{days} "
        f"-printf '%A+ %s %p\\n' 2>/dev/null | sort | head -50"
    )

    result = _run_remote(cfg, cmd)

    if result.returncode != 0 or not result.stdout.strip():
        click.echo("No at-risk files found. Everything looks safe.")
        return

    click.echo(f"{'Last Access':<22} {'Size':>10}  Path")
    click.echo("-" * 70)

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

    click.echo(f"\nThese files may be deleted in {15 - days} days or less!")
    click.echo("Run 'rcac-keepalive touch' to update access times.")


@cli.command("setup-cron")
@click.option("--interval", "-i", default=7, help="Run every N days (default: 7)")
def setup_cron(interval: int) -> None:
    """Set up a recurring SLURM job to keep files alive."""
    cfg = get_rcac_config()

    require_user(cfg)

    user = cfg["ssh"]["user"]
    # Still the pre-port literal, and still wrong for Gilbreth, whose scratch
    # root is /scratch/gilbreth/<user>. C6 replaces all three occurrences in
    # this module with config.scratch_root() (AC-6.2). The header below does
    # NOT go through here: it renders `scratch_root` from paths.scratch, which
    # is what AC-6.4's "inherits every AC-3 guarantee" requires. Until C6 lands
    # the generated job therefore names two different roots -- the correct one
    # in -o/-e, the stale one in the body.
    scratch = f"/scratch/{user}"

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

    # The batch job that runs it.  Rendered from a SubmissionPlan through the
    # same shared header as every other job, so it inherits every AC-3
    # guarantee -- including the ones this command used to get wrong: a
    # partition that does not exist on Gilbreth, no account at all, and a job
    # class the scheduler has no flag for.
    #
    # `-p` prefers whatever the user already put in their config over the
    # fallback: someone who knows which GPU type their account can reach knows
    # better than a constant here.  The `slurm.partition` key is optional and
    # `configs/rcac.yaml` ships without it.
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

    click.echo(f"Keepalive scripts created on RCAC:")
    click.echo(f"  Script: {remote_script}")
    click.echo(f"  SLURM:  {remote_slurm}")
    click.echo()
    click.echo(f"To run manually:")
    click.echo(f"  ssh gilbreth 'sbatch {remote_slurm}'")
    click.echo()
    click.echo(f"To set up periodic execution, add to crontab on RCAC:")
    click.echo(f"  ssh gilbreth 'crontab -l 2>/dev/null; echo \"0 0 */{interval} * * sbatch {remote_slurm}\"' | ssh gilbreth 'crontab -'")
    click.echo()
    click.echo(f"Or submit it now:")

    result = _run_remote(cfg, f"sbatch {remote_slurm}")
    if result.returncode == 0:
        click.echo(f"  Submitted: {result.stdout.strip()}")
    else:
        click.echo(f"  Submit failed: {result.stderr}")
