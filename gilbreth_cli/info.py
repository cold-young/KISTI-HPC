"""Cluster information: accounts, partitions, nodes, quota, scratch purge and a setup doctor."""

from __future__ import annotations

import shlex
import sys

import click

from .config import ENV_ACCOUNT, ENV_USER, get_config, get_config_path, require_user
from .remote import is_connected, run_remote


@click.group()
def cli() -> None:
    """Gilbreth cluster information and setup checks."""


@cli.command()
def accounts() -> None:
    """Your SLURM accounts and GPU balance (slist)."""
    cfg = get_config()
    require_user(cfg)
    result = run_remote(cfg, "slist")
    click.echo(result.stdout or result.stderr, err=result.returncode != 0)
    if result.returncode != 0:
        sys.exit(1)
    else:
        click.echo(f"Use one of these as -A / {ENV_ACCOUNT} / slurm.account in {get_config_path()}")


@cli.command()
def partitions() -> None:
    """Partition cheat-sheet plus live idle/alloc GPU counts (sinfo)."""
    cfg = get_config()
    click.echo(f"{'Partition':<12}{'GPU':<20}{'GPUs/node':<66}QOS")
    for name, row in (cfg.get("partitions") or {}).items():
        click.echo(f"{name:<12}{row.get('gpu', ''):<20}{row.get('gpus_per_node', '')!s:<66}{row.get('qos', '')}")
    if cfg["ssh"].get("user") and is_connected(cfg):
        click.echo("\nLive node state (sinfo):")
        result = run_remote(cfg, "sinfo -o '%.12P %.6a %.10l %.6D %.10T %.30G' | sort")
        click.echo(result.stdout or result.stderr)
    else:
        click.echo("\n(connect with gilbreth-connect to see live sinfo output)")


@cli.command()
@click.option("--partition", "-p", default=None)
def nodes(partition: str | None) -> None:
    """Node features / GPU inventory (sfeatures)."""
    cfg = get_config()
    require_user(cfg)
    result = run_remote(cfg, "sfeatures" + (f" -p {shlex.quote(partition)}" if partition else ""))
    click.echo(result.stdout or result.stderr)


@cli.command()
def quota() -> None:
    """Home / scratch usage (myquota)."""
    cfg = get_config()
    require_user(cfg)
    result = run_remote(cfg, "myquota")
    click.echo(result.stdout or result.stderr)


@cli.command("purge-check")
@click.option("--days", "-d", default=50, help="Flag files not accessed for more than N days (purge is at 60)")
@click.option("--limit", default=40)
def purge_check(days: int, limit: int) -> None:
    """List scratch files close to the 60-day inactivity purge so you can pull or archive them."""
    cfg = get_config()
    require_user(cfg)
    scratch = cfg["paths"]["scratch"]
    cmd = (f"find {shlex.quote(scratch)} -xdev -type f -atime +{days} -not -path '*/envs/*' -not -path '*/.cache/*' "
           f"-printf '%A+ %10s %p\\n' 2>/dev/null | sort | head -n {limit}")
    result = run_remote(cfg, cmd)
    out = result.stdout.strip()
    if not out:
        click.echo(f"No files older than {days} days in {scratch}.")
        return
    click.echo(f"Files in {scratch} not accessed for >{days} days (purged at 60):\n")
    click.echo(out)
    click.echo("\nPull them (gilbreth-sync pull) or archive to Fortress/Depot. Do not 'touch' files to dodge the purge; RCAC policy forbids it.")


@cli.command()
def doctor() -> None:
    """Check local config, SSH session, account, scratch, modules and apptainer."""
    cfg = get_config()
    ok = True

    def report(label: str, good: bool, detail: str = ""):
        nonlocal ok
        ok = ok and good
        click.echo(f"  [{'OK' if good else 'FAIL'}] {label}{(': ' + detail) if detail else ''}")

    click.echo("Local configuration")
    report(f"username ({ENV_USER})", bool(cfg["ssh"].get("user")), cfg["ssh"].get("user") or "not set")
    report(f"account ({ENV_ACCOUNT})", bool(cfg["slurm"].get("account")), cfg["slurm"].get("account") or "not set (gilbreth-info accounts)")
    if not cfg["ssh"].get("user"):
        sys.exit(1)

    click.echo("Connection")
    connected = is_connected(cfg)
    report("ControlMaster session", connected, "" if connected else "run gilbreth-connect")
    if not connected:
        sys.exit(1)

    checks = {
        "hostname": "hostname",
        "scratch": f"test -d {cfg['paths']['scratch']} && echo {cfg['paths']['scratch']}",
        "slist": "command -v slist",
        f"module {cfg['environment'].get('conda_module', 'conda')}": f"module load {cfg['environment'].get('conda_module', 'conda')} 2>&1 && command -v conda",
        "module cuda": "module load cuda 2>&1 && echo ${CUDA_HOME:-loaded}",
        "apptainer": "command -v apptainer && apptainer --version",
        "sinteractive": "command -v sinteractive",
    }
    click.echo("Remote")
    for label, cmd in checks.items():
        result = run_remote(cfg, cmd)
        good = result.returncode == 0 and bool(result.stdout.strip())
        detail = (result.stdout.strip() or result.stderr.strip()).splitlines()[-1:] if (result.stdout or result.stderr) else [""]
        report(label, good, detail[0] if detail else "")
    if cfg["slurm"].get("account"):
        result = run_remote(cfg, f"sacctmgr -n show assoc user=$USER account={cfg['slurm']['account']} format=account")
        report(f"account '{cfg['slurm']['account']}' usable", bool(result.stdout.strip()), result.stdout.strip() or "not found in sacctmgr")
    click.echo("\nAll good." if ok else "\nSome checks failed; see above.")
    sys.exit(0 if ok else 1)
