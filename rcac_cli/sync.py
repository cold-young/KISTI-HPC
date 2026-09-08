"""Code synchronization between local and RCAC HPC via rsync."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import click

from .config import get_defaults, get_rcac_config, require_user, resolve_path
from .ssh import globus_guidance, ssh_opt_str, transfer_host


def _build_exclude_args(defaults: dict) -> list[str]:
    """Build rsync --exclude arguments from config."""
    args = []
    for pattern in defaults.get("sync", {}).get("exclude", []):
        args.extend(["--exclude", pattern])
    return args


def _build_include_args(defaults: dict) -> list[str]:
    """Build rsync --include arguments for pull (results only)."""
    args = []
    for pattern in defaults.get("sync", {}).get("pull_include", []):
        args.extend(["--include", pattern])
    return args


def _tree_size(path: Path) -> int:
    """Total bytes of a file or of every regular file under a directory.

    Used only to decide whether to print the Globus guidance (AC-2.5), so a
    file that vanishes mid-walk is skipped rather than raising.
    """
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        try:
            if child.is_file() and not child.is_symlink():
                total += child.stat().st_size
        except OSError:
            continue
    return total


def _get_remote_path(cfg: dict, project: str) -> str:
    """Build remote project path."""
    user = cfg["ssh"]["user"]
    host = cfg["ssh"]["login_host"]
    project_dir = resolve_path(cfg, "projects") + f"/{project}/"
    return f"{user}@{host}:{project_dir}"


def _detect_project_name(local_path: Path) -> str:
    """Detect project name from local directory."""
    return local_path.resolve().name


def _build_ssh_option(cfg: dict) -> str:
    """Build SSH command string for rsync -e option."""
    return ssh_opt_str(cfg)


def _run_rsync(args: list[str], cfg: dict | None = None, dry_run: bool = False) -> int:
    """Execute rsync with given arguments."""
    cmd = ["rsync", "-avz", "--progress"]
    if cfg:
        cmd.extend(["-e", _build_ssh_option(cfg)])
    if dry_run:
        cmd.append("--dry-run")
    cmd.extend(args)
    result = subprocess.run(cmd)
    return result.returncode


@click.group()
def cli() -> None:
    """Sync code between local machine and RCAC HPC."""


@cli.command()
@click.argument("local_path", default=".", type=click.Path(exists=True))
@click.option("--project", "-p", default=None, help="Remote project name (default: directory name)")
@click.option("--dry-run", is_flag=True, help="Show what would be transferred")
@click.option("--delete", is_flag=True, help="Delete remote files not present locally")
def push(local_path: str, project: str | None, dry_run: bool, delete: bool) -> None:
    """Push local code to RCAC (local -> remote)."""
    cfg = get_rcac_config()
    defaults = get_defaults()

    require_user(cfg)

    local = Path(local_path).resolve()
    project = project or _detect_project_name(local)
    remote = _get_remote_path(cfg, project)

    click.echo(f"Pushing {local}/ -> {remote}")

    args = _build_exclude_args(defaults)
    if delete:
        args.append("--delete")
    args.extend([f"{local}/", remote])

    rc = _run_rsync(args, cfg=cfg, dry_run=dry_run)
    if rc == 0:
        click.echo("Push complete.")
    else:
        click.echo("Push failed.", err=True)
        sys.exit(rc)


@cli.command()
@click.argument("local_path", default=".", type=click.Path(exists=True))
@click.option("--project", "-p", default=None, help="Remote project name")
@click.option("--all-files", is_flag=True, help="Pull all files (not just results)")
@click.option("--dry-run", is_flag=True, help="Show what would be transferred")
def pull(local_path: str, project: str | None, all_files: bool, dry_run: bool) -> None:
    """Pull results from RCAC (remote -> local)."""
    cfg = get_rcac_config()
    defaults = get_defaults()

    require_user(cfg)

    local = Path(local_path).resolve()
    project = project or _detect_project_name(local)
    remote = _get_remote_path(cfg, project)

    click.echo(f"Pulling {remote} -> {local}/")

    args = []
    if not all_files:
        args = _build_include_args(defaults)
        args.extend(["--include", "*/"])
        args.extend(["--exclude", "*"])
    args.extend([remote, f"{local}/"])

    rc = _run_rsync(args, cfg=cfg, dry_run=dry_run)
    if rc == 0:
        click.echo("Pull complete.")
    else:
        click.echo("Pull failed.", err=True)
        sys.exit(rc)


@cli.command()
@click.argument("local_path", default=".", type=click.Path(exists=True))
@click.option("--project", "-p", default=None, help="Remote project name")
@click.option("--interval", "-i", default=2.0, help="Check interval in seconds")
def watch(local_path: str, project: str | None, interval: float) -> None:
    """Watch local directory and auto-push changes to RCAC."""
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

    cfg = get_rcac_config()
    defaults = get_defaults()

    require_user(cfg)

    local = Path(local_path).resolve()
    project = project or _detect_project_name(local)
    remote = _get_remote_path(cfg, project)
    exclude_args = _build_exclude_args(defaults)

    click.echo(f"Watching {local}/ for changes...")
    click.echo(f"Target: {remote}")
    click.echo("Press Ctrl+C to stop.\n")

    class SyncHandler(FileSystemEventHandler):
        def __init__(self):
            self._pending = False
            self._last_sync = 0.0

        def on_any_event(self, event):
            if event.is_directory:
                return
            src = event.src_path
            # Skip excluded patterns (derived from defaults.yaml sync.exclude)
            exclude_patterns = defaults.get("sync", {}).get("exclude", [])
            skip = [p.strip("/").strip("*").strip(".") for p in exclude_patterns if p.strip("/").strip("*").strip(".")]
            if any(s in src for s in skip):
                return
            self._pending = True

        def sync_if_pending(self):
            now = time.time()
            if self._pending and (now - self._last_sync) > interval:
                self._pending = False
                self._last_sync = now
                args = exclude_args + [f"{local}/", remote]
                ssh_opt = _build_ssh_option(cfg)
                cmd = ["rsync", "-avz", "--quiet", "-e", ssh_opt] + args
                result = subprocess.run(cmd, capture_output=True, text=True)
                ts = time.strftime("%H:%M:%S")
                if result.returncode == 0:
                    click.echo(f"  [{ts}] Synced.")
                else:
                    click.echo(f"  [{ts}] Sync failed: {result.stderr}")

    handler = SyncHandler()
    observer = Observer()
    observer.schedule(handler, str(local), recursive=True)
    observer.start()

    try:
        while True:
            handler.sync_if_pending()
            time.sleep(0.5)
    except KeyboardInterrupt:
        click.echo("\nStopping watch.")
        observer.stop()
    observer.join()


@cli.command("push-data")
@click.argument("local_data_path", type=click.Path(exists=True))
@click.option("--dry-run", is_flag=True)
def push_data(local_data_path: str, dry_run: bool) -> None:
    """Push large data to RCAC (use Globus for bulk transfers)."""
    # The old --via flag chose between a login node and a data-transfer node.
    # Gilbreth documents no transfer node, so the flag went with the config key
    # that fed it: there is exactly one direct route, and Globus for anything
    # bulky (AC-2.5). Leaving the flag as a one-choice no-op would have implied
    # a choice the cluster does not offer.
    cfg = get_rcac_config()
    user = require_user(cfg)

    host = transfer_host(cfg)
    data_dir = resolve_path(cfg, "data")
    remote = f"{user}@{host}:{data_dir}/"

    local = Path(local_data_path).resolve()
    click.echo(f"Pushing data {local}/ -> {remote}")

    guidance = globus_guidance(_tree_size(local))
    if guidance:
        click.echo(guidance)

    args = [f"{local}/", remote]
    rc = _run_rsync(args, cfg=cfg, dry_run=dry_run)
    if rc == 0:
        click.echo("Data push complete.")
    else:
        click.echo("Data push failed.", err=True)
        sys.exit(rc)
