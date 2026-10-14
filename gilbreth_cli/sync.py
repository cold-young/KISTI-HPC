"""Code / result synchronisation between the local machine and Gilbreth scratch."""

from __future__ import annotations

import fnmatch
import shlex
import subprocess
import sys
import time
from pathlib import Path

import click

from .config import get_config, require_user, safe_name
from .remote import rsync, run_remote, ssh_option_string, ssh_target


def exclude_args(cfg: dict) -> list[str]:
    args: list[str] = []
    for pattern in cfg.get("sync", {}).get("exclude", []):
        args.extend(["--exclude", pattern])
    return args


def include_args(cfg: dict) -> list[str]:
    args: list[str] = []
    for pattern in cfg.get("sync", {}).get("pull_include", []):
        args.extend(["--include", pattern])
    args.extend(["--include", "*/", "--exclude", "*"])
    return args


def remote_project_dir(cfg: dict, project: str) -> str:
    return f"{cfg['paths']['projects']}/{project}"


def project_name(local: Path, override: str | None) -> str:
    return safe_name(override or local.resolve().name)


def is_excluded(rel_path: str, patterns: list[str]) -> bool:
    """Apply rsync-style exclude patterns to a path relative to the synced root.

    ``dir/`` matches a directory anywhere in the path, ``*.ext`` matches the basename,
    anything else matches a path component.
    """
    parts = rel_path.replace("\\", "/").split("/")
    for pat in patterns:
        if pat.endswith("/"):
            if pat.rstrip("/") in parts[:-1]:
                return True
        elif any(fnmatch.fnmatch(part, pat) for part in parts):
            return True
    return False


@click.group()
def cli() -> None:
    """Sync code between your machine and Gilbreth (/scratch/gilbreth/<user>/projects)."""


@cli.command()
@click.argument("local_path", default=".", type=click.Path(exists=True))
@click.option("--project", "-p", default=None, help="Remote project name (default: directory name)")
@click.option("--dry-run", is_flag=True, help="Show what would be transferred")
@click.option("--delete", is_flag=True, help="Delete remote files that no longer exist locally")
def push(local_path: str, project: str | None, dry_run: bool, delete: bool) -> None:
    """Push local code to Gilbreth (local -> scratch)."""
    cfg = get_config()
    require_user(cfg)
    local = Path(local_path).resolve()
    project = project_name(local, project)
    remote_dir = remote_project_dir(cfg, project)

    run_remote(cfg, f"mkdir -p {shlex.quote(remote_dir)}")
    extra = exclude_args(cfg)
    if delete:
        extra.append("--delete")
    click.echo(f"Pushing {local}/ -> gilbreth:{remote_dir}/")
    result = rsync(cfg, f"{local}/", f"{ssh_target(cfg)}:{remote_dir}/", extra=extra, dry_run=dry_run)
    if result.returncode != 0:
        click.echo("Push failed.", err=True)
        sys.exit(result.returncode)
    click.echo("Push complete.")


@cli.command()
@click.argument("local_path", default=".", type=click.Path())
@click.option("--project", "-p", default=None, help="Remote project name (default: directory name)")
@click.option("--all-files", is_flag=True, help="Pull everything, not only result patterns")
@click.option("--dry-run", is_flag=True)
def pull(local_path: str, project: str | None, all_files: bool, dry_run: bool) -> None:
    """Pull results from Gilbreth (scratch -> local)."""
    cfg = get_config()
    require_user(cfg)
    local = Path(local_path).resolve()
    local.mkdir(parents=True, exist_ok=True)
    project = project_name(local, project)
    remote_dir = remote_project_dir(cfg, project)

    extra = ["--prune-empty-dirs"] + ([] if all_files else include_args(cfg))
    click.echo(f"Pulling gilbreth:{remote_dir}/ -> {local}/")
    result = rsync(cfg, f"{ssh_target(cfg)}:{remote_dir}/", f"{local}/", extra=extra, dry_run=dry_run)
    if result.returncode != 0:
        click.echo("Pull failed.", err=True)
        sys.exit(result.returncode)
    click.echo("Pull complete.")


@cli.command()
@click.argument("local_path", default=".", type=click.Path(exists=True))
@click.option("--project", "-p", default=None)
@click.option("--interval", "-i", default=2.0, help="Debounce interval in seconds")
def watch(local_path: str, project: str | None, interval: float) -> None:
    """Watch a directory and push every change automatically."""
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

    cfg = get_config()
    require_user(cfg)
    local = Path(local_path).resolve()
    project = project_name(local, project)
    remote_dir = remote_project_dir(cfg, project)
    run_remote(cfg, f"mkdir -p {shlex.quote(remote_dir)}")
    extra = exclude_args(cfg)
    patterns = list(cfg.get("sync", {}).get("exclude", []))

    click.echo(f"Watching {local}/ -> gilbreth:{remote_dir}/ (Ctrl+C to stop)")

    class Handler(FileSystemEventHandler):
        pending = False
        last = 0.0

        def on_any_event(self, event):
            if event.is_directory:
                return
            try:
                rel = str(Path(str(event.src_path)).resolve().relative_to(local))
            except ValueError:
                rel = str(event.src_path)
            if is_excluded(rel, patterns):
                return
            self.pending = True

        def flush(self):
            now = time.time()
            if self.pending and now - self.last > interval:
                self.pending = False
                self.last = now
                cmd = ["rsync", "-az", "--quiet", "-e", ssh_option_string(cfg)] + extra + [f"{local}/", f"{ssh_target(cfg)}:{remote_dir}/"]
                res = subprocess.run(cmd, capture_output=True, text=True, check=False)
                ts = time.strftime("%H:%M:%S")
                click.echo(f"  [{ts}] synced" if res.returncode == 0 else f"  [{ts}] sync failed: {res.stderr.strip()}")

    handler = Handler()
    observer = Observer()
    observer.schedule(handler, str(local), recursive=True)
    observer.start()
    try:
        while True:
            handler.flush()
            time.sleep(0.5)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


@cli.command("push-data")
@click.argument("local_data_path", type=click.Path(exists=True))
@click.option("--dest", default=None, help="Remote directory (default: paths.data)")
@click.option("--dry-run", is_flag=True)
def push_data(local_data_path: str, dest: str | None, dry_run: bool) -> None:
    """Push a dataset to /scratch/gilbreth/<user>/data (large transfers: consider Globus)."""
    cfg = get_config()
    require_user(cfg)
    local = Path(local_data_path).resolve()
    if local.is_dir():
        remote_dir = dest or f"{cfg['paths']['data']}/{local.name}"
        src = f"{local}/"
    else:
        remote_dir = dest or cfg["paths"]["data"]
        src = str(local)
    run_remote(cfg, f"mkdir -p {shlex.quote(remote_dir)}")
    click.echo(f"Pushing {src} -> gilbreth:{remote_dir}/")
    click.echo("Tip: for >100 GB use Globus collection 'Purdue Gilbreth Cluster - Scratch'.")
    result = rsync(cfg, src, f"{ssh_target(cfg)}:{remote_dir}/", dry_run=dry_run)
    sys.exit(result.returncode)
