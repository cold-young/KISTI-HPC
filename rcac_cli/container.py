"""Singularity container management for RCAC HPC."""

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path

import click

from .config import get_rcac_config, require_user, scratch_root
from .ssh import globus_guidance, ssh_opt_argv, ssh_opt_str, ssh_target, transfer_host


def _ssh_cmd(cfg: dict) -> list[str]:
    """Build base SSH command using ControlMaster."""
    return ["ssh", *ssh_opt_argv(cfg), ssh_target(cfg)]


def _run_remote(cfg: dict, command: str) -> subprocess.CompletedProcess:
    cmd = _ssh_cmd(cfg) + [command]
    return subprocess.run(cmd, capture_output=True, text=True)


@click.group()
def cli() -> None:
    """Manage Singularity containers on RCAC HPC."""


@cli.command()
@click.argument("def_file", type=click.Path(exists=True))
@click.option("--name", "-n", default=None, help="Output image name (default: <def_file>.sif)")
@click.option("--remote-build", is_flag=True, help="Build on RCAC instead of locally")
def build(def_file: str, name: str | None, remote_build: bool) -> None:
    """Build a Singularity image from a definition file.

    For local builds, Singularity/Apptainer must be installed locally.
    Use --remote-build to build on RCAC (requires interactive session).
    """
    cfg = get_rcac_config()
    def_path = Path(def_file)
    image_name = name or def_path.stem + ".sif"

    if remote_build:
        require_user(cfg)

        user = cfg["ssh"]["user"]
        scratch = scratch_root(cfg)
        remote_def = f"{scratch}/.rcac-tools/{def_path.name}"
        remote_sif = f"{scratch}/.rcac-tools/{image_name}"

        # Upload definition file
        click.echo(f"Uploading {def_path.name} to RCAC...")
        _run_remote(cfg, f"mkdir -p {scratch}/.rcac-tools")
        ssh = cfg["ssh"]
        subprocess.run(
            ["scp", "-P", str(ssh["port"]), str(def_path),
             f"{user}@{ssh['login_host']}:{remote_def}"],
            capture_output=True, text=True,
        )

        click.echo(f"Building {image_name} on RCAC (this may take a while)...")
        click.echo("Note: For large builds, consider using an interactive session:")
        click.echo(f"  srun --partition=amd_a100nv_8 --ntasks=1 --cpus-per-task=4 --time=02:00:00 --comment singularity --pty bash")
        click.echo(f"  singularity build {remote_sif} {remote_def}")

        result = _run_remote(cfg, f"singularity build {remote_sif} {remote_def}")
        click.echo(result.stdout)
        if result.returncode != 0:
            click.echo(f"Build failed: {result.stderr}", err=True)
            sys.exit(1)
        click.echo(f"Image built: {remote_sif}")
    else:
        click.echo(f"Building {image_name} locally...")
        result = subprocess.run(
            ["singularity", "build", image_name, str(def_path)],
            capture_output=False,
        )
        if result.returncode != 0:
            click.echo("Local build failed. Try --remote-build or install Singularity/Apptainer.", err=True)
            sys.exit(1)
        click.echo(f"Image built: {image_name}")


@cli.command()
@click.argument("image", type=click.Path(exists=True))
@click.option("--dest", "-d", default=None,
              help="Remote path (default: <paths.scratch>/.rcac-tools/)")
def push(image: str, dest: str | None) -> None:
    """Upload a Singularity image to RCAC."""
    cfg = get_rcac_config()

    require_user(cfg)

    user = cfg["ssh"]["user"]
    image_path = Path(image)

    # Gilbreth has no data-transfer node, so this goes over the login host.
    host = transfer_host(cfg)
    remote_dest = dest or f"{scratch_root(cfg)}/.rcac-tools/{image_path.name}"

    size_bytes = image_path.stat().st_size
    click.echo(f"Uploading {image_path.name} to RCAC...")
    click.echo(f"  Size: {size_bytes / (1024**3):.1f} GB")
    guidance = globus_guidance(size_bytes)
    if guidance:
        click.echo(guidance)

    # B4/AC-2.9: without -e this rsync opens its own SSH connection and costs
    # the user a second MFA prompt. Route it through the ControlMaster.
    result = subprocess.run(
        ["rsync", "-avz", "--progress",
         "-e", ssh_opt_str(cfg),
         str(image_path), f"{user}@{host}:{remote_dest}"],
    )

    if result.returncode == 0:
        click.echo(f"Uploaded to: {remote_dest}")
    else:
        click.echo("Upload failed.", err=True)
        sys.exit(1)


@cli.command()
@click.argument("image_path")
@click.argument("command", nargs=-1)
@click.option("--bind", "-B", multiple=True,
              help="Bind mount paths (e.g., <paths.scratch>:/data)")
@click.option("--nv", is_flag=True, default=True, help="Enable NVIDIA GPU support (default: on)")
def run(image_path: str, command: tuple[str, ...], bind: tuple[str, ...], nv: bool) -> None:
    """Run a command inside a Singularity container on RCAC.

    IMAGE_PATH is the remote path to the .sif file.
    """
    cfg = get_rcac_config()

    require_user(cfg)

    parts = ["singularity", "exec"]
    if nv:
        parts.append("--nv")
    for b in bind:
        parts.extend(["--bind", b])
    parts.append(image_path)
    parts.extend(command)

    remote_cmd = shlex.join(parts)
    click.echo(f"Running: {remote_cmd}")
    result = _run_remote(cfg, remote_cmd)
    click.echo(result.stdout)
    if result.stderr:
        click.echo(result.stderr, err=True)


@cli.command("list")
def list_images() -> None:
    """List Singularity images on RCAC."""
    cfg = get_rcac_config()

    require_user(cfg)

    result = _run_remote(
        cfg,
        f"find {scratch_root(cfg)} -name '*.sif' -printf '%s %p\\n' "
        "2>/dev/null | sort -rn",
    )

    if result.returncode != 0 or not result.stdout.strip():
        click.echo("No Singularity images found.")
        return

    click.echo(f"{'Size':>10}  Path")
    click.echo("-" * 60)
    for line in result.stdout.strip().split("\n"):
        if line.strip():
            size_bytes, path = line.split(None, 1)
            size_gb = int(size_bytes) / (1024**3)
            click.echo(f"{size_gb:>9.1f}G  {path}")
