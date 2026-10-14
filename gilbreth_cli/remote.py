"""Shared SSH / rsync helpers. Every remote call in gilbreth-cli goes through here
so tests can stub a single seam and so the ControlMaster socket is reused."""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path


def socket_path(cfg: dict) -> Path:
    ssh = cfg["ssh"]
    resolved = (
        ssh["control_path"]
        .replace("%r", ssh["user"])
        .replace("%h", ssh["login_host"])
        .replace("%p", str(ssh["port"]))
    )
    return Path(os.path.expanduser(resolved))


def ssh_target(cfg: dict) -> str:
    return f"{cfg['ssh']['user']}@{cfg['ssh']['login_host']}"


def ssh_base(cfg: dict, tty: bool = False) -> list[str]:
    ssh = cfg["ssh"]
    cmd = [
        "ssh",
        "-o", f"ControlPath={ssh['control_path']}",
        "-o", "ControlMaster=auto",
        "-o", f"ControlPersist={ssh.get('control_persist', '12h')}",
        "-p", str(ssh["port"]),
    ]
    if tty:
        cmd.append("-t")
    cmd.append(ssh_target(cfg))
    return cmd


def ssh_option_string(cfg: dict) -> str:
    """The ``-e`` argument for rsync."""
    ssh = cfg["ssh"]
    return (f"ssh -o ControlPath={ssh['control_path']} -o ControlMaster=auto "
            f"-o ControlPersist={ssh.get('control_persist', '12h')} -p {ssh['port']}")


def run_remote(cfg: dict, command: str, check: bool = False) -> subprocess.CompletedProcess:
    """Run a shell command on the Gilbreth login node and capture its output.

    The command is wrapped in a login shell so ``module`` and SLURM commands
    (``slist``, ``sinteractive`` ...) resolve exactly as in an interactive session.
    """
    wrapped = f"bash -lc {shlex.quote(command)}"
    result = subprocess.run(ssh_base(cfg) + [wrapped], capture_output=True, text=True, check=False)
    if check and result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"remote command failed: {command}")
    return result


def run_remote_interactive(cfg: dict, command: str | None = None) -> int:
    """Run a remote command attached to the local terminal (for shells, tail -f ...)."""
    cmd = ssh_base(cfg, tty=True)
    if command:
        cmd.append(f"bash -lc {shlex.quote(command)}")
    return subprocess.run(cmd, check=False).returncode


def is_connected(cfg: dict) -> bool:
    sock = socket_path(cfg)
    if not sock.exists():
        return False
    result = subprocess.run(
        ["ssh", "-O", "check", "-S", str(sock), ssh_target(cfg)],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def rsync(
    cfg: dict,
    src: str,
    dst: str,
    extra: list[str] | None = None,
    quiet: bool = False,
    dry_run: bool = False,
    capture: bool = False,
) -> subprocess.CompletedProcess:
    cmd = ["rsync", "-az"]
    cmd.append("--quiet" if quiet else "-v")
    if not quiet:
        cmd.append("--progress")
    if dry_run:
        cmd.append("--dry-run")
    cmd.extend(["-e", ssh_option_string(cfg)])
    cmd.extend(extra or [])
    cmd.extend([src, dst])
    return subprocess.run(cmd, capture_output=capture, text=True, check=False)


def upload(cfg: dict, local: str | Path, remote: str, quiet: bool = True) -> bool:
    """Copy a local file or directory to ``remote`` (absolute path on Gilbreth)."""
    result = rsync(cfg, str(local), f"{ssh_target(cfg)}:{remote}", quiet=quiet, capture=True)
    return result.returncode == 0


def download(cfg: dict, remote: str, local: str | Path, quiet: bool = False) -> bool:
    result = rsync(cfg, f"{ssh_target(cfg)}:{remote}", str(local), quiet=quiet)
    return result.returncode == 0
