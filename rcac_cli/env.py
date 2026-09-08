"""Conda environment management on RCAC HPC."""

import shlex
import subprocess
import sys
from pathlib import Path

import click
import yaml

from .config import get_defaults, get_rcac_config

CONDA_BIN = "/apps/applications/Miniconda/25.11.1/bin/conda"


def _ssh_cmd(cfg: dict) -> list[str]:
    """Build base SSH command using ControlMaster."""
    ssh = cfg["ssh"]
    return [
        "ssh",
        "-o", f"ControlPath={ssh['control_path']}",
        "-p", str(ssh["port"]),
        f"{ssh['user']}@{ssh['login_host']}",
    ]


def _run_remote(cfg: dict, command: str) -> subprocess.CompletedProcess:
    """Execute a command on RCAC via SSH."""
    cmd = _ssh_cmd(cfg) + [command]
    return subprocess.run(cmd, capture_output=True, text=True)


def _get_conda_path(defaults: dict) -> str:
    """Get conda binary path from config."""
    return defaults.get("environment", {}).get("conda_path", CONDA_BIN)


@click.group()
def cli() -> None:
    """Manage Conda environments on RCAC HPC."""


@cli.command()
@click.option("--from", "from_file", default="environment.yml", type=click.Path(),
              help="Conda environment file (default: environment.yml)")
@click.option("--name", "-n", default=None, help="Environment name (default: from YAML)")
def setup(from_file: str, name: str) -> None:
    """Create or update a Conda environment on RCAC from environment.yml."""
    cfg = get_rcac_config()
    defaults = get_defaults()

    if not cfg["ssh"]["user"]:
        click.echo("Error: RCAC_USER not set.")
        sys.exit(1)

    local_file = Path(from_file)
    if not local_file.exists():
        click.echo(f"Error: {from_file} not found.")
        sys.exit(1)

    ssh = cfg["ssh"]
    user = ssh["user"]
    host = ssh["login_host"]
    remote_tmp = f"/scratch/{user}/.rcac-tools/environment.yml"
    conda = _get_conda_path(defaults)

    # Upload environment.yml to RCAC
    click.echo("Uploading environment file...")
    _run_remote(cfg, f"mkdir -p /scratch/{user}/.rcac-tools")
    ssh_opt = f"ssh -o ControlPath={ssh['control_path']} -p {ssh['port']}"
    rsync_cmd = ["rsync", "-az", "-e", ssh_opt, str(local_file), f"{user}@{host}:{remote_tmp}"]
    result = subprocess.run(rsync_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        click.echo(f"Error uploading file: {result.stderr}", err=True)
        sys.exit(1)

    # Parse env name from local YAML
    with open(local_file) as _f:
        env_name_from_yaml = yaml.safe_load(_f).get("name", "")

    effective_name = name or env_name_from_yaml
    name_flag = f"--name {shlex.quote(effective_name)}" if effective_name else ""

    remote_cmd = (
        f"if {conda} env list | grep -q {shlex.quote(effective_name)}; then "
        f"echo 'Updating existing environment...'; "
        f"{conda} env update {name_flag} --file {remote_tmp} --prune; "
        f"else "
        f"echo 'Creating new environment...'; "
        f"{conda} env create {name_flag} --file {remote_tmp}; "
        f"fi"
    )

    click.echo("Setting up Conda environment on RCAC...")
    result = _run_remote(cfg, remote_cmd)
    click.echo(result.stdout)
    if result.returncode != 0:
        click.echo(f"Error: {result.stderr}", err=True)
        sys.exit(1)
    click.echo("Environment setup complete.")


@cli.command()
@click.argument("env_name")
def activate(env_name: str) -> None:
    """Print the activation commands for a RCAC environment.

    For pre-built environments (pytorch, tensorflow), use module load.
    For user environments, use conda run.
    """
    defaults = get_defaults()
    modules = defaults.get("environment", {}).get("default_modules", [])

    # Check if env_name matches a known conda module
    module_envs = {
        "pytorch": "conda/pytorch_2.9.1_cuda12",
        "pytorch_cuda12": "conda/pytorch_2.9.1_cuda12",
        "pytorch_cuda13": "conda/pytorch_2.9.1_cuda13",
        "tensorflow": "conda/tensorflow_2.20.0",
    }

    if env_name in module_envs:
        click.echo("# Pre-built environment (use in SLURM scripts):")
        click.echo(f"module load {module_envs[env_name]}")
    else:
        click.echo("# User conda environment (use in SLURM scripts):")
        for m in modules:
            click.echo(f"module load {m}")
        click.echo(f"# Then run commands with:")
        click.echo(f"{CONDA_BIN} run -n {env_name} python your_script.py")
        click.echo()
        click.echo("# Or set PATH directly:")
        click.echo(f"export PATH=$HOME/.conda/envs/{env_name}/bin:$PATH")


@cli.command()
@click.argument("env_name")
@click.option("--output", "-o", default="environment.yml", help="Output file path")
def export(env_name: str, output: str) -> None:
    """Export a Conda environment from RCAC to a local YAML file."""
    cfg = get_rcac_config()
    defaults = get_defaults()

    if not cfg["ssh"]["user"]:
        click.echo("Error: RCAC_USER not set.")
        sys.exit(1)

    conda = _get_conda_path(defaults)
    remote_cmd = f"{conda} env export --name {shlex.quote(env_name)} --no-builds"
    result = _run_remote(cfg, remote_cmd)

    if result.returncode != 0:
        click.echo(f"Error: {result.stderr}", err=True)
        sys.exit(1)

    Path(output).write_text(result.stdout)
    click.echo(f"Environment exported to {output}")


@cli.command("list")
def list_envs() -> None:
    """List Conda environments on RCAC."""
    cfg = get_rcac_config()
    defaults = get_defaults()

    if not cfg["ssh"]["user"]:
        click.echo("Error: RCAC_USER not set.")
        sys.exit(1)

    conda = _get_conda_path(defaults)
    result = _run_remote(cfg, f"{conda} env list")

    if result.returncode == 0:
        click.echo(result.stdout)
    else:
        click.echo(f"Error: {result.stderr}", err=True)

    # Also show available module-based environments
    click.echo("\nPre-built environments (use via module load):")
    mod_result = _run_remote(cfg, "module avail conda 2>&1")
    if mod_result.stdout:
        click.echo(mod_result.stdout)
    elif mod_result.stderr:
        # module avail outputs to stderr
        for line in mod_result.stderr.split("\n"):
            if "conda/" in line or "---" in line:
                click.echo(f"  {line.strip()}")


@cli.command("modules")
def list_modules() -> None:
    """Show available pre-built conda module environments on RCAC."""
    cfg = get_rcac_config()

    if not cfg["ssh"]["user"]:
        click.echo("Error: RCAC_USER not set.")
        sys.exit(1)

    click.echo("Available pre-built environments (module load):\n")
    result = _run_remote(cfg, "module avail conda 2>&1")
    output = result.stdout or result.stderr
    for line in output.split("\n"):
        if line.strip():
            click.echo(f"  {line.strip()}")

    click.echo("\nUsage in experiment.yaml:")
    click.echo("  environment:")
    click.echo("    modules:")
    click.echo("      - conda/pytorch_2.9.1_cuda12")
