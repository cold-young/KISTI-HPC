"""Conda environment management on Neuron HPC."""

import shlex
import subprocess
import sys
from pathlib import Path

import click
import yaml

from .config import get_defaults, get_neuron_config


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
    """Execute a command on Neuron via SSH."""
    cmd = _ssh_cmd(cfg) + [command]
    return subprocess.run(cmd, capture_output=True, text=True)


@click.group()
def cli() -> None:
    """Manage Conda environments on Neuron HPC."""


@cli.command()
@click.option("--from", "from_file", default="environment.yml", type=click.Path(),
              help="Conda environment file (default: environment.yml)")
@click.option("--name", "-n", default=None, help="Environment name (default: from YAML)")
def setup(from_file: str, name: str) -> None:
    """Create or update a Conda environment on Neuron from environment.yml."""
    cfg = get_neuron_config()
    defaults = get_defaults()

    if not cfg["ssh"]["user"]:
        click.echo("Error: NEURON_USER not set.")
        sys.exit(1)

    local_file = Path(from_file)
    if not local_file.exists():
        click.echo(f"Error: {from_file} not found.")
        sys.exit(1)

    # Upload environment.yml to Neuron
    ssh = cfg["ssh"]
    user = ssh["user"]
    host = ssh["login_host"]
    remote_tmp = f"/scratch/{user}/.neuron-tools/environment.yml"

    click.echo("Uploading environment file...")
    _run_remote(cfg, f"mkdir -p /scratch/{user}/.neuron-tools")
    scp_cmd = ["scp", "-P", str(ssh["port"]), str(local_file), f"{user}@{host}:{remote_tmp}"]
    result = subprocess.run(scp_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        click.echo(f"Error uploading file: {result.stderr}", err=True)
        sys.exit(1)

    # Build module load chain
    modules = defaults.get("environment", {}).get("default_modules", [])
    module_cmds = "module purge && " + " && ".join(f"module load {m}" for m in modules) if modules else ""

    # Parse env name from local YAML to avoid fragile shell parsing
    with open(local_file) as _f:
        env_name_from_yaml = yaml.safe_load(_f).get("name", "")

    effective_name = name or env_name_from_yaml
    name_flag = f"--name {shlex.quote(effective_name)}" if effective_name else ""
    remote_cmd = f"""
{module_cmds}
if conda env list | grep -q {shlex.quote(effective_name)}; then
    echo "Updating existing environment..."
    conda env update {name_flag} --file {remote_tmp} --prune
else
    echo "Creating new environment..."
    conda env create {name_flag} --file {remote_tmp}
fi
"""

    click.echo("Setting up Conda environment on Neuron...")
    result = _run_remote(cfg, remote_cmd)
    click.echo(result.stdout)
    if result.returncode != 0:
        click.echo(f"Error: {result.stderr}", err=True)
        sys.exit(1)
    click.echo("Environment setup complete.")


@cli.command()
@click.argument("env_name")
def activate(env_name: str) -> None:
    """Print the activation commands for a Neuron environment."""
    defaults = get_defaults()
    modules = defaults.get("environment", {}).get("default_modules", [])

    lines = ["module purge"]
    lines.extend(f"module load {m}" for m in modules)
    lines.append(f"source activate {env_name}")

    click.echo("# Run the following on Neuron (or add to your SLURM script):")
    for line in lines:
        click.echo(line)


@cli.command()
@click.argument("env_name")
@click.option("--output", "-o", default="environment.yml", help="Output file path")
def export(env_name: str, output: str) -> None:
    """Export a Conda environment from Neuron to a local YAML file."""
    cfg = get_neuron_config()

    if not cfg["ssh"]["user"]:
        click.echo("Error: NEURON_USER not set.")
        sys.exit(1)

    remote_cmd = f"conda env export --name {shlex.quote(env_name)} --no-builds"
    result = _run_remote(cfg, remote_cmd)

    if result.returncode != 0:
        click.echo(f"Error: {result.stderr}", err=True)
        sys.exit(1)

    Path(output).write_text(result.stdout)
    click.echo(f"Environment exported to {output}")


@cli.command("list")
def list_envs() -> None:
    """List Conda environments on Neuron."""
    cfg = get_neuron_config()

    if not cfg["ssh"]["user"]:
        click.echo("Error: NEURON_USER not set.")
        sys.exit(1)

    result = _run_remote(cfg, "conda env list")
    if result.returncode == 0:
        click.echo(result.stdout)
    else:
        click.echo(f"Error: {result.stderr}", err=True)
