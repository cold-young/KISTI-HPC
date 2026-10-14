"""Python environment management on Gilbreth (conda in scratch, conda-env-mod, venv)."""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

import click
import yaml

from .config import get_config, require_user
from .remote import run_remote, upload


def _conda_prelude(cfg: dict) -> str:
    """Load the conda module and keep package / pip caches on scratch (home is only 25 GB)."""
    module = cfg.get("environment", {}).get("conda_module", "conda")
    scratch = cfg["paths"]["scratch"]
    # NB: the conda module exports CONDA_PKGS_DIRS/CONDA_ENVS_PATH pointing at $HOME, so override *after* loading it.
    return (f"module load {module} && "
            f"export CONDA_PKGS_DIRS={scratch}/.conda/pkgs PIP_CACHE_DIR={scratch}/.cache/pip && "
            f"mkdir -p {scratch}/.conda/pkgs {scratch}/.cache/pip && "
            f"eval \"$(conda shell.bash hook)\"")


def _env_prefix(cfg: dict, name: str) -> str:
    return name if name.startswith("/") else f"{cfg['paths']['envs']}/{name}"


@click.group()
def cli() -> None:
    """Create and inspect Python environments on Gilbreth."""


@cli.command()
@click.option("--from", "from_file", default="environment.yml", type=click.Path(), help="Conda environment.yml")
@click.option("--name", "-n", default=None, help="Env name (default: name: in the YAML)")
@click.option("--requirements", "-r", default=None, type=click.Path(), help="Also pip install -r this file")
@click.option("--python", "python_version", default=None, help="Create a bare env with this Python instead of a YAML")
def setup(from_file: str, name: str | None, requirements: str | None, python_version: str | None) -> None:
    """Create or update a conda env under /scratch/gilbreth/<user>/envs (keeps the 25 GB home quota free).

    Note: compute nodes have internet access on Gilbreth, but creating envs on the
    login node is fine and faster. Large envs can take 10-20 minutes.
    """
    cfg = get_config()
    require_user(cfg)
    envs_dir = cfg["paths"]["envs"]
    tools_dir = f"{cfg['paths']['scratch']}/.gilbreth-cli"
    run_remote(cfg, f"mkdir -p {envs_dir} {tools_dir}")

    if python_version:
        if not name:
            click.echo("--name is required with --python", err=True)
            sys.exit(1)
        prefix = _env_prefix(cfg, name)
        q = shlex.quote(prefix)
        create = (f"if [ -d {q} ]; then echo Updating {q}; conda install -y --prefix {q} python={shlex.quote(python_version)}; "
                  f"else echo Creating {q}; conda create -y --prefix {q} python={shlex.quote(python_version)}; fi")
    else:
        local = Path(from_file)
        if not local.exists():
            click.echo(f"{from_file} not found. Use --python 3.11 --name myenv for a bare env.", err=True)
            sys.exit(1)
        with open(local) as f:
            spec = yaml.safe_load(f) or {}
        name = name or spec.get("name")
        if not name:
            click.echo("Environment name missing: pass --name or add 'name:' to the YAML", err=True)
            sys.exit(1)
        prefix = _env_prefix(cfg, name)
        remote_yaml = f"{tools_dir}/{local.name}"
        if not upload(cfg, local, remote_yaml):
            click.echo("Upload failed.", err=True)
            sys.exit(1)
        q = shlex.quote(prefix)
        create = (f"if [ -d {q} ]; then echo Updating {q}; conda env update --prefix {q} --file {shlex.quote(remote_yaml)} --prune; "
                  f"else echo Creating {q}; conda env create --prefix {q} --file {shlex.quote(remote_yaml)}; fi")

    cmd = f"{_conda_prelude(cfg)} && {create}"
    if requirements:
        remote_req = f"{tools_dir}/{Path(requirements).name}"
        if not upload(cfg, requirements, remote_req):
            click.echo("Upload failed.", err=True)
            sys.exit(1)
        cmd += f" && conda activate {shlex.quote(prefix)} && pip install -r {shlex.quote(remote_req)}"

    click.echo(f"Setting up {prefix} on Gilbreth (this can take a while) ...")
    result = run_remote(cfg, cmd)
    click.echo(result.stdout)
    if result.returncode != 0:
        click.echo(result.stderr, err=True)
        sys.exit(1)
    click.echo(f"Done. In experiment.yaml use:\n  environment:\n    type: conda\n    name: {name}")


@cli.command("list")
def list_envs() -> None:
    """List conda envs in scratch and in $HOME."""
    cfg = get_config()
    require_user(cfg)
    envs_dir = cfg["paths"]["envs"]
    result = run_remote(cfg, f"{_conda_prelude(cfg)} && echo '# scratch envs ({envs_dir})' && ls -1 {shlex.quote(envs_dir)} 2>/dev/null; echo; conda env list")
    click.echo(result.stdout or result.stderr, err=result.returncode != 0)
    if result.returncode != 0:
        sys.exit(1)


@cli.command()
@click.argument("name")
def activate(name: str) -> None:
    """Print the shell lines needed to activate an env in a job or interactive shell."""
    cfg = get_config()
    require_user(cfg)
    module = cfg.get("environment", {}).get("conda_module", "conda")
    click.echo(f"module load {module}")
    click.echo('eval "$(conda shell.bash hook)"')
    click.echo(f"conda activate {_env_prefix(cfg, name)}")


@cli.command()
@click.argument("name")
@click.option("--yes", "-y", is_flag=True)
def remove(name: str, yes: bool) -> None:
    """Delete a scratch conda env."""
    cfg = get_config()
    require_user(cfg)
    prefix = _env_prefix(cfg, name)
    if not yes:
        click.confirm(f"Delete {prefix} on Gilbreth?", abort=True)
    result = run_remote(cfg, f"{_conda_prelude(cfg)} && conda env remove -y --prefix {shlex.quote(prefix)}")
    click.echo(result.stdout if result.returncode == 0 else result.stderr, err=result.returncode != 0)


@cli.command()
@click.argument("query", default="")
def modules(query: str) -> None:
    """Search available modules (module spider), e.g. gilbreth-env modules cuda."""
    cfg = get_config()
    require_user(cfg)
    cmd = f"module -t avail {shlex.quote(query)} 2>&1" if not query else f"module spider {shlex.quote(query)} 2>&1"
    result = run_remote(cfg, cmd)
    click.echo(result.stdout or result.stderr)


@cli.command()
@click.argument("name")
@click.option("--output", "-o", default="environment.yml")
def export(name: str, output: str) -> None:
    """Export a scratch conda env to a local environment.yml."""
    cfg = get_config()
    require_user(cfg)
    result = run_remote(cfg, f"{_conda_prelude(cfg)} && conda env export --prefix {_env_prefix(cfg, name)} --no-builds")
    if result.returncode != 0:
        click.echo(result.stderr, err=True)
        sys.exit(1)
    text = "\n".join(line for line in result.stdout.splitlines() if not line.startswith("prefix:"))
    Path(output).write_text(f"name: {Path(name).name}\n" + text.split("\n", 1)[1] if text.startswith("name:") else text)
    click.echo(f"Exported to {output}")
