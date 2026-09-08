"""Python environment management on Gilbreth, via ``conda-env-mod``.

RCAC documents exactly one supported way to build a user Python environment on
Gilbreth, and it is not a hand-rolled conda install on a shared filesystem::

    module purge
    module load anaconda            # + cuda cudnn for a GPU build
    conda-env-mod create -n NAME
    module load use.own
    module load conda-env/NAME-pyX.Y

That sequence is AC-5.1 and it is what :func:`setup` emits; :func:`activate`
emits its last two lines and nothing else (AC-5.1b).  Three properties of this
module follow from RCAC's own guidance rather than from taste:

* **No module name or version is written here.**  The only names spelled are
  ``anaconda``, ``cuda``, ``cudnn``, ``use.own`` and the ``conda-env/`` prefix
  -- the generic Lmod names, never a version.  RCAC's instruction is explicit:
  *do not invent module names or versions*.  :func:`list_modules` asks the
  cluster what exists instead of shipping a table that expires.
* **``pyX.Y`` is never guessed** (AC-5.2).  It comes from ``--py``, else from
  the cluster itself, else the command fails with a message naming ``--py``.
  Composing a module name out of a locally-plausible Python version is how you
  get ``module load`` failing at line one of every job.
* **Nothing here writes to a shell start-up file** (AC-5.11).  RCAC documents
  ``module load`` inside job scripts and interactive sessions, never in a login
  start-up file, because a module loaded at login silently follows the user
  into every job they ever submit.

The previous cluster's ``CONDA_BIN`` path and the ``conda_path`` config key it
read are both gone: they named a conda installation that does not exist on
Gilbreth, and a default pointing at a non-existent binary fails later and less
legibly than having no default at all.
"""

from __future__ import annotations

import re
import shlex
import subprocess
import sys
from pathlib import Path

import click
import yaml

from .config import get_rcac_config, require_user, scratch_root
from .ssh import ssh_opt_argv, ssh_opt_str, ssh_target

#: The generic Lmod names.  No versions: Lmod resolves the site default, which
#: is the only version this code can name without inventing one.
ANACONDA_MODULE = "anaconda"
GPU_MODULES = ("cuda", "cudnn")

#: ``use.own`` puts the user's own generated modules on ``MODULEPATH``.  Every
#: environment ``conda-env-mod`` creates is invisible until this is loaded --
#: which is why it is half of :func:`activate`'s two-line contract.
USE_OWN_MODULE = "use.own"

#: The module namespace ``conda-env-mod`` generates into.  The full name is
#: ``conda-env/<name>-py<X.Y>``; the ``pyX.Y`` half is never composed from a
#: guess (AC-5.2).
CONDA_ENV_NAMESPACE = "conda-env"

#: Ask the cluster which Python the ``anaconda`` module actually provides.
#: This is the "else discovery" branch of AC-5.2 -- the answer comes from the
#: interpreter that will build the environment, not from this machine.
_PY_VERSION_PROBE = (
    f"module purge && module load {ANACONDA_MODULE} && "
    "python -c 'import sys; print(\"%d.%d\" % sys.version_info[:2])'"
)

_PY_VERSION_RE = re.compile(r"^\d+\.\d+(?:\.\d+)*$")


def _ssh_cmd(cfg: dict) -> list[str]:
    """Build base SSH command using ControlMaster."""
    return ["ssh", *ssh_opt_argv(cfg), ssh_target(cfg)]


def _run_remote(cfg: dict, command: str) -> subprocess.CompletedProcess:
    """Execute a command on Gilbreth via SSH.

    The command is wrapped in a login shell because ``module`` is an Lmod shell
    function: a bare ``ssh host 'module load anaconda'`` runs in a shell where
    that function was never defined and fails with "command not found", which
    reads like the module is missing when it is not.
    """
    cmd = _ssh_cmd(cfg) + ["bash -l -c " + shlex.quote(command)]
    return subprocess.run(cmd, capture_output=True, text=True)


def conda_env_module(name: str, py_version: str) -> str:
    """The module name ``conda-env-mod`` generates for *name* under *py_version*."""
    return f"{CONDA_ENV_NAMESPACE}/{name}-py{py_version}"


def setup_commands(name: str, py_version: str, with_gpu: bool = False) -> list[str]:
    """AC-5.1 -- the documented creation sequence, in order.

    Returned as a list rather than echoed in place so the ordering is a value a
    test can assert on, instead of a property of six ``click.echo`` calls that
    someone can reorder without noticing.
    """
    commands = [
        "module purge",
        f"module load {ANACONDA_MODULE}",
    ]
    if with_gpu:
        commands.append("module load " + " ".join(GPU_MODULES))
    commands.append(f"conda-env-mod create -n {name}")
    commands.extend(activate_commands(name, py_version))
    return commands


def activate_commands(name: str, py_version: str) -> list[str]:
    """AC-5.1b -- exactly the two lines ``rcac-env activate`` prints."""
    return [
        f"module load {USE_OWN_MODULE}",
        f"module load {conda_env_module(name, py_version)}",
    ]


def _resolve_py_version(cfg: dict, py_version: str | None) -> str:
    """AC-5.2 -- ``--py``, else discovery, else an actionable failure.

    There is deliberately no fallback to this machine's own Python: the module
    name has to match what ``conda-env-mod`` generated on Gilbreth, and a
    locally-plausible version is exactly the kind of guess that produces a
    ``module load`` failure the user cannot debug from the error text.
    """
    if py_version:
        return str(py_version).strip()

    result = _run_remote(cfg, _PY_VERSION_PROBE)
    for line in reversed((result.stdout or "").strip().splitlines()):
        candidate = line.strip()
        if _PY_VERSION_RE.match(candidate):
            return ".".join(candidate.split(".")[:2])

    detail = (result.stderr or result.stdout or "").strip().splitlines()
    raise click.ClickException(
        "could not determine which Python the `anaconda` module provides, so "
        f"the `{CONDA_ENV_NAMESPACE}/<name>-pyX.Y` module name cannot be "
        "composed. Pass it explicitly with `--py X.Y` -- the value is the one "
        "printed by `module avail conda-env` on Gilbreth after "
        "`module load use.own`."
        + (f"\nremote said: {detail[-1]}" if detail else "")
    )


@click.group()
def cli() -> None:
    """Manage Python environments on Gilbreth (conda-env-mod)."""


@cli.command()
@click.option("--from", "from_file", default="environment.yml", type=click.Path(),
              help="Conda environment file (default: environment.yml)")
@click.option("--name", "-n", default=None, help="Environment name (default: from YAML)")
@click.option("--py", "py_version", default=None,
              help="Python X.Y for the generated conda-env/<name>-pyX.Y module. "
                   "Discovered from the cluster's `anaconda` module when omitted; "
                   "never guessed.")
@click.option("--gpu", "with_gpu", is_flag=True,
              help="Also `module load cuda cudnn` before creating the environment.")
def setup(from_file: str, name: str, py_version: str, with_gpu: bool) -> None:
    """Create a Gilbreth environment with conda-env-mod (AC-5.1).

    Prints the exact module sequence it is about to run, then runs it on the
    cluster.  If ``environment.yml`` is present its packages are installed into
    the new environment once the environment's own module is loaded.
    """
    cfg = get_rcac_config()

    require_user(cfg)

    local_file = Path(from_file)
    env_name_from_yaml = ""
    if local_file.exists():
        with open(local_file) as _f:
            env_name_from_yaml = (yaml.safe_load(_f) or {}).get("name", "")
    elif not name:
        click.echo(
            f"Error: {from_file} not found and no --name given, so the "
            "environment has no name. Pass `--name <env>`.",
            err=True,
        )
        sys.exit(1)

    effective_name = name or env_name_from_yaml
    if not effective_name:
        click.echo(
            f"Error: {from_file} declares no `name:` and no --name was given.",
            err=True,
        )
        sys.exit(1)

    py = _resolve_py_version(cfg, py_version)

    ssh = cfg["ssh"]
    user = ssh["user"]
    host = ssh["login_host"]
    scratch = scratch_root(cfg)
    remote_tmp = f"{scratch}/.rcac-tools/environment.yml"

    commands = setup_commands(effective_name, py, with_gpu=with_gpu)

    if local_file.exists():
        click.echo("Uploading environment file...")
        _run_remote(cfg, f"mkdir -p {scratch}/.rcac-tools")
        rsync_cmd = [
            "rsync", "-az",
            "-e", ssh_opt_str(cfg),
            str(local_file), f"{user}@{host}:{remote_tmp}",
        ]
        result = subprocess.run(rsync_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            click.echo(f"Error uploading file: {result.stderr}", err=True)
            sys.exit(1)
        commands.append(
            f"conda env update --name {shlex.quote(effective_name)} "
            f"--file {remote_tmp} --prune"
        )

    click.echo(f"Environment: {effective_name} (python {py})")
    click.echo("Running on Gilbreth:")
    for command in commands:
        click.echo(command)

    result = _run_remote(cfg, " && ".join(commands))
    click.echo(result.stdout)
    if result.returncode != 0:
        click.echo(f"Error: {result.stderr}", err=True)
        sys.exit(1)
    click.echo("Environment setup complete.")
    click.echo(f"Load it in a job script with `rcac-env activate {effective_name}`.")


@cli.command()
@click.argument("env_name")
@click.option("--py", "py_version", default=None,
              help="Python X.Y of the conda-env/<name>-pyX.Y module. Discovered "
                   "from the cluster when omitted; never guessed.")
def activate(env_name: str, py_version: str) -> None:
    """Print the two module loads that activate ENV_NAME in a job script.

    AC-5.1b fixes the contract: stdout is exactly ``module load use.own``
    followed by ``module load conda-env/<name>-pyX.Y``, and nothing else, so
    the output can be pasted straight into an sbatch script or consumed by
    ``$(rcac-env activate NAME)``.  Everything explanatory -- and every failure
    -- goes to stderr.
    """
    if py_version:
        py = str(py_version).strip()
    else:
        cfg = get_rcac_config()
        require_user(cfg)
        py = _resolve_py_version(cfg, None)

    for command in activate_commands(env_name, py):
        click.echo(command)


@cli.command()
@click.argument("env_name")
@click.option("--output", "-o", default="environment.yml", help="Output file path")
def export(env_name: str, output: str) -> None:
    """Export a Gilbreth environment to a local YAML file."""
    cfg = get_rcac_config()

    require_user(cfg)

    remote_cmd = (
        f"module purge && module load {ANACONDA_MODULE} && "
        f"conda env export --name {shlex.quote(env_name)} --no-builds"
    )
    result = _run_remote(cfg, remote_cmd)

    if result.returncode != 0:
        click.echo(f"Error: {result.stderr}", err=True)
        sys.exit(1)

    Path(output).write_text(result.stdout)
    click.echo(f"Environment exported to {output}")


@cli.command("list")
def list_envs() -> None:
    """List the environments visible to your account on Gilbreth."""
    cfg = get_rcac_config()

    require_user(cfg)

    result = _run_remote(
        cfg, f"module purge && module load {ANACONDA_MODULE} && conda env list"
    )

    if result.returncode == 0:
        click.echo(result.stdout)
    else:
        click.echo(f"Error: {result.stderr}", err=True)

    # The environments `conda-env-mod` generated are modules, not conda envs,
    # and are invisible until `use.own` is on MODULEPATH.
    click.echo(f"\nGenerated environment modules (after `module load {USE_OWN_MODULE}`):")
    mod_result = _run_remote(
        cfg,
        f"module load {USE_OWN_MODULE} && module avail {CONDA_ENV_NAMESPACE} 2>&1",
    )
    output = mod_result.stdout or mod_result.stderr
    for line in output.split("\n"):
        if line.strip():
            click.echo(f"  {line.strip()}")


#: What :func:`list_modules` asks the cluster about when given no argument.
#: These are Lmod *queries*, not a table of what is installed: every name here
#: is a generic module family RCAC documents for Gilbreth, and the versions --
#: the part that goes stale -- come back from `module avail` (AC-5.8).
MODULE_QUERIES = (
    f"module avail {ANACONDA_MODULE}",
    *(f"module avail {m}" for m in GPU_MODULES),
    f"module load {USE_OWN_MODULE} && module avail {CONDA_ENV_NAMESPACE}",
    "module avail ngc",
)


@cli.command("modules")
@click.argument("pattern", required=False)
def list_modules(pattern: str) -> None:
    """Ask Gilbreth which modules exist (AC-5.8).

    With PATTERN, runs ``module spider PATTERN``.  Without one, runs
    ``module avail`` for the families a Python/GPU job needs.  Nothing is
    printed from a local table: RCAC's guidance is not to invent module names
    or versions, and the only defence against a stale table is not to keep one.
    """
    cfg = get_rcac_config()

    require_user(cfg)

    if pattern:
        queries = [f"module spider {shlex.quote(pattern)}"]
    else:
        queries = list(MODULE_QUERIES)

    for query in queries:
        click.echo(f"$ {query}")
        result = _run_remote(cfg, f"{query} 2>&1")
        output = result.stdout or result.stderr
        for line in output.split("\n"):
            if line.strip():
                click.echo(f"  {line.strip()}")
        click.echo()

    click.echo("Usage in experiment.yaml -- copy names verbatim from above:")
    click.echo("  environment:")
    click.echo("    modules:")
    click.echo("      - <a name `module avail` actually printed>")
