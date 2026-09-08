"""Apptainer container management for Gilbreth.

Gilbreth's container runtime is **Apptainer** (1.3.3).  Three facts about it
shape both this module and the four job templates, and each is recorded here
rather than in the templates so the templates stay readable and the provenance
stays in one place (AC-5.6, AC-5.7).

* **No ``module load`` is needed.**  Apptainer is installed system-wide and
  ``apptainer`` is directly callable on the front-ends and the compute nodes.
  RCAC's documentation contains no ``module load apptainer`` anywhere, so a
  generated job script must not emit one either (AC-5.5) -- loading a module
  that does not exist fails the job at line one.
  Source: RCAC's Gilbreth user guide, docs.rcac.purdue.edu.

* **Apptainer auto-binds** ``/etc/resolv.conf``, ``/etc/hosts``,
  ``/home/$USER``, ``/apps``, ``/scratch`` and ``/depot``.  An explicit
  ``--bind`` for any of those is dead weight at best, and at worst shadows a
  path the runtime had already mounted, so neither this module's ``run`` nor
  any template emits one (AC-5.6).  ``-B/--bind`` survives as a flag only for
  a path *outside* that set.
  Source: RCAC's Gilbreth user guide, docs.rcac.purdue.edu.

* **``--nv``** exposes the host NVIDIA driver inside the container, and its
  provenance is worth stating exactly, because it is weaker than the other two:
  ``--nv`` does *not* appear on the Gilbreth guide itself.  It is documented on
  RCAC's own life-sciences applications guide, whose ``apptainer exec --nv``
  example is annotated "Bell/Gilbreth only".  That is the strongest RCAC-owned
  statement available offline, so ``--nv`` is used and defaulted on; plan row
  L3 confirms it on the cluster (AC-5.7).

The previous cluster's runtime name is gone from this module entirely; a
pre-port ``environment.type`` naming it is rejected by ``rcac-exp`` with a
message naming the replacement, using the one allowlisted literal in
:mod:`rcac_cli._legacy` (AC-5.9).
"""

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path

import click

from .config import CredentialError, get_rcac_config, require_account, require_user, scratch_root
from .gilbreth import build_directives
from .ssh import globus_guidance, ssh_opt_argv, ssh_opt_str, ssh_target, transfer_host

#: The allocation a *fallback* interactive build asks for, if the rootless
#: front-end build turns out to need one (plan row L17).  `a10` is the smallest
#: documented card and `standby` is uncharged with a 4 hour cap -- two orders of
#: magnitude more than an image build needs, and billed to nobody.  These are
#: only ever printed, never submitted, and they go through
#: :func:`rcac_cli.gilbreth.build_directives` so the hint cannot drift into a
#: directive set the cluster would reject.
BUILD_FALLBACK_PARTITION = "a10"
BUILD_FALLBACK_QOS = "standby"
BUILD_FALLBACK_TIME = "02:00:00"

#: Printed in place of ``-A`` when no account is resolvable.  A hint is not a
#: submission, so an unset ``RCAC_ACCOUNT`` must not turn a failed build into a
#: credential error (P4).
ACCOUNT_PLACEHOLDER = "YOUR_ACCOUNT"


def _ssh_cmd(cfg: dict) -> list[str]:
    """Build base SSH command using ControlMaster."""
    return ["ssh", *ssh_opt_argv(cfg), ssh_target(cfg)]


def _run_remote(cfg: dict, command: str) -> subprocess.CompletedProcess:
    cmd = _ssh_cmd(cfg) + [command]
    return subprocess.run(cmd, capture_output=True, text=True)


def interactive_build_hint(cfg: dict, remote_sif: str, remote_def: str) -> list[str]:
    """The fallback shown when a rootless front-end build fails.

    RCAC documents Apptainer as rootless, which implies a front-end build needs
    no allocation at all and this hint is dead weight.  *Implies* is not
    *states*, and the plan routes the question to live row L17 rather than
    deleting the path on an assumption -- so the direct build is tried first
    and this is printed only if it fails.

    The srun line is composed from a validated
    :class:`~rcac_cli.gilbreth.SubmissionPlan`, so it carries Gilbreth's
    ``-A``/``-p``/``-q`` triple and ``--gpus-per-node``, and cannot regress to
    the previous cluster's single-partition form or to a ``--comment``
    annotation Gilbreth has no flag for.
    """
    try:
        account = require_account(cfg)
    except CredentialError:
        account = ACCOUNT_PLACEHOLDER

    plan = build_directives(
        account=account,
        partition=BUILD_FALLBACK_PARTITION,
        qos=BUILD_FALLBACK_QOS,
        time=BUILD_FALLBACK_TIME,
        gpus_per_node=1,
    )
    srun = (
        f"srun -A {plan.account} -p {plan.partition} -q {plan.qos} "
        f"--gpus-per-node={plan.gpus_per_node} --mem={plan.mem} "
        f"--time={plan.time} --pty bash"
    )
    lines = [
        "The rootless build on the front-end failed.",
        "Apptainer is rootless on Gilbreth, so a build should not need an "
        "allocation. If this was a resource limit rather than a permission "
        "error, retry from an interactive session:",
        f"  {srun}",
        f"  apptainer build {remote_sif} {remote_def}",
    ]
    if account == ACCOUNT_PLACEHOLDER:
        lines.append(
            f"  ({ACCOUNT_PLACEHOLDER} is a placeholder: run `slist` on the "
            "cluster for your allocations, or set RCAC_ACCOUNT.)"
        )
    return lines


@click.group()
def cli() -> None:
    """Manage Apptainer containers on Gilbreth."""


@cli.command()
@click.argument("def_file", type=click.Path(exists=True))
@click.option("--name", "-n", default=None, help="Output image name (default: <def_file>.sif)")
@click.option("--remote-build", is_flag=True, help="Build on Gilbreth instead of locally")
def build(def_file: str, name: str | None, remote_build: bool) -> None:
    """Build an Apptainer image from a definition file.

    Local builds need apptainer installed on this machine.  ``--remote-build``
    uploads the definition and builds on a Gilbreth front-end directly -- no
    allocation is requested, because RCAC documents Apptainer as rootless.  If
    that build fails, the interactive-allocation fallback is printed rather
    than assumed; see :func:`interactive_build_hint` and plan row L17.
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
        click.echo(f"Uploading {def_path.name} to Gilbreth...")
        _run_remote(cfg, f"mkdir -p {scratch}/.rcac-tools")
        ssh = cfg["ssh"]
        subprocess.run(
            ["scp", "-P", str(ssh["port"]), str(def_path),
             f"{user}@{ssh['login_host']}:{remote_def}"],
            capture_output=True, text=True,
        )

        click.echo(f"Building {image_name} on Gilbreth (this may take a while)...")
        click.echo(f"  apptainer build {remote_sif} {remote_def}")

        result = _run_remote(cfg, f"apptainer build {remote_sif} {remote_def}")
        click.echo(result.stdout)
        if result.returncode != 0:
            click.echo(f"Build failed: {result.stderr}", err=True)
            for line in interactive_build_hint(cfg, remote_sif, remote_def):
                click.echo(line, err=True)
            sys.exit(1)
        click.echo(f"Image built: {remote_sif}")
    else:
        click.echo(f"Building {image_name} locally...")
        result = subprocess.run(
            ["apptainer", "build", image_name, str(def_path)],
            capture_output=False,
        )
        if result.returncode != 0:
            click.echo("Local build failed. Try --remote-build, or install "
                       "apptainer locally.", err=True)
            sys.exit(1)
        click.echo(f"Image built: {image_name}")


@cli.command()
@click.argument("image", type=click.Path(exists=True))
@click.option("--dest", "-d", default=None,
              help="Remote path (default: <paths.scratch>/.rcac-tools/)")
def push(image: str, dest: str | None) -> None:
    """Upload an Apptainer image (.sif) to Gilbreth."""
    cfg = get_rcac_config()

    require_user(cfg)

    user = cfg["ssh"]["user"]
    image_path = Path(image)

    # Gilbreth has no data-transfer node, so this goes over the login host.
    host = transfer_host(cfg)
    remote_dest = dest or f"{scratch_root(cfg)}/.rcac-tools/{image_path.name}"

    size_bytes = image_path.stat().st_size
    click.echo(f"Uploading {image_path.name} to Gilbreth...")
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
              help="Bind a path Apptainer does NOT auto-bind. /home, /scratch, "
                   "/apps and /depot are already mounted; passing them again is "
                   "at best a no-op.")
@click.option("--nv", is_flag=True, default=True, help="Enable NVIDIA GPU support (default: on)")
def run(image_path: str, command: tuple[str, ...], bind: tuple[str, ...], nv: bool) -> None:
    """Run a command inside an Apptainer container on Gilbreth.

    IMAGE_PATH is the remote path to the .sif file.
    """
    cfg = get_rcac_config()

    require_user(cfg)

    parts = ["apptainer", "exec"]
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
    """List Apptainer images (.sif) on Gilbreth."""
    cfg = get_rcac_config()

    require_user(cfg)

    result = _run_remote(
        cfg,
        f"find {scratch_root(cfg)} -name '*.sif' -printf '%s %p\\n' "
        "2>/dev/null | sort -rn",
    )

    if result.returncode != 0 or not result.stdout.strip():
        click.echo("No Apptainer images found.")
        return

    click.echo(f"{'Size':>10}  Path")
    click.echo("-" * 60)
    for line in result.stdout.strip().split("\n"):
        if line.strip():
            size_bytes, path = line.split(None, 1)
            size_gb = int(size_bytes) / (1024**3)
            click.echo(f"{size_gb:>9.1f}G  {path}")
