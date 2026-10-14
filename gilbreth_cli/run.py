"""Quick one-off jobs and interactive GPU shells without an experiment.yaml."""

from __future__ import annotations

import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import click

from .config import get_config, require_account, require_user, safe_name
from .experiment import _sbatch, _unique_run_dir, effective_env, render_or_exit
from .remote import rsync, run_remote, ssh_base, ssh_target, upload
from .scheduler import resolve_qos


def _common_options(fn):
    opts = [
        click.option("--account", "-A", default=None, help="SLURM account (default: GILBRETH_ACCOUNT)"),
        click.option("--partition", "-p", default=None, help="a10 | a30 | a100-40gb | a100-80gb | h100 | training"),
        click.option("--qos", "-q", default=None, help="normal | standby | training"),
        click.option("--gpus", "-g", default=None, type=int, help="GPUs per node"),
        click.option("--cpus-per-gpu", default=None, type=int),
        click.option("--mem", default=None, help="e.g. 32G"),
        click.option("--time", "-t", "time_limit", default=None, help="e.g. 02:00:00"),
        click.option("--constraint", "-C", default=None, help="e.g. nvlink"),
    ]
    for opt in reversed(opts):
        fn = opt(fn)
    return fn


def _merge_slurm(cfg: dict, **overrides) -> dict:
    s = dict(cfg["slurm"])
    for k, v in overrides.items():
        if v is not None:
            s[k] = v
    s["account"] = require_account(cfg, s.get("account"))
    s["qos"] = resolve_qos(s["partition"], s.get("qos"))
    return s


@click.command(context_settings={"ignore_unknown_options": True, "allow_interspersed_args": False})
@click.argument("command", nargs=-1, required=True, type=click.UNPROCESSED)
@click.option("--project", default=None, help="Remote project dir name (default: current directory name)")
@click.option("--name", "-J", default=None, help="Job name")
@click.option("--env", "env_name", default=None, help="conda env name / path (overrides config)")
@click.option("--env-type", default=None, type=click.Choice(["conda", "conda-env-mod", "apptainer", "venv", "none"]))
@click.option("--container", default=None, help="Apptainer .sif path (with --env-type apptainer)")
@click.option("--torchrun", is_flag=True, help="Wrap the command with torchrun --nproc_per_node=<gpus>")
@click.option("--sync/--no-sync", default=True, help="rsync the current directory to the project first")
@click.option("--dry-run", is_flag=True)
@_common_options
def run(command, project, name, env_name, env_type, container, torchrun, sync, dry_run,
        account, partition, qos, gpus, cpus_per_gpu, mem, time_limit, constraint) -> None:
    """Submit a single command as a batch job, e.g.

        gilbreth-run -g 1 -t 02:00:00 python train.py --lr 1e-4

    gilbreth-run options go BEFORE the command; everything after the first non-option word is
    passed to the job untouched. The current directory is synced to
    /scratch/gilbreth/<user>/projects/<dir name> and the command runs there.
    """
    cfg = get_config()
    require_user(cfg)
    slurm = _merge_slurm(cfg, account=account, partition=partition, qos=qos, gpus=gpus,
                         cpus_per_gpu=cpus_per_gpu, mem=mem, time=time_limit, constraint=constraint)
    env = dict(cfg.get("environment", {}))
    if env_type:
        env["type"] = env_type
    if env_name:
        env["name"] = env_name
    if container:
        env["container"] = container
        env["type"] = "apptainer"
    env = effective_env(env, "gilbreth-run")
    if env.get("type") == "apptainer" and not env.get("container"):
        raise click.UsageError("--env-type apptainer needs --container PATH.sif")

    local = Path.cwd().resolve()
    project = safe_name(project or local.name)
    project_dir = f"{cfg['paths']['projects']}/{project}"
    exp_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_name = safe_name(name) if name else f"{project}_run"
    cmd_str = shlex.join(command)

    kwargs = {
        "project": project, "exp_id": exp_id, "job_name": job_name,
        "account": slurm["account"], "partition": slurm["partition"], "qos": slurm["qos"],
        "nodes": int(slurm.get("nodes", 1)), "gpus": int(slurm["gpus"]), "cpus_per_gpu": int(slurm.get("cpus_per_gpu", 8)),
        "mem": str(slurm["mem"]), "time": str(slurm["time"]), "constraint": slurm.get("constraint") or "",
        "env_type": env.get("type", "none"), "env_name": env.get("name") or "", "container": env.get("container") or "",
        "modules": env.get("modules"),
    }
    if torchrun:
        if command[0] in ("python", "python3"):
            command = command[1:]
        kwargs.update(launcher="torchrun", script=command[0], args=shlex.join(command[1:]))
    else:
        kwargs.update(command=cmd_str)
    script_text = render_or_exit(cfg, **kwargs)

    local_run = _unique_run_dir(f"{project}_{exp_id}")
    local_run.mkdir(parents=True, exist_ok=True)
    local_script = local_run / "job.sh"
    local_script.write_text(script_text)

    click.echo(f"Job       : {job_name}  ->  {project_dir}")
    click.echo(f"SLURM     : -A {slurm['account']} -p {slurm['partition']} -q {slurm['qos']} gpus={slurm['gpus']} mem={slurm['mem']} time={slurm['time']}")
    click.echo(f"Command   : {cmd_str}")
    if dry_run:
        click.echo(f"\nDry run. Script: {local_script}")
        return

    res = run_remote(cfg, f"mkdir -p {project_dir}/scripts {cfg['paths']['logs']}")
    if res.returncode != 0:
        click.echo(f"Could not create remote directories: {res.stderr.strip()}", err=True)
        sys.exit(1)
    if sync:
        from .sync import exclude_args
        res = rsync(cfg, f"{local}/", f"{ssh_target(cfg)}:{project_dir}/", extra=exclude_args(cfg), quiet=True, capture=True)
        if res.returncode != 0:
            click.echo(f"Sync failed: {res.stderr.strip()}", err=True)
            sys.exit(1)
    remote_script = f"{project_dir}/scripts/run_{exp_id}.sh"
    if not upload(cfg, local_script, remote_script):
        click.echo("Upload failed.", err=True)
        sys.exit(1)
    job_id, err = _sbatch(cfg, remote_script)
    if not job_id:
        click.echo(f"sbatch failed: {err}", err=True)
        sys.exit(1)
    click.echo(f"Submitted job {job_id}.  Logs: gilbreth-exp logs {job_id}")


@click.command()
@click.option("--login", is_flag=True, help="Plain SSH shell on the login node (has one A30 for quick tests)")
@click.option("--cd", "cd_project", default=None, help="Project dir to start in (default: current directory name)")
@_common_options
def shell(login, cd_project, account, partition, qos, gpus, cpus_per_gpu, mem, time_limit, constraint) -> None:
    """Open an interactive GPU shell on a compute node (sinteractive), or --login for the front-end."""
    cfg = get_config()
    require_user(cfg)
    project = safe_name(cd_project or Path.cwd().resolve().name)
    project_dir = f"{cfg['paths']['projects']}/{project}"
    if login:
        rc = subprocess.run(ssh_base(cfg, tty=True) + [f"cd {shlex.quote(project_dir)} 2>/dev/null; exec bash -l"], check=False).returncode
        sys.exit(rc)
    slurm = _merge_slurm(cfg, account=account, partition=partition, qos=qos, gpus=gpus,
                         cpus_per_gpu=cpus_per_gpu, mem=mem, time=time_limit or "01:00:00", constraint=constraint)
    sint = (f"sinteractive -A {slurm['account']} -p {slurm['partition']} -q {slurm['qos']} "
            f"-N {slurm.get('nodes', 1)} --gpus-per-node={slurm['gpus']} --cpus-per-gpu={slurm.get('cpus_per_gpu', 8)} "
            f"--mem={slurm['mem']} -t {slurm['time']}")
    if slurm.get("constraint"):
        sint += f" -C {slurm['constraint']}"
    click.echo(f"Requesting: {sint}")
    click.echo("Waiting for a node ... (Ctrl+C to abort)")
    remote = f"cd {shlex.quote(project_dir)} 2>/dev/null; {sint}"
    rc = subprocess.run(ssh_base(cfg, tty=True) + [f"bash -lc {shlex.quote(remote)}"], check=False).returncode
    sys.exit(rc)
