"""Experiment manager: experiment.yaml -> sweep configs -> sbatch on Gilbreth."""

from __future__ import annotations

import itertools
import json
import random
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import click
import yaml

from .config import get_config, require_account, require_user, safe_name
from .remote import (
    download,
    rsync,
    run_remote,
    run_remote_interactive,
    ssh_base,
    ssh_target,
    upload,
)
from .scheduler import render_job, resolve_qos

TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL",
                   "BOOT_FAIL", "DEADLINE", "PREEMPTED"}

LOCAL_RUNS_DIR = Path.home() / ".gilbreth" / "runs"

EXAMPLE_EXPERIMENT = """\
# gilbreth-exp launch experiment.yaml
project: my_project            # remote dir: /scratch/gilbreth/<user>/projects/my_project

slurm:
  # account: mygroup           # default: GILBRETH_ACCOUNT / configs/gilbreth.yaml
  partition: a100-80gb         # a10 | a30 | a100-40gb | a100-80gb | h100 | training
  qos: normal                  # normal | standby (free, <=4h) | training
  nodes: 1
  gpus: 1                      # --gpus-per-node
  cpus_per_gpu: 8
  mem: 64G
  time: "04:00:00"
  # constraint: nvlink

environment:
  type: conda                  # conda | conda-env-mod | apptainer | venv | none
  name: my_env                 # conda env created with `gilbreth-env setup`
  modules: [cuda]
  launcher: python             # torchrun for multi-GPU DDP

script: train.py
args: "--config {config_path} --lr {learning_rate}"   # placeholders = base_params / sweep keys / config_path / exp_id

base_params:
  epochs: 10
  batch_size: 64
  seed: 42

sweep:
  method: grid                 # grid | random (count: N)
  mode: array                  # array (one sbatch --array) | separate (one sbatch per config)
  parameters:
    learning_rate: [1.0e-4, 3.0e-4]   # 1.0e-4, not 1e-4: PyYAML would read 1e-4 as a string

# Optional multi-stage pipeline (each stage waits for the previous one with afterok)
# pipeline:
#   - name: train
#     script: train.py
#     args: "--config {config_path}"
#   - name: eval
#     script: evaluate.py
#     args: "--run {exp_id}"
"""


class _StrictDict(dict):
    def __missing__(self, key):
        raise KeyError(key)


def generate_sweep_configs(sweep: dict, base_params: dict) -> list[dict]:
    params = sweep.get("parameters") or {}
    if not params:
        return [dict(base_params)]
    keys = list(params)
    values = [v if isinstance(v, list) else [v] for v in (params[k] for k in keys)]
    method = sweep.get("method", "grid")
    if method == "random":
        rng = random.Random(sweep.get("seed"))
        combos = [tuple(rng.choice(v) for v in values) for _ in range(int(sweep.get("count", 10)))]
    else:
        combos = list(itertools.product(*values))
    configs = []
    for combo in combos:
        cfg = dict(base_params)
        cfg.update(zip(keys, combo))
        configs.append(cfg)
    return configs


def format_args(raw: str, ctx: dict, where: str) -> str:
    try:
        return raw.format_map(_StrictDict(ctx))
    except KeyError as e:
        click.echo(f"[{where}] unknown placeholder {{{e.args[0]}}}; available: {sorted(ctx)}", err=True)
        sys.exit(2)


def _slurm_settings(cfg: dict, exp: dict, overrides: dict) -> dict:
    d = dict(cfg["slurm"])
    d.update({k: v for k, v in (exp.get("slurm") or {}).items() if v is not None})
    d.update({k: v for k, v in overrides.items() if v is not None})
    d["account"] = require_account(cfg, d.get("account"))
    return d


def _env_settings(cfg: dict, exp: dict) -> dict:
    d = dict(cfg.get("environment", {}))
    d.update({k: v for k, v in (exp.get("environment") or {}).items() if v is not None})
    return effective_env(d, "experiment.yaml")
def effective_env(env: dict, where: str = "") -> dict:
    """Downgrade a conda/venv env with no name to 'none' (with a warning) instead of rendering a broken activate."""
    env = dict(env)
    if env.get("type") in ("conda", "conda-env-mod", "venv") and not env.get("name"):
        click.echo(f"{where + ': ' if where else ''}environment.type={env['type']} but no environment.name; "
                   f"running with the module/system python. Set environment.name (or --env NAME).", err=True)
        env["type"] = "none"
    return env



def _unique_run_dir(name: str) -> Path:
    path = LOCAL_RUNS_DIR / name
    n = 1
    while path.exists():
        path = LOCAL_RUNS_DIR / f"{name}_{n}"
        n += 1
    return path


def parse_sacct_states(output: str, pending: set[str]) -> dict[str, tuple[str, str, str]]:
    """Map each watched job id to (state, exit_code, elapsed) once *every* sacct row of it
    (array tasks like 123_0, 123_[1-5]) is in a terminal state."""
    rows: dict[str, list[tuple[str, str, str]]] = {}
    for line in output.strip().splitlines():
        jid, state, exit_code, elapsed = (line.split("|") + ["", "", ""])[:4]
        parent = jid.split("_")[0].strip()
        if parent in pending:
            rows.setdefault(parent, []).append((state.split()[0] if state else "", exit_code, elapsed))
    done = {}
    for jid, states in rows.items():
        if states and all(st[0] in TERMINAL_STATES for st in states):
            worst = next((st for st in states if st[0] != "COMPLETED"), states[0])
            done[jid] = worst
    return done


def render_or_exit(cfg: dict, **kwargs) -> str:
    """render_job with validation errors turned into a clean exit instead of a traceback."""
    try:
        return render_job(cfg, **kwargs)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(2)


def _sbatch(cfg: dict, remote_script: str) -> tuple[str | None, str]:
    result = run_remote(cfg, f"sbatch --parsable {shlex.quote(remote_script)}")
    if result.returncode != 0:
        return None, result.stderr.strip()
    return result.stdout.strip().split(";")[0], ""


@click.group()
def cli() -> None:
    """Submit and track training jobs on Gilbreth."""


@cli.command()
@click.option("--output", "-o", default="experiment.yaml", type=click.Path())
def init(output: str) -> None:
    """Write a starter experiment.yaml into the current directory."""
    path = Path(output)
    if path.exists():
        click.echo(f"{path} already exists.", err=True)
        sys.exit(1)
    path.write_text(EXAMPLE_EXPERIMENT)
    click.echo(f"Wrote {path}. Edit it, then: gilbreth-exp launch {path}")


@cli.command()
@click.argument("experiment_file", type=click.Path(exists=True))
@click.option("--dry-run", is_flag=True, help="Render scripts locally without submitting")
@click.option("-n", "--max-jobs", default=None, type=int, help="Only the first N sweep combinations")
@click.option("--account", "-A", default=None)
@click.option("--partition", "-p", default=None)
@click.option("--qos", "-q", default=None)
@click.option("--gpus", default=None, type=int)
@click.option("--time", "time_limit", default=None)
@click.option("--mem", default=None)
@click.option("--sync/--no-sync", default=True, help="rsync the experiment.yaml directory to the project first")
def launch(experiment_file, dry_run, max_jobs, account, partition, qos, gpus, time_limit, mem, sync) -> None:
    """Launch experiments described by an experiment.yaml."""
    cfg = get_config()
    user = require_user(cfg)
    exp_path = Path(experiment_file).resolve()
    with open(exp_path) as f:
        exp = yaml.safe_load(f) or {}

    project = safe_name(exp.get("project") or exp_path.parent.name)
    slurm = _slurm_settings(cfg, exp, {
        "account": account, "partition": partition, "qos": qos, "gpus": gpus, "time": time_limit, "mem": mem,
    })
    env = _env_settings(cfg, exp)
    base_params = exp.get("base_params") or {}
    sweep = exp.get("sweep") or {}
    pipeline = exp.get("pipeline") or []
    sweep_mode = sweep.get("mode", "array" if not pipeline else "separate")

    configs = generate_sweep_configs(sweep, base_params)
    if max_jobs:
        configs = configs[:max_jobs]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    paths = cfg["paths"]
    project_dir = f"{paths['projects']}/{project}"
    remote_cfg_dir = f"{project_dir}/configs/{timestamp}"
    remote_script_dir = f"{project_dir}/scripts/{timestamp}"
    local_run = _unique_run_dir(f"{project}_{timestamp}")
    (local_run / "configs").mkdir(parents=True, exist_ok=True)
    (local_run / "scripts").mkdir(parents=True, exist_ok=True)

    click.echo(f"Project   : {project}  ->  gilbreth:{project_dir}")
    slurm["qos"] = resolve_qos(slurm["partition"], slurm.get("qos"))
    click.echo(f"SLURM     : -A {slurm['account']} -p {slurm['partition']} -q {slurm['qos']} "
               f"gpus={slurm['gpus']} mem={slurm['mem']} time={slurm['time']}")
    click.echo(f"Env       : {env.get('type')} {env.get('name') or env.get('container') or ''}")
    click.echo(f"Configs   : {len(configs)} ({'pipeline' if pipeline else sweep_mode})")

    for i, c in enumerate(configs):
        c["_exp_id"] = f"{timestamp}_{i:03d}"
        with open(local_run / "configs" / f"config_{i:03d}.yaml", "w") as f:
            yaml.safe_dump(c, f, default_flow_style=False)

    common = {
        "account": slurm["account"], "partition": slurm["partition"], "qos": slurm.get("qos"),
        "nodes": int(slurm.get("nodes", 1)), "gpus": int(slurm["gpus"]), "cpus_per_gpu": int(slurm.get("cpus_per_gpu", 8)),
        "mem": str(slurm["mem"]), "time": str(slurm["time"]), "constraint": slurm.get("constraint") or "",
        "env_type": env.get("type", "none"), "env_name": env.get("name") or "", "container": env.get("container") or "",
        "modules": env.get("modules"), "launcher": env.get("launcher", "python"), "env_vars": exp.get("env_vars"),
    }

    scripts: list[tuple[str, str, str | None]] = []  # (local file, label, depends-on label)
    default_script = exp.get("script", "train.py")
    default_args = exp.get("args", "--config {config_path}")

    for stage in pipeline:
        if not isinstance(stage, dict) or not stage.get("name") or not stage.get("script"):
            click.echo(f"pipeline stages need 'name' and 'script': {stage!r}", err=True)
            sys.exit(2)
    if pipeline:
        for i, c in enumerate(configs):
            exp_id = c["_exp_id"]
            config_path = f"{remote_cfg_dir}/config_{i:03d}.yaml"
            prev = None
            for stage in pipeline:
                ctx = {**{k: v for k, v in c.items() if not k.startswith("_")}, "exp_id": exp_id, "config_path": config_path}
                label = f"{i:03d}_{stage['name']}"
                text = render_or_exit(cfg, project=project, exp_id=exp_id, stage_name=stage["name"],
                                  job_name=f"{project}_{stage['name']}_{exp_id}",
                                  script=stage["script"], args=format_args(stage.get("args", ""), ctx, label),
                                  config_path=config_path, dependency_job_id=f"__{prev}__" if prev else None, **common)
                path = local_run / "scripts" / f"job_{label}.sh"
                path.write_text(text)
                scripts.append((str(path), label, prev))
                prev = label
    elif sweep_mode == "array" and len(configs) > 1:
        # One sbatch --array; per-task args are pre-rendered into args_NNN.txt next to the configs.
        for i, c in enumerate(configs):
            config_path = f"{remote_cfg_dir}/config_{i:03d}.yaml"
            ctx = {**{k: v for k, v in c.items() if not k.startswith("_")}, "exp_id": c["_exp_id"], "config_path": config_path}
            (local_run / "configs" / f"args_{i:03d}.txt").write_text(format_args(default_args, ctx, c["_exp_id"]) + "\n")
        text = render_or_exit(cfg, project=project, exp_id=timestamp, job_name=f"{project}_{timestamp}",
                          script=default_script, args='$(cat "$ARGS_FILE")',
                          config_dir=remote_cfg_dir, array_max=len(configs) - 1,
                          array_parallel=sweep.get("parallel"), **common)
        path = local_run / "scripts" / "job_array.sh"
        path.write_text(text)
        scripts.append((str(path), "array", None))
    else:
        for i, c in enumerate(configs):
            exp_id = c["_exp_id"]
            config_path = f"{remote_cfg_dir}/config_{i:03d}.yaml"
            ctx = {**{k: v for k, v in c.items() if not k.startswith("_")}, "exp_id": exp_id, "config_path": config_path}
            text = render_or_exit(cfg, project=project, exp_id=exp_id, script=default_script,
                              args=format_args(default_args, ctx, exp_id), config_path=config_path, **common)
            path = local_run / "scripts" / f"job_{i:03d}.sh"
            path.write_text(text)
            scripts.append((str(path), f"{i:03d}", None))

    if dry_run:
        for path, label, _ in scripts:
            click.echo(f"  [{label}] {path}")
        click.echo(f"\nDry run. Scripts and configs in {local_run}")
        return

    if sync:
        from .sync import exclude_args

        click.echo(f"Syncing {exp_path.parent}/ -> {project_dir}/ ...")
        run_remote(cfg, f"mkdir -p {project_dir}")
        res = rsync(cfg, f"{exp_path.parent}/", f"{ssh_target(cfg)}:{project_dir}/", extra=exclude_args(cfg), quiet=True, capture=True)
        if res.returncode != 0:
            click.echo(f"Sync failed: {res.stderr.strip()}", err=True)
            sys.exit(1)

    res = run_remote(cfg, f"mkdir -p {remote_cfg_dir} {remote_script_dir} {paths['logs']} {project_dir}/experiments")
    if res.returncode != 0:
        click.echo(f"Could not create remote directories: {res.stderr.strip()}", err=True)
        sys.exit(1)
    if not upload(cfg, f"{local_run}/configs/", f"{remote_cfg_dir}/") or not upload(cfg, f"{local_run}/scripts/", f"{remote_script_dir}/"):
        click.echo("Upload of configs/scripts failed.", err=True)
        sys.exit(1)

    job_ids: dict[str, str] = {}
    failures = 0
    for path, label, depends in scripts:
        remote_script = f"{remote_script_dir}/{Path(path).name}"
        if depends:
            dep_id = job_ids.get(depends)
            if not dep_id:
                click.echo(f"  [{label}] skipped (dependency {depends} not submitted)")
                continue
            res = run_remote(cfg, f"sed -i 's/__{depends}__/{dep_id}/' {shlex.quote(remote_script)}")
            if res.returncode != 0:
                click.echo(f"  [{label}] FAILED to wire dependency: {res.stderr.strip()}", err=True)
                failures += 1
                continue
        job_id, err = _sbatch(cfg, remote_script)
        if job_id:
            job_ids[label] = job_id
            click.echo(f"  [{label}] submitted: job {job_id}")
        else:
            click.echo(f"  [{label}] FAILED: {err}", err=True)
            failures += 1

    tracking = {
        "project": project, "timestamp": timestamp, "user": user, "jobs": job_ids,
        "num_configs": len(configs), "experiment_file": str(exp_path), "slurm": slurm,
    }
    (local_run / "tracking.json").write_text(json.dumps(tracking, indent=2, default=str))
    upload(cfg, f"{local_run}/tracking.json", f"{project_dir}/experiments/{timestamp}_tracking.json")
    if job_ids:
        click.echo(f"\nSubmitted {len(job_ids)} job(s). Track with: gilbreth-exp status | gilbreth-exp logs <JOB_ID>")
    if failures:
        click.echo(f"{failures} submission(s) failed.", err=True)
        sys.exit(1)


@cli.command()
@click.option("--all-users", is_flag=True, help="Show the whole partition queue instead of only your jobs")
@click.option("--partition", "-p", default=None)
def status(all_users: bool, partition: str | None) -> None:
    """List your queued / running jobs (squeue)."""
    cfg = get_config()
    user = require_user(cfg)
    fmt = "%.10i %.28j %.10P %.8T %.10M %.10l %.5D %.4b %R"
    cmd = f"squeue -o '{fmt}'" + ("" if all_users else f" -u {user}") + (f" -p {partition}" if partition else "")
    result = run_remote(cfg, cmd)
    if result.returncode != 0:
        click.echo(result.stderr.strip(), err=True)
        sys.exit(1)
    lines = result.stdout.strip().splitlines()
    click.echo("\n".join(lines) if len(lines) > 1 else "No jobs in queue.")


@cli.command()
@click.option("--days", "-d", default=3, help="Look back N days")
def history(days: int) -> None:
    """Show recently finished jobs (sacct)."""
    cfg = get_config()
    user = require_user(cfg)
    cmd = (f"sacct -u {user} -S now-{days}days -X "
           f"--format=JobID%14,JobName%30,Partition%10,State%12,ExitCode,Elapsed,AllocTRES%30")
    result = run_remote(cfg, cmd)
    click.echo(result.stdout if result.returncode == 0 else result.stderr, err=result.returncode != 0)


def _find_log(cfg: dict, job_id: str) -> str | None:
    """Resolve the stdout file of a job (first array task for array ids); None if it does not exist yet."""
    base = job_id.split("_")[0]
    result = run_remote(cfg, f"scontrol show job {shlex.quote(job_id)} 2>/dev/null | sed -n 's/^ *StdOut=//p' | head -1")
    path = result.stdout.strip()
    if path and run_remote(cfg, f"test -f {shlex.quote(path)}").returncode == 0:
        return path
    logs = cfg["paths"]["logs"]
    result = run_remote(cfg, f"ls -t {logs}/*_{base}.out {logs}/*_{base}_*.out 2>/dev/null | head -1")
    return result.stdout.strip() or None


@cli.command()
@click.argument("job_id")
@click.option("--follow/--no-follow", default=True, help="Keep streaming (default) or print the tail once")
@click.option("--lines", "-n", default=50)
def logs(job_id: str, follow: bool, lines: int) -> None:
    """Show (and follow) the stdout log of a job."""
    cfg = get_config()
    require_user(cfg)
    log = _find_log(cfg, job_id)
    if not log:
        click.echo(f"No log found for job {job_id} (not started yet?)")
        return
    click.echo(f"== {log}")
    if follow:
        try:
            run_remote_interactive(cfg, f"tail -n {lines} -f {shlex.quote(log)}")
        except KeyboardInterrupt:
            pass
    else:
        click.echo(run_remote(cfg, f"tail -n {lines} {shlex.quote(log)}").stdout)


@cli.command()
@click.argument("job_id")
def cancel(job_id: str) -> None:
    """Cancel one job, or 'all' for every job of yours."""
    cfg = get_config()
    user = require_user(cfg)
    if job_id == "all":
        click.confirm(f"Cancel ALL jobs of {user}?", abort=True)
        result = run_remote(cfg, f"scancel -u {user}")
    else:
        result = run_remote(cfg, f"scancel {shlex.quote(job_id)}")
    click.echo(f"Cancelled: {job_id}" if result.returncode == 0 else result.stderr.strip(), err=result.returncode != 0)


@cli.command()
@click.argument("job_ids", nargs=-1)
@click.option("--interval", "-i", default=30)
@click.option("--pull", default=None, type=click.Path(), help="Local dir to pull <project>/experiments into when done")
@click.option("--project", default=None, help="Project name for --pull")
def watch(job_ids: tuple[str, ...], interval: int, pull: str | None, project: str | None) -> None:
    """Poll jobs until they finish, optionally pulling results afterwards."""
    cfg = get_config()
    user = require_user(cfg)
    pending = {j.split("_")[0] for j in job_ids}
    click.echo(f"Watching {'jobs ' + ','.join(job_ids) if job_ids else 'all jobs of ' + user} (Ctrl+C to stop)")
    try:
        while True:
            if pending:
                result = run_remote(cfg, f"sacct -j {','.join(sorted(pending))} -X --parsable2 --noheader --format=JobID,State,ExitCode,Elapsed")
                if result.returncode != 0:
                    click.echo(f"  sacct failed: {result.stderr.strip()}", err=True)
                    sys.exit(1)
                for jid, (state, exit_code, elapsed) in parse_sacct_states(result.stdout, pending).items():
                    pending.discard(jid)
                    click.echo(f"  [{time.strftime('%H:%M:%S')}] job {jid}: {state} (exit {exit_code}, {elapsed})")
                if not pending:
                    break
            else:
                result = run_remote(cfg, f"squeue -h -u {user} -o '%i'")
                if result.returncode != 0:
                    click.echo(f"  squeue failed: {result.stderr.strip()}", err=True)
                    sys.exit(1)
                running = [j for j in result.stdout.split() if j]
                if not running:
                    click.echo("No jobs in queue.")
                    break
                click.echo(f"  [{time.strftime('%H:%M:%S')}] {len(running)} job(s) queued/running")
            time.sleep(interval)
    except KeyboardInterrupt:
        click.echo("\nStopped.")
        return
    if pull:
        project = project or Path(pull).resolve().name
        remote = f"{cfg['paths']['projects']}/{project}/experiments/"
        Path(pull).mkdir(parents=True, exist_ok=True)
        click.echo(f"Pulling {remote} -> {pull}/")
        download(cfg, remote, f"{pull}/")


@cli.command()
@click.argument("job_id")
def attach(job_id: str) -> None:
    """Open an interactive shell inside a running job's node (srun --jobid)."""
    cfg = get_config()
    require_user(cfg)
    cmd = f"srun --jobid={shlex.quote(job_id)} --overlap --pty bash -l"
    subprocess.run(ssh_base(cfg, tty=True) + [f"bash -lc {shlex.quote(cmd)}"], check=False)
