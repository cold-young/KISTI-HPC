"""Experiment manager: parse experiment.yaml, generate sweep configs, submit SLURM jobs."""

from __future__ import annotations

import itertools
import json
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import click
import yaml

from .config import get_defaults, get_neuron_config
from .scheduler import (
    generate_job_array_script,
    generate_multi_gpu_script,
    generate_pipeline_script,
    generate_single_gpu_script,
)


def _ssh_cmd(cfg: dict) -> list[str]:
    ssh = cfg["ssh"]
    return [
        "ssh",
        "-o", f"ControlPath={ssh['control_path']}",
        "-p", str(ssh["port"]),
        f"{ssh['user']}@{ssh['login_host']}",
    ]


def _run_remote(cfg: dict, command: str) -> subprocess.CompletedProcess:
    # Ensure commands run from scratch directory (KISTI sbatch requirement)
    user = cfg["ssh"]["user"]
    wrapped = f"cd /scratch/{user} && {command}"
    cmd = _ssh_cmd(cfg) + [wrapped]
    return subprocess.run(cmd, capture_output=True, text=True)


def _generate_sweep_configs(sweep: dict, base_params: dict) -> list[dict]:
    """Generate parameter combinations from sweep config."""
    method = sweep.get("method", "grid")
    params = sweep.get("parameters", {})

    if not params:
        return [base_params.copy()]

    keys = list(params.keys())
    values = [params[k] if isinstance(params[k], list) else [params[k]] for k in keys]

    if method == "grid":
        combos = list(itertools.product(*values))
    elif method == "random":
        n = sweep.get("count", 10)
        combos = []
        for _ in range(n):
            combo = tuple(random.choice(v) for v in values)
            combos.append(combo)
    else:
        combos = list(itertools.product(*values))

    configs = []
    for combo in combos:
        cfg = base_params.copy()
        for k, v in zip(keys, combo):
            cfg[k] = v
        configs.append(cfg)

    return configs


def _write_config_file(config: dict, config_dir: Path, index: int) -> Path:
    """Write a single experiment config to YAML file."""
    config_path = config_dir / f"config_{index:03d}.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    return config_path


@click.group()
def cli() -> None:
    """Manage experiments on Neuron HPC."""


@cli.command()
@click.argument("experiment_file", type=click.Path(exists=True))
@click.option("--dry-run", is_flag=True, help="Generate scripts without submitting")
@click.option("-n", "--max-jobs", default=None, type=int, help="Limit number of sweep combinations")
@click.option("--partition", default=None, help="Override SLURM partition (e.g., amd_a100nv_8)")
@click.option("--gpus", "gpu_override", default=None, type=int, help="Override number of GPUs")
def launch(experiment_file: str, dry_run: bool, max_jobs: int | None, partition: str | None, gpu_override: int | None) -> None:
    """Launch experiments from an experiment.yaml file."""
    cfg = get_neuron_config()

    if not cfg["ssh"]["user"]:
        click.echo("Error: NEURON_USER not set.")
        sys.exit(1)

    with open(experiment_file) as f:
        exp = yaml.safe_load(f)

    project = exp["project"]
    slurm = exp.get("slurm", {})
    defaults = get_defaults()
    slurm_defaults = defaults.get("slurm", {})

    partition = partition or slurm.get("partition", slurm_defaults.get("partition", "amd_a100nv_8"))
    nodes = slurm.get("nodes", slurm_defaults.get("nodes", 1))
    gpus = gpu_override or slurm.get("gpus", slurm_defaults.get("gpus", 2))
    cpus_per_gpu = slurm.get("cpus_per_gpu", slurm_defaults.get("cpus_per_gpu", 8))
    time_limit = slurm.get("time", slurm_defaults.get("time", "24:00:00"))
    comment = slurm.get("comment", slurm_defaults.get("comment", "pytorch"))

    env_cfg = exp.get("environment", {})
    conda_env = env_cfg.get("name", "base")
    modules = env_cfg.get("modules", defaults.get("environment", {}).get("default_modules", []))

    base_params = exp.get("base_params", {})
    sweep = exp.get("sweep", {})
    pipeline = exp.get("pipeline", [])

    # Generate sweep configs
    configs = _generate_sweep_configs(sweep, base_params)
    if max_jobs:
        configs = configs[:max_jobs]

    click.echo(f"Project: {project}")
    click.echo(f"Parameter combinations: {len(configs)}")

    # Create experiment directory structure
    user = cfg["ssh"]["user"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_base = f"/scratch/{user}/projects/{project}"
    config_remote_dir = f"{exp_base}/configs/{timestamp}"
    scripts_dir = Path(f"/tmp/neuron_exp_{timestamp}")
    scripts_dir.mkdir(parents=True, exist_ok=True)

    # Ensure remote directories
    _run_remote(cfg, f"mkdir -p {config_remote_dir} /scratch/{user}/logs {exp_base}/experiments")

    job_ids = []

    for i, config in enumerate(configs):
        exp_id = f"{timestamp}_{i:03d}"
        config["_exp_id"] = exp_id
        config_path = f"{config_remote_dir}/config_{i:03d}.yaml"

        # Write config locally then upload
        local_config = scripts_dir / f"config_{i:03d}.yaml"
        with open(local_config, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        if pipeline:
            # Pipeline mode: chain stages with dependencies
            prev_job_id = None
            for stage in pipeline:
                stage_name = stage["name"]
                script = stage["script"]
                args = stage.get("args", "").replace("{config_path}", config_path)

                slurm_script = generate_pipeline_script(
                    project=project,
                    exp_id=exp_id,
                    stage_name=stage_name,
                    script=script,
                    args=args,
                    dependency_job_id=prev_job_id,
                    partition=partition,
                    nodes=nodes,
                    gpus=gpus,
                    cpus_per_gpu=cpus_per_gpu,
                    time=time_limit,
                    comment=comment,
                    modules=modules,
                    conda_env=conda_env,
                    config_path=config_path,
                )

                local_script = scripts_dir / f"job_{i:03d}_{stage_name}.sh"
                local_script.write_text(slurm_script)

                if not dry_run:
                    # Upload config and script
                    _upload_file(cfg, str(local_config), config_path)
                    remote_script = f"{exp_base}/scripts/job_{i:03d}_{stage_name}.sh"
                    _run_remote(cfg, f"mkdir -p {exp_base}/scripts")
                    _upload_file(cfg, str(local_script), remote_script)

                    # Submit job
                    result = _run_remote(cfg, f"sbatch {remote_script}")
                    if result.returncode == 0:
                        job_id = result.stdout.strip().split()[-1]
                        prev_job_id = job_id
                        job_ids.append(job_id)
                        click.echo(f"  [{exp_id}/{stage_name}] Submitted: Job {job_id}")
                    else:
                        click.echo(f"  [{exp_id}/{stage_name}] Failed: {result.stderr}", err=True)
                        break
                else:
                    click.echo(f"  [{exp_id}/{stage_name}] Script: {local_script}")
        else:
            # Single stage mode (no pipeline defined)
            train_script = "train.py"
            args = f"--config {config_path}"

            gen_fn = generate_multi_gpu_script if gpus > 1 else generate_single_gpu_script
            slurm_script = gen_fn(
                project=project,
                exp_id=exp_id,
                script=train_script,
                args=args,
                partition=partition,
                nodes=nodes,
                gpus=gpus,
                cpus_per_gpu=cpus_per_gpu,
                time=time_limit,
                comment=comment,
                modules=modules,
                conda_env=conda_env,
                config_path=config_path,
            )

            local_script = scripts_dir / f"job_{i:03d}.sh"
            local_script.write_text(slurm_script)

            if not dry_run:
                _upload_file(cfg, str(local_config), config_path)
                remote_script = f"{exp_base}/scripts/job_{i:03d}.sh"
                _run_remote(cfg, f"mkdir -p {exp_base}/scripts")
                _upload_file(cfg, str(local_script), remote_script)

                result = _run_remote(cfg, f"sbatch {remote_script}")
                if result.returncode == 0:
                    job_id = result.stdout.strip().split()[-1]
                    job_ids.append(job_id)
                    click.echo(f"  [{exp_id}] Submitted: Job {job_id}")
                else:
                    click.echo(f"  [{exp_id}] Failed: {result.stderr}", err=True)
            else:
                click.echo(f"  [{exp_id}] Script: {local_script}")

    # Save job tracking info
    tracking = {
        "project": project,
        "timestamp": timestamp,
        "job_ids": job_ids,
        "num_configs": len(configs),
        "experiment_file": experiment_file,
    }
    tracking_file = scripts_dir / "tracking.json"
    with open(tracking_file, "w") as f:
        json.dump(tracking, f, indent=2)

    if dry_run:
        click.echo(f"\nDry run complete. Scripts in: {scripts_dir}")
    else:
        click.echo(f"\nSubmitted {len(job_ids)} jobs: {job_ids}")
        # Also save tracking remotely
        remote_tracking = f"{exp_base}/experiments/{timestamp}/tracking.json"
        _run_remote(cfg, f"mkdir -p {exp_base}/experiments/{timestamp}")
        _upload_file(cfg, str(tracking_file), remote_tracking)


def _upload_file(cfg: dict, local_path: str, remote_path: str) -> bool:
    """Upload a file to Neuron via rsync. Returns True on success."""
    ssh = cfg["ssh"]
    ssh_cmd = f"ssh -o ControlPath={ssh['control_path']} -p {ssh['port']}"
    cmd = [
        "rsync", "-az",
        "-e", ssh_cmd,
        local_path,
        f"{ssh['user']}@{ssh['login_host']}:{remote_path}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        click.echo(f"Upload failed ({local_path}): {result.stderr}", err=True)
        return False
    return True


@cli.command()
def status() -> None:
    """Show status of all running experiments."""
    cfg = get_neuron_config()

    if not cfg["ssh"]["user"]:
        click.echo("Error: NEURON_USER not set.")
        sys.exit(1)

    result = _run_remote(cfg, "squeue -u $USER -o '%.10i %.30j %.8T %.10M %.6D %R' --noheader")
    if result.returncode != 0:
        click.echo(f"Error: {result.stderr}", err=True)
        sys.exit(1)

    lines = result.stdout.strip().split("\n")
    if not lines or not lines[0].strip():
        click.echo("No running jobs.")
        return

    click.echo(f"{'Job ID':<12} {'Name':<32} {'Status':<10} {'Time':<12} {'Nodes':<6} {'Reason'}")
    click.echo("-" * 90)
    for line in lines:
        if line.strip():
            click.echo(line)


@cli.command()
@click.argument("job_id")
def logs(job_id: str) -> None:
    """Stream logs for a specific job."""
    cfg = get_neuron_config()

    if not cfg["ssh"]["user"]:
        click.echo("Error: NEURON_USER not set.")
        sys.exit(1)

    user = cfg["ssh"]["user"]
    result = _run_remote(cfg, f"ls -t /scratch/{user}/logs/*_{job_id}.out 2>/dev/null | head -1")

    if result.returncode != 0 or not result.stdout.strip():
        click.echo(f"No log file found for job {job_id}.")
        return

    log_file = result.stdout.strip()
    click.echo(f"Streaming: {log_file}\n")

    # Use SSH to tail the log
    ssh = cfg["ssh"]
    cmd = _ssh_cmd(cfg) + [f"tail -f {log_file}"]
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        click.echo("\nStopped streaming.")


@cli.command()
@click.argument("project")
@click.option("--auto-pull", default=None, type=click.Path(), help="Local dir to auto-download results when jobs finish")
@click.option("--interval", "-i", default=30, help="Poll interval in seconds (default: 30)")
@click.option("--jobs", "-j", default=None, help="Comma-separated job IDs to watch (default: all user jobs)")
def watch(project: str, auto_pull: str | None, interval: int, jobs: str | None) -> None:
    """Watch running jobs and auto-pull results when they complete."""
    cfg = get_neuron_config()

    if not cfg["ssh"]["user"]:
        click.echo("Error: NEURON_USER not set.")
        sys.exit(1)

    user = cfg["ssh"]["user"]
    watch_ids = set(jobs.split(",")) if jobs else None
    completed_ids: set[str] = set()

    click.echo(f"Watching jobs for project '{project}'...")
    if auto_pull:
        click.echo(f"Auto-pull to: {auto_pull}")
    click.echo(f"Poll interval: {interval}s (Ctrl+C to stop)\n")

    try:
        while True:
            # Get current job states via sacct
            if watch_ids:
                job_filter = ",".join(watch_ids - completed_ids)
                if not job_filter:
                    click.echo("All watched jobs completed.")
                    break
                result = _run_remote(cfg, f"sacct -j {job_filter} --format=JobID%15,JobName%30,State%12,ExitCode,Elapsed --noheader --parsable2")
            else:
                result = _run_remote(cfg, f"squeue -u {user} -o '%i|%j|%T|%M' --noheader")

            if result.returncode != 0:
                time.sleep(interval)
                continue

            lines = [l for l in result.stdout.strip().split("\n") if l.strip()]

            if watch_ids:
                # sacct mode: detect newly completed jobs
                for line in lines:
                    parts = line.split("|")
                    if len(parts) < 4:
                        continue
                    job_id = parts[0].strip()
                    if "." in job_id:  # skip sub-steps like 717773.batch
                        continue
                    state = parts[2].strip()
                    if state in ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT") and job_id not in completed_ids:
                        exit_code = parts[3].strip()
                        elapsed = parts[4].strip() if len(parts) > 4 else ""
                        ts = time.strftime("%H:%M:%S")
                        icon = "OK" if state == "COMPLETED" else "FAIL"
                        click.echo(f"  [{ts}] [{icon}] Job {job_id} {state} (exit: {exit_code}, time: {elapsed})")
                        completed_ids.add(job_id)

                        if auto_pull and state == "COMPLETED":
                            _auto_pull_results(cfg, project, auto_pull)

                # Check if all done
                remaining = watch_ids - completed_ids
                if not remaining:
                    click.echo("\nAll watched jobs completed.")
                    break
            else:
                # squeue mode: show running jobs, exit when none left
                running = [l for l in lines if project in l]
                if not running:
                    click.echo("No running jobs found for this project.")
                    if auto_pull:
                        _auto_pull_results(cfg, project, auto_pull)
                    break
                ts = time.strftime("%H:%M:%S")
                click.echo(f"  [{ts}] {len(running)} job(s) running...")

            time.sleep(interval)

    except KeyboardInterrupt:
        click.echo("\nStopped watching.")
        if auto_pull:
            click.echo("Pulling latest results...")
            _auto_pull_results(cfg, project, auto_pull)


def _auto_pull_results(cfg: dict, project: str, local_dir: str) -> None:
    """Pull experiment results from Neuron to local directory."""
    from pathlib import Path
    ssh = cfg["ssh"]
    user = ssh["user"]
    remote = f"{user}@{ssh['login_host']}:/scratch/{user}/projects/{project}/experiments/"
    local = Path(local_dir).resolve()
    local.mkdir(parents=True, exist_ok=True)

    ssh_opt = f"ssh -o ControlPath={ssh['control_path']} -p {ssh['port']}"
    cmd = [
        "rsync", "-avz", "--progress",
        "-e", ssh_opt,
        remote,
        f"{local}/",
    ]
    click.echo(f"  Pulling results → {local}/")
    result = subprocess.run(cmd)
    if result.returncode == 0:
        click.echo("  Pull complete.")
    else:
        click.echo("  Pull failed.", err=True)


@cli.command()
@click.argument("job_id")
def cancel(job_id: str) -> None:
    """Cancel a specific job or all jobs."""
    cfg = get_neuron_config()

    if not cfg["ssh"]["user"]:
        click.echo("Error: NEURON_USER not set.")
        sys.exit(1)

    if job_id == "all":
        result = _run_remote(cfg, "scancel -u $USER")
    else:
        result = _run_remote(cfg, f"scancel {job_id}")

    if result.returncode == 0:
        click.echo(f"Cancelled: {job_id}")
    else:
        click.echo(f"Error: {result.stderr}", err=True)
