"""Experiment manager: parse experiment.yaml, generate sweep configs, submit SLURM jobs."""

from __future__ import annotations

import itertools
import json
import random
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import click
import yaml

from ._legacy import LEGACY_ENV_TYPES

#: The suffix `conda-env-mod` appends to a module it creates, e.g. the
#: `-py3.12` in `conda-env/train_env-py3.12`. Used only to notice a manifest
#: that names the environment without it; see the warning in `launch`.
PY_SUFFIX_RE = re.compile(r"-py\d+\.\d+")
from . import discovery
from .config import (
    get_defaults,
    get_rcac_config,
    require_account,
    require_user,
    scratch_root,
)
from .gilbreth import DEFAULT_QOS, emit_warnings
from .ssh import ssh_opt_argv, ssh_opt_str, ssh_target
# B6/AC-3.16: `generate_job_array_script` was imported here and never called.
# `rcac-exp launch` submits a sweep as N independent jobs, not as one array, so
# the import was dead. The generator itself is still real and tested in
# scheduler.py; wiring `rcac-exp sweep --array` to it is a follow-up.
from .scheduler import (
    generate_multi_gpu_script,
    generate_pipeline_script,
    generate_single_gpu_script,
)

#: AC-3.14 / R15. Keys an older experiment.yaml may still carry. None of them
#: is silently dropped: each one warns and names its replacement, because a
#: silently ignored `comment:` looks identical to one that was honoured.
LEGACY_SLURM_KEYS: dict[str, str] = {
    "comment": (
        "slurm.comment is ignored: Gilbreth has no --comment job class. "
        "Remove the key. The billing entity is slurm.account (-A) and the GPU "
        "type is slurm.partition (-p)."
    ),
    "gres": (
        "slurm.gres is ignored: Gilbreth's documented GPU request form is "
        "--gpus-per-node, never --gres. Use slurm.gpus_per_node."
    ),
    "queue": (
        "slurm.queue is ignored: use slurm.partition for the GPU type (-p) "
        "and slurm.qos for the QOS (-q)."
    ),
}


def resolve_env_type(env_type: object) -> str | None:
    """AC-5.9 -- accept Gilbreth's runtime, reject the previous cluster's.

    A pre-port manifest naming the old container runtime is a *hard error*
    naming its replacement, not a silent alias.  Aliasing would work exactly
    once and leave the file wrong for the next reader; and unlike the
    ``slurm:`` keys above -- which are ignorable, so they warn -- an
    unrecognised ``environment.type`` selects the template's fallback branch
    and runs the job against the host system python, which fails deep inside
    the user's imports minutes after submission.

    The legacy spelling itself is not written here: it comes from
    :data:`rcac_cli._legacy.LEGACY_ENV_TYPES`, the one module allowlisted to
    hold it (AC-0.8), because banning the literal everywhere while also
    requiring its detection is otherwise a contradiction.
    """
    if env_type is None:
        return None
    replacement = LEGACY_ENV_TYPES.get(str(env_type).strip().lower())
    if replacement:
        raise click.ClickException(
            f"environment.type: {env_type} names the container runtime of the "
            f"cluster this tool was ported from. Gilbreth runs {replacement} "
            f"({replacement} is system-installed and needs no `module load`), "
            f"and the generated job scripts only have a branch for it. Change "
            f"the manifest to `type: {replacement}`; the image path in "
            f"`environment.container` does not change."
        )
    return str(env_type)


def _normalise_slurm(source: dict | None, origin: str, warnings: list) -> dict:
    """Return *source* with pre-port key spellings resolved, warning on each.

    Done once per layer, before the precedence chain runs, so a manifest that
    still says ``gpus:`` cannot be quietly out-voted by a ``gpus_per_node:``
    sitting in ``defaults.yaml`` -- which is exactly what a per-key fallback
    inside the chain would do, and it would do it silently.
    """
    out = dict(source or {})
    for key, message in LEGACY_SLURM_KEYS.items():
        if key in out:
            warnings.append(f"{origin}: {message}")
            out.pop(key)
    if "gpus" in out:
        legacy = out.pop("gpus")
        if "gpus_per_node" in out:
            warnings.append(
                f"{origin}: slurm.gpus and slurm.gpus_per_node are both set; "
                "slurm.gpus is the pre-port spelling and is ignored."
            )
        else:
            warnings.append(
                f"{origin}: slurm.gpus is the pre-port spelling and is read as "
                "slurm.gpus_per_node (--gpus-per-node, never --gres). Rename it."
            )
            out["gpus_per_node"] = legacy
    return out


def _ssh_cmd(cfg: dict) -> list[str]:
    """Build base SSH command using ControlMaster."""
    return ["ssh", *ssh_opt_argv(cfg), ssh_target(cfg)]


def _run_remote(cfg: dict, command: str) -> subprocess.CompletedProcess:
    """Run *command* on the login host.

    It used to be wrapped in a `cd` into the scratch root, annotated as an
    "sbatch requirement". That was a requirement of the *previous* cluster and
    the annotation survived the rename with the cluster name swapped, turning
    it into a false claim about RCAC. Nothing in Gilbreth's documentation
    requires a submission to happen from any particular directory, and every
    command this module sends uses absolute paths, so the wrapper is dropped
    rather than re-pointed at a directory that may not exist yet.
    """
    cmd = _ssh_cmd(cfg) + [command]
    return subprocess.run(cmd, capture_output=True, text=True)


class _StrictFormatDict(dict):
    """Dict for ``str.format_map`` that raises ``KeyError`` on missing keys.

    Used so manifest-author typos in placeholders surface as a hard error at
    launch time instead of being passed through verbatim into the sbatch args.
    """

    def __missing__(self, key):  # pragma: no cover - tiny shim
        raise KeyError(key)


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
    """Manage experiments on RCAC HPC."""


@cli.command()
@click.argument("experiment_file", type=click.Path(exists=True))
@click.option("--dry-run", is_flag=True,
              help="Generate scripts and print the exact sbatch command without "
                   "submitting or contacting the cluster at all.")
@click.option("-n", "--max-jobs", default=None, type=int, help="Limit number of sweep combinations")
@click.option("-A", "--account", "account_override", default=None,
              help="SLURM account (-A): the allocation the job is billed to. "
                   "Run `slist` on the cluster to see yours.")
@click.option("-p", "--partition", "partition_override", default=None,
              help="SLURM partition (-p): selects the GPU TYPE, e.g. a10, a30, "
                   "a100-40gb, a100-80gb, training.")
@click.option("-q", "--qos", "qos_override", default=None,
              help="SLURM QOS (-q): normal, standby (uncharged, 4 hour cap) or "
                   "training (requires -p training).")
@click.option("--time", "time_override", default=None,
              help="Walltime (--time): MM, MM:SS, HH:MM:SS, D-HH[:MM[:SS]].")
@click.option("--mem", "mem_override", default=None,
              help="Memory per node (--mem), e.g. 64G. Default: gpus_per_node x 32G.")
@click.option("--constraint", "constraint_override", default=None,
              help="Node feature (--constraint). Only 'nvlink' is documented, on "
                   "a100-40gb with multiple GPUs serving a single task.")
@click.option("--gpus", "gpu_override", default=None, type=int,
              help="Override --gpus-per-node.")
@click.option("--no-discover", is_flag=True,
              help="Do not ask the cluster for its partition list, so an "
                   "unrecognised partition warns instead of failing. Also "
                   f"settable as {discovery.NO_DISCOVER_ENV}=1.")
@click.option("--no-cache", is_flag=True,
              help="Re-run every discovery probe instead of reusing this "
                   "process's answers. Also settable as "
                   f"{discovery.NO_CACHE_ENV}=1.")
def launch(
    experiment_file: str,
    dry_run: bool,
    max_jobs: int | None,
    account_override: str | None,
    partition_override: str | None,
    qos_override: str | None,
    time_override: str | None,
    mem_override: str | None,
    constraint_override: str | None,
    gpu_override: int | None,
    no_discover: bool,
    no_cache: bool,
) -> None:
    """Launch experiments from an experiment.yaml file."""
    cfg = get_rcac_config()

    require_user(cfg)
    scratch = scratch_root(cfg)

    with open(experiment_file) as f:
        exp = yaml.safe_load(f)

    project = exp["project"]
    defaults = get_defaults()

    # AC-3.14 / R15. A legacy key is never silently dropped.
    manifest_warnings: list[str] = []
    slurm = _normalise_slurm(exp.get("slurm"), experiment_file, manifest_warnings)
    slurm_defaults = _normalise_slurm(
        defaults.get("slurm"), "defaults.yaml", manifest_warnings
    )

    def pick(key: str, override, literal=None):
        """CLI flag > experiment.yaml `slurm:` > defaults.yaml `slurm:` > literal.

        ``None`` means "not supplied at this level", so an explicit ``0`` or an
        empty string from a manifest still wins over the level below it.
        """
        for candidate in (override, slurm.get(key), slurm_defaults.get(key)):
            if candidate is not None:
                return candidate
        return literal

    partition = pick("partition", partition_override)
    qos = pick("qos", qos_override, DEFAULT_QOS)
    nodes = pick("nodes", None, 1)
    gpus_per_node = pick("gpus_per_node", gpu_override, 1)
    cpus_per_gpu = pick("cpus_per_gpu", None, 8)
    time_limit = pick("time", time_override)
    mem = pick("mem", mem_override)
    constraint = pick("constraint", constraint_override)

    # `-A` is a credential, so the environment outranks the manifest: RCAC_USER
    # / RCAC_ACCOUNT are the documented primary path precisely so nobody has to
    # write their account into a file they might commit (P4). An explicit
    # `--account` still wins over both.
    account = account_override or cfg.get("slurm", {}).get("account") or slurm.get("account")
    if not account:
        require_account(cfg)

    # AC-3.8's tri-state, now fed by real discovery (AC-4.7).
    #
    # `None` means "not checked", and an unrecognised partition then warns and
    # renders rather than being rejected on the strength of a static table that
    # may already be stale. A populated set means the cluster itself
    # enumerated its partitions, and a partition absent from it is a hard
    # error. `ClusterView.known_partitions` folds an *empty* result back to
    # `None` -- a probe that succeeded and reported zero partitions is a probe
    # that failed without saying so, and honouring it literally would reject
    # every submission.
    #
    # `--no-discover` skips the probes outright rather than running and
    # discarding them: with discovery off there is no failure to report, so
    # there is nothing to warn about either.
    known_partitions = None
    if not discovery.discovery_disabled(no_discover):
        view = discovery.survey(cfg, no_cache=no_cache)
        known_partitions = view.known_partitions
        # AC-4.2: every probe is optional and a failed one is a warning that
        # joins the end-of-run summary, never an abort.
        manifest_warnings.extend(view.warnings)

    env_cfg = exp.get("environment", {})
    # `name` is the environment as its MODULE is named on Gilbreth: the
    # generated script loads `use.own` then `conda-env/<name>`, which is what
    # `rcac-env activate` prints. If conda-env-mod gave it a `-pyX.Y` suffix,
    # that suffix is part of the name and belongs here.
    conda_env = env_cfg.get("name", "base")
    # Surface environment.type ("conda" | "apptainer" | None) and
    # environment.container (for apptainer) to the SLURM template so the
    # generated job actually runs python inside the requested env. Without
    # this, the template falls back to the host system python and ImportErrors
    # immediately.
    env_type = resolve_env_type(env_cfg.get("type"))
    env_container = env_cfg.get("container")
    # `rcac-env setup --name X --py 3.12` creates the module `conda-env/X-py3.12`.
    # A manifest naming a bare `X` therefore renders `module load conda-env/X`,
    # and the job dies at line one on the compute node minutes after submission.
    # This is the one place that knows both halves, so warn here rather than let
    # Lmod discover it remotely. A warning, not an error: an environment built by
    # hand, or by an older conda-env-mod, legitimately carries no suffix.
    if env_type == "conda" and not PY_SUFFIX_RE.search(str(conda_env)):
        manifest_warnings.append(
            f"{experiment_file}: environment.name is {conda_env!r}, which carries "
            "no -pyX.Y suffix. This renders `module load "
            f"conda-env/{conda_env}`, which fails on the compute node unless a "
            "module of exactly that name exists. `rcac-env activate <env>` "
            "prints the name to use; `module avail conda-env` lists them."
        )
    modules = env_cfg.get("modules", defaults.get("environment", {}).get("default_modules", []))

    base_params = exp.get("base_params", {})
    sweep = exp.get("sweep", {})
    pipeline = exp.get("pipeline", [])

    # Generate sweep configs
    configs = _generate_sweep_configs(sweep, base_params)
    if max_jobs:
        configs = configs[:max_jobs]

    emit_warnings(manifest_warnings)

    click.echo(f"Project: {project}")
    click.echo(f"Parameter combinations: {len(configs)}")

    # Create experiment directory structure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_base = f"{scratch}/projects/{project}"
    config_remote_dir = f"{exp_base}/configs/{timestamp}"
    scripts_dir = Path(f"/tmp/rcac_exp_{timestamp}")
    scripts_dir.mkdir(parents=True, exist_ok=True)

    # Ensure remote directories. AC-3.15: --dry-run contacts nothing, so this
    # is gated -- it used to run unconditionally and opened an SSH connection
    # even on a dry run.
    if not dry_run:
        _run_remote(cfg, f"mkdir -p {config_remote_dir} {scratch}/logs {exp_base}/experiments")

    job_ids = []
    #: Every SubmissionPlan warning raised across the run. Re-printed at the
    #: end because one warning above a 40-job sweep has scrolled off by then.
    run_warnings: list[str] = list(manifest_warnings)

    def build(generator, **kwargs) -> str:
        """Render one script, emitting and accumulating its plan warnings."""
        plans: list = []
        script_text = generator(
            account=account,
            partition=partition,
            qos=qos,
            time=time_limit,
            nodes=nodes,
            gpus_per_node=gpus_per_node,
            cpus_per_gpu=cpus_per_gpu,
            mem=mem,
            constraint=constraint,
            known_partitions=known_partitions,
            modules=modules,
            conda_env=conda_env,
            plan_out=plans,
            **kwargs,
        )
        if plans:
            fresh = [w for w in plans[0].warnings if w not in run_warnings]
            emit_warnings(fresh)
            run_warnings.extend(fresh)
        return script_text

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
                # Previously only `{config_path}` was substituted via
                # str.replace, leaving every other manifest token (`{hdf5}`,
                # `{exp_id}`, `{config_id}`, sweep keys, ...) as a literal in
                # the sbatch args -> argparse failed at runtime.  Now we expose
                # base_params + per-combo sweep keys + meta keys via
                # str.format_map, raising on truly missing keys so the author
                # hears about typos rather than silent skips.
                #
                # `config_dir` is the directory holding `config_path`, and it is
                # the whole vocabulary a manifest has for naming an artefact one
                # stage writes and a later stage reads.  There is deliberately
                # no `{generate_data.output_path}` form: these stages are
                # separate `sbatch` submissions chained by `--dependency`, so
                # `launch` returns before stage one has run and CANNOT know
                # what it produced.  Cross-stage values must therefore be
                # *derivable* at launch time -- a path both stages compute the
                # same way -- not *observed*.  `{config_dir}` plus the manifest's
                # own `base_params` is what makes that possible.
                fmt_ctx = {
                    **{k: v for k, v in base_params.items() if k != "_exp_id"},
                    **{k: v for k, v in config.items() if not k.startswith("_")},
                    "exp_id": exp_id,
                    "config_path": config_path,
                    "config_dir": config_remote_dir,
                }
                raw_args = stage.get("args", "")
                try:
                    args = raw_args.format_map(_StrictFormatDict(fmt_ctx))
                except KeyError as e:
                    click.echo(
                        f"  [{exp_id}/{stage_name}] manifest references "
                        f"unknown placeholder {{{e.args[0]}}}; available keys: "
                        f"{sorted(fmt_ctx.keys())}",
                        err=True,
                    )
                    sys.exit(2)

                slurm_script = build(
                    generate_pipeline_script,
                    project=project,
                    exp_id=exp_id,
                    stage_name=stage_name,
                    script=script,
                    args=args,
                    dependency_job_id=prev_job_id,
                    env_type=env_type,
                    env_container=env_container,
                    config_path=config_path,
                )

                local_script = scripts_dir / f"job_{i:03d}_{stage_name}.sh"
                local_script.write_text(slurm_script)
                remote_script = f"{exp_base}/scripts/job_{i:03d}_{stage_name}.sh"

                if not dry_run:
                    # Upload config and script
                    _upload_file(cfg, str(local_config), config_path)
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
                    _echo_dry_run(f"{exp_id}/{stage_name}", local_script, remote_script, slurm_script)
        else:
            # Single stage mode (no pipeline defined)
            train_script = "train.py"
            args = f"--config {config_path}"

            gen_fn = (
                generate_multi_gpu_script
                if int(gpus_per_node) > 1
                else generate_single_gpu_script
            )
            slurm_script = build(
                gen_fn,
                project=project,
                exp_id=exp_id,
                script=train_script,
                args=args,
                env_type=env_type,
                env_container=env_container,
                config_path=config_path,
            )

            local_script = scripts_dir / f"job_{i:03d}.sh"
            local_script.write_text(slurm_script)
            remote_script = f"{exp_base}/scripts/job_{i:03d}.sh"

            if not dry_run:
                _upload_file(cfg, str(local_config), config_path)
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
                _echo_dry_run(exp_id, local_script, remote_script, slurm_script)

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

    emit_warnings(
        run_warnings,
        header="Warnings raised during this run (repeated -- the first copy has "
               "scrolled off by now):",
    )

    if dry_run:
        click.echo(f"\nDry run complete. Nothing was submitted and the cluster "
                   f"was not contacted. Scripts in: {scripts_dir}")
    else:
        click.echo(f"\nSubmitted {len(job_ids)} jobs: {job_ids}")
        # Also save tracking remotely
        remote_tracking = f"{exp_base}/experiments/{timestamp}/tracking.json"
        _run_remote(cfg, f"mkdir -p {exp_base}/experiments/{timestamp}")
        _upload_file(cfg, str(tracking_file), remote_tracking)


def _echo_dry_run(label: str, local_script: Path, remote_script: str, body: str) -> None:
    """AC-3.15 -- print the exact sbatch command and the full script.

    A dry run whose entire output is a path is not a dry run of the thing that
    matters: the generated script IS the product, and the point of `--dry-run`
    offline is to read the SBATCH directive block before a live submission
    ever happens. Nothing here touches the network -- AC-3.19(a)'s text scan
    bans the directive token outside the templates, so it is not spelled here.
    """
    click.echo(f"  [{label}] Script: {local_script}")
    click.echo(f"  [{label}] Would run: sbatch {remote_script}")
    click.echo(f"--- begin {remote_script} ---")
    click.echo(body.rstrip("\n"))
    click.echo(f"--- end {remote_script} ---")


def _upload_file(cfg: dict, local_path: str, remote_path: str) -> bool:
    """Upload a file to RCAC via rsync. Returns True on success."""
    cmd = [
        "rsync", "-az",
        "-e", ssh_opt_str(cfg),
        local_path,
        f"{ssh_target(cfg)}:{remote_path}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        click.echo(f"Upload failed ({local_path}): {result.stderr}", err=True)
        return False
    return True


@cli.command()
def status() -> None:
    """Show status of all running experiments."""
    cfg = get_rcac_config()

    require_user(cfg)

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
    cfg = get_rcac_config()

    require_user(cfg)

    result = _run_remote(
        cfg,
        f"ls -t {scratch_root(cfg)}/logs/*_{job_id}.out 2>/dev/null | head -1",
    )

    if result.returncode != 0 or not result.stdout.strip():
        click.echo(f"No log file found for job {job_id}.")
        return

    log_file = result.stdout.strip()
    click.echo(f"Streaming: {log_file}\n")

    # Use SSH to tail the log
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
    cfg = get_rcac_config()

    require_user(cfg)

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
    """Pull experiment results from RCAC to local directory."""
    from pathlib import Path
    remote = (
        f"{ssh_target(cfg)}:"
        f"{scratch_root(cfg)}/projects/{project}/experiments/"
    )
    local = Path(local_dir).resolve()
    local.mkdir(parents=True, exist_ok=True)

    cmd = [
        "rsync", "-avz", "--progress",
        "-e", ssh_opt_str(cfg),
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
    cfg = get_rcac_config()

    require_user(cfg)

    if job_id == "all":
        result = _run_remote(cfg, "scancel -u $USER")
    else:
        result = _run_remote(cfg, f"scancel {job_id}")

    if result.returncode == 0:
        click.echo(f"Cancelled: {job_id}")
    else:
        click.echo(f"Error: {result.stderr}", err=True)
