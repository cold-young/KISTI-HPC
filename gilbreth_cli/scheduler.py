"""SLURM script generation for Gilbreth (single Jinja2 template, several modes)."""

from __future__ import annotations

import shlex
from pathlib import Path

from jinja2 import Environment, FileSystemLoader


def template_dir() -> Path:
    return Path(__file__).parent.parent / "templates" / "gilbreth"


def _env() -> Environment:
    env = Environment(
        loader=FileSystemLoader(str(template_dir())),
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["shell_quote"] = shlex.quote
    return env


def resolve_qos(partition: str, qos: str | None) -> str:
    """Gilbreth: the ``training`` partition only accepts ``-q training``."""
    if partition == "training":
        return "training"
    return qos or "normal"


def resolve_env_ref(cfg: dict, env_type: str, name: str | None) -> str:
    """Translate ``environment.name`` into the argument the template needs.

    conda: an absolute prefix, or ``<paths.envs>/<name>`` for a bare name.
    venv : an absolute path, or ``<paths.envs>/<name>``.
    conda-env-mod: the module suffix printed by ``conda-env-mod create`` (e.g. ``my_env-py3.12.3``).
    """
    name = name or ""
    if env_type in ("conda", "venv") and name and not name.startswith("/"):
        return f"{cfg['paths']['envs']}/{name}"
    return name


def render_job(
    cfg: dict,
    *,
    project: str,
    exp_id: str,
    job_name: str | None = None,
    script: str = "",
    args: str = "",
    command: str = "",
    account: str,
    partition: str,
    qos: str | None = None,
    nodes: int = 1,
    gpus: int = 1,
    cpus_per_gpu: int = 8,
    mem: str = "64G",
    time: str = "04:00:00",
    constraint: str = "",
    env_type: str = "none",
    env_name: str = "",
    container: str = "",
    modules: list[str] | None = None,
    launcher: str = "python",
    env_vars: dict | None = None,
    config_path: str = "",
    config_dir: str = "",
    stage_name: str = "",
    dependency_job_id: str | None = None,
    array_max: int | None = None,
    array_parallel: int | None = None,
    workdir: str | None = None,
) -> str:
    """Render a complete sbatch script for Gilbreth."""
    if not script and not command:
        raise ValueError("render_job needs either script= or command=")
    if gpus < 1:
        raise ValueError("Gilbreth jobs must request at least one GPU (--gpus-per-node)")
    if launcher not in ("python", "torchrun"):
        raise ValueError(f"unknown launcher: {launcher}")
    if env_type not in ("conda", "conda-env-mod", "apptainer", "venv", "none"):
        raise ValueError(f"unknown environment.type: {env_type}")
    if env_type == "apptainer" and not container:
        raise ValueError("environment.type=apptainer requires environment.container")
    if env_type in ("conda", "conda-env-mod", "venv") and not env_name:
        raise ValueError(f"environment.type={env_type} requires environment.name")

    paths = cfg["paths"]
    env_cfg = cfg.get("environment", {})
    merged_env_vars = dict(env_cfg.get("env_vars", {}) or {})
    merged_env_vars.update(env_vars or {})

    template = _env().get_template("job.sh.j2")
    return template.render(
        job_name=job_name or f"{project}_{exp_id}",
        project=project,
        exp_id=exp_id,
        script=script,
        args=args,
        command=command,
        account=account,
        partition=partition,
        qos=resolve_qos(partition, qos),
        nodes=nodes,
        gpus=gpus,
        cpus_per_gpu=cpus_per_gpu,
        mem=mem,
        time=time,
        constraint=constraint or "",
        env_type=env_type,
        env_ref=resolve_env_ref(cfg, env_type, env_name),
        conda_module=env_cfg.get("conda_module", "conda"),
        base_module=env_cfg.get("base_module", "rcac"),
        container=container,
        modules=modules if modules is not None else env_cfg.get("modules", []),
        launcher=launcher,
        env_vars=merged_env_vars,
        config_path=config_path,
        config_dir=config_dir,
        stage_name=stage_name,
        dependency_job_id=dependency_job_id,
        array_max=array_max,
        array_parallel=array_parallel,
        workdir=workdir or f"{paths['projects']}/{project}",
        logs_dir=paths["logs"],
        scratch=paths["scratch"],
        depot=paths.get("depot") or "",
        mail_user=cfg["slurm"].get("mail_user") or "",
        mail_type=cfg["slurm"].get("mail_type", "END,FAIL"),
    )
