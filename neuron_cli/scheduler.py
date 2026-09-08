"""SLURM script generator using Jinja2 templates."""

from __future__ import annotations

import os
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined


def get_template_dir() -> Path:
    """Return the templates directory path."""
    return Path(__file__).parent.parent / "templates"


def render_slurm_script(
    template_name: str,
    template_dir: str | Path | None = None,
    config: dict | None = None,
    **kwargs,
) -> str:
    """Render a SLURM script from a Jinja2 template.

    ``template_dir`` and ``config`` are injection points for tests: when either
    is omitted it falls back to the on-disk default, so default behaviour is
    unchanged.
    """
    # Inject user if not provided
    if "user" not in kwargs:
        if config is None:
            from .config import get_neuron_config
            config = get_neuron_config()
        kwargs["user"] = config["ssh"]["user"]
    if template_dir is None:
        template_dir = get_template_dir()
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        keep_trailing_newline=True,
        undefined=StrictUndefined,
    )
    template = env.get_template(template_name)
    return template.render(**kwargs)


def generate_single_gpu_script(
    project: str,
    exp_id: str,
    script: str,
    args: str = "",
    partition: str = "amd_a100nv_8",
    nodes: int = 1,
    gpus: int = 1,
    cpus_per_gpu: int = 8,
    time: str = "24:00:00",
    comment: str = "pytorch",
    modules: list[str] | None = None,
    conda_env: str = "base",
    env_type: str | None = None,
    env_container: str | None = None,
    config_path: str = "",
    template_dir: str | Path | None = None,
    config: dict | None = None,
) -> str:
    """Generate a single-GPU SLURM script."""
    return render_slurm_script(
        "single_gpu.sh.j2",
        template_dir=template_dir,
        config=config,
        project=project,
        exp_id=exp_id,
        script=script,
        args=args,
        partition=partition,
        nodes=nodes,
        gpus=gpus,
        cpus_per_gpu=cpus_per_gpu,
        time=time,
        comment=comment,
        modules=modules or ["cuda/12.4.1"],
        conda_env=conda_env,
        env_type=env_type,
        env_container=env_container,
        config_path=config_path,
    )


def generate_multi_gpu_script(
    project: str,
    exp_id: str,
    script: str,
    args: str = "",
    partition: str = "amd_a100nv_8",
    nodes: int = 1,
    gpus: int = 2,
    cpus_per_gpu: int = 8,
    time: str = "24:00:00",
    comment: str = "pytorch",
    modules: list[str] | None = None,
    conda_env: str = "base",
    env_type: str | None = None,
    env_container: str | None = None,
    config_path: str = "",
    template_dir: str | Path | None = None,
    config: dict | None = None,
) -> str:
    """Generate a multi-GPU distributed training SLURM script."""
    return render_slurm_script(
        "multi_gpu.sh.j2",
        template_dir=template_dir,
        config=config,
        project=project,
        exp_id=exp_id,
        script=script,
        args=args,
        partition=partition,
        nodes=nodes,
        gpus=gpus,
        cpus_per_gpu=cpus_per_gpu,
        time=time,
        comment=comment,
        modules=modules or ["cuda/12.4.1"],
        conda_env=conda_env,
        env_type=env_type,
        env_container=env_container,
        config_path=config_path,
    )


def generate_job_array_script(
    project: str,
    script: str,
    config_dir: str,
    array_max: int,
    args: str = "",
    array_simultaneous: int | None = None,
    partition: str = "amd_a100nv_8",
    nodes: int = 1,
    gpus: int = 1,
    cpus_per_gpu: int = 8,
    time: str = "24:00:00",
    comment: str = "pytorch",
    modules: list[str] | None = None,
    conda_env: str = "base",
    template_dir: str | Path | None = None,
    config: dict | None = None,
) -> str:
    """Generate a SLURM job array script for parameter sweeps."""
    return render_slurm_script(
        "job_array.sh.j2",
        template_dir=template_dir,
        config=config,
        project=project,
        script=script,
        config_dir=config_dir,
        array_max=array_max,
        args=args,
        array_simultaneous=array_simultaneous,
        partition=partition,
        nodes=nodes,
        gpus=gpus,
        cpus_per_gpu=cpus_per_gpu,
        time=time,
        comment=comment,
        modules=modules or ["cuda/12.4.1"],
        conda_env=conda_env,
    )


def generate_pipeline_script(
    project: str,
    exp_id: str,
    stage_name: str,
    script: str,
    args: str = "",
    dependency_job_id: str | None = None,
    partition: str = "amd_a100nv_8",
    nodes: int = 1,
    gpus: int = 1,
    cpus_per_gpu: int = 8,
    time: str = "24:00:00",
    comment: str = "pytorch",
    modules: list[str] | None = None,
    conda_env: str = "base",
    env_type: str | None = None,
    env_container: str | None = None,
    config_path: str = "",
    template_dir: str | Path | None = None,
    config: dict | None = None,
) -> str:
    """Generate a pipeline stage SLURM script with optional dependency."""
    return render_slurm_script(
        "pipeline.sh.j2",
        template_dir=template_dir,
        config=config,
        project=project,
        exp_id=exp_id,
        stage_name=stage_name,
        script=script,
        args=args,
        dependency_job_id=dependency_job_id,
        partition=partition,
        nodes=nodes,
        gpus=gpus,
        cpus_per_gpu=cpus_per_gpu,
        time=time,
        comment=comment,
        modules=modules or ["cuda/12.4.1"],
        conda_env=conda_env,
        env_type=env_type,
        env_container=env_container,
        config_path=config_path,
    )
