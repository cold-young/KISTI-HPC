"""SLURM script generator using Jinja2 templates."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader


def get_template_dir() -> Path:
    """Return the templates directory path."""
    return Path(__file__).parent.parent / "templates"


def render_slurm_script(template_name: str, **kwargs) -> str:
    """Render a SLURM script from a Jinja2 template."""
    env = Environment(
        loader=FileSystemLoader(str(get_template_dir())),
        keep_trailing_newline=True,
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
    config_path: str = "",
) -> str:
    """Generate a single-GPU SLURM script."""
    return render_slurm_script(
        "single_gpu.sh.j2",
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
    config_path: str = "",
) -> str:
    """Generate a multi-GPU distributed training SLURM script."""
    return render_slurm_script(
        "multi_gpu.sh.j2",
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
) -> str:
    """Generate a SLURM job array script for parameter sweeps."""
    return render_slurm_script(
        "job_array.sh.j2",
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
    config_path: str = "",
) -> str:
    """Generate a pipeline stage SLURM script with optional dependency."""
    return render_slurm_script(
        "pipeline.sh.j2",
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
        config_path=config_path,
    )
