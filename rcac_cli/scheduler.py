"""SLURM script generation for Gilbreth.

**Every generator here renders from a
:class:`rcac_cli.gilbreth.SubmissionPlan`.**  That is the whole point of this
module's shape: :func:`rcac_cli.gilbreth.build_directives` is the choke point,
and the ``generate_*_script`` functions call it themselves rather than
accepting pre-validated directives from a caller.  A caller therefore cannot
bypass validation without bypassing this module entirely -- which is what
AC-3.19's two assertions look for.

C3c added the fifth generator, :func:`generate_keepalive_script`.  It is here
and not in ``keepalive.py`` so that the directive token lives only in
``templates/``: after C3c no ``.py`` file in the package spells it at all,
which is the form AC-3.19(a)'s text scan takes.

The signatures changed materially at C3b and are deliberately *not* backwards
compatible:

* ``account`` (``-A``), ``partition`` (``-p``) and ``time`` are **required**
  and carry no defaults.  The pre-port ``partition="amd_a100nv_8"`` default
  named the previous cluster's partition; keeping any default here would mean
  silently billing a job to, or running it on, something the user did not
  choose.  ``build_directives`` raises with an actionable message instead.
* ``comment`` is gone.  ``--comment`` carried the previous cluster's job class.
* ``gpus`` became ``gpus_per_node``; Gilbreth's documented GPU request form is
  ``--gpus-per-node=N``, never ``--gres``.
* ``modules`` now defaults to ``[]``, not ``["cuda/12.4.1"]``.  That module
  name was the previous cluster's and emitting it here would put a
  ``module load`` of a possibly non-existent module at the top of every job.
  RCAC's own guidance is explicit: do not invent module names.
* ``job_name`` is a parameter on all four generators (AC-3.18), composed in
  Python rather than in Jinja so the three distinct compositions stay distinct.

Paths come from ``paths.scratch`` via :func:`rcac_cli.config.scratch_root`
(B9/AC-3.12).  No template writes a scratch path literal any more, so changing
``paths.scratch`` changes every rendered path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from .config import get_rcac_config, scratch_root
from .gilbreth import DEFAULT_QOS, SubmissionPlan, build_directives


def get_template_dir() -> Path:
    """Return the packaged templates directory.

    Templates live *inside* the package (``rcac_cli/templates/``) so they ship
    in the wheel; before this they sat next to it in the source tree and were
    silently excluded from every install (B3/R2).
    """
    return Path(__file__).resolve().parent / "templates"


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

    ``user`` and ``scratch_root`` are injected from the config when the caller
    does not supply them.  ``scratch_root`` goes through
    :func:`rcac_cli.config.scratch_root`, which raises rather than formatting a
    literal ``{user}`` or a ``//`` into a rendered ``-o`` path (AC-3.17).
    AC-3.19(a) bans the directive token itself outside ``templates/`` and
    ``gilbreth.py``, so it is not spelled here.
    """
    loaded: dict | None = config
    if "user" not in kwargs or "scratch_root" not in kwargs:
        if loaded is None:
            loaded = get_rcac_config()
    if "user" not in kwargs:
        kwargs["user"] = loaded["ssh"]["user"]
    if "scratch_root" not in kwargs:
        kwargs["scratch_root"] = scratch_root(loaded)
    if template_dir is None:
        template_dir = get_template_dir()
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        keep_trailing_newline=True,
        undefined=StrictUndefined,
    )
    template = env.get_template(template_name)
    return template.render(**kwargs)


def _plan_context(plan: SubmissionPlan, *, job_name: str, cpus_per_gpu: int) -> dict:
    """The template variables the shared ``_sbatch_header.j2`` consumes.

    Everything the header emits comes from *plan* except two values:

    ``job_name``
        AC-3.18.  The three compositions (``{project}_{exp_id}``,
        ``{project}_sweep``, ``{project}_{stage_name}_{exp_id}``) are a
        presentation concern, not a submission rule, so they are composed by
        the four generators below and passed in.

    ``cpus_per_gpu``
        Deliberately absent from :class:`SubmissionPlan`: no documented
        Gilbreth rule constrains it, and putting an unconstrained value on the
        plan would suggest ``build_directives`` had checked it.  It is threaded
        separately for exactly that reason.

    The directive values come from :meth:`SubmissionPlan.to_slurm_dict`, not
    from a dict literal spelled here.  That is C3c's doing and it is what makes
    AC-3.19(b) exact rather than allowlisted: the header context and the
    ``slurm:`` mapping ``simulator.py`` writes into generated YAML are now the
    *same* mapping, so "no module outside ``gilbreth.py`` builds a dict literal
    carrying a ``partition`` or ``qos`` key" holds with zero exemptions.

    ``constraint`` is restored when :meth:`to_slurm_dict` omitted it.  The
    header evaluates ``{% if constraint %}`` on every render and ``jinja2``
    runs under ``StrictUndefined``, so absent is not the same as ``None`` here
    -- absent raises.  Suppression must reach the template as a value.
    """
    context = plan.to_slurm_dict()
    context.setdefault("constraint", None)
    context["job_name"] = job_name
    context["cpus_per_gpu"] = cpus_per_gpu
    return context


def _build(
    *,
    account: str,
    partition: str,
    time: str,
    qos: str,
    gpus_per_node: int,
    nodes: int,
    mem: Optional[str],
    constraint: Optional[str],
    known_partitions,
) -> SubmissionPlan:
    """The single call into the choke point, shared by all four generators."""
    return build_directives(
        account=account,
        partition=partition,
        time=time,
        qos=qos,
        gpus_per_node=gpus_per_node,
        nodes=nodes,
        mem=mem,
        constraint=constraint,
        known_partitions=known_partitions,
    )


def generate_single_gpu_script(
    project: str,
    exp_id: str,
    script: str,
    *,
    account: str,
    partition: str,
    time: str,
    args: str = "",
    qos: str = DEFAULT_QOS,
    nodes: int = 1,
    gpus_per_node: int = 1,
    cpus_per_gpu: int = 8,
    mem: str | None = None,
    constraint: str | None = None,
    known_partitions: Sequence[str] | None = None,
    job_name: str | None = None,
    modules: list[str] | None = None,
    conda_env: str = "base",
    env_type: str | None = None,
    env_container: str | None = None,
    config_path: str = "",
    template_dir: str | Path | None = None,
    config: dict | None = None,
    plan_out: list | None = None,
) -> str:
    """Generate a single-GPU SLURM script.

    ``plan_out``, when given, receives the :class:`SubmissionPlan` so the caller
    can read ``plan.warnings`` without re-running validation.
    """
    plan = _build(
        account=account,
        partition=partition,
        time=time,
        qos=qos,
        gpus_per_node=gpus_per_node,
        nodes=nodes,
        mem=mem,
        constraint=constraint,
        known_partitions=known_partitions,
    )
    if plan_out is not None:
        plan_out.append(plan)
    return render_slurm_script(
        "single_gpu.sh.j2",
        template_dir=template_dir,
        config=config,
        project=project,
        exp_id=exp_id,
        script=script,
        args=args,
        modules=modules or [],
        conda_env=conda_env,
        env_type=env_type,
        env_container=env_container,
        config_path=config_path,
        **_plan_context(
            plan,
            job_name=job_name or f"{project}_{exp_id}",
            cpus_per_gpu=cpus_per_gpu,
        ),
    )


def generate_multi_gpu_script(
    project: str,
    exp_id: str,
    script: str,
    *,
    account: str,
    partition: str,
    time: str,
    args: str = "",
    qos: str = DEFAULT_QOS,
    nodes: int = 1,
    gpus_per_node: int = 2,
    cpus_per_gpu: int = 8,
    mem: str | None = None,
    constraint: str | None = None,
    known_partitions: Sequence[str] | None = None,
    job_name: str | None = None,
    modules: list[str] | None = None,
    conda_env: str = "base",
    env_type: str | None = None,
    env_container: str | None = None,
    config_path: str = "",
    template_dir: str | Path | None = None,
    config: dict | None = None,
    plan_out: list | None = None,
) -> str:
    """Generate a multi-GPU distributed training SLURM script."""
    plan = _build(
        account=account,
        partition=partition,
        time=time,
        qos=qos,
        gpus_per_node=gpus_per_node,
        nodes=nodes,
        mem=mem,
        constraint=constraint,
        known_partitions=known_partitions,
    )
    if plan_out is not None:
        plan_out.append(plan)
    return render_slurm_script(
        "multi_gpu.sh.j2",
        template_dir=template_dir,
        config=config,
        project=project,
        exp_id=exp_id,
        script=script,
        args=args,
        modules=modules or [],
        conda_env=conda_env,
        env_type=env_type,
        env_container=env_container,
        config_path=config_path,
        **_plan_context(
            plan,
            job_name=job_name or f"{project}_{exp_id}",
            cpus_per_gpu=cpus_per_gpu,
        ),
    )


def generate_job_array_script(
    project: str,
    script: str,
    config_dir: str,
    array_max: int,
    *,
    account: str,
    partition: str,
    time: str,
    args: str = "",
    array_simultaneous: int | None = None,
    qos: str = DEFAULT_QOS,
    nodes: int = 1,
    gpus_per_node: int = 1,
    cpus_per_gpu: int = 8,
    mem: str | None = None,
    constraint: str | None = None,
    known_partitions: Sequence[str] | None = None,
    job_name: str | None = None,
    modules: list[str] | None = None,
    conda_env: str = "base",
    template_dir: str | Path | None = None,
    config: dict | None = None,
    plan_out: list | None = None,
) -> str:
    """Generate a SLURM job array script for parameter sweeps.

    Not currently reachable from any CLI command: ``rcac-exp launch`` submits a
    sweep as N independent jobs rather than as one array.  It is kept because
    the template is real and tested; wiring ``rcac-exp sweep --array`` to it is
    a follow-up, not dead code to delete blind.
    """
    plan = _build(
        account=account,
        partition=partition,
        time=time,
        qos=qos,
        gpus_per_node=gpus_per_node,
        nodes=nodes,
        mem=mem,
        constraint=constraint,
        known_partitions=known_partitions,
    )
    if plan_out is not None:
        plan_out.append(plan)
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
        modules=modules or [],
        conda_env=conda_env,
        **_plan_context(
            plan,
            job_name=job_name or f"{project}_sweep",
            cpus_per_gpu=cpus_per_gpu,
        ),
    )


def generate_pipeline_script(
    project: str,
    exp_id: str,
    stage_name: str,
    script: str,
    *,
    account: str,
    partition: str,
    time: str,
    args: str = "",
    dependency_job_id: str | None = None,
    qos: str = DEFAULT_QOS,
    nodes: int = 1,
    gpus_per_node: int = 1,
    cpus_per_gpu: int = 8,
    mem: str | None = None,
    constraint: str | None = None,
    known_partitions: Sequence[str] | None = None,
    job_name: str | None = None,
    modules: list[str] | None = None,
    conda_env: str = "base",
    env_type: str | None = None,
    env_container: str | None = None,
    config_path: str = "",
    template_dir: str | Path | None = None,
    config: dict | None = None,
    plan_out: list | None = None,
) -> str:
    """Generate a pipeline stage SLURM script with optional dependency."""
    plan = _build(
        account=account,
        partition=partition,
        time=time,
        qos=qos,
        gpus_per_node=gpus_per_node,
        nodes=nodes,
        mem=mem,
        constraint=constraint,
        known_partitions=known_partitions,
    )
    if plan_out is not None:
        plan_out.append(plan)
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
        modules=modules or [],
        conda_env=conda_env,
        env_type=env_type,
        env_container=env_container,
        config_path=config_path,
        **_plan_context(
            plan,
            job_name=job_name or f"{project}_{stage_name}_{exp_id}",
            cpus_per_gpu=cpus_per_gpu,
        ),
    )


def generate_keepalive_script(
    keepalive_script: str,
    *,
    account: str,
    partition: str,
    time: str,
    qos: str = DEFAULT_QOS,
    nodes: int = 1,
    gpus_per_node: int = 1,
    cpus_per_gpu: int = 1,
    mem: str | None = None,
    constraint: str | None = None,
    known_partitions: Sequence[str] | None = None,
    job_name: str | None = None,
    template_dir: str | Path | None = None,
    config: dict | None = None,
    plan_out: list | None = None,
) -> str:
    """Generate the batch script for ``rcac-keepalive setup-cron`` (AC-6.4).

    The fifth generator, and the reason C3c exists.  ``keepalive.py`` built
    this header by hand from an f-string, which made it the one place in the
    package where a submission reached a cluster without passing
    :func:`rcac_cli.gilbreth.build_directives` -- and it showed: the header it
    built named the previous cluster's partition and a job class Gilbreth does
    not have.

    It lives here rather than in ``keepalive.py`` for the same reason the other
    four do: the choke point is called through :func:`_build`, once, and
    ``keepalive.py`` is left holding a *policy* (which QOS a file-touching job
    deserves) rather than a header.

    ``keepalive_script`` is the remote path of the shell script the job runs.
    """
    plan = _build(
        account=account,
        partition=partition,
        time=time,
        qos=qos,
        gpus_per_node=gpus_per_node,
        nodes=nodes,
        mem=mem,
        constraint=constraint,
        known_partitions=known_partitions,
    )
    if plan_out is not None:
        plan_out.append(plan)
    return render_slurm_script(
        "keepalive.sh.j2",
        template_dir=template_dir,
        config=config,
        keepalive_script=keepalive_script,
        **_plan_context(
            plan,
            job_name=job_name or "rcac_keepalive",
            cpus_per_gpu=cpus_per_gpu,
        ),
    )
