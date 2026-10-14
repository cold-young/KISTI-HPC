"""Experiment results collection, comparison, and ranking."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import click

from .config import get_rcac_config, require_user, scratch_root
from .ssh import ssh_cmd, ssh_opt_str, ssh_target


def _run_remote(cfg: dict, command: str) -> subprocess.CompletedProcess:
    cmd = [*ssh_cmd(cfg), command]
    return subprocess.run(cmd, capture_output=True, text=True)


def _fetch_experiment_results(cfg: dict, project: str) -> list[dict]:
    """Fetch all experiment results from RCAC."""
    exp_dir = f"{scratch_root(cfg)}/projects/{project}/experiments"

    # Step 1: Find all result files
    find_cmd = f"find {exp_dir} -name 'eval_results.json' 2>/dev/null"
    find_result = _run_remote(cfg, find_cmd)
    if find_result.returncode != 0 or not find_result.stdout.strip():
        return []

    # Step 2: Read all files in one cat call
    files = [f for f in find_result.stdout.strip().split("\n") if f.strip()]
    if not files:
        return []
    cat_cmd = "; ".join(f"echo; echo __FILE__:{f}; cat {f}" for f in files)
    result = _run_remote(cfg, cat_cmd)
    if result.returncode != 0 or not result.stdout.strip():
        return []

    results = []
    current_file = None
    current_content = []

    def _process_block(filepath: str, lines: list[str]) -> None:
        text = "\n".join(lines).strip()
        if not text:
            return
        try:
            data = json.loads(text)
            data["_source_file"] = filepath
            parts = filepath.split("/")
            for i, p in enumerate(parts):
                if p == "experiments" and i + 1 < len(parts):
                    data["_exp_id"] = parts[i + 1]
                    break
            results.append(data)
        except json.JSONDecodeError:
            pass

    for line in result.stdout.split("\n"):
        if line.startswith("__FILE__:"):
            if current_file:
                _process_block(current_file, current_content)
            current_file = line[len("__FILE__:"):]
            current_content = []
        else:
            current_content.append(line)

    if current_file:
        _process_block(current_file, current_content)

    return results


@click.group()
def cli() -> None:
    """View and compare experiment results."""


@cli.command()
@click.argument("project")
@click.option("--sort", "-s", default=None, help="Sort by metric name (descending)")
@click.option("--limit", "-l", default=20, help="Max results to show")
def results(project: str, sort: str | None, limit: int) -> None:
    """Show all experiment results for a project."""
    cfg = get_rcac_config()

    require_user(cfg)

    data = _fetch_experiment_results(cfg, project)
    if not data:
        click.echo("No results found.")
        return

    if sort:
        data.sort(key=lambda x: x.get(sort, 0), reverse=True)

    # Determine columns from first result
    skip_keys = {"_source_file", "_exp_id"}
    sample = data[0]
    columns = ["_exp_id"] + [k for k in sample if k not in skip_keys and not k.startswith("_")]

    # Print header
    header = " | ".join(f"{c:<16}" for c in columns)
    click.echo(header)
    click.echo("-" * len(header))

    for row in data[:limit]:
        values = []
        for c in columns:
            v = row.get(c, "N/A")
            if isinstance(v, float):
                values.append(f"{v:<16.4f}")
            else:
                values.append(f"{str(v):<16}")
        click.echo(" | ".join(values))


@cli.command()
@click.argument("project")
@click.argument("exp_ids", nargs=-1, required=True)
def compare(project: str, exp_ids: tuple[str, ...]) -> None:
    """Compare specific experiments side by side."""
    cfg = get_rcac_config()

    require_user(cfg)

    data = _fetch_experiment_results(cfg, project)
    if not data:
        click.echo("No results found.")
        return

    # Filter to requested experiments
    filtered = [d for d in data if d.get("_exp_id") in exp_ids]
    if not filtered:
        click.echo(f"No matching experiments found for: {exp_ids}")
        return

    # Collect all metric keys
    all_keys = set()
    for d in filtered:
        all_keys.update(k for k in d if not k.startswith("_"))

    click.echo(f"{'Metric':<20}" + " | ".join(f"{d.get('_exp_id', '?'):<20}" for d in filtered))
    click.echo("-" * (20 + 23 * len(filtered)))

    for key in sorted(all_keys):
        values = []
        for d in filtered:
            v = d.get(key, "N/A")
            if isinstance(v, float):
                values.append(f"{v:<20.4f}")
            else:
                values.append(f"{str(v):<20}")
        click.echo(f"{key:<20}" + " | ".join(values))


@cli.command()
@click.argument("project")
@click.option("--metric", "-m", default="accuracy", help="Metric to rank by")
@click.option("--top", "-t", default=5, help="Number of top results to show")
def best(project: str, metric: str, top: int) -> None:
    """Show top-performing experiments."""
    cfg = get_rcac_config()

    require_user(cfg)

    data = _fetch_experiment_results(cfg, project)
    if not data:
        click.echo("No results found.")
        return

    # Filter to results that have the metric
    with_metric = [d for d in data if metric in d]
    if not with_metric:
        click.echo(f"No results with metric '{metric}' found.")
        return

    with_metric.sort(key=lambda x: x[metric], reverse=True)

    click.echo(f"Top {top} by {metric}:\n")
    for i, d in enumerate(with_metric[:top], 1):
        exp_id = d.get("_exp_id", "unknown")
        val = d[metric]
        click.echo(f"  {i}. {exp_id}: {metric}={val:.4f}" if isinstance(val, float) else f"  {i}. {exp_id}: {metric}={val}")

        # Show other metrics too
        for k, v in d.items():
            if k != metric and not k.startswith("_"):
                if isinstance(v, float):
                    click.echo(f"     {k}: {v:.4f}")
                elif isinstance(v, (int, str)):
                    click.echo(f"     {k}: {v}")
        click.echo()


@cli.command("sync-wandb")
@click.argument("project")
@click.option("--exp-id", default=None, help="Specific experiment ID")
def sync_wandb(project: str, exp_id: str | None) -> None:
    """Download W&B offline runs from RCAC and sync locally."""
    cfg = get_rcac_config()

    require_user(cfg)

    remote_wandb = f"{scratch_root(cfg)}/projects/{project}/wandb"

    if exp_id:
        remote_wandb = f"{remote_wandb}/*{exp_id}*"

    local_wandb = Path(f"./wandb_sync/{project}")
    local_wandb.mkdir(parents=True, exist_ok=True)

    click.echo("Downloading W&B runs from RCAC...")
    # B4/AC-2.9: without -e this rsync opens its own SSH connection and
    # costs the user a second MFA prompt. Route it through the ControlMaster.
    cmd = [
        "rsync", "-avz", "--progress",
        "-e", ssh_opt_str(cfg),
        f"{ssh_target(cfg)}:{remote_wandb}/",
        str(local_wandb) + "/",
    ]
    result = subprocess.run(cmd)

    if result.returncode == 0:
        click.echo(f"\nDownloaded to {local_wandb}")
        click.echo("To sync with W&B cloud, run:")
        click.echo(f"  cd {local_wandb} && wandb sync .")
    else:
        click.echo("Download failed.", err=True)
