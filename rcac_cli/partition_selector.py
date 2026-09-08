"""Smart partition selector based on queue status and GPU requirements."""

from __future__ import annotations

import re
import subprocess
import sys

import click

from .config import get_rcac_config, require_user
from .ssh import ssh_opt_argv, ssh_target

# Partition specs: (gpu_type, gpus_per_node, cores, relative_perf, cost_tier)
PARTITION_INFO = {
    "amd_a100nv_8":  {"gpu": "A100", "gpus_per_node": 8, "cores": 64, "perf": 1.0, "cost": 2},
    "eme_h200nv_8":  {"gpu": "H200", "gpus_per_node": 8, "cores": 96, "perf": 1.8, "cost": 4},
    "amd_h200nv_8":  {"gpu": "H200", "gpus_per_node": 8, "cores": 96, "perf": 1.8, "cost": 4},
    "cas_v100nv_8":  {"gpu": "V100", "gpus_per_node": 8, "cores": 32, "perf": 0.5, "cost": 1},
    "cas_v100_4":    {"gpu": "V100", "gpus_per_node": 4, "cores": 40, "perf": 0.5, "cost": 1},
    "cas_v100_2":    {"gpu": "V100", "gpus_per_node": 2, "cores": 32, "perf": 0.5, "cost": 1},
}


def _run_remote(cfg: dict, command: str) -> subprocess.CompletedProcess:
    cmd = ["ssh", *ssh_opt_argv(cfg), ssh_target(cfg), command]
    return subprocess.run(cmd, capture_output=True, text=True)


def _parse_sinfo(output: str) -> dict[str, dict]:
    """Parse sinfo output into partition availability data."""
    partitions = {}
    for line in output.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        # PARTITION AVAIL NODES STATE NODELIST GRES
        partition = parts[0].rstrip("*")
        avail = parts[1]
        try:
            node_count = int(parts[2])
        except ValueError:
            continue
        state = parts[3]

        if partition not in partitions:
            partitions[partition] = {"idle": 0, "mix": 0, "alloc": 0, "total": 0, "avail": avail}

        partitions[partition]["total"] += node_count
        if state in ("idle", "idle~"):
            partitions[partition]["idle"] += node_count
        elif state in ("mix", "mixed"):
            partitions[partition]["mix"] += node_count
        elif state in ("alloc", "allocated"):
            partitions[partition]["alloc"] += node_count

    return partitions


def _score_partition(
    name: str,
    info: dict,
    queue_data: dict,
    gpus_needed: int,
    prefer_cost: bool = False,
) -> float | None:
    """Score a partition for a given job. Higher = better."""
    if name not in PARTITION_INFO:
        return None

    spec = PARTITION_INFO[name]
    q = queue_data.get(name, {})

    # Can this partition serve the request?
    if gpus_needed > spec["gpus_per_node"]:
        return None

    if q.get("avail") == "down":
        return None

    # Availability score (0-50): prefer partitions with idle nodes
    idle = q.get("idle", 0)
    mix = q.get("mix", 0)
    avail_score = idle * 10 + mix * 3

    # Performance score (0-30)
    perf_score = spec["perf"] * 30

    # Cost efficiency score (0-20): lower cost = higher score
    if prefer_cost:
        cost_score = (5 - spec["cost"]) * 5
    else:
        cost_score = 10  # neutral when not cost-sensitive

    return avail_score + perf_score + cost_score


def recommend_partition(
    cfg: dict,
    gpus_needed: int = 2,
    prefer_cost: bool = False,
) -> list[tuple[str, float, dict]]:
    """Return ranked list of (partition_name, score, details)."""
    result = _run_remote(cfg, "sinfo -o '%P %a %D %T %N %G' --noheader")
    if result.returncode != 0:
        return []

    queue_data = _parse_sinfo(result.stdout)
    scored = []

    for name in PARTITION_INFO:
        score = _score_partition(name, PARTITION_INFO[name], queue_data, gpus_needed, prefer_cost)
        if score is not None:
            spec = PARTITION_INFO[name]
            q = queue_data.get(name, {})
            details = {
                "gpu": spec["gpu"],
                "idle_nodes": q.get("idle", 0),
                "mix_nodes": q.get("mix", 0),
                "gpus_per_node": spec["gpus_per_node"],
                "cost_tier": spec["cost"],
                "perf": spec["perf"],
            }
            scored.append((name, score, details))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


@click.command("partition")
@click.option("--gpus", "-g", default=2, help="Number of GPUs needed")
@click.option("--prefer-cost", is_flag=True, help="Prefer cheaper partitions over performance")
def select_partition(gpus: int, prefer_cost: bool) -> None:
    """Recommend the best partition based on current queue status."""
    cfg = get_rcac_config()

    require_user(cfg)

    click.echo(f"Analyzing partitions for {gpus} GPU(s)...\n")

    ranked = recommend_partition(cfg, gpus_needed=gpus, prefer_cost=prefer_cost)

    if not ranked:
        click.echo("No suitable partitions found. Check SSH connection.")
        sys.exit(1)

    click.echo(f"{'Rank':<5} {'Partition':<18} {'GPU':<6} {'Idle':<6} {'Mix':<5} {'Perf':<6} {'Cost':<6} {'Score':<8}")
    click.echo("-" * 70)

    for i, (name, score, d) in enumerate(ranked, 1):
        cost_label = ["", "Low", "Med", "", "High"][d["cost_tier"]]
        marker = " <-- recommended" if i == 1 else ""
        click.echo(
            f"{i:<5} {name:<18} {d['gpu']:<6} {d['idle_nodes']:<6} {d['mix_nodes']:<5} "
            f"{d['perf']:<6.1f} {cost_label:<6} {score:<8.1f}{marker}"
        )

    best = ranked[0]
    click.echo(f"\nRecommended: --partition={best[0]}")
    click.echo(f"  Use: rcac-exp launch experiment.yaml --partition {best[0]}")
