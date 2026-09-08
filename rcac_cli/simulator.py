"""Simulator integration for Genesis, MuJoCo, and Isaac Gym on RCAC HPC.

Provides environment validation, data generation pipeline support,
and simulator-specific SLURM configuration helpers.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import click
import yaml

from .config import get_defaults, get_rcac_config, require_user
from .ssh import ssh_opt_argv, ssh_target

# Simulator compatibility matrix with RCAC GPUs
SIMULATOR_COMPAT = {
    "genesis": {
        "display_name": "Genesis",
        "gpu_support": {"V100": True, "A100": True, "H200": True},
        "rt_core_required": False,
        "container": "genesis-sim.sif",
        "pip_package": "genesis-world",
        "slurm_comment": "genesis",
        "notes": "Pure CUDA, no RT Core needed. 10-80x faster than Isaac Gym.",
    },
    "mujoco": {
        "display_name": "MuJoCo (MJX)",
        "gpu_support": {"V100": True, "A100": True, "H200": True},
        "rt_core_required": False,
        "container": None,
        "pip_package": "mujoco",
        "slurm_comment": "mujoco",
        "notes": "GPU accel via MJX (JAX backend). Broad physics sim.",
    },
    "isaac_gym": {
        "display_name": "Isaac Gym Preview",
        "gpu_support": {"V100": True, "A100": True, "H200": True},
        "rt_core_required": False,
        "container": "isaacgym-preview.sif",
        "pip_package": None,  # manual install from NVIDIA
        "slurm_comment": "isaacgym",
        "notes": "DEPRECATED. Use for legacy code only. Pin via Singularity.",
    },
    "isaac_sim": {
        "display_name": "Isaac Sim",
        "gpu_support": {"V100": False, "A100": False, "H200": False},
        "rt_core_required": True,
        "container": None,
        "pip_package": None,
        "slurm_comment": "isaacsim",
        "notes": "NOT SUPPORTED on RCAC. Requires RT Core GPUs.",
    },
    "pybullet": {
        "display_name": "PyBullet",
        "gpu_support": {"V100": True, "A100": True, "H200": True},
        "rt_core_required": False,
        "container": None,
        "pip_package": "pybullet",
        "slurm_comment": "pybullet",
        "notes": "Lightweight sim. Limited GPU acceleration.",
    },
}

# Partition -> GPU type mapping
PARTITION_GPU = {
    "amd_a100nv_8": "A100",
    "eme_h200nv_8": "H200",
    "amd_h200nv_8": "H200",
    "cas_v100nv_8": "V100",
    "cas_v100_4": "V100",
    "cas_v100_2": "V100",
}


def _run_remote(cfg: dict, command: str) -> subprocess.CompletedProcess:
    cmd = ["ssh", *ssh_opt_argv(cfg), ssh_target(cfg), command]
    return subprocess.run(cmd, capture_output=True, text=True)


def check_compatibility(simulator: str, partition: str) -> dict:
    """Check if a simulator is compatible with a given partition's GPU."""
    sim = SIMULATOR_COMPAT.get(simulator)
    if not sim:
        return {"compatible": False, "reason": f"Unknown simulator: {simulator}"}

    gpu_type = PARTITION_GPU.get(partition)
    if not gpu_type:
        return {"compatible": False, "reason": f"Unknown partition: {partition}"}

    supported = sim["gpu_support"].get(gpu_type, False)
    if not supported:
        return {
            "compatible": False,
            "reason": f"{sim['display_name']} is not supported on {gpu_type} ({partition}). {sim['notes']}",
        }

    return {
        "compatible": True,
        "simulator": sim["display_name"],
        "gpu": gpu_type,
        "partition": partition,
        "container": sim["container"],
        "comment": sim["slurm_comment"],
    }


def generate_datagen_config(
    simulator: str,
    env_name: str,
    num_episodes: int = 10000,
    output_dir: str = "generated_data",
    extra_args: dict | None = None,
) -> dict:
    """Generate a data generation configuration for a simulator."""
    config = {
        "simulator": simulator,
        "env_name": env_name,
        "num_episodes": num_episodes,
        "output_dir": output_dir,
        "render": False,  # headless on HPC
        "num_envs": 256,  # parallel environments
    }
    if extra_args:
        config.update(extra_args)
    return config


@click.group()
def cli() -> None:
    """Simulator management for RCAC HPC."""


@cli.command("compat")
@click.argument("simulator", type=click.Choice(list(SIMULATOR_COMPAT.keys())))
@click.option("--partition", "-p", default="amd_a100nv_8", help="Target partition")
def check_compat(simulator: str, partition: str) -> None:
    """Check simulator compatibility with a RCAC partition."""
    result = check_compatibility(simulator, partition)

    if result["compatible"]:
        click.echo(f"{result['simulator']}: COMPATIBLE with {result['gpu']} ({partition})")
        if result.get("container"):
            click.echo(f"  Recommended container: {result['container']}")
        click.echo(f"  SLURM --comment: {result['comment']}")
    else:
        click.echo(f"INCOMPATIBLE: {result['reason']}")


@cli.command("list")
def list_simulators() -> None:
    """Show all simulators and their RCAC GPU compatibility."""
    click.echo(f"{'Simulator':<22} {'V100':<8} {'A100':<8} {'H200':<8} {'Notes'}")
    click.echo("-" * 90)

    for key, sim in SIMULATOR_COMPAT.items():
        v100 = "OK" if sim["gpu_support"]["V100"] else "NO"
        a100 = "OK" if sim["gpu_support"]["A100"] else "NO"
        h200 = "OK" if sim["gpu_support"]["H200"] else "NO"
        click.echo(f"{sim['display_name']:<22} {v100:<8} {a100:<8} {h200:<8} {sim['notes']}")


@cli.command("check-env")
@click.argument("simulator", type=click.Choice(["genesis", "mujoco", "isaac_gym", "pybullet"]))
def check_env(simulator: str) -> None:
    """Verify simulator is installed on RCAC."""
    cfg = get_rcac_config()

    require_user(cfg)

    sim = SIMULATOR_COMPAT[simulator]
    pkg = sim["pip_package"]

    if not pkg:
        click.echo(f"{sim['display_name']} requires manual installation.")
        if simulator == "isaac_gym":
            click.echo("  Download from NVIDIA and install in Singularity container.")
            click.echo("  See: containers/isaacgym-preview.def")
        return

    result = _run_remote(cfg, f"python -c 'import {pkg.replace('-', '_')}; print(\"OK\")' 2>&1")

    if "OK" in result.stdout:
        click.echo(f"{sim['display_name']}: INSTALLED")
    else:
        click.echo(f"{sim['display_name']}: NOT FOUND")
        click.echo(f"  Install: pip install {pkg}")
        if sim["container"]:
            click.echo(f"  Or use container: rcac-container build containers/{sim['container'].replace('.sif', '.def')}")


@cli.command("init-pipeline")
@click.argument("simulator", type=click.Choice(["genesis", "mujoco", "isaac_gym"]))
@click.option("--env", "env_name", default="manipulation", help="Environment name")
@click.option("--episodes", default=10000, help="Number of data generation episodes")
@click.option("--output", "-o", default="experiment_sim.yaml", help="Output experiment file")
@click.option("--partition", "-p", default="amd_a100nv_8", help="Target partition")
def init_pipeline(simulator: str, env_name: str, episodes: int, output: str, partition: str) -> None:
    """Generate a simulation -> training -> evaluation experiment.yaml."""
    compat = check_compatibility(simulator, partition)
    if not compat["compatible"]:
        click.echo(f"Error: {compat['reason']}")
        sys.exit(1)

    sim = SIMULATOR_COMPAT[simulator]

    # Build the experiment YAML
    experiment = {
        "project": f"{env_name}_{simulator}",
        "description": f"{sim['display_name']} simulation + training pipeline for {env_name}",
        "slurm": {
            "partition": partition,
            "nodes": 1,
            "gpus": 2,
            "cpus_per_gpu": 8,
            "time": "24:00:00",
            "comment": sim["slurm_comment"],
        },
        "environment": {
            "type": "singularity" if sim["container"] else "conda",
            "name": f"{simulator}_env",
            "modules": ["cuda/12.4.1"],
        },
        "base_params": {
            "dataset": f"/scratch/$USER/data/{env_name}_{simulator}",
            "epochs": 100,
            "batch_size": 64,
            "optimizer": "adamw",
            "seed": 42,
        },
        "pipeline": [
            {
                "name": "generate_data",
                "script": f"{simulator}_datagen.py",
                "args": f"--env {env_name} --num_episodes {episodes} --output {{config_path}}/../data",
            },
            {
                "name": "train",
                "script": "train.py",
                "depends_on": "generate_data",
                "args": "--config {config_path} --dataset {generate_data.output_path}",
            },
            {
                "name": "evaluate_sim",
                "script": f"{simulator}_eval.py",
                "depends_on": "train",
                "args": "--checkpoint {train.best_model} --env " + env_name,
            },
        ],
    }

    # Add container config if applicable
    if sim["container"]:
        experiment["environment"]["container"] = f"/scratch/$USER/.rcac-tools/{sim['container']}"

    with open(output, "w") as f:
        yaml.dump(experiment, f, default_flow_style=False, sort_keys=False)

    click.echo(f"Pipeline experiment file created: {output}")
    click.echo(f"\nPipeline stages:")
    for i, stage in enumerate(experiment["pipeline"], 1):
        dep = f" (after: {stage['depends_on']})" if stage.get("depends_on") else ""
        click.echo(f"  {i}. {stage['name']}: {stage['script']}{dep}")
    click.echo(f"\nTo launch: rcac-exp launch {output}")
    click.echo(f"To dry-run: rcac-exp launch {output} --dry-run")


@cli.command("scaffold")
@click.argument("simulator", type=click.Choice(["genesis", "mujoco", "isaac_gym"]))
@click.option("--env", "env_name", default="manipulation", help="Environment name")
@click.option("--output-dir", "-o", default=".", help="Output directory for scaffold files")
def scaffold(simulator: str, env_name: str, output_dir: str) -> None:
    """Generate starter Python scripts for a sim-to-real pipeline."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sim = SIMULATOR_COMPAT[simulator]

    # Data generation script
    datagen_file = out / f"{simulator}_datagen.py"
    datagen_file.write_text(_scaffold_datagen(simulator, env_name))
    click.echo(f"  Created: {datagen_file}")

    # Evaluation script
    eval_file = out / f"{simulator}_eval.py"
    eval_file.write_text(_scaffold_eval(simulator, env_name))
    click.echo(f"  Created: {eval_file}")

    # Train script (generic)
    train_file = out / "train.py"
    if not train_file.exists():
        train_file.write_text(_scaffold_train())
        click.echo(f"  Created: {train_file}")

    click.echo(f"\nScaffold complete for {sim['display_name']} / {env_name}")
    click.echo(f"Next: rcac-sim init-pipeline {simulator} --env {env_name}")


def _scaffold_datagen(simulator: str, env_name: str) -> str:
    """Generate a data generation script template."""
    if simulator == "genesis":
        return f'''"""Genesis data generation for {env_name}."""

import argparse
import json
from pathlib import Path

import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="{env_name}")
    parser.add_argument("--num_episodes", type=int, default=10000)
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--output", default="generated_data")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import genesis as gs
        gs.init(backend=gs.cuda)

        # TODO: Define your Genesis scene and robot
        # scene = gs.Scene(...)
        # robot = scene.add_entity(gs.Morphology(...))

        print(f"Generating {{args.num_episodes}} episodes with {{args.num_envs}} parallel envs...")

        all_observations = []
        all_actions = []

        for ep in range(0, args.num_episodes, args.num_envs):
            batch_size = min(args.num_envs, args.num_episodes - ep)
            # TODO: Run simulation and collect data
            # obs, actions = run_episode_batch(scene, robot, batch_size)
            # all_observations.append(obs)
            # all_actions.append(actions)

            if (ep + batch_size) % 1000 == 0:
                print(f"  Episodes: {{ep + batch_size}}/{{args.num_episodes}}")

        # Save collected data
        # np.savez(output_dir / "dataset.npz",
        #          observations=np.concatenate(all_observations),
        #          actions=np.concatenate(all_actions))

        metadata = {{
            "simulator": "genesis",
            "env": args.env,
            "num_episodes": args.num_episodes,
            "output_path": str(output_dir),
        }}
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Data saved to {{output_dir}}")

    except ImportError:
        print("ERROR: Genesis not installed. Run: pip install genesis-world")
        raise SystemExit(1)

if __name__ == "__main__":
    main()
'''
    elif simulator == "mujoco":
        return f'''"""MuJoCo data generation for {env_name}."""

import argparse
import json
from pathlib import Path

import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="{env_name}")
    parser.add_argument("--num_episodes", type=int, default=10000)
    parser.add_argument("--output", default="generated_data")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import mujoco

        # TODO: Load your MuJoCo model
        # model = mujoco.MjModel.from_xml_path("your_model.xml")
        # data = mujoco.MjData(model)

        print(f"Generating {{args.num_episodes}} episodes...")

        for ep in range(args.num_episodes):
            # TODO: Run episode, collect observations and actions
            # mujoco.mj_step(model, data)
            pass

        metadata = {{
            "simulator": "mujoco",
            "env": args.env,
            "num_episodes": args.num_episodes,
            "output_path": str(output_dir),
        }}
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Data saved to {{output_dir}}")

    except ImportError:
        print("ERROR: MuJoCo not installed. Run: pip install mujoco")
        raise SystemExit(1)

if __name__ == "__main__":
    main()
'''
    else:  # isaac_gym
        return f'''"""Isaac Gym Preview data generation for {env_name}."""

import argparse
import json
from pathlib import Path

import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="{env_name}")
    parser.add_argument("--num_episodes", type=int, default=10000)
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--output", default="generated_data")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from isaacgym import gymapi

        gym = gymapi.acquire_gym()
        sim_params = gymapi.SimParams()
        sim_params.use_gpu_pipeline = True

        # TODO: Create sim and environment
        # sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

        print(f"Generating {{args.num_episodes}} episodes with {{args.num_envs}} parallel envs...")
        print("NOTE: Isaac Gym Preview is deprecated. Consider migrating to Genesis.")

        metadata = {{
            "simulator": "isaac_gym",
            "env": args.env,
            "num_episodes": args.num_episodes,
            "output_path": str(output_dir),
        }}
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Data saved to {{output_dir}}")

    except ImportError:
        print("ERROR: Isaac Gym not installed. Install from NVIDIA package.")
        raise SystemExit(1)

if __name__ == "__main__":
    main()
'''


def _scaffold_eval(simulator: str, env_name: str) -> str:
    """Generate an evaluation script template."""
    return f'''"""Simulation evaluation for {env_name} using {simulator}."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--env", default="{env_name}")
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    print(f"Evaluating {{args.checkpoint}} in {{args.env}}...")

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = YourModel()
    # model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    # model.eval()

    # TODO: Run evaluation episodes in simulation
    success_count = 0
    total_reward = 0.0

    for ep in range(args.num_episodes):
        # TODO: Reset environment, run policy, check success
        # obs = env.reset()
        # done = False
        # while not done:
        #     action = model(obs)
        #     obs, reward, done, info = env.step(action)
        #     total_reward += reward
        # if info.get("success"):
        #     success_count += 1
        pass

    results = {{
        "checkpoint": args.checkpoint,
        "env": args.env,
        "num_episodes": args.num_episodes,
        "success_rate": success_count / max(args.num_episodes, 1),
        "avg_reward": total_reward / max(args.num_episodes, 1),
        "accuracy": success_count / max(args.num_episodes, 1),  # alias for results.py
    }}

    output_path = Path(args.checkpoint).parent / "eval_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Success rate: {{results['success_rate']:.2%}}")
    print(f"Avg reward: {{results['avg_reward']:.2f}}")
    print(f"Results saved to {{output_path}}")

if __name__ == "__main__":
    main()
'''


def _scaffold_train() -> str:
    """Generate a generic training script template."""
    return '''"""Generic training script for sim-to-real pipeline."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Experiment config YAML")
    parser.add_argument("--dataset", default=None, help="Dataset path override")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    dataset_path = args.dataset or config.get("dataset", "")
    epochs = config.get("epochs", 100)
    batch_size = config.get("batch_size", 64)
    lr = config.get("learning_rate", 1e-4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    print(f"Config: epochs={epochs}, batch_size={batch_size}, lr={lr}")
    print(f"Dataset: {dataset_path}")

    # TODO: Load dataset
    # dataset = YourDataset(dataset_path)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # TODO: Define model
    # model = YourModel(**config).to(device)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    checkpoint_dir = Path(args.config).parent / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    train_log = []

    try:
        import wandb
        wandb.init(mode="offline", config=config)
    except ImportError:
        wandb = None

    for epoch in range(epochs):
        # TODO: Training loop
        # epoch_loss = train_one_epoch(model, dataloader, optimizer, device)
        epoch_loss = 0.0  # placeholder

        train_log.append({"epoch": epoch, "loss": epoch_loss})

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({}, checkpoint_dir / "best_model.pth")

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

        if wandb:
            wandb.log({"epoch": epoch, "loss": epoch_loss})

    # Save training log
    log_path = Path(args.config).parent / "train_log.json"
    with open(log_path, "w") as f:
        json.dump(train_log, f, indent=2)

    if wandb:
        wandb.finish()

    print(f"Training complete. Best loss: {best_loss:.4f}")
    print(f"Checkpoint: {checkpoint_dir / 'best_model.pth'}")

if __name__ == "__main__":
    main()
'''
