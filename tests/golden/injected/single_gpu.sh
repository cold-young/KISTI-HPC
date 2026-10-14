#!/bin/bash
#SBATCH -J golden_project_exp_001
#SBATCH -A golden_account
#SBATCH -p a100-40gb
#SBATCH -q normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH -o /scratch/gilbreth/golden_user/logs/%x_%j.out
#SBATCH -e /scratch/gilbreth/golden_user/logs/%x_%j.err

# === Environment Setup ===


# === Working Directory ===
cd /scratch/gilbreth/golden_user/projects/golden_project

# === Experiment ===
export EXPERIMENT_ID="exp_001"
export CONFIG_PATH="configs/exp_001.yaml"
export WANDB_MODE=offline

# === Python invocation === (honours environment.type)

python train.py --lr 1e-4


# === Record Completion ===
EXIT_CODE=$?
echo "EXPERIMENT_DONE: $EXPERIMENT_ID exit_code=${EXIT_CODE}" >> /scratch/gilbreth/golden_user/logs/completed.log
exit ${EXIT_CODE}
