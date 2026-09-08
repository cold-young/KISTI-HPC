#!/bin/bash
#SBATCH -J golden_project_exp_001
#SBATCH -A golden_account
#SBATCH -p a100-40gb
#SBATCH -q normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --constraint=nvlink
#SBATCH -o /scratch/gilbreth/golden_user/logs/%x_%j.out
#SBATCH -e /scratch/gilbreth/golden_user/logs/%x_%j.err

# === Environment Setup ===


# === Working Directory ===
cd /scratch/gilbreth/golden_user/projects/golden_project

# === Experiment ===
export EXPERIMENT_ID="exp_001"
export CONFIG_PATH="configs/exp_001.yaml"
export WANDB_MODE=offline
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

# === Distributed launch === (honours environment.type)

torchrun --nproc_per_node=2 --nnodes=1 \
    --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
    train.py --lr 1e-4


# === Record Completion ===
EXIT_CODE=$?
echo "EXPERIMENT_DONE: $EXPERIMENT_ID exit_code=${EXIT_CODE}" >> /scratch/gilbreth/golden_user/logs/completed.log
exit ${EXIT_CODE}
