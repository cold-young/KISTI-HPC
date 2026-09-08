#!/bin/bash
#SBATCH -J golden_project_exp_001
#SBATCH -p amd_a100nv_8
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=8
#SBATCH --time=24:00:00
#SBATCH --comment pytorch
#SBATCH -o /scratch/YOUR_ID/logs/%x_%j.out
#SBATCH -e /scratch/YOUR_ID/logs/%x_%j.err

# === Environment Setup ===

module load cuda/12.4.1


# === Working Directory ===
cd /scratch/YOUR_ID/projects/golden_project

# === Experiment ===
export EXPERIMENT_ID="exp_001"
export CONFIG_PATH="configs/exp_001.yaml"
export WANDB_MODE=offline
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

# === Distributed launch === (BUG-K1 fix: respect environment.type)

torchrun --nproc_per_node=4 --nnodes=2 \
    --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
    train.py --lr 1e-4


# === Record Completion ===
EXIT_CODE=$?
echo "EXPERIMENT_DONE: $EXPERIMENT_ID exit_code=${EXIT_CODE}" >> /scratch/YOUR_ID/logs/completed.log
exit ${EXIT_CODE}
