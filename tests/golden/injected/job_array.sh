#!/bin/bash
#SBATCH -J golden_project_sweep
#SBATCH -p amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=24:00:00
#SBATCH --comment pytorch
#SBATCH --array=0-7%2

#SBATCH -o /scratch/golden_user/logs/%x_%A_%a.out
#SBATCH -e /scratch/golden_user/logs/%x_%A_%a.err

# === Environment Setup ===

module load cuda/12.4.1


export PATH=$HOME/.conda/envs/golden_env/bin:$PATH


# === Working Directory ===
cd /scratch/golden_user/projects/golden_project

# === Select Config by Array Index ===
CONFIG_DIR="configs/sweep"
CONFIG_FILE="${CONFIG_DIR}/config_$(printf '%03d' ${SLURM_ARRAY_TASK_ID}).yaml"

export EXPERIMENT_ID="golden_project_${SLURM_ARRAY_TASK_ID}"
export CONFIG_PATH="${CONFIG_FILE}"
export WANDB_MODE=offline

python train.py --config ${CONFIG_FILE} --epochs 10

# === Record Completion ===
EXIT_CODE=$?
echo "EXPERIMENT_DONE: $EXPERIMENT_ID exit_code=${EXIT_CODE}" >> /scratch/golden_user/logs/completed.log
exit ${EXIT_CODE}
