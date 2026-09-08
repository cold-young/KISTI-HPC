#!/bin/bash
#SBATCH -J golden_project_train_exp_001
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

# === Stage: train ===
export EXPERIMENT_ID="exp_001"
export STAGE_NAME="train"
export CONFIG_PATH="configs/exp_001.yaml"
export WANDB_MODE=offline

# === Python invocation ===
# Wraps `python {script} {args}` according to environment.type declared in the
# experiment.yaml. Without this branch, the host system python is used and
# every K-UBM import fails immediately. (BUG-K1 fix.)

# Fallback: rely on PATH/modules already loaded above.
python train.py 


EXIT_CODE=$?
echo "STAGE_DONE: ${STAGE_NAME} exp=${EXPERIMENT_ID} exit_code=${EXIT_CODE}" >> /scratch/gilbreth/golden_user/logs/completed.log
exit ${EXIT_CODE}
