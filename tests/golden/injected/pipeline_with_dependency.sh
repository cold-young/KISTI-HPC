#!/bin/bash
#SBATCH -J golden_project_evaluate_exp_001
#SBATCH -p amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=24:00:00
#SBATCH --comment pytorch

#SBATCH --dependency=afterok:12345

#SBATCH -o /scratch/golden_user/logs/%x_%j.out
#SBATCH -e /scratch/golden_user/logs/%x_%j.err

# === Environment Setup ===

module load cuda/12.4.1


# === Working Directory ===
cd /scratch/golden_user/projects/golden_project

# === Stage: evaluate ===
export EXPERIMENT_ID="exp_001"
export STAGE_NAME="evaluate"
export CONFIG_PATH="configs/exp_001.yaml"
export WANDB_MODE=offline

# === Python invocation ===
# Wraps `python {script} {args}` according to environment.type declared in the
# experiment.yaml. Without this branch, the host system python is used and
# every K-UBM import fails immediately. (BUG-K1 fix.)

# Fallback: rely on PATH/modules already loaded above.
python evaluate.py 


EXIT_CODE=$?
echo "STAGE_DONE: ${STAGE_NAME} exp=${EXPERIMENT_ID} exit_code=${EXIT_CODE}" >> /scratch/golden_user/logs/completed.log
exit ${EXIT_CODE}
