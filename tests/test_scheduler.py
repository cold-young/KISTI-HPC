"""Tests for SLURM script generation."""

from rcac_cli.scheduler import generate_single_gpu_script, generate_pipeline_script


def test_single_gpu_script_contains_required_fields():
    script = generate_single_gpu_script(
        project="test_project",
        exp_id="exp_001",
        script="train.py",
        args="--lr 1e-4",
        conda_env="test_env",
    )
    assert "#SBATCH -J test_project_exp_001" in script
    assert "#SBATCH -p amd_a100nv_8" in script
    assert "#SBATCH --comment pytorch" in script
    assert "python train.py --lr 1e-4" in script
    assert "module load cuda/12.4.1" in script


def test_pipeline_script_with_dependency():
    script = generate_pipeline_script(
        project="test_project",
        exp_id="exp_001",
        stage_name="evaluate",
        script="evaluate.py",
        dependency_job_id="12345",
        conda_env="test_env",
    )
    assert "#SBATCH --dependency=afterok:12345" in script
    assert "evaluate" in script


def test_pipeline_script_without_dependency():
    script = generate_pipeline_script(
        project="test_project",
        exp_id="exp_001",
        stage_name="train",
        script="train.py",
        conda_env="test_env",
    )
    assert "--dependency" not in script
