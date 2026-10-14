import pytest

from gilbreth_cli.scheduler import render_job, resolve_env_ref, resolve_qos


def _render(cfg, **kw):
    base = {"project": "proj", "exp_id": "20260101_000", "script": "train.py", "args": "--lr 1e-4",
            "account": "primelab", "partition": "a100-80gb"}
    base.update(kw)
    return render_job(cfg, **base)


def test_header_has_every_required_gilbreth_flag(gcfg_env):
    s = _render(gcfg_env, qos="standby", gpus=2, mem="32G", time="01:00:00")
    for line in ["#SBATCH -A primelab", "#SBATCH -p a100-80gb", "#SBATCH -q standby", "#SBATCH --gpus-per-node=2",
                 "#SBATCH --mem=32G", "#SBATCH --time=01:00:00", "#SBATCH --cpus-per-gpu=8"]:
        assert line in s, line
    assert "-o /scratch/gilbreth/alice/logs/%x_%j.out" in s
    assert "cd /scratch/gilbreth/alice/projects/proj ||" in s
    assert "mkdir -p /scratch" not in s
    assert "python train.py --lr 1e-4" in s
    assert "--constraint" not in s
    assert "--dependency" not in s


def test_training_partition_forces_training_qos(gcfg_env):
    assert resolve_qos("training", "normal") == "training"
    assert resolve_qos("a10", None) == "normal"
    s = _render(gcfg_env, partition="training", qos="normal", gpus=4)
    assert "#SBATCH -q training" in s


def test_conda_env_in_scratch(gcfg_env):
    s = _render(gcfg_env, env_type="conda", env_name="myenv", modules=["cuda", "cudnn"])
    assert s.index("module load rcac") < s.index("module load conda ||") < s.index("module load cuda ||") < s.index("module load cudnn ||")
    assert "command -v conda >/dev/null ||" in s
    assert 'eval "$(conda shell.bash hook)"' in s
    assert "conda activate /scratch/gilbreth/alice/envs/myenv" in s
    assert resolve_env_ref(gcfg_env, "conda", "/abs/env") == "/abs/env"


def test_conda_env_mod(gcfg_env):
    gcfg_env["environment"]["conda_module"] = "anaconda"
    s = _render(gcfg_env, env_type="conda-env-mod", env_name="myenv-py3.12.3")
    assert "module load anaconda" in s
    assert "module load use.own" in s
    assert "module load conda-env/myenv-py3.12.3" in s


def test_apptainer_binds_scratch_and_depot(gcfg_env):
    gcfg_env["paths"]["depot"] = "/depot/primelab"
    s = _render(gcfg_env, env_type="apptainer", container="/scratch/gilbreth/alice/img.sif")
    assert "apptainer exec --nv --bind /scratch/gilbreth/alice --bind /depot/primelab /scratch/gilbreth/alice/img.sif" in s
    assert "$RUNNER python train.py" in s
    with pytest.raises(ValueError):
        _render(gcfg_env, env_type="apptainer")


def test_torchrun_single_and_multi_node(gcfg_env):
    s = _render(gcfg_env, launcher="torchrun", gpus=4)
    assert "torchrun --standalone --nproc_per_node=4 train.py --lr 1e-4" in s
    s = _render(gcfg_env, launcher="torchrun", gpus=4, nodes=2)
    assert "srun $RUNNER torchrun" in s and "--nnodes=$SLURM_NNODES --nproc_per_node=4" in s
    assert "#SBATCH --nodes=2" in s


def test_dependency_constraint_mail(gcfg_env):
    gcfg_env["slurm"]["mail_user"] = "alice@purdue.edu"
    s = _render(gcfg_env, dependency_job_id="4242", constraint="nvlink", stage_name="eval")
    assert "#SBATCH --dependency=afterok:4242" in s
    assert "#SBATCH --constraint=nvlink" in s
    assert "#SBATCH --mail-user=alice@purdue.edu" in s
    assert 'export STAGE_NAME="eval"' in s


def test_array_mode(gcfg_env):
    s = _render(gcfg_env, array_max=5, array_parallel=2, config_dir="/scratch/gilbreth/alice/projects/proj/configs/x",
                args='$(cat "$ARGS_FILE")')
    assert "#SBATCH --array=0-5%2" in s
    assert "%x_%A_%a.out" in s
    assert "config_$(printf '%03d' $SLURM_ARRAY_TASK_ID).yaml" in s
    assert 'eval "set -- $(cat "$ARGS_FILE")"' in s and 'python train.py "$@"' in s
    assert 'export ARGS_FILE="/scratch/gilbreth/alice/projects/proj/configs/x/args_$(printf' in s


def test_command_mode_is_shell_quoted(gcfg_env):
    s = _render(gcfg_env, script="", command="python train.py --name 'a b'")
    assert "bash -c 'python train.py --name '\"'\"'a b'\"'\"''" in s


def test_env_vars_merged_and_validation(gcfg_env):
    s = _render(gcfg_env, env_vars={"FOO": 'say "hi"'})
    assert "export FOO='say \"hi\"'" in s and "export PIP_CACHE_DIR=/scratch/gilbreth/alice/.cache/pip" in s
    with pytest.raises(ValueError):
        _render(gcfg_env, gpus=0)
    with pytest.raises(ValueError):
        _render(gcfg_env, launcher="mpirun")
    with pytest.raises(ValueError):
        _render(gcfg_env, script="", command="")
    with pytest.raises(ValueError):
        _render(gcfg_env, env_type="conda", env_name="")
