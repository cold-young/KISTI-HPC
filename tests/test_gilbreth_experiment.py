import subprocess

import yaml
from click.testing import CliRunner

from gilbreth_cli import experiment
from gilbreth_cli import run as run_mod
from gilbreth_cli.experiment import generate_sweep_configs


def test_grid_and_random_sweeps():
    base = {"epochs": 1}
    grid = generate_sweep_configs({"method": "grid", "parameters": {"lr": [1, 2], "wd": [0.1]}}, base)
    assert len(grid) == 2 and all(c["epochs"] == 1 and c["wd"] == 0.1 for c in grid)
    rnd = generate_sweep_configs({"method": "random", "count": 5, "seed": 0, "parameters": {"lr": [1, 2, 3]}}, base)
    assert len(rnd) == 5 and all(c["lr"] in (1, 2, 3) for c in rnd)
    assert generate_sweep_configs({}, base) == [base]


def _write_exp(tmp_path, **extra):
    exp = {
        "project": "demo",
        "slurm": {"partition": "a30", "gpus": 1, "time": "00:30:00", "mem": "16G"},
        "environment": {"type": "conda", "name": "demo_env"},
        "script": "train.py",
        "args": "--config {config_path} --lr {lr}",
        "base_params": {"epochs": 2},
        "sweep": {"method": "grid", "parameters": {"lr": [0.1, 0.2, 0.3]}},
    }
    exp.update(extra)
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(exp))
    return path


def test_launch_array_dry_run(gcfg_env, fake_remote, isolated_runs, tmp_path):
    path = _write_exp(tmp_path)
    result = CliRunner().invoke(experiment.cli, ["launch", str(path), "--dry-run"])
    assert result.exit_code == 0, result.output
    run_dir = next(isolated_runs.iterdir())
    script = (run_dir / "scripts" / "job_array.sh").read_text()
    assert "#SBATCH --array=0-2" in script
    assert "#SBATCH -p a30" in script and "#SBATCH -A primelab" in script
    args = sorted((run_dir / "configs").glob("args_*.txt"))
    assert len(args) == 3
    assert args[1].read_text().strip().endswith("--lr 0.2")
    assert "config_001.yaml" in args[1].read_text()
    assert fake_remote["remote"] == []  # dry run never touches the cluster


def test_launch_separate_submits_each_config(gcfg_env, fake_remote, isolated_runs, tmp_path):
    path = _write_exp(tmp_path, sweep={"method": "grid", "mode": "separate", "parameters": {"lr": [0.1, 0.2]}})
    result = CliRunner().invoke(experiment.cli, ["launch", str(path), "-n", "2", "--qos", "standby"])
    assert result.exit_code == 0, result.output
    sbatch_calls = [c for c in fake_remote["remote"] if c.startswith("sbatch")]
    assert len(sbatch_calls) == 2
    assert "submitted: job" in result.output
    run_dir = next(isolated_runs.iterdir())
    assert "#SBATCH -q standby" in (run_dir / "scripts" / "job_000.sh").read_text()
    assert (run_dir / "tracking.json").exists()
    # source dir synced, configs and scripts uploaded
    assert fake_remote["rsync"] and fake_remote["uploads"]


def test_launch_pipeline_wires_dependencies(gcfg_env, fake_remote, isolated_runs, tmp_path):
    path = _write_exp(tmp_path, sweep={}, pipeline=[
        {"name": "train", "script": "train.py", "args": "--config {config_path}"},
        {"name": "eval", "script": "eval.py", "args": "--run {exp_id}"},
    ])
    result = CliRunner().invoke(experiment.cli, ["launch", str(path)])
    assert result.exit_code == 0, result.output
    seds = [c for c in fake_remote["remote"] if c.startswith("sed -i")]
    assert len(seds) == 1 and "__000_train__" in seds[0]
    run_dir = next(isolated_runs.iterdir())
    assert "afterok:__000_train__" in (run_dir / "scripts" / "job_000_eval.sh").read_text()


def test_unknown_placeholder_fails_fast(gcfg_env, fake_remote, isolated_runs, tmp_path):
    path = _write_exp(tmp_path, args="--config {config_path} --x {nope}")
    result = CliRunner().invoke(experiment.cli, ["launch", str(path), "--dry-run"])
    assert result.exit_code == 2
    assert "unknown placeholder {nope}" in result.output


def test_missing_account_is_reported(gcfg_env, fake_remote, isolated_runs, tmp_path, monkeypatch):
    monkeypatch.delenv("GILBRETH_ACCOUNT")
    path = _write_exp(tmp_path)
    result = CliRunner().invoke(experiment.cli, ["launch", str(path), "--dry-run"])
    assert result.exit_code == 1 and "SLURM account not set" in result.output


def test_gilbreth_run_command(gcfg_env, fake_remote, isolated_runs, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(run_mod.run, ["-g", "2", "-t", "01:00:00", "-p", "a10", "--", "python", "train.py", "--lr", "1e-3"])
    assert result.exit_code == 0, result.output
    script = (next(isolated_runs.iterdir()) / "job.sh").read_text()
    assert "#SBATCH --gpus-per-node=2" in script and "#SBATCH -p a10" in script
    assert "python train.py --lr 1e-3" in script
    assert any(c.startswith("sbatch") for c in fake_remote["remote"])
    assert "Submitted job" in result.output


def test_gilbreth_run_torchrun(gcfg_env, fake_remote, isolated_runs, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(run_mod.run, ["--torchrun", "-g", "4", "--dry-run", "--", "python", "train.py", "--lr", "1e-3"])
    assert result.exit_code == 0, result.output
    script = (next(isolated_runs.iterdir()) / "job.sh").read_text()
    assert "torchrun --standalone --nproc_per_node=4 train.py --lr 1e-3" in script


def test_exp_init_writes_template(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(experiment.cli, ["init"])
    assert result.exit_code == 0
    assert yaml.safe_load((tmp_path / "experiment.yaml").read_text())["project"] == "my_project"


def test_missing_env_name_falls_back_to_system_python(gcfg_env, fake_remote, isolated_runs, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(run_mod.run, ["--dry-run", "--", "python", "-c", "print(1)"])
    assert result.exit_code == 0, result.output
    assert "no environment.name" in result.output
    script = (next(isolated_runs.iterdir()) / "job.sh").read_text()
    assert "conda activate" not in script


def test_run_options_before_command_and_quoting(gcfg_env, fake_remote, isolated_runs, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # no "--" separator; the wrapped command's own -t / --lr must not be eaten by gilbreth-run
    result = CliRunner().invoke(run_mod.run, ["-g", "1", "-t", "00:10:00", "--dry-run", "python", "-c", "print('hi')", "-t", "5", "--lr", "1e-3"])
    assert result.exit_code == 0, result.output
    script = (next(isolated_runs.iterdir()) / "job.sh").read_text()
    assert "#SBATCH --time=00:10:00" in script
    # execute the generated run line locally: bash must see exactly the original argv
    line = next(ln for ln in script.splitlines() if ln.startswith("$RUNNER bash -c "))
    proc = subprocess.run(["bash", "-c", "RUNNER=; " + line.replace("python -c", "python3 -c", 1)], check=False, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "hi"


def test_run_apptainer_without_container_is_usage_error(gcfg_env, fake_remote, isolated_runs, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(run_mod.run, ["--env-type", "apptainer", "--dry-run", "python", "x.py"])
    assert result.exit_code == 2 and "--container" in result.output


def test_parse_sacct_states_handles_array_jobs():
    from gilbreth_cli.experiment import parse_sacct_states

    out = "100_0|COMPLETED|0:0|00:01:00\n100_1|RUNNING|0:0|00:00:30\n101|FAILED|1:0|00:00:05\n"
    done = parse_sacct_states(out, {"100", "101"})
    assert done == {"101": ("FAILED", "1:0", "00:00:05")}
    out = "100_0|COMPLETED|0:0|00:01:00\n100_1|CANCELLED by 123|0:0|00:00:30\n"
    assert parse_sacct_states(out, {"100"})["100"][0] == "CANCELLED"


def test_launch_rejects_bad_pipeline_stage(gcfg_env, fake_remote, isolated_runs, tmp_path):
    path = _write_exp(tmp_path, sweep={}, pipeline=[{"name": "train"}])
    result = CliRunner().invoke(experiment.cli, ["launch", str(path), "--dry-run"])
    assert result.exit_code == 2 and "pipeline stages need" in result.output


def test_launch_reports_failed_sbatch(gcfg_env, fake_remote, isolated_runs, tmp_path, monkeypatch):
    import subprocess

    def failing(cfg, command, check=False):
        fake_remote["remote"].append(command)
        rc = 1 if command.startswith("sbatch") else 0
        return subprocess.CompletedProcess([], rc, stdout="", stderr="sbatch: error: Invalid account")

    monkeypatch.setattr(experiment, "run_remote", failing)
    path = _write_exp(tmp_path, sweep={"method": "grid", "mode": "separate", "parameters": {"lr": [0.1]}})
    result = CliRunner().invoke(experiment.cli, ["launch", str(path)])
    assert result.exit_code == 1 and "Invalid account" in result.output


def test_project_names_are_sanitised(gcfg_env, fake_remote, isolated_runs, tmp_path):
    path = _write_exp(tmp_path, project="My Project (v2)")
    result = CliRunner().invoke(experiment.cli, ["launch", str(path), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "projects/My_Project_v2" in result.output
