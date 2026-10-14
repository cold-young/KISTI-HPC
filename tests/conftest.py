"""Shared fixtures: a Gilbreth config that never touches the network."""

import pytest

from gilbreth_cli import config as gcfg


@pytest.fixture
def gcfg_env(monkeypatch):
    monkeypatch.setenv(gcfg.ENV_USER, "alice")
    monkeypatch.setenv(gcfg.ENV_ACCOUNT, "primelab")
    monkeypatch.delenv(gcfg.ENV_CONFIG, raising=False)
    return gcfg.get_config()


@pytest.fixture
def fake_remote(monkeypatch):
    """Stub every SSH / rsync seam and record the remote commands that were issued."""
    import subprocess

    calls = {"remote": [], "uploads": [], "rsync": []}

    def run_remote(cfg, command, check=False):
        calls["remote"].append(command)
        if command.startswith("sbatch"):
            return subprocess.CompletedProcess([], 0, stdout=f"{1000 + len(calls['remote'])}\n", stderr="")
        return subprocess.CompletedProcess([], 0, stdout="", stderr="")

    def upload(cfg, local, remote, quiet=True):
        calls["uploads"].append((str(local), remote))
        return True

    def rsync(cfg, src, dst, extra=None, quiet=False, dry_run=False, capture=False):
        calls["rsync"].append((src, dst))
        return subprocess.CompletedProcess([], 0, stdout="", stderr="")

    for mod in ("gilbreth_cli.experiment", "gilbreth_cli.run", "gilbreth_cli.env", "gilbreth_cli.info", "gilbreth_cli.sync"):
        m = __import__(mod, fromlist=["x"])
        for name, fn in (("run_remote", run_remote), ("upload", upload), ("rsync", rsync)):
            if hasattr(m, name):
                monkeypatch.setattr(m, name, fn)
    return calls


@pytest.fixture
def isolated_runs(monkeypatch, tmp_path):
    from gilbreth_cli import experiment

    runs = tmp_path / "runs"
    monkeypatch.setattr(experiment, "LOCAL_RUNS_DIR", runs)  # run.py goes through experiment._unique_run_dir
    return runs
