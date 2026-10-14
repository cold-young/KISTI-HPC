import subprocess

from click.testing import CliRunner

from gilbreth_cli import sync
from gilbreth_cli.sync import is_excluded

PATTERNS = [".git/", "__pycache__/", "*.pyc", "wandb/", "*.pt", "*.pth", "outputs/", "logs/"]


def test_is_excluded_matches_rsync_semantics():
    assert is_excluded("model.pt", PATTERNS)
    assert is_excluded("ckpt/best.pth", PATTERNS)
    assert is_excluded("outputs/run1/x.txt", PATTERNS)
    assert is_excluded(".git/HEAD", PATTERNS)
    # substring false positives from the old implementation
    assert not is_excluded("classifier.py", PATTERNS)
    assert not is_excluded("scripts/train.py", PATTERNS)
    assert not is_excluded("src/prompt.py", PATTERNS)
    assert not is_excluded("logs.py", PATTERNS)
    assert not is_excluded("my_logs_dir/notes.txt", PATTERNS)


def test_push_data_accepts_a_single_file(gcfg_env, fake_remote, tmp_path):
    f = tmp_path / "sim.sif"
    f.write_bytes(b"x")
    result = CliRunner().invoke(sync.cli, ["push-data", str(f)])
    assert result.exit_code == 0, result.output
    src, dst = fake_remote["rsync"][-1]
    assert src == str(f)
    assert dst.endswith(":/scratch/gilbreth/alice/data/")
    assert fake_remote["remote"][-1] == "mkdir -p /scratch/gilbreth/alice/data"


def test_push_uses_trailing_slash_and_excludes(gcfg_env, fake_remote, tmp_path, monkeypatch):
    captured = {}

    def rsync(cfg, src, dst, extra=None, quiet=False, dry_run=False, capture=False):
        captured.update(src=src, dst=dst, extra=extra)
        return subprocess.CompletedProcess([], 0, stdout="", stderr="")

    monkeypatch.setattr(sync, "rsync", rsync)
    proj = tmp_path / "proj"
    proj.mkdir()
    result = CliRunner().invoke(sync.cli, ["push", str(proj)])
    assert result.exit_code == 0, result.output
    assert captured["src"] == f"{proj}/"
    assert captured["dst"] == "alice@gilbreth.rcac.purdue.edu:/scratch/gilbreth/alice/projects/proj/"
    assert "--exclude" in captured["extra"] and "wandb/" in captured["extra"]
