from gilbreth_cli import config as gcfg


def test_env_overrides_and_user_substitution(gcfg_env):
    cfg = gcfg_env
    assert cfg["ssh"]["user"] == "alice"
    assert cfg["slurm"]["account"] == "primelab"
    assert cfg["paths"]["scratch"] == "/scratch/gilbreth/alice"
    assert cfg["paths"]["envs"] == "/scratch/gilbreth/alice/envs"
    assert cfg["environment"]["env_vars"]["HF_HOME"].startswith("/scratch/gilbreth/alice")


def test_missing_user_exits(monkeypatch):
    import pytest

    monkeypatch.delenv(gcfg.ENV_USER, raising=False)
    cfg = gcfg.get_config()
    cfg["ssh"]["user"] = ""
    with pytest.raises(SystemExit):
        gcfg.require_user(cfg)


def test_missing_account_exits(gcfg_env):
    import pytest

    gcfg_env["slurm"]["account"] = ""
    with pytest.raises(SystemExit):
        gcfg.require_account(gcfg_env)
    assert gcfg.require_account(gcfg_env, "other") == "other"


def test_ssh_config_block_is_idempotent(gcfg_env, tmp_path):
    from gilbreth_cli.connect import MARKER_BEGIN, write_ssh_config

    path = tmp_path / "config"
    path.write_text("Host other\n    HostName example.com\n")
    write_ssh_config(gcfg_env, path)
    write_ssh_config(gcfg_env, path)
    text = path.read_text()
    assert text.count(MARKER_BEGIN) == 1
    assert "Host other" in text
    assert "HostName gilbreth.rcac.purdue.edu" in text
    assert "User alice" in text
