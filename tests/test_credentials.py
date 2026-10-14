"""No real credential may reach the repository (AC-2.6).

``configs/rcac.yaml`` is git-tracked.  The documented primary path is the
environment, which needs no file edit -- but nothing stops a user from typing
their Purdue id into the file and committing it.

The plan recorded this honestly as its weakest guarantee: with no CI in this
repo (``.github/workflows/deploy.yml`` is an orphan mkdocs job), a check that
only runs in CI is decorative.  So the check lives in
``scripts/check_credentials.py`` and is wired into a **pre-commit hook**, and
these tests exercise the same code the hook runs -- including a seeded bad
commit, so "the hook blocks" is asserted rather than asserted-about.

The check *parses* the YAML.  The regex form it replaced was verified vacuous:
it passed against ``user: "abc"``.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKER = REPO_ROOT / "scripts" / "check_credentials.py"
HOOK = REPO_ROOT / ".githooks" / "pre-commit"


def load_checker():
    """Import ``scripts/check_credentials.py`` by path.

    It is deliberately not a package module: the hook must be able to run it
    with a bare interpreter from a fresh clone, before anything is installed.
    """
    spec = importlib.util.spec_from_file_location("check_credentials", CHECKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check_credentials = load_checker()


# --------------------------------------------------------------------------
# AC-2.6 -- the parse gate
# --------------------------------------------------------------------------


def test_every_tracked_config_carries_only_placeholders():
    """The live assertion: this repo, right now, holds no credential."""
    problems = check_credentials.run(check_credentials.config_paths(), staged=False)
    assert not problems, "\n".join(problems)


def test_the_gate_covers_both_the_checkout_and_the_packaged_configs():
    """``rcac_cli/configs/`` ships inside the wheel, so it is equally dangerous.

    Guarding only ``configs/`` would leave a hole exactly the size of the files
    a user actually installs.
    """
    paths = check_credentials.config_paths()
    assert "configs/rcac.yaml" in paths
    assert "rcac_cli/configs/rcac.yaml" in paths


def test_the_gate_is_not_vacuous_against_a_real_looking_value():
    """The control that iteration 1's regex failed.

    A gate that cannot fail is worse than no gate, because it reads like
    protection.  This seeds the exact value shape a user would type.
    """
    text = yaml.safe_dump(
        {"ssh": {"user": "abc"}, "slurm": {"account": "YOUR_QUEUE"}}
    )
    problems = check_credentials.check_document(
        "seeded.yaml", text, check_credentials.placeholders()
    )
    assert len(problems) == 1
    assert "ssh.user" in problems[0]
    assert "RCAC_USER" in problems[0]


def test_the_gate_catches_the_account_too():
    text = yaml.safe_dump(
        {"ssh": {"user": "YOUR_PURDUE_ID"}, "slurm": {"account": "bagchi-gpu"}}
    )
    problems = check_credentials.check_document(
        "seeded.yaml", text, check_credentials.placeholders()
    )
    assert len(problems) == 1
    assert "slurm.account" in problems[0]
    assert "RCAC_ACCOUNT" in problems[0]


def test_trailing_whitespace_does_not_defeat_the_gate():
    """Placeholder membership is tested after ``.strip()``.

    Not hypothetical: the shipped YAML carried trailing whitespace after the
    value, which is what makes the ``.strip()`` in the comparison load-bearing
    rather than defensive.
    """
    text = 'ssh:\n  user: "YOUR_PURDUE_ID"   \nslurm:\n  account: "YOUR_QUEUE"\n'
    assert not check_credentials.check_document(
        "seeded.yaml", text, check_credentials.placeholders()
    )


def test_an_absent_field_is_not_a_violation():
    """``defaults.yaml`` has neither field; it must not be reported."""
    text = yaml.safe_dump({"slurm": {"nodes": 1}})
    assert not check_credentials.check_document(
        "seeded.yaml", text, check_credentials.placeholders()
    )


# --------------------------------------------------------------------------
# The hook -- asserted by making it block a real commit
# --------------------------------------------------------------------------


def git(*args, cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.update(
        GIT_AUTHOR_NAME="t",
        GIT_AUTHOR_EMAIL="t@example.invalid",
        GIT_COMMITTER_NAME="t",
        GIT_COMMITTER_EMAIL="t@example.invalid",
    )
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=check, env=env
    )


@pytest.fixture
def hooked_clone(tmp_path):
    """A throwaway repo carrying this repo's hook, checker, and package.

    Runs against a real ``git commit`` rather than calling the hook directly,
    because "the script exits 1" and "git refuses the commit" are different
    claims and only the second one is the guarantee.
    """
    repo = tmp_path / "repo"
    (repo / "configs").mkdir(parents=True)
    (repo / "scripts").mkdir()
    (repo / ".githooks").mkdir()
    (repo / "rcac_cli").mkdir()

    (repo / "scripts" / "check_credentials.py").write_text(
        CHECKER.read_text(), encoding="utf-8"
    )
    hook = repo / ".githooks" / "pre-commit"
    hook.write_text(HOOK.read_text(), encoding="utf-8")
    hook.chmod(0o755)

    # The checker imports PLACEHOLDERS from the package, so the package has to
    # be importable. Copy the real module rather than stubbing it: a stub would
    # let the two definitions drift, which is the thing the import prevents.
    (repo / "rcac_cli" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "rcac_cli" / "config.py").write_text(
        (REPO_ROOT / "rcac_cli" / "config.py").read_text(), encoding="utf-8"
    )

    # The hook looks for `.venv/bin/python` because the checker imports
    # PLACEHOLDERS from the package (single source of truth) and PyYAML to
    # parse -- a bare system python3 has neither, and blocking is the correct
    # response to "the check could not run". A shim rather than a symlink: a
    # symlinked interpreter has no pyvenv.cfg beside it, so it silently
    # resolves to the base prefix and loses exactly the packages it was
    # chosen for.
    (repo / ".venv" / "bin").mkdir(parents=True)
    shim = repo / ".venv" / "bin" / "python"
    shim.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n', encoding="utf-8")
    shim.chmod(0o755)

    git("init", "-q", cwd=repo)
    git("config", "core.hooksPath", ".githooks", cwd=repo)
    (repo / "configs" / "rcac.yaml").write_text(
        (REPO_ROOT / "configs" / "rcac.yaml").read_text(), encoding="utf-8"
    )
    git("add", "-A", cwd=repo)
    git("commit", "-q", "-m", "baseline", cwd=repo)
    return repo


def test_hook_allows_a_commit_that_keeps_the_placeholders(hooked_clone):
    """The other half of the control: the hook must not block ordinary work."""
    (hooked_clone / "configs" / "rcac.yaml").write_text(
        (REPO_ROOT / "configs" / "rcac.yaml").read_text() + "\n# a harmless comment\n",
        encoding="utf-8",
    )
    git("add", "-A", cwd=hooked_clone)

    result = git("commit", "-m", "harmless", cwd=hooked_clone, check=False)

    assert result.returncode == 0, result.stdout + result.stderr


def test_hook_blocks_a_commit_carrying_a_real_credential(hooked_clone):
    """The guarantee, end to end: git itself refuses."""
    config = hooked_clone / "configs" / "rcac.yaml"
    config.write_text(
        config.read_text().replace('user: "YOUR_PURDUE_ID"', 'user: "cahn"'),
        encoding="utf-8",
    )
    git("add", "-A", cwd=hooked_clone)
    head_before = git("rev-parse", "HEAD", cwd=hooked_clone).stdout.strip()

    result = git("commit", "-m", "leak", cwd=hooked_clone, check=False)

    assert result.returncode != 0, "the hook let a credential through"
    assert "ssh.user" in result.stderr
    assert "RCAC_USER" in result.stderr
    head_after = git("rev-parse", "HEAD", cwd=hooked_clone).stdout.strip()
    assert head_after == head_before, "a commit was created despite the hook"


def test_hook_reads_the_index_not_the_working_tree(hooked_clone):
    """Only staged bytes can be published.

    A real id in the working tree is harmless -- it never leaves the machine.
    Blocking on it would train the user to reach for ``--no-verify``, which
    would disable the check for the case that matters.
    """
    config = hooked_clone / "configs" / "rcac.yaml"
    other = hooked_clone / "notes.txt"
    other.write_text("unrelated\n", encoding="utf-8")
    git("add", "notes.txt", cwd=hooked_clone)
    config.write_text(
        config.read_text().replace('user: "YOUR_PURDUE_ID"', 'user: "cahn"'),
        encoding="utf-8",
    )  # dirty, but NOT staged

    result = git("commit", "-m", "unrelated", cwd=hooked_clone, check=False)

    assert result.returncode == 0, result.stdout + result.stderr


def test_hook_is_executable_and_documents_its_installer():
    """A hook that is not executable is silently skipped by git."""
    assert os.access(HOOK, os.X_OK), f"{HOOK} is not executable; git would skip it"
    assert check_credentials.INSTALL_HINT in HOOK.read_text()


def test_checker_reports_an_unimportable_package_rather_than_passing(monkeypatch):
    """"The check could not run" must never look like "the check passed"."""
    monkeypatch.setitem(sys.modules, "rcac_cli.config", None)
    with pytest.raises(Exception):
        check_credentials.placeholders()
