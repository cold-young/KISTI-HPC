"""AC-1.3 -- every console script answers ``--help`` and exits 0.

This is the only criterion in the suite that runs the *installed* entry points
rather than importing the package, and that distinction is the whole point.

``pytest`` cannot see the failure this catches.  When it runs, the repository
root is already on ``sys.path``, so ``import rcac_cli`` resolves from the
source tree whether or not the package is installed.  A console script has no
such luck: its ``sys.path[0]`` is the ``bin/`` directory it lives in, so it
finds ``rcac_cli`` only through the install.  The two can therefore disagree,
and on this machine they repeatedly have -- a green suite of several hundred
tests alongside twelve entry points that all died with ``ModuleNotFoundError``.

The mechanism, recorded because the symptom is otherwise unattributable: the
virtualenv's ``.pth`` files acquire the macOS ``UF_HIDDEN`` flag (the checkout
lives under ``~/Desktop``, which iCloud syncs), and CPython 3.14's ``site.py``
silently skips hidden ``.pth`` files.  The editable install is not uninstalled,
it is *deactivated*, and nothing says so.  :data:`REMEDY` is the fix, and
:func:`_diagnose` prints it, because a reader who hits this needs the command
rather than the diagnosis.

AC-1.3 was, until this module existed, neither machine-checked nor tagged as
excused -- it was enforced only by whoever remembered to loop over the scripts
by hand.  :func:`test_ac_1_3_is_not_excused_from_machine_checking` pins the
resolution so it cannot drift back.
"""

from __future__ import annotations

import os
import subprocess
import sys
import sysconfig
import tomllib
from pathlib import Path

import pytest

from tests.test_repo_hygiene import HYGIENE_SCOPE, UNCHECKED_AC_IDS

ROOT = Path(__file__).resolve().parent.parent

#: Where the interpreter running this suite keeps its installed entry points.
#: Taken from ``sysconfig`` rather than from ``sys.executable``: resolving that
#: path follows the virtualenv's symlink back to the base interpreter, whose
#: ``bin/`` holds no ``rcac-*`` at all.
BIN_DIR = Path(sysconfig.get_path("scripts"))

#: The venv-local repair for the deactivated-editable-install failure.  Derived
#: from the running interpreter rather than written out, so it names the
#: directory that is actually broken instead of the one that was broken the day
#: this was written.
REMEDY = f"chflags -R nohidden {sysconfig.get_paths()['purelib']}"


def _console_scripts() -> dict[str, str]:
    """``[project.scripts]``, read from ``pyproject.toml`` (AC-1.2's source)."""
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["project"]["scripts"]


SCRIPTS = sorted(_console_scripts())


def _diagnose(script: str, exe: Path, proc: subprocess.CompletedProcess) -> str:
    stderr = (proc.stderr or "").strip()
    body = (
        f"AC-1.3: `{script} --help` exited {proc.returncode}, not 0.\n"
        f"  script:   {exe}\n"
        f"  stderr:   {stderr[-2000:] or '(empty)'}\n"
    )
    if "ModuleNotFoundError" in stderr or "No module named" in stderr:
        body += (
            "\n  The package is not importable from the entry point, which is\n"
            "  NOT the same as not being installed. On macOS the venv's .pth\n"
            "  files acquire the UF_HIDDEN flag (iCloud syncs ~/Desktop) and\n"
            "  CPython 3.14's site.py silently skips hidden .pth files, so the\n"
            "  editable install deactivates. pytest masks it because the repo\n"
            "  root is already on sys.path -- which is exactly why this test\n"
            "  runs the script instead of importing the module.\n"
            f"\n  Remedy:\n      {REMEDY}\n"
            "\n  Run that immediately before any console-script check; the flag\n"
            "  has come back mid-session, with the suite staying green.\n"
        )
    else:
        body += (
            "\n  If the script is missing entirely, the package is not\n"
            "  installed into the interpreter running this suite:\n"
            f"      {sys.executable} -m pip install -e '.[dev]'\n"
            f"  If it exists but cannot import rcac_cli, try:\n      {REMEDY}\n"
        )
    return body


def test_the_script_list_is_not_empty() -> None:
    """A vacuity guard (AC-0.7's shape), not a second count assertion.

    ``tests/test_readme.py`` owns AC-1.2 -- exactly twelve scripts, all
    ``rcac-*``.  This one exists only so an empty ``[project.scripts]`` table
    would fail here rather than silently parametrise the test below over
    nothing.
    """
    assert SCRIPTS, "pyproject.toml declares no console scripts at all"
    assert len(SCRIPTS) >= 12, SCRIPTS


@pytest.mark.parametrize("script", SCRIPTS)
def test_console_script_responds_to_help(script: str, tmp_path: Path) -> None:
    """AC-1.3 -- the installed entry point runs and exits 0.

    Two deliberate choices in how the subprocess is built, both of which exist
    to stop this test masking the failure the way ``pytest`` does:

    * ``cwd`` is a temporary directory, never the checkout.  From inside the
      repository a stray ``''`` on ``sys.path`` would import ``rcac_cli`` from
      the source tree and the entry point would appear healthy while being
      broken for every user who runs it from anywhere else.
    * ``PYTHONPATH`` is stripped for the same reason -- it is the other way the
      source tree reaches a subprocess that the install did not put it in.
    """
    exe = BIN_DIR / script
    assert exe.exists(), (
        f"AC-1.3: {script} is declared in pyproject.toml but no executable "
        f"exists at {exe}.\n"
        f"  Install the package into the interpreter running this suite:\n"
        f"      {sys.executable} -m pip install -e '.[dev]'"
    )

    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    proc = subprocess.run(
        [str(exe), "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, _diagnose(script, exe, proc)
    assert proc.stdout.strip(), f"{script} --help printed nothing"
    assert "Usage:" in proc.stdout, (
        f"{script} --help exited 0 but printed no usage line:\n{proc.stdout}"
    )


def test_ac_1_3_is_not_excused_from_machine_checking() -> None:
    """AC-0.3(b) -- and this module is what makes that claim true.

    AC-0.3(b) exists so the plan's prose cannot drift from what is actually
    checked.  AC-1.3 drifted anyway, in the one direction the mechanism does
    not cover: it was absent from :data:`UNCHECKED_AC_IDS` *and* absent from
    every test, so nothing in the repository was in a position to notice.
    Asserting it here ties the resolution to the module that resolves it -- a
    later edit that files AC-1.3 as excused has to delete this test to do it.
    """
    assert "AC-1.3" not in UNCHECKED_AC_IDS, (
        "AC-1.3 is machine-checked by this module; it must not be listed among "
        "the criteria excused from machine checking"
    )
    excused = {row.ac_id for row in HYGIENE_SCOPE if row.kind in {"manual", "shell"}}
    assert "AC-1.3" not in excused, (
        f"AC-1.3 is tagged as an unchecked hygiene row: {sorted(excused)}"
    )
