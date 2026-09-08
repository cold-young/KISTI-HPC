#!/usr/bin/env python3
"""Refuse to let a real Purdue credential into the repository (AC-2.6).

``configs/rcac.yaml`` is git-tracked and ships with placeholders.  The
documented primary path is the environment, which needs no file edit -- but a
user who edits the file anyway dirties the tree and can commit their real
career-account id or Slurm account.

This checker parses the YAML and asserts that ``ssh.user`` and
``slurm.account`` are still placeholder members after ``.strip()``.  It parses
rather than greps because the original regex form was verified *vacuous*: it
passed against ``user: "abc"``.

Two modes:

``--worktree`` (default)
    Check the files on disk.  This is what the test suite calls.

``--staged``
    Check the versions in the git index, which is what the pre-commit hook
    calls.  The distinction matters: a user can have a real id in the working
    tree (harmless, it never leaves the machine) while staging only unrelated
    hunks, and only the staged bytes are what a commit would publish.

Install the hook (one line, from the repo root)::

    git config core.hooksPath .githooks

CI runs this same check on every push.  The hook is the half that acts before a
credential is committed at all; CI can only report one that already landed.
Running it by hand::

    python3 scripts/check_credentials.py --staged
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Every tracked YAML that can carry a credential.  ``rcac_cli/configs/`` holds
#: the copies that ship inside the wheel, so it is exactly as dangerous as the
#: checkout copy and is checked identically.
CONFIG_GLOBS = ("configs/*.yaml", "rcac_cli/configs/*.yaml")

#: ``(dotted path, environment variable)`` for each guarded field.
GUARDED_FIELDS = (
    ("ssh.user", "RCAC_USER"),
    ("slurm.account", "RCAC_ACCOUNT"),
)

INSTALL_HINT = "git config core.hooksPath .githooks"


def placeholders() -> frozenset:
    """The placeholder set, imported from the package so it cannot drift.

    Falls back to nothing -- an unimportable package is reported as a failure
    rather than silently waved through, because "the check could not run" and
    "the check passed" must never look the same.
    """
    sys.path.insert(0, str(REPO_ROOT))
    from rcac_cli.config import PLACEHOLDERS

    return PLACEHOLDERS


def config_paths() -> list[str]:
    """Repo-relative paths of every config this checker guards."""
    found: list[str] = []
    for glob in CONFIG_GLOBS:
        found += [
            p.relative_to(REPO_ROOT).as_posix() for p in sorted(REPO_ROOT.glob(glob))
        ]
    return found


def staged_paths() -> list[str]:
    """The guarded configs that are actually staged for commit."""
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    staged = set(result.stdout.split())
    return [p for p in config_paths() if p in staged]


def read_staged(rel_path: str) -> str:
    """The staged bytes of *rel_path*, which are what a commit would publish."""
    result = subprocess.run(
        ["git", "show", f":{rel_path}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def dig(doc: object, dotted: str) -> Optional[object]:
    """Walk a dotted path through nested mappings; ``None`` if any hop is missing."""
    node = doc
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def check_document(rel_path: str, text: str, allowed: frozenset) -> list[str]:
    """Violations in one parsed config."""
    import yaml

    doc = yaml.safe_load(text) or {}
    problems: list[str] = []
    for dotted, env_var in GUARDED_FIELDS:
        value = dig(doc, dotted)
        if value is None:
            continue  # the field is absent; nothing to leak
        if str(value).strip() not in allowed:
            problems.append(
                f"{rel_path}: `{dotted}` is set to a real value. "
                f"Restore the placeholder and export {env_var} instead."
            )
    return problems


def run(paths: Iterable[str], staged: bool) -> list[str]:
    allowed = placeholders()
    problems: list[str] = []
    for rel_path in paths:
        text = read_staged(rel_path) if staged else (REPO_ROOT / rel_path).read_text()
        problems += check_document(rel_path, text, allowed)
    return problems


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--staged",
        action="store_true",
        help="check the git index instead of the working tree (pre-commit mode)",
    )
    args = parser.parse_args(argv)

    paths = staged_paths() if args.staged else config_paths()
    if not paths:
        return 0

    try:
        problems = run(paths, staged=args.staged)
    except Exception as exc:  # noqa: BLE001 - any failure here blocks, by design
        print(
            f"Cannot verify credentials ({type(exc).__name__}: {exc}).\n"
            "  A check that could not run must not look like a check that "
            "passed, so this blocks.\n"
            f"  Set up the dev environment (see the plan's bootstrap) or "
            f"install PyYAML, then retry.\n"
            f"  Installed hook: {INSTALL_HINT}",
            file=sys.stderr,
        )
        return 1
    if not problems:
        return 0

    print("Refusing the commit: a credential would be committed.", file=sys.stderr)
    for problem in problems:
        print(f"  {problem}", file=sys.stderr)
    print(
        "\nCredentials belong in the environment, not in a tracked file:\n"
        "  export RCAC_USER=<your Purdue career account id>\n"
        "  export RCAC_ACCOUNT=<your Slurm account>\n"
        "The environment always wins over the config file, so you never need "
        "to edit it.\n",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
