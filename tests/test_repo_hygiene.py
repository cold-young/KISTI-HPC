"""Repo-wide content hygiene gates (AC-0).

This module is the mechanism named by principle P6 of
``.omc/plans/rcac-cli-gilbreth-port.md``: *gates are tests, not shell, and
scope is derived, not declared.*  Every repo-wide content assertion in the
``neuron-cli`` -> ``rcac-cli`` port lives here, expressed in ``pathlib`` + ``re``
(AC-0.1).  Nothing shells out to ``grep``.

The single source of truth is :data:`HYGIENE_SCOPE`.  Each row names an
acceptance criterion, the earliest commit at which it can hold, the paths it
covers, its allowlist, and its *kind* (``test`` / ``manual`` / ``shell``).
Three properties follow from the table rather than from prose (AC-0.3):

(a) The runtime package root is *resolved*, never hardcoded -- :data:`PKG` is
    ``neuron_cli`` today and becomes ``rcac_cli`` at C1 with no edit here.
(b) ``manual`` and ``shell`` criteria are tagged in the table, so the plan's
    prose cannot drift away from what is actually machine-checked.
(c) ``.md`` rows carry ``earliest_commit = "C7"``; the READMEs are not
    rewritten until then.

:data:`ACTIVE_COMMIT` gates which rows are *enforced*.  A row whose
``earliest_commit`` is later than :data:`ACTIVE_COMMIT` is filtered out of the
active set -- it is not enforced and it is not skipped.  ``pytest.skip`` is
never used here; AC-7.6 requires the suite to report zero skips.

The class-closing assertion is :func:`test_meta_scope_is_derived_from_change_lists`
(AC-0.6).  It runs against *every* row at *every* commit, enforced or not, and
turns a scope error into an authoring-time test failure instead of a review
finding three iterations later.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, NamedTuple, Optional, Sequence, Tuple

import pytest

# --------------------------------------------------------------------------
# Roots
# --------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

#: AC-0.3(a) -- the runtime package root, resolved rather than declared.  This
#: module must work unchanged at C0 (``neuron_cli/``) and from C1
#: (``rcac_cli/``).  This is the ONLY place either name may be written.
PKG = next(p for p in ("rcac_cli", "neuron_cli") if (ROOT / p).is_dir())


# --------------------------------------------------------------------------
# AC-0.2 -- the exclusion walk
# --------------------------------------------------------------------------

#: Directory *names* pruned anywhere in the tree.  ``.omc/`` is load-bearing:
#: ``.omc/plans/`` holds the port plan, which uses the word "neuron" hundreds
#: of times and would make every ban row fail.
EXCLUDED_DIR_NAMES = frozenset(
    {".git", ".venv", ".omc", ".history", "__pycache__", "build", "dist"}
)

#: Directory *suffixes* pruned anywhere in the tree (``*.egg-info/``).
EXCLUDED_DIR_SUFFIXES: Tuple[str, ...] = (".egg-info",)

#: Repo-relative directory prefixes pruned.  ``tests/golden/`` holds rendered
#: snapshots that are full of banned literals by construction.
EXCLUDED_REL_PREFIXES: Tuple[str, ...] = ("tests/golden",)

#: Individual files pruned.  This module is the scanner, not the scanned: it
#: must spell out every banned literal in order to ban it, and :data:`PKG` must
#: name both packages in order to resolve between them.  This is the same
#: argument AC-0.8 makes for ``_legacy.py``, but structural rather than a
#: row allowlist -- an allowlist entry would exempt the file from one row,
#: whereas the gate definitions belong outside the corpus entirely.
EXCLUDED_REL_FILES: Tuple[str, ...] = ("tests/test_repo_hygiene.py",)


# --------------------------------------------------------------------------
# Commit ordering
# --------------------------------------------------------------------------

COMMIT_ORDER: Tuple[str, ...] = (
    "C0",
    "C1",
    "C2",
    "C3a",
    "C3b",
    "C3c",
    "C4",
    "C5",
    "C6",
    "C7",
)

#: The commit currently being authored.  Bump this as each commit lands; every
#: row whose ``earliest_commit`` is <= this value becomes enforced.  Rows above
#: it are inactive *by construction* -- filtered out, never skipped (AC-7.6).
ACTIVE_COMMIT = "C1"


def commit_index(commit: str) -> int:
    """Position of *commit* in :data:`COMMIT_ORDER`."""
    try:
        return COMMIT_ORDER.index(commit)
    except ValueError:  # pragma: no cover - guarded by test_commit_ids_are_known
        raise AssertionError(
            f"unknown commit id {commit!r}; known ids are {list(COMMIT_ORDER)}"
        ) from None


# --------------------------------------------------------------------------
# AC-0.6 -- commit change lists
#
# Seeded from section 4 of the plan.  Patterns are repo-relative globs; ``{pkg}``
# is substituted with the resolved package root, so a single entry describes the
# path on both sides of the C1 rename.  These are *authoring* declarations: the
# meta-test asserts that no file may still carry a banned literal past the
# commit whose change list claims to touch it.
# --------------------------------------------------------------------------

COMMIT_CHANGE_LISTS: dict[str, set[str]] = {
    # C0 -- harden the baseline.
    "C0": {
        "tests/test_repo_hygiene.py",
        "tests/test_rendered_shape.py",
        "tests/test_golden.py",
        "tests/test_scheduler.py",
        "tests/golden/**",
        "{pkg}/scheduler.py",
    },
    # C1 -- mechanical rename across 24 files.  `git mv neuron_cli rcac_cli`
    # and `git mv configs/neuron.yaml configs/rcac.yaml` mean the whole package
    # plus both config filenames are in scope.
    "C1": {
        "{pkg}/**",
        "pyproject.toml",
        "configs/neuron.yaml",
        "configs/rcac.yaml",
        "configs/defaults.yaml",
        "examples/experiment_simple.yaml",
        "examples/experiment_genesis.yaml",
        "containers/*.def",
        "tests/test_scheduler.py",
        "tests/test_repo_hygiene.py",
        "tests/test_rendered_shape.py",
        "tests/test_golden.py",
    },
    # C2 -- config, credentials, ssh.py, _legacy.py, packaging.  Step 8 moves
    # `templates/` into the package, so both sides of that move are touched.
    "C2": {
        "configs/rcac.yaml",
        "{pkg}/config.py",
        "{pkg}/ssh.py",
        "{pkg}/_legacy.py",
        "{pkg}/connect.py",
        "{pkg}/container.py",
        "{pkg}/sync.py",
        "{pkg}/results.py",
        "{pkg}/partition_selector.py",
        "{pkg}/env.py",
        "{pkg}/keepalive.py",
        "{pkg}/experiment.py",
        "{pkg}/simulator.py",
        "templates/**",
        "{pkg}/templates/**",
        "{pkg}/configs/*.yaml",
        "pyproject.toml",
        "MANIFEST.in",
    },
    # C3a -- gilbreth.py, wired to nothing.
    "C3a": {
        "{pkg}/gilbreth.py",
        "tests/test_gilbreth.py",
    },
    # C3b -- primary submission path.
    "C3b": {
        "{pkg}/templates/**",
        "{pkg}/scheduler.py",
        "{pkg}/experiment.py",
        "configs/defaults.yaml",
        "examples/*.yaml",
        "tests/test_scheduler.py",
    },
    # C3c -- the two non-scheduler producers.
    "C3c": {
        "{pkg}/keepalive.py",
        "{pkg}/simulator.py",
    },
    # C4 -- environment and container runtime.  This is where
    # `templates/pipeline.sh.j2:42` ("KISTI's tight quota") is finally deleted.
    "C4": {
        "{pkg}/env.py",
        "{pkg}/container.py",
        "{pkg}/experiment.py",
        "{pkg}/templates/**",
        "examples/experiment_genesis.yaml",
        "containers/*.def",
    },
    # C5 -- runtime discovery and partition selection.
    "C5": {
        "{pkg}/discovery.py",
        "{pkg}/partition_selector.py",
        "{pkg}/gilbreth.py",
        "tests/fixtures/**",
    },
    # C6 -- retention, simulator, transport.
    "C6": {
        "{pkg}/keepalive.py",
        "{pkg}/simulator.py",
        "{pkg}/sync.py",
        "{pkg}/connect.py",
        "{pkg}/notify.py",
    },
    # C7 -- documentation.  The `.md` layer enters scope here and nowhere
    # earlier (AC-0.3(c)).
    "C7": {
        "README.md",
        "README.ko.md",
        "containers/*.def",
        "examples/*.yaml",
    },
}


# --------------------------------------------------------------------------
# AC-0.3 -- the scope table
# --------------------------------------------------------------------------

#: AC-0.8 -- ``{pkg}/_legacy.py`` is the sole path-exempt module.  It holds
#: ``LEGACY_SSH_MARKER`` and ``LEGACY_ENV_TYPES``, the two literals that AC-2.10
#: and AC-5.9 cannot do their job without.  It does not exist until C2; the
#: allowlist is a pattern list and tolerates that.
LEGACY_ALLOWLIST: Tuple[str, ...] = ("{pkg}/_legacy.py",)

#: AC-0.5 -- ``*.yml`` is scanned alongside ``*.yaml`` and ``*.def`` alongside
#: source.  These bundles exist so no row can forget an extension.
SOURCE_GLOBS: Tuple[str, ...] = (
    "{pkg}/**/*.py",
    "tests/**/*.py",
)
CONFIG_GLOBS: Tuple[str, ...] = (
    "configs/**/*.yaml",
    "configs/**/*.yml",
    "examples/**/*.yaml",
    "examples/**/*.yml",
    "pyproject.toml",
)

#: The cluster-connection config, carved out of AC-1.6 exactly as the templates
#: are.  ``configs/rcac.yaml`` carries ``ssh.login_host`` /``ssh.dm_host``, whose
#: *values* are the KISTI hostnames themselves -- ``neuron.ksc.re.kr`` and
#: ``neuron-dm.ksc.re.kr``.  There is no ``rcac`` spelling of a hostname: the
#: value can only be *replaced* with the Gilbreth host, which is a semantic
#: claim P3 bars from the C1 rename commit (and which C2 step 1 makes, together
#: with the ``Host rcac`` -> ``Host gilbreth`` alias switch in C2 step 7).  The
#: file therefore enters AC-1.6's scope at C2, not C1.
CLUSTER_CONFIG_GLOBS: Tuple[str, ...] = (
    "configs/rcac.yaml",
    "configs/neuron.yaml",
)

#: :data:`CONFIG_GLOBS` minus :data:`CLUSTER_CONFIG_GLOBS`.  Enumerated rather
#: than subtracted because ``configs/**/*.yaml`` would re-match the carve-out.
NON_CLUSTER_CONFIG_GLOBS: Tuple[str, ...] = (
    "configs/defaults.yaml",
    "examples/**/*.yaml",
    "examples/**/*.yml",
    "pyproject.toml",
)

CONTAINER_GLOBS: Tuple[str, ...] = ("containers/**/*.def",)
TEMPLATE_GLOBS: Tuple[str, ...] = (
    "templates/**/*.j2",
    "{pkg}/templates/**/*.j2",
)
MARKDOWN_GLOBS: Tuple[str, ...] = ("*.md",)

#: The history fence -- the only region of a README where legacy branding may
#: survive (AC-1.7, AC-8.4).
HISTORY_FENCE = ("<!-- history:begin -->", "<!-- history:end -->")


class HygieneRow(NamedTuple):
    """One row of :data:`HYGIENE_SCOPE`.

    The five fields required by AC-0.3 come first.  ``pattern`` carries the
    banned regex (``None`` for ``manual`` / ``shell`` rows, which have nothing
    to scan) and ``fence`` names a comment fence whose contents are exempt.
    """

    ac_id: str
    earliest_commit: str
    path_globs: Tuple[str, ...]
    allowlist: Tuple[str, ...]
    kind: str  # "test" | "manual" | "shell"
    pattern: Optional[str] = None
    fence: Optional[Tuple[str, str]] = None
    note: str = ""


HYGIENE_SCOPE: Tuple[HygieneRow, ...] = (
    # ---------------------------------------------------------------- AC-1
    HygieneRow(
        ac_id="AC-1.4",
        earliest_commit="C1",
        path_globs=SOURCE_GLOBS,
        allowlist=(),
        kind="test",
        pattern=r"NEURON_USER",
        note="30 sites across 10 modules; RCAC_USER/RCAC_ACCOUNT replace it.",
    ),
    HygieneRow(
        ac_id="AC-1.5",
        earliest_commit="C1",
        path_globs=SOURCE_GLOBS + CONFIG_GLOBS,
        allowlist=(),
        kind="test",
        pattern=r"neuron-tools",
        note="12 sites: container 5, keepalive 3, env 2, simulator 1, "
        "examples/experiment_genesis.yaml:18.",
    ),
    HygieneRow(
        ac_id="AC-1.6",
        earliest_commit="C1",
        path_globs=SOURCE_GLOBS + NON_CLUSTER_CONFIG_GLOBS + CONTAINER_GLOBS,
        allowlist=LEGACY_ALLOWLIST,
        kind="test",
        pattern=r"(?i)neuron|kisti",
        note="24 files at C0 across *.py/*.j2/*.yaml/*.yml/*.toml/*.def. "
        "Templates are carved out into AC-1.6-templates and the cluster "
        "config into AC-1.6-cluster-config below.",
    ),
    HygieneRow(
        ac_id="AC-1.6-cluster-config",
        earliest_commit="C2",
        path_globs=CLUSTER_CONFIG_GLOBS,
        allowlist=(),
        kind="test",
        pattern=r"(?i)neuron|kisti",
        note="configs/rcac.yaml:3-4 hold ssh.login_host / ssh.dm_host, whose "
        "values are neuron.ksc.re.kr and neuron-dm.ksc.re.kr. A hostname has "
        "no rcac spelling -- it can only be REPLACED with the Gilbreth host, "
        "which is semantic work P3 bars from the C1 rename commit, and which "
        "C2 step 1 performs. Same shape as the AC-1.6-templates carve-out.",
    ),
    HygieneRow(
        ac_id="AC-1.6-templates",
        earliest_commit="C4",
        path_globs=TEMPLATE_GLOBS,
        allowlist=LEGACY_ALLOWLIST,
        kind="test",
        pattern=r"(?i)neuron|kisti",
        note="templates/pipeline.sh.j2:42 says \"KISTI's tight quota\". There is "
        "no rcac equivalent of that sentence, so the line can only be DELETED, "
        "which is semantic work P3 bars from the C1 rename commit. The "
        "templates enter package scope at C2 and are cleaned at C4.",
    ),
    HygieneRow(
        ac_id="AC-1.7",
        earliest_commit="C7",
        path_globs=MARKDOWN_GLOBS,
        allowlist=(),
        kind="test",
        pattern=r"(?i)neuron|kisti",
        fence=HISTORY_FENCE,
        note="AC-0.3(c): the READMEs are not rewritten until C7. Legacy "
        "branding survives only inside the history fence.",
    ),
    HygieneRow(
        ac_id="AC-1.8",
        earliest_commit="C1",
        path_globs=(),
        allowlist=(),
        kind="shell",
        note="`git log --follow` shows pre-rename commits. Not expressible as a "
        "content scan; run the command from section 6.1.",
    ),
    # ---------------------------------------------------------------- AC-2
    HygieneRow(
        ac_id="AC-2.4",
        earliest_commit="C2",
        path_globs=SOURCE_GLOBS + CONFIG_GLOBS,
        allowlist=LEGACY_ALLOWLIST,
        kind="test",
        pattern=r"\bdm_host\b",
        note="Three consumers: container.py:107, sync.py:219, connect.py:44. "
        "All land in C2 with the key removal (R10).",
    ),
    HygieneRow(
        ac_id="AC-2.7",
        earliest_commit="C2",
        path_globs=(),
        allowlist=(),
        kind="shell",
        note="A built wheel contains every source template by name, and "
        "{pkg}/configs/*.yaml. Requires `python -m build`; see section 6.1.",
    ),
    # ---------------------------------------------------------------- AC-5
    HygieneRow(
        ac_id="AC-5.3",
        earliest_commit="C4",
        path_globs=SOURCE_GLOBS + TEMPLATE_GLOBS,
        allowlist=LEGACY_ALLOWLIST,
        kind="test",
        pattern=r"Miniconda|CONDA_ENVS_PATH|\.conda/envs",
        note="Source and rendered output. conda-env-mod replaces all three.",
    ),
    HygieneRow(
        ac_id="AC-5.4",
        earliest_commit="C4",
        path_globs=SOURCE_GLOBS + CONFIG_GLOBS + CONTAINER_GLOBS + TEMPLATE_GLOBS,
        allowlist=LEGACY_ALLOWLIST,
        kind="test",
        pattern=r"(?i)singularity",
        note="28 occurrences across 8 files, 14 in container.py alone. "
        "`.sif` is retained. _legacy.py is allowlisted per AC-0.8.",
    ),
    HygieneRow(
        ac_id="AC-5.11",
        earliest_commit="C4",
        path_globs=SOURCE_GLOBS + TEMPLATE_GLOBS,
        allowlist=(),
        kind="test",
        pattern=r"bashrc|bash_profile|zshrc|\.profile\b|>>\s*~/\.",
        note="No package file writes to or references shell profiles. The "
        "broad form matched profile.d/conda.sh, which C4 step 3 deletes in the "
        "same commit -- latent, not a live defect.",
    ),
    HygieneRow(
        ac_id="AC-5.12",
        earliest_commit="C4",
        path_globs=SOURCE_GLOBS + CONTAINER_GLOBS + TEMPLATE_GLOBS,
        allowlist=(),
        kind="test",
        pattern=r"/apps/ngc/images",
        note="A don't-introduce constraint: /apps/ngc/images/ is "
        "Gautschi-documented only and must not be assumed on Gilbreth. "
        "Currently zero occurrences.",
    ),
    # ---------------------------------------------------------------- AC-6
    HygieneRow(
        ac_id="AC-6.1",
        earliest_commit="C6",
        path_globs=SOURCE_GLOBS + TEMPLATE_GLOBS,
        allowlist=(),
        kind="test",
        pattern=r"15-day|15 day|15일",
        note="Gilbreth scratch purges at 60 days, not KISTI's 15. The *.md "
        "half of this criterion is carried by AC-8.4 at C7.",
    ),
    # ---------------------------------------------------------------- AC-8
    HygieneRow(
        ac_id="AC-8.1",
        earliest_commit="C7",
        path_globs=(),
        allowlist=(),
        kind="manual",
        note="Both READMEs rewritten for Gilbreth. Reviewer: Critic at final "
        "review.",
    ),
    HygieneRow(
        ac_id="AC-8.4",
        earliest_commit="C7",
        path_globs=MARKDOWN_GLOBS,
        allowlist=(),
        kind="test",
        pattern=r"(?i)15-day|15 day|15일|home01|OTP\b|DM node",
        fence=HISTORY_FENCE,
        note="KISTI-only content only inside the history fence.",
    ),
    HygieneRow(
        ac_id="AC-8.5",
        earliest_commit="C7",
        path_globs=MARKDOWN_GLOBS,
        allowlist=(),
        kind="test",
        pattern=r"(?i)boilerkey",
        note="Purdue's MFA is never called BoilerKey. Currently zero "
        "occurrences -- a don't-introduce constraint.",
    ),
    HygieneRow(
        ac_id="AC-8.6",
        earliest_commit="C7",
        path_globs=(),
        allowlist=(),
        kind="manual",
        note="The live-verification section has one row per section 6.2 item. "
        "Reviewer: whoever runs the live pass.",
    ),
)

#: AC-0.3(b) -- the criteria that are deliberately NOT machine-checked.  This
#: mirrors the `[shell]` / `[manual]` tags in section 3 of the plan; the test
#: below asserts the two cannot drift apart.
UNCHECKED_AC_IDS = frozenset({"AC-1.8", "AC-2.7", "AC-8.1", "AC-8.6"})

#: AC-0.7 -- a scan over zero files that passes silently is the exact failure
#: this module exists to kill.
MIN_PY_FILES_IN_SCOPE = 10


# --------------------------------------------------------------------------
# Scanning primitives -- all root-parameterised so the negative test (AC-0.9)
# can point them at a seeded tmp_path.
# --------------------------------------------------------------------------


class Violation(NamedTuple):
    """One banned match, carrying everything AC-0.4 requires in a message."""

    ac_id: str
    rel_path: str
    lineno: int
    matched: str
    line: str

    def render(self) -> str:
        return f"  {self.rel_path}:{self.lineno}: matched {self.matched!r} in: {self.line.strip()}"


def _expand(pattern: str) -> str:
    """Substitute the resolved package root into a glob pattern."""
    return pattern.format(pkg=PKG)


def _compile_glob(pattern: str) -> re.Pattern[str]:
    """Compile a repo-relative glob into a regex.

    ``**`` crosses directory separators; ``*`` and ``?`` do not.
    """
    out = ["^"]
    i = 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append(r"(?:[^/]+/)*")
            i += 3
        elif pattern.startswith("**", i):
            out.append(r".*")
            i += 2
        elif pattern[i] == "*":
            out.append(r"[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append(r"[^/]")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    out.append("$")
    return re.compile("".join(out))


def glob_match(rel_path: str, patterns: Iterable[str]) -> bool:
    """True if *rel_path* matches any of *patterns* (after ``{pkg}`` expansion)."""
    return any(_compile_glob(_expand(p)).match(rel_path) for p in patterns)


def _is_excluded(rel_parts: Sequence[str], rel_posix: str) -> bool:
    """AC-0.2 -- the exclusion walk."""
    if rel_posix in EXCLUDED_REL_FILES:
        return True
    for part in rel_parts:
        if part in EXCLUDED_DIR_NAMES:
            return True
        if part.endswith(EXCLUDED_DIR_SUFFIXES):
            return True
    for prefix in EXCLUDED_REL_PREFIXES:
        if rel_posix == prefix or rel_posix.startswith(prefix + "/"):
            return True
    return False


def iter_files(root: Path) -> list[str]:
    """Every non-excluded file under *root*, as sorted repo-relative posix paths."""
    found: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        rel_posix = rel.as_posix()
        # The directory components are all but the last part.
        if _is_excluded(rel.parts[:-1], rel_posix):
            continue
        found.append(rel_posix)
    return sorted(found)


def _read_lines(path: Path) -> Optional[list[str]]:
    """Text lines of *path*, or ``None`` if it is not decodable text."""
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except (UnicodeDecodeError, OSError):
        return None


def _blank_fenced_lines(lines: list[str], fence: Tuple[str, str]) -> list[str]:
    """Blank out fenced regions, preserving line numbering (AC-0.4)."""
    begin, end = fence
    out: list[str] = []
    inside = False
    for line in lines:
        if not inside and begin in line:
            inside = True
            out.append("")
            continue
        if inside:
            out.append("")
            if end in line:
                inside = False
            continue
        out.append(line)
    return out


def resolve_scope(root: Path, row: HygieneRow) -> list[str]:
    """Files under *root* covered by *row*, with its allowlist removed."""
    return [
        rel
        for rel in iter_files(root)
        if glob_match(rel, row.path_globs) and not glob_match(rel, row.allowlist)
    ]


def scan(root: Path, row: HygieneRow) -> list[Violation]:
    """Every banned match for *row* under *root*.

    *root* is a parameter, not a module constant, so :func:`scan` can be
    pointed at a seeded ``tmp_path`` -- this is what makes AC-0.9's negative
    test a real control rather than a tautology.
    """
    if row.pattern is None:
        return []
    banned = re.compile(row.pattern)
    violations: list[Violation] = []
    for rel in resolve_scope(root, row):
        lines = _read_lines(root / rel)
        if lines is None:
            continue
        if row.fence is not None:
            lines = _blank_fenced_lines(lines, row.fence)
        for lineno, line in enumerate(lines, start=1):
            for match in banned.finditer(line):
                violations.append(
                    Violation(row.ac_id, rel, lineno, match.group(0), line)
                )
    return violations


def assert_no_violations(row: HygieneRow, violations: Sequence[Violation]) -> None:
    """AC-0.4 -- fail naming every offending ``path:line`` and matched text."""
    if not violations:
        return
    shown = list(violations[:40])
    body = "\n".join(v.render() for v in shown)
    if len(violations) > len(shown):
        body += f"\n  ... and {len(violations) - len(shown)} more"
    raise AssertionError(
        f"{row.ac_id}: {len(violations)} banned match(es) for /{row.pattern}/ "
        f"(enforced from {row.earliest_commit}):\n{body}\n"
        f"  note: {row.note}"
    )


def changed_paths_through(commit: str) -> set[str]:
    """Union of the change lists of every commit at or before *commit*."""
    limit = commit_index(commit)
    union: set[str] = set()
    for cid, paths in COMMIT_CHANGE_LISTS.items():
        if commit_index(cid) <= limit:
            union |= paths
    return union


def active_rows(commit: str = ACTIVE_COMMIT) -> list[HygieneRow]:
    """Rows enforced at *commit*.

    Rows above *commit* are filtered out of the active set.  They are inactive
    by construction -- never via ``pytest.skip``, which AC-7.6 forbids.
    """
    limit = commit_index(commit)
    return [
        row
        for row in HYGIENE_SCOPE
        if row.kind == "test" and commit_index(row.earliest_commit) <= limit
    ]


# ==========================================================================
# Tests
# ==========================================================================


def test_package_root_resolves_to_exactly_one_package():
    """AC-0.3(a) -- runtime package-root resolution."""
    candidates = [p for p in ("rcac_cli", "neuron_cli") if (ROOT / p).is_dir()]
    assert len(candidates) == 1, (
        f"expected exactly one package root under {ROOT}, found {candidates}. "
        "During C1 both may exist momentarily; the rename must be a single "
        "`git mv`, not a copy."
    )
    assert PKG == candidates[0]
    assert (ROOT / PKG / "__init__.py").is_file()


def test_exclusion_walk_prunes_every_named_directory():
    """AC-0.2 -- the walk excludes the nine named locations."""
    scanned = iter_files(ROOT)
    assert scanned, "the exclusion walk returned no files at all"

    forbidden_names = set(EXCLUDED_DIR_NAMES)
    offenders = []
    for rel in scanned:
        parts = Path(rel).parts[:-1]
        if forbidden_names.intersection(parts):
            offenders.append(rel)
        elif any(p.endswith(EXCLUDED_DIR_SUFFIXES) for p in parts):
            offenders.append(rel)
        elif any(
            rel.startswith(prefix + "/") for prefix in EXCLUDED_REL_PREFIXES
        ):
            offenders.append(rel)
        elif rel in EXCLUDED_REL_FILES:
            offenders.append(rel)
    assert not offenders, (
        "the exclusion walk leaked files from excluded directories:\n  "
        + "\n  ".join(offenders[:20])
    )

    # The plan document itself is the canonical thing that must never be
    # scanned: it uses the banned literals hundreds of times.
    plan = ROOT / ".omc" / "plans" / "rcac-cli-gilbreth-port.md"
    if plan.is_file():
        assert plan.relative_to(ROOT).as_posix() not in scanned


def test_self_exclusion_is_narrow_and_necessary():
    """This module is the scanner, not the scanned -- but only this module.

    ``test_repo_hygiene.py`` spells out every banned literal by construction,
    and :data:`PKG` must name both package roots to resolve between them.  The
    exclusion is asserted to be exactly one file so it cannot quietly grow into
    a way of hiding real violations.
    """
    assert EXCLUDED_REL_FILES == ("tests/test_repo_hygiene.py",)

    self_rel = Path(__file__).resolve().relative_to(ROOT).as_posix()
    assert self_rel in EXCLUDED_REL_FILES, (
        f"this module lives at {self_rel} but EXCLUDED_REL_FILES names "
        f"{EXCLUDED_REL_FILES}; the self-exclusion has gone stale"
    )
    assert self_rel not in iter_files(ROOT)

    # The exclusion is load-bearing: this file really does carry the literals.
    banned = re.compile(r"(?i)neuron|kisti")
    assert banned.search(Path(__file__).read_text(encoding="utf-8")), (
        "this module no longer contains the banned literals, so the "
        "self-exclusion is dead weight and should be removed"
    )


def test_vacuity_guard_scope_contains_enough_python_files():
    """AC-0.7 -- a scan over zero files passing silently is the failure to kill."""
    in_scope: set[str] = set()
    for row in HYGIENE_SCOPE:
        if row.kind != "test":
            continue
        in_scope.update(resolve_scope(ROOT, row))
    py_files = sorted(f for f in in_scope if f.endswith(".py"))
    assert len(py_files) >= MIN_PY_FILES_IN_SCOPE, (
        f"resolved hygiene scope holds only {len(py_files)} .py file(s), "
        f"expected >= {MIN_PY_FILES_IN_SCOPE}. The scope table has gone vacuous "
        f"(PKG={PKG!r}, ROOT={ROOT}). Files found: {py_files}"
    )


def test_scope_table_is_well_formed():
    """Structural invariants the rest of the module relies on."""
    seen: set[str] = set()
    for row in HYGIENE_SCOPE:
        assert row.ac_id not in seen, f"duplicate row for {row.ac_id}"
        seen.add(row.ac_id)
        assert row.earliest_commit in COMMIT_ORDER, (
            f"{row.ac_id}: unknown earliest_commit {row.earliest_commit!r}"
        )
        assert row.kind in {"test", "manual", "shell"}, (
            f"{row.ac_id}: unknown kind {row.kind!r}"
        )
        assert row.note, f"{row.ac_id}: every row must record why it exists"
        if row.kind == "test":
            assert row.pattern, f"{row.ac_id}: a test row needs a ban pattern"
            assert row.path_globs, f"{row.ac_id}: a test row needs path globs"
            re.compile(row.pattern)
        else:
            assert row.pattern is None, (
                f"{row.ac_id}: a {row.kind} row must not carry a scan pattern"
            )

    for cid in COMMIT_CHANGE_LISTS:
        assert cid in COMMIT_ORDER, f"unknown commit id in change lists: {cid!r}"
    assert set(COMMIT_CHANGE_LISTS) == set(COMMIT_ORDER), (
        "every commit needs a change list entry, even if empty: missing "
        f"{sorted(set(COMMIT_ORDER) - set(COMMIT_CHANGE_LISTS))}"
    )
    assert ACTIVE_COMMIT in COMMIT_ORDER


def test_manual_and_shell_criteria_are_tagged():
    """AC-0.3(b) -- the plan's prose cannot drift from the table."""
    tagged = {row.ac_id for row in HYGIENE_SCOPE if row.kind in {"manual", "shell"}}
    assert tagged == UNCHECKED_AC_IDS, (
        "the set of criteria excused from machine checking has drifted from "
        f"section 3 of the plan.\n  table says: {sorted(tagged)}\n"
        f"  plan says:  {sorted(UNCHECKED_AC_IDS)}"
    )
    for row in HYGIENE_SCOPE:
        if row.kind in {"manual", "shell"}:
            assert not row.path_globs, (
                f"{row.ac_id} is tagged {row.kind} but declares path globs; "
                "if it is scannable it should be kind='test'"
            )


def test_markdown_criteria_are_deferred_to_c7():
    """AC-0.3(c) -- ``.md`` rows carry ``earliest_commit = 'C7'``."""
    for row in HYGIENE_SCOPE:
        if row.kind != "test":
            continue
        if any("*.md" in g for g in row.path_globs):
            assert row.earliest_commit == "C7", (
                f"{row.ac_id} scans *.md but is enforced from "
                f"{row.earliest_commit}; the READMEs are not rewritten until C7"
            )


def test_yml_and_def_extensions_are_scanned():
    """AC-0.5 -- ``*.yml`` alongside ``*.yaml``, ``*.def`` alongside source."""
    config_globs = set(CONFIG_GLOBS)
    for yaml_glob in [g for g in config_globs if g.endswith("*.yaml")]:
        assert yaml_glob[: -len("yaml")] + "yml" in config_globs, (
            f"{yaml_glob} has no *.yml counterpart"
        )
    assert any(g.endswith("*.def") for g in CONTAINER_GLOBS)

    # And the extensions must actually resolve to files today, or the
    # extension coverage is decorative.
    all_files = iter_files(ROOT)
    assert any(glob_match(f, CONTAINER_GLOBS) for f in all_files), (
        "no *.def file resolved; containers/ coverage is vacuous"
    )
    assert any(f.endswith((".yaml", ".yml")) for f in all_files)


def test_active_ban_rows_are_clean():
    """Enforce every row at or below :data:`ACTIVE_COMMIT`.

    Rows above it are filtered out here rather than skipped, so
    ``pytest --strict-markers`` reports zero skips (AC-7.6).
    """
    rows = active_rows()
    inactive = [row for row in HYGIENE_SCOPE if row.kind == "test" and row not in rows]
    for row in inactive:
        assert commit_index(row.earliest_commit) > commit_index(ACTIVE_COMMIT), (
            f"{row.ac_id} was filtered out of the active set but its "
            f"earliest_commit ({row.earliest_commit}) is not after "
            f"ACTIVE_COMMIT ({ACTIVE_COMMIT})"
        )
    for row in rows:
        assert_no_violations(row, scan(ROOT, row))


@pytest.mark.parametrize(
    "row", [r for r in HYGIENE_SCOPE if r.kind == "test"], ids=lambda r: r.ac_id
)
def test_meta_scope_is_derived_from_change_lists(row: HygieneRow):
    """AC-0.6 -- the class-closing assertion.

    For every row, each path still matching the ban pattern must be named in
    the change list of some commit at or before the row's ``earliest_commit``.
    A file that is dirty but that no scheduled commit touches is a scope error,
    and this turns it into an authoring-time test failure rather than a review
    finding at iteration 4.

    This runs for *every* row at *every* commit, enforced or not.  That is the
    point: it is what makes ``ACTIVE_COMMIT`` safe to leave behind the plan.
    """
    dirty = sorted({v.rel_path for v in scan(ROOT, row)})
    if not dirty:
        return

    scheduled = changed_paths_through(row.earliest_commit)
    unscheduled = [p for p in dirty if not glob_match(p, scheduled)]
    assert not unscheduled, (
        f"{row.ac_id}: {len(unscheduled)} path(s) still match /{row.pattern}/ "
        f"but are not named in the change list of any commit <= "
        f"{row.earliest_commit}:\n  "
        + "\n  ".join(unscheduled)
        + "\n  Either the file must be added to a commit's change list, or the "
        "row's earliest_commit is wrong.\n"
        f"  note: {row.note}"
    )


def test_templates_row_is_deferred_to_c4():
    """The concrete instance AC-0.6 exists to have caught.

    ``pipeline.sh.j2:42`` reads "KISTI's tight quota".  There is no ``rcac``
    equivalent sentence, so C1 cannot rename it -- only C4 can delete it.  The
    main AC-1.6 row must therefore not cover ``*.j2``.
    """
    main = next(r for r in HYGIENE_SCOPE if r.ac_id == "AC-1.6")
    templates = next(r for r in HYGIENE_SCOPE if r.ac_id == "AC-1.6-templates")

    assert templates.earliest_commit == "C4"
    assert not any(g.endswith("*.j2") for g in main.path_globs), (
        "AC-1.6 must not cover *.j2; templates carry earliest_commit=C4"
    )

    pipeline = [
        rel
        for rel in resolve_scope(ROOT, templates)
        if rel.endswith("pipeline.sh.j2")
    ]
    assert pipeline, (
        "pipeline.sh.j2 did not resolve into the templates row; the glob has "
        f"drifted (globs={templates.path_globs}, PKG={PKG!r})"
    )
    # Guard the premise: if the KISTI sentence is gone, this carve-out should
    # be collapsed back into AC-1.6 rather than left as dead scope.
    assert scan(ROOT, templates), (
        "the templates carve-out no longer matches anything; fold "
        "AC-1.6-templates back into AC-1.6"
    )


def test_cluster_config_row_is_deferred_to_c2():
    """The second instance AC-0.6 caught, and the plan had not.

    Section 4's C1 step list names ``configs/neuron.yaml`` only in the ``git mv``.
    But AC-1.6 scans ``configs/**/*.yaml`` from C1, and the file's
    ``login_host`` / ``dm_host`` *values* are the KISTI hostnames.  Renaming a
    hostname is not a rename, it is a replacement, so C1 cannot satisfy AC-1.6
    over this file and C2 step 1 must.
    """
    main = next(r for r in HYGIENE_SCOPE if r.ac_id == "AC-1.6")
    cluster = next(r for r in HYGIENE_SCOPE if r.ac_id == "AC-1.6-cluster-config")

    assert cluster.earliest_commit == "C2"
    assert not glob_match("configs/rcac.yaml", main.path_globs), (
        "AC-1.6 must not cover the cluster config; it carries "
        "earliest_commit=C2"
    )
    # The carve-out must not silently widen: every other configs/ YAML stays
    # in the main row, or the deferral has become a blanket exemption.
    for rel in iter_files(ROOT):
        if rel.startswith("configs/") and rel.endswith((".yaml", ".yml")):
            assert glob_match(rel, main.path_globs) ^ glob_match(
                rel, CLUSTER_CONFIG_GLOBS
            ), f"{rel} is in both or neither of the AC-1.6 rows"

    # Guard the premise: once C2 replaces the hostnames this carve-out is dead
    # scope and must be folded back into AC-1.6.
    assert scan(ROOT, cluster), (
        "the cluster-config carve-out no longer matches anything; fold "
        "AC-1.6-cluster-config back into AC-1.6"
    )


def test_legacy_allowlist_is_the_sole_exemption():
    """AC-0.8 -- and it must tolerate the file not existing yet."""
    exempting = [r for r in HYGIENE_SCOPE if r.allowlist]
    assert exempting, "no row allowlists _legacy.py"
    for row in exempting:
        assert row.allowlist == LEGACY_ALLOWLIST, (
            f"{row.ac_id} declares an allowlist other than _legacy.py: "
            f"{row.allowlist}"
        )
    legacy = _expand(LEGACY_ALLOWLIST[0])
    assert legacy.endswith("/_legacy.py")
    # Not asserted to exist: it arrives in C2.
    assert glob_match(legacy, LEGACY_ALLOWLIST)


# --------------------------------------------------------------------------
# AC-0.9 / AC-7.9 -- the negative test
# --------------------------------------------------------------------------


def _seed_root(tmp_path: Path) -> Path:
    """A miniature repo carrying one real violation and four decoys."""
    (tmp_path / PKG).mkdir()
    (tmp_path / PKG / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / PKG / "clean.py").write_text(
        "import os\n\n\ndef ok():\n    return os.sep\n", encoding="utf-8"
    )
    # The seeded violation: line 3, matched text "Neuron".
    (tmp_path / PKG / "dirty.py").write_text(
        'import os\n\nCLUSTER = "Neuron"\n', encoding="utf-8"
    )
    # Decoys inside excluded directories -- must NOT be reported.
    (tmp_path / ".omc" / "plans").mkdir(parents=True)
    (tmp_path / ".omc" / "plans" / "plan.py").write_text(
        "neuron kisti neuron\n", encoding="utf-8"
    )
    (tmp_path / "tests" / "golden").mkdir(parents=True)
    (tmp_path / "tests" / "golden" / "snap.py").write_text(
        "KISTI\n", encoding="utf-8"
    )
    (tmp_path / PKG / "__pycache__").mkdir()
    (tmp_path / PKG / "__pycache__" / "cached.py").write_text(
        "neuron\n", encoding="utf-8"
    )
    (tmp_path / f"{PKG}.egg-info").mkdir()
    (tmp_path / f"{PKG}.egg-info" / "meta.py").write_text(
        "neuron\n", encoding="utf-8"
    )
    return tmp_path


_NEGATIVE_ROW = HygieneRow(
    ac_id="AC-0.9",
    earliest_commit="C0",
    path_globs=("{pkg}/**/*.py",),
    allowlist=LEGACY_ALLOWLIST,
    kind="test",
    pattern=r"(?i)neuron|kisti",
    note="negative control fixture",
)


def test_negative_seeded_violation_is_reported(tmp_path):
    """AC-0.9 / AC-7.9 -- the scanner is proven to fire on a seeded violation.

    This is a control in both directions: the equality pins over-reporting (a
    broken exclusion walk fails it) and :func:`pytest.raises` pins
    under-reporting (a scanner that finds nothing fails it).  It is not a
    tautology -- nothing here asserts a property the scanner defines.
    """
    root = _seed_root(tmp_path)
    violations = scan(root, _NEGATIVE_ROW)

    assert [(v.rel_path, v.lineno, v.matched) for v in violations] == [
        (f"{PKG}/dirty.py", 3, "Neuron")
    ], f"scanner mis-reported the seeded tree: {violations}"

    with pytest.raises(AssertionError) as excinfo:
        assert_no_violations(_NEGATIVE_ROW, violations)

    message = str(excinfo.value)
    # AC-0.4 -- path:line and the matched text must be in the message.
    assert f"{PKG}/dirty.py:3" in message
    assert "'Neuron'" in message
    assert 'CLUSTER = "Neuron"' in message
    assert "AC-0.9" in message


def test_negative_clean_tree_produces_no_violation(tmp_path):
    """The other half of the control: a clean tree must pass."""
    root = _seed_root(tmp_path)
    (root / PKG / "dirty.py").write_text(
        'import os\n\nCLUSTER = "gilbreth"\n', encoding="utf-8"
    )
    violations = scan(root, _NEGATIVE_ROW)
    assert violations == [], (
        "the scanner reported violations in a clean tree; the exclusion walk "
        f"or the glob is over-matching: {violations}"
    )
    assert_no_violations(_NEGATIVE_ROW, violations)


def test_negative_allowlisted_path_is_exempt(tmp_path):
    """AC-0.8 -- the allowlist works, and does not require the file to exist."""
    root = _seed_root(tmp_path)
    (root / PKG / "dirty.py").write_text("clean\n", encoding="utf-8")
    (root / PKG / "_legacy.py").write_text(
        'LEGACY_SSH_MARKER = "# neuron-cli managed"\n', encoding="utf-8"
    )
    assert scan(root, _NEGATIVE_ROW) == []

    unexempted = _NEGATIVE_ROW._replace(allowlist=())
    assert [v.rel_path for v in scan(root, unexempted)] == [f"{PKG}/_legacy.py"], (
        "removing the allowlist did not surface _legacy.py, so the allowlist "
        "was never doing anything"
    )


def test_negative_fence_exempts_only_fenced_lines(tmp_path):
    """The history fence must exempt its contents and nothing else."""
    root = tmp_path
    (root / "README.md").write_text(
        "\n".join(
            [
                "# Title",  # 1
                "<!-- history:begin -->",  # 2
                "This used to be neuron-cli for KISTI.",  # 3 - exempt
                "<!-- history:end -->",  # 4
                "Runs on neuron today.",  # 5 - violation
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    row = HygieneRow(
        ac_id="AC-1.7",
        earliest_commit="C7",
        path_globs=("*.md",),
        allowlist=(),
        kind="test",
        pattern=r"(?i)neuron|kisti",
        fence=HISTORY_FENCE,
        note="fence control fixture",
    )
    assert [(v.lineno, v.matched) for v in scan(root, row)] == [(5, "neuron")]
