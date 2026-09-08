"""README criteria that a ban-pattern scan cannot express (AC-8.2/8.3/8.7/8.9/8.10).

:mod:`tests.test_repo_hygiene` owns every criterion of the form "this literal
must not appear here" -- AC-1.7, AC-8.4 and AC-8.5 are rows in its scope table.
The criteria in *this* module are the opposite shape: they assert that
something **is** present, and each needs a parser rather than a regex.

* **AC-8.2** -- every content line inside a ``facts`` fence carries a
  ``docs.rcac.purdue.edu`` URL.  This is the criterion that makes the fact
  tables auditable instead of merely confident.
* **AC-8.3** -- the command reference names all twelve console scripts, and the
  list is read out of ``pyproject.toml`` rather than copied into the test.
* **AC-8.7** -- both shipped examples still parse through ``rcac-exp launch
  --dry-run``.
* **AC-8.9** -- the two READMEs share one heading *sequence*.
* **AC-8.10** -- the README states the front-end count and what ControlMaster
  does about it.

Both READMEs are checked by every one of them.  A criterion that held only for
the English file would let the translated one drift into being wrong in a
language most reviewers of this repo do not read.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
from click.testing import CliRunner

from rcac_cli.experiment import cli as exp_cli

ROOT = Path(__file__).resolve().parent.parent

#: Both READMEs, always checked as a pair.
READMES = ("README.md", "README.ko.md")

#: The fact fences.  Several pairs per file: a fence wraps exactly one complete
#: markdown table, because an HTML comment *between* a table's delimiter row and
#: its body stops the table rendering as a table.  Wrapping whole tables keeps
#: the rendered page intact and keeps every data row inside the checked region.
FACTS_BEGIN = "<!-- facts:begin -->"
FACTS_END = "<!-- facts:end -->"

#: The host every fenced line must cite.
SOURCE_HOST = "docs.rcac.purdue.edu"

#: A markdown table's delimiter row, e.g. ``|---|---:|---|``.  It is the one
#: line inside a fence that cannot carry a URL: markdown requires it verbatim
#: between the header and the body.  It is exempted by *shape*, not by
#: position, so a fact row can never be mistaken for one.
TABLE_DELIMITER = re.compile(r"^\s*\|[\s:|-]+\|\s*$")

#: An ATX heading, matched only outside fenced code blocks.
HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


def _lines(name: str) -> list[str]:
    return (ROOT / name).read_text(encoding="utf-8").splitlines()


def _fenced_regions(lines: list[str], name: str) -> list[list[tuple[int, str]]]:
    """Every ``facts`` region of *lines*, as ``(lineno, text)`` pairs.

    Raises on an unbalanced or nested fence rather than returning fewer
    regions: a dropped ``facts:end`` would otherwise silently shrink what
    AC-8.2 checks, which is the vacuity failure AC-0.7 exists to refuse.
    """
    regions: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] | None = None
    for lineno, line in enumerate(lines, start=1):
        if FACTS_BEGIN in line:
            assert current is None, f"{name}:{lineno}: nested {FACTS_BEGIN}"
            current = []
            continue
        if FACTS_END in line:
            assert current is not None, f"{name}:{lineno}: stray {FACTS_END}"
            regions.append(current)
            current = None
            continue
        if current is not None:
            current.append((lineno, line))
    assert current is None, f"{name}: {FACTS_BEGIN} with no matching {FACTS_END}"
    return regions


def _headings(lines: list[str]) -> list[tuple[int, int, str]]:
    """``(lineno, level, text)`` for every heading outside a fenced code block.

    Stripping fences is not a refinement, it is the whole check.  ``grep -c
    '^#'`` counts shell comments inside the bash examples and returns a number
    far larger than the heading count -- and it returned the *same* wrong
    number for both files before this port, which is exactly how a scalar gate
    passes while the two documents disagree.
    """
    out: list[tuple[int, int, str]] = []
    in_code = False
    for lineno, line in enumerate(lines, start=1):
        if line.lstrip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        match = HEADING.match(line)
        if match:
            out.append((lineno, len(match.group(1)), match.group(2)))
    return out


def _console_scripts() -> dict[str, str]:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["project"]["scripts"]


# --------------------------------------------------------------------------
# AC-8.2
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", READMES)
def test_every_fact_line_cites_an_rcac_url(name: str) -> None:
    """AC-8.2 -- no unsourced claim survives inside a ``facts`` fence."""
    lines = _lines(name)
    regions = _fenced_regions(lines, name)
    assert regions, f"{name} has no {FACTS_BEGIN} region at all"

    content = [
        (lineno, text)
        for region in regions
        for lineno, text in region
        if text.strip() and not TABLE_DELIMITER.match(text)
    ]
    # Vacuity guard, in the spirit of AC-0.7: a fence pair around nothing
    # passes every assertion below without checking a single fact.
    assert len(content) >= 20, (
        f"{name}: only {len(content)} fact line(s) inside {len(regions)} "
        "fence(s); the fact tables are supposed to live in there"
    )

    unsourced = [(n, t) for n, t in content if SOURCE_HOST not in t]
    assert not unsourced, (
        f"AC-8.2: {len(unsourced)} line(s) inside a {name} facts fence carry no "
        f"{SOURCE_HOST} URL. Either cite the page, or move the claim out of the "
        "fence and into the live-verification checklist:\n"
        + "\n".join(f"  {name}:{n}: {t.strip()[:110]}" for n, t in unsourced[:20])
    )


@pytest.mark.parametrize("name", READMES)
def test_fact_delimiter_exemption_cannot_hide_a_fact(name: str) -> None:
    """The one exempted shape must stay incapable of carrying a claim.

    ``TABLE_DELIMITER`` is the single carve-out in the check above.  If it ever
    matched a row with words in it, AC-8.2 would have a hole the size of that
    row, so the exemption is pinned here rather than trusted.
    """
    assert TABLE_DELIMITER.match("|---|---|---|")
    assert TABLE_DELIMITER.match("| :--- | ---: |")
    assert not TABLE_DELIMITER.match("| `a10` | A10, 24 GB | 16 |")
    assert not TABLE_DELIMITER.match("| purge policy | 60 days |")

    for region in _fenced_regions(_lines(name), name):
        for lineno, text in region:
            if TABLE_DELIMITER.match(text):
                assert not re.search(r"[0-9A-Za-z]", text), (
                    f"{name}:{lineno}: a delimiter row with content in it"
                )


# --------------------------------------------------------------------------
# AC-8.3
# --------------------------------------------------------------------------


def test_pyproject_declares_exactly_twelve_rcac_scripts() -> None:
    """AC-1.2 restated where AC-8.3 reads it, so the count cannot drift."""
    scripts = _console_scripts()
    assert len(scripts) == 12, sorted(scripts)
    assert all(entry.startswith("rcac-") for entry in scripts), sorted(scripts)


@pytest.mark.parametrize("name", READMES)
def test_command_reference_names_every_console_script(name: str) -> None:
    """AC-8.3 -- the reference is cross-checked against ``pyproject.toml``.

    The expected list is *read from the packaging metadata*, so adding a
    thirteenth entry point fails this test until it is documented, in both
    languages.
    """
    text = (ROOT / name).read_text(encoding="utf-8")
    missing = sorted(entry for entry in _console_scripts() if entry not in text)
    assert not missing, (
        f"AC-8.3: {name} documents no {missing} -- every console script "
        "declared in pyproject.toml needs an entry in the command reference"
    )


# --------------------------------------------------------------------------
# AC-8.9
# --------------------------------------------------------------------------


def test_readmes_share_one_heading_sequence() -> None:
    """AC-8.9 -- identical heading **level and order**, fenced blocks stripped."""
    english, korean = (_headings(_lines(name)) for name in READMES)
    assert len(english) == len(korean), (
        f"AC-8.9: README.md has {len(english)} headings, README.ko.md has "
        f"{len(korean)}"
    )
    for index, (left, right) in enumerate(zip(english, korean)):
        assert left[1] == right[1], (
            f"AC-8.9: heading #{index + 1} diverges in level -- "
            f"README.md:{left[0]} is h{left[1]} {left[2]!r} but "
            f"README.ko.md:{right[0]} is h{right[1]} {right[2]!r}"
        )


def test_the_scalar_heading_count_is_not_a_valid_gate() -> None:
    """Why the test above compares a sequence and not a number.

    ``grep -c '^#'`` counts every ``#`` that starts a line, including shell
    comments inside the bash examples.  It has agreed across the two files
    while overcounting both -- a gate that is green for a reason unrelated to
    the property it claims to check.  This pins the gap so nobody replaces the
    sequence comparison with the cheaper number.
    """
    for name in READMES:
        lines = _lines(name)
        naive = sum(1 for line in lines if line.startswith("#"))
        real = len(_headings(lines))
        assert naive > real, (
            f"{name}: the naive count ({naive}) no longer exceeds the real "
            f"heading count ({real}); re-read this test before deleting it"
        )


# --------------------------------------------------------------------------
# AC-8.10
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", READMES)
def test_readme_states_the_front_end_count_and_what_pins_the_session(
    name: str,
) -> None:
    """AC-8.10 -- 4 front ends, and ControlMaster pinning the session to one.

    Both halves matter and neither implies the other: the count without the
    consequence is trivia, and the consequence without the count reads like a
    convenience feature instead of the thing that stops ``sbatch`` and
    ``squeue`` landing on different machines.
    """
    text = (ROOT / name).read_text(encoding="utf-8")
    assert re.search(r"\*\*4\s*(nodes|대)\*\*", text), (
        f"AC-8.10: {name} does not state that Gilbreth has 4 front-end nodes"
    )
    assert "ControlMaster" in text, f"AC-8.10: {name} never names ControlMaster"
    assert re.search(r"(?i)(pins?|고정)", text), (
        f"AC-8.10: {name} does not say the session is pinned to one front end"
    )


# --------------------------------------------------------------------------
# AC-8.7
# --------------------------------------------------------------------------

#: The same offline environment the scheduler CLI tests use: a resolved user
#: and account so the run reaches the renderer rather than exiting from the
#: credential check, and discovery pinned off so the suite never reaches out
#: to a Purdue login node.
CLI_ENV = {
    "RCAC_USER": "testid",
    "RCAC_ACCOUNT": "testqueue",
    "RCAC_CONFIG_DIR": str(ROOT / "configs"),
    "RCAC_NO_DISCOVER": "1",
}

EXAMPLES = sorted((ROOT / "examples").glob("*.yaml"))


def test_the_examples_directory_is_not_empty() -> None:
    """AC-8.7 parametrizes over a glob; an empty glob would pass silently."""
    assert len(EXAMPLES) >= 2, [p.name for p in EXAMPLES]


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda p: p.name)
def test_shipped_example_parses_through_dry_run(example: Path) -> None:
    """AC-8.7 -- every shipped example still launches, as shipped.

    ``--dry-run`` renders the script and prints the submission command without
    contacting the cluster, so this asserts on the artefact a live front end
    would see.  ``-n 1`` keeps the sweep to one combination; the sweep itself
    is covered elsewhere.
    """
    result = CliRunner().invoke(
        exp_cli,
        ["launch", str(example), "--dry-run", "-n", "1"],
        env=CLI_ENV,
    )
    assert result.exit_code == 0, f"{example.name}:\n{result.output}"
    assert "#SBATCH -A testqueue" in result.output, result.output
    assert "#SBATCH -p " in result.output, result.output
