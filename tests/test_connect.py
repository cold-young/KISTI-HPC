"""The generated SSH config block, and the removal counterpart it never had.

**AC-2.12 (B13)** existed because ``_generate_ssh_config`` had *no* content
assertion at all.  It emitted two ``Host`` stanzas, the second pointing at a
data-transfer node that Gilbreth does not document, and between the rename
commit and this one it would have happily emitted the new alias with the old
cluster's ``HostName`` and nothing would have noticed.

**AC-2.10 (B11)** is the removal function.  ``_write_ssh_config`` appended a
block to a file the user also hand-maintains and offered no way back out;
``disconnect`` only tore down the socket.  A user returning from the pre-port
CLI kept a stale alias pointing at a cluster this tool no longer talks to
(R17), which is why removal must handle the legacy marker too.

The legacy block below is assembled from :mod:`rcac_cli._legacy` rather than
typed out.  That module is the sole path allowlisted from the branding ban
(AC-0.8) precisely so that no other file -- including this one -- has to spell
the literals in order to work with them.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from rcac_cli._legacy import LEGACY_SSH_END_MARKER, LEGACY_SSH_MARKER
from rcac_cli.connect import (
    MANAGED_MARKERS,
    SSH_CONFIG_END_MARKER,
    SSH_CONFIG_MARKER,
    _generate_ssh_config,
    disconnect,
    strip_managed_blocks,
)

CFG = {
    "ssh": {
        "login_host": "gilbreth.rcac.purdue.edu",
        "user": "testid",
        "port": 22,
        "control_persist": "4h",
        "control_path": "~/.ssh/sockets/%r@%h-%p",
        "server_alive_interval": 60,
        "server_alive_count_max": 10,
    },
}

#: A hostname the pre-port CLI would have written into the removed second
#: stanza. Deliberately a placeholder rather than the real former value: the
#: test is about block boundaries, not about that cluster.
LEGACY_TRANSFER_HOSTNAME = "legacy-transfer.example.invalid"

BEFORE = "Host work\n    HostName work.example.com\n    User someone\n"
AFTER = "Host other\n    HostName other.example.com\n"


def legacy_block() -> str:
    """The two-stanza block the pre-port CLI appended, markers included."""
    return "\n".join(
        [
            LEGACY_SSH_MARKER,
            "Host legacy",
            "    HostName legacy-login.example.invalid",
            "    User testid",
            "    Port 22",
            "    ControlMaster auto",
            "    ControlPath ~/.ssh/sockets/%r@%h-%p",
            "    ControlPersist 4h",
            "",
            "Host legacy-dm",
            f"    HostName {LEGACY_TRANSFER_HOSTNAME}",
            "    User testid",
            "    ControlPersist 8h",
            LEGACY_SSH_END_MARKER,
        ]
    )


# --------------------------------------------------------------------------
# AC-2.12 (B13) -- content assertions on the generated block
# --------------------------------------------------------------------------


def test_generated_config_has_exactly_one_host_stanza():
    """It emitted two until this commit; the second fed a node that does not exist."""
    stanzas = [
        line
        for line in _generate_ssh_config(CFG).splitlines()
        if line.startswith("Host ")
    ]
    assert stanzas == ["Host gilbreth"]


def test_generated_hostname_equals_the_configured_login_host():
    """Closes the window in which the alias and the HostName could disagree."""
    hostnames = [
        line.split(maxsplit=1)[1]
        for line in _generate_ssh_config(CFG).splitlines()
        if line.strip().startswith("HostName ")
    ]
    assert hostnames == [CFG["ssh"]["login_host"]]


def test_the_alias_matches_the_cluster_it_points_at():
    """``Host gilbreth`` is only honest because ``login_host`` is Gilbreth now."""
    block = _generate_ssh_config(CFG)
    assert "Host gilbreth" in block
    assert CFG["ssh"]["login_host"].startswith("gilbreth.")


def test_no_generated_line_has_an_empty_value():
    """A directive with a missing value is silently ignored by ssh.

    Same failure shape as an empty ``#SBATCH`` directive: it looks plausible
    and only misbehaves later, on a connection the user is waiting on.
    """
    empty = [
        line
        for line in _generate_ssh_config(CFG).splitlines()
        if line.strip()
        and not line.startswith("#")
        and len(line.split()) < 2
    ]
    assert not empty, empty


def test_marker_pair_is_balanced():
    block = _generate_ssh_config(CFG)
    assert block.count(SSH_CONFIG_MARKER) == 1
    assert block.count(SSH_CONFIG_END_MARKER) == 1
    assert block.index(SSH_CONFIG_MARKER) < block.index(SSH_CONFIG_END_MARKER)


def test_generated_block_carries_the_control_master_settings():
    """The 4-front-end problem: without ControlMaster the session is unpinned."""
    block = _generate_ssh_config(CFG)
    assert "    ControlMaster auto" in block
    assert f"    ControlPath {CFG['ssh']['control_path']}" in block
    assert f"    ControlPersist {CFG['ssh']['control_persist']}" in block


# --------------------------------------------------------------------------
# AC-2.10 (B11) -- removal, for both markers
# --------------------------------------------------------------------------


def test_removal_handles_the_legacy_block_and_preserves_its_surroundings(tmp_path):
    """AC-2.10 -- the block goes, everything else is byte-preserved (R17)."""
    config = tmp_path / "config"
    original = BEFORE + "\n" + legacy_block() + "\n\n" + AFTER
    config.write_text(original, encoding="utf-8")

    cleaned, removed = strip_managed_blocks(config.read_text())

    assert removed == 1
    assert LEGACY_SSH_MARKER not in cleaned
    assert LEGACY_SSH_END_MARKER not in cleaned
    assert LEGACY_TRANSFER_HOSTNAME not in cleaned, (
        "the stanza for the old cluster's transfer node survived removal"
    )
    assert BEFORE in cleaned, "content before the block was not byte-preserved"
    assert AFTER in cleaned, "content after the block was not byte-preserved"


def test_removal_handles_the_current_block(tmp_path):
    """The counterpart must work on what this version actually writes."""
    original = BEFORE + "\n" + _generate_ssh_config(CFG) + "\n\n" + AFTER

    cleaned, removed = strip_managed_blocks(original)

    assert removed == 1
    assert "Host gilbreth" not in cleaned
    assert BEFORE in cleaned
    assert AFTER in cleaned


def test_removal_handles_both_blocks_at_once(tmp_path):
    """A user who ran the old CLI and then this one carries both."""
    original = (
        BEFORE
        + "\n"
        + legacy_block()
        + "\n\n"
        + _generate_ssh_config(CFG)
        + "\n\n"
        + AFTER
    )

    cleaned, removed = strip_managed_blocks(original)

    assert removed == 2
    for begin, end in MANAGED_MARKERS:
        assert begin not in cleaned
        assert end not in cleaned
    assert BEFORE in cleaned
    assert AFTER in cleaned


def test_removal_is_a_no_op_on_an_unmanaged_config():
    """A config this tool never touched must come back byte-identical."""
    original = BEFORE + "\n" + AFTER

    cleaned, removed = strip_managed_blocks(original)

    assert removed == 0
    assert cleaned == original


def test_unterminated_block_is_removed_to_end_of_file():
    """The user deleted the End marker by hand.

    Leaving the half-block behind would make ``--setup-config`` refuse forever
    (the begin marker is still there) while ssh reads a truncated stanza.
    """
    original = BEFORE + "\n" + legacy_block().replace(LEGACY_SSH_END_MARKER, "")

    cleaned, removed = strip_managed_blocks(original)

    assert removed == 1
    assert LEGACY_SSH_MARKER not in cleaned
    assert LEGACY_TRANSFER_HOSTNAME not in cleaned
    assert BEFORE in cleaned


def test_round_trip_write_then_remove_restores_the_original(tmp_path, monkeypatch):
    """``_write_ssh_config`` and ``_remove_ssh_config`` must be inverses.

    Not exactly byte-inverse: the write adds two newlines as a separator and
    the removal deliberately leaves whitespace alone rather than guessing which
    blank lines were the user's.  So the assertion is on the non-blank content.
    """
    from rcac_cli import connect as connect_mod

    home = tmp_path / "home"
    (home / ".ssh").mkdir(parents=True)
    config = home / ".ssh" / "config"
    original = BEFORE + "\n" + AFTER
    config.write_text(original, encoding="utf-8")
    monkeypatch.setattr(connect_mod.Path, "home", classmethod(lambda cls: home))

    connect_mod._write_ssh_config(CFG)
    assert SSH_CONFIG_MARKER in config.read_text()

    connect_mod._remove_ssh_config()

    def significant(text: str) -> list[str]:
        return [line for line in text.splitlines() if line.strip()]

    assert significant(config.read_text()) == significant(original)


def test_disconnect_remove_config_is_the_user_facing_route(tmp_path, monkeypatch):
    """B11 -- there must be a command, not just a private function."""
    from rcac_cli import connect as connect_mod

    home = tmp_path / "home"
    (home / ".ssh").mkdir(parents=True)
    config = home / ".ssh" / "config"
    config.write_text(BEFORE + "\n" + legacy_block() + "\n" + AFTER, encoding="utf-8")
    monkeypatch.setattr(connect_mod.Path, "home", classmethod(lambda cls: home))
    monkeypatch.setenv("RCAC_USER", "testid")
    monkeypatch.delenv("RCAC_CONFIG_DIR", raising=False)
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(disconnect, ["--remove-config"])

    assert result.exit_code == 0, result.output
    assert LEGACY_SSH_MARKER not in config.read_text()
    assert BEFORE in config.read_text()


def test_write_drops_a_stale_legacy_block_before_appending(tmp_path, monkeypatch):
    """R17 -- two aliases must never both answer for the same session."""
    from rcac_cli import connect as connect_mod

    home = tmp_path / "home"
    (home / ".ssh").mkdir(parents=True)
    config = home / ".ssh" / "config"
    config.write_text(BEFORE + "\n" + legacy_block() + "\n", encoding="utf-8")
    monkeypatch.setattr(connect_mod.Path, "home", classmethod(lambda cls: home))

    connect_mod._write_ssh_config(CFG)

    text = config.read_text()
    assert LEGACY_SSH_MARKER not in text
    assert LEGACY_TRANSFER_HOSTNAME not in text
    assert SSH_CONFIG_MARKER in text
    assert BEFORE in text


@pytest.mark.parametrize("begin,end", MANAGED_MARKERS)
def test_every_managed_marker_pair_is_removable(begin, end):
    """Adding a marker pair without teaching removal about it is the B11 shape."""
    text = f"keep\n{begin}\nHost x\n{end}\nkeep2\n"

    cleaned, removed = strip_managed_blocks(text)

    assert removed == 1
    assert cleaned == "keep\nkeep2\n"
