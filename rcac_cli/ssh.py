"""The single source of SSH connection options (B12, AC-2.11).

Before this module, five hand-rolled ControlMaster strings existed in three
different shapes across twelve call sites: a bare ``-o ControlPath=`` argv
pair, the same thing spelled into an ``ssh ...`` string for ``rsync -e``, and
one full five-option list in :mod:`rcac_cli.connect`.  Two ``rsync``
invocations had no ``-e`` at all, which cost the user a second MFA prompt
(B4/AC-2.9).  Everything now routes through the two entry points here, so the
option set is defined once and the two encodings cannot drift.

**Two sites are deliberately NOT converted.**
``connect._generate_ssh_config`` writes SSH *config-file* directives --
``    ControlPath <path>``: indented, space-separated, no ``-o``.  Those are a
different grammar and cannot be produced by an option builder; converting them
was the mistake that made the original count 14 instead of 12.
"""

from __future__ import annotations

#: Above this, tell the user to use Globus instead of rsync (AC-2.5).  A
#: heuristic, not a cluster fact: Gilbreth has no documented data-transfer
#: node, so rsync over the login host is the only direct route and it is the
#: wrong tool for a bulk dataset.
GLOBUS_THRESHOLD_BYTES = 10 * 1024**3

#: Purdue's Globus front door and the Gilbreth collection name.
GLOBUS_ENDPOINT = "transfer.rcac.purdue.edu"
GLOBUS_COLLECTION = "Gilbreth Cluster Collection"


def ssh_options(cfg: dict) -> list[tuple[str, str]]:
    """The canonical ``(key, value)`` SSH options, in a stable order."""
    ssh = cfg["ssh"]
    return [
        ("ControlMaster", "auto"),
        ("ControlPath", str(ssh["control_path"])),
        ("ControlPersist", str(ssh["control_persist"])),
        ("ServerAliveInterval", str(ssh["server_alive_interval"])),
        ("ServerAliveCountMax", str(ssh["server_alive_count_max"])),
    ]


def ssh_opt_argv(cfg: dict) -> list[str]:
    """Options and port as argv fragments, for a list passed to ``subprocess``.

    :func:`ssh_cmd` composes this with the target; prefer it over rebuilding
    the prefix by hand.
    """
    argv: list[str] = []
    for key, value in ssh_options(cfg):
        argv += ["-o", f"{key}={value}"]
    argv += ["-p", str(cfg["ssh"]["port"])]
    return argv


def ssh_opt_str(cfg: dict) -> str:
    """The same options as one string, for ``rsync -e``.

    Deliberately **unquoted**: rsync splits its ``-e`` value on whitespace with
    ``strtok`` and does no quote processing at all, so a ``shlex``-quoted
    string would reach ssh with the quote characters still in the argument.
    ``control_path`` therefore must not contain whitespace -- which is also
    true of the plain-``ssh`` form, since ssh reads it as one argv element.
    """
    return "ssh " + " ".join(ssh_opt_argv(cfg))


def ssh_target(cfg: dict) -> str:
    """``user@host`` for the Gilbreth login node."""
    ssh = cfg["ssh"]
    return f"{ssh['user']}@{ssh['login_host']}"


def ssh_cmd(cfg: dict) -> list[str]:
    """The ``ssh`` argv prefix: the binary, the options, and the target.

    Four modules each carried a private copy of this three-element list, so a
    change to the prefix meant four edits and the copies could drift the way
    the option strings this module exists to unify once did.  Callers append
    the remote command::

        subprocess.run([*ssh_cmd(cfg), command], capture_output=True)
    """
    return ["ssh", *ssh_opt_argv(cfg), ssh_target(cfg)]


def transfer_host(cfg: dict) -> str:
    """The host to use for a data transfer.

    Gilbreth has **no documented data-transfer node**, so this is the login
    host.  The function exists rather than being inlined so that the day RCAC
    documents one, there is a single place to change -- and so that the removal
    of the old cluster's transfer-node config key could not leave a
    ``KeyError`` behind in its three consumers (R10).
    """
    return cfg["ssh"]["login_host"]


def globus_guidance(total_bytes: int) -> str:
    """Guidance to print when a transfer is large enough to belong on Globus.

    Empty string below :data:`GLOBUS_THRESHOLD_BYTES`, so callers can
    unconditionally ``if guidance: click.echo(guidance)``.
    """
    if total_bytes < GLOBUS_THRESHOLD_BYTES:
        return ""
    gib = total_bytes / 1024**3
    return (
        f"\nThis transfer is {gib:.1f} GiB. Gilbreth has no data-transfer node,\n"
        f"so rsync would run over the login node. For bulk data use Globus:\n"
        f"  1. Open {GLOBUS_ENDPOINT}\n"
        f'  2. Select the "{GLOBUS_COLLECTION}" collection\n'
        f"  3. Transfer into the destination shown above\n"
    )
