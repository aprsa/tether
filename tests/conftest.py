"""Live-test rig: a real Slurm cluster in a container.

Replaces the old shell rig entirely. There is one mechanism now -- docker
compose -- and what it runs is a genuine slurmctld/slurmd pair, real conda, and
real environment modules, so the live tests exercise the same machinery tether
will meet on an HPC rather than a shell script pretending to be one.

The suite brings the cluster up on demand and leaves it running, because a
warm container answers in milliseconds and tearing it down between runs would
only slow the loop. Drive it by hand with:

    docker compose -f tests/cluster/docker-compose.yml up -d --wait
    docker compose -f tests/cluster/docker-compose.yml down

Live tests skip when Docker is unavailable. Set `TETHER_REQUIRE_LIVE=1` to turn
that skip into a hard error -- layer 3 has no other coverage, so a silent skip
in CI would mean `server.py` is effectively untested.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
from pathlib import Path

import asyncssh
import pytest

HERE = Path(__file__).parent
# The cluster's Docker artifacts live together in tests/cluster/; only this
# file has to sit in tests/, because pytest applies a conftest to the tests
# below it and test_live.py is a sibling, not a child.
CLUSTER = HERE / 'cluster'
COMPOSE = CLUSTER / 'docker-compose.yml'
RIG = CLUSTER / '.rig'

HOST = '127.0.0.1'
PORT = int(os.environ.get('TETHER_RIG_PORT', '2222'))
ALIAS = 'cluster'
SSH_CONFIG = RIG / 'ssh_config'

#: Paths baked into the image. Tests assert against these.
VENV = '/opt/venvs/example'
CONDA_BASE = '/opt/conda'
CONDA_ENV = 'phoebe-dev'
MODULE = 'tether-probe/1.0'
MODULES_INIT = '/etc/profile.d/modules.sh'

REQUIRE_LIVE = bool(os.environ.get('TETHER_REQUIRE_LIVE'))


def _compose(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ['docker', 'compose', '-f', str(COMPOSE), *args],
        capture_output=True,
        text=True,
        check=check,
    )


def _write_key_material() -> None:
    """Generate the client and host keys, host-side.

    Doing this here rather than in the container means the bind mount can be
    read-only: nothing inside ever writes back, so there is no dance over which
    uid owns the generated files. asyncssh is already a dependency, so this
    needs no `ssh-keygen` on the host either.
    """
    RIG.mkdir(exist_ok=True)
    if not (RIG / 'ssh_config').exists():
        client = asyncssh.generate_private_key('ssh-ed25519')
        host = asyncssh.generate_private_key('ssh-ed25519')

        (RIG / 'id_ed25519').write_bytes(client.export_private_key('openssh'))
        (RIG / 'id_ed25519').chmod(0o600)
        (RIG / 'authorized_keys').write_bytes(client.export_public_key('openssh'))
        (RIG / 'hostkey').write_bytes(host.export_private_key('openssh'))
        (RIG / 'hostkey').chmod(0o600)
        # sshd checks the private key against the .pub sitting beside it, so
        # both have to be shipped.
        (RIG / 'hostkey.pub').write_bytes(host.export_public_key('openssh'))

        pub = host.export_public_key('openssh').decode().strip()
        (RIG / 'known_hosts').write_text(f'[{HOST}]:{PORT} {pub}\n')

        # The tests connect to an alias through this file, which also exercises
        # the ssh_config fall-through tether advertises, and keeps ~/.ssh out
        # of it entirely.
        (RIG / 'ssh_config').write_text(
            f'Host {ALIAS}\n'
            f'    HostName {HOST}\n'
            f'    Port {PORT}\n'
            f'    User tether\n'
            f'    IdentityFile {RIG / "id_ed25519"}\n'
            f'    IdentitiesOnly yes\n'
            f'    UserKnownHostsFile {RIG / "known_hosts"}\n'
        )


def _unavailable() -> str | None:
    """Why the rig cannot run, or None if it can."""
    if not shutil.which('docker'):
        return 'docker is not installed'
    probe = subprocess.run(
        ['docker', 'info', '--format', '{{.ServerVersion}}'],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if probe.returncode != 0:
        return f'docker daemon unreachable: {probe.stderr.strip().splitlines()[-1:]}'
    return None


@pytest.fixture(scope='session')
def rig() -> str:
    """Ensure the cluster is up; yield the ssh_config the tests connect through.

    Session-scoped and idempotent: `up --wait` on an already-healthy container
    returns immediately, so the cost is paid once per machine, not per run.
    """
    why = _unavailable()
    if why:
        if REQUIRE_LIVE:
            raise RuntimeError(f'TETHER_REQUIRE_LIVE is set but {why}')
        pytest.skip(f'no rig: {why}')

    _write_key_material()

    # `up --wait` blocks on the healthcheck, which asserts a node is idle and
    # sshd is answering -- so no test races the scheduler's startup.
    # It also builds the image on first run, which is slow exactly once.
    done = _compose('up', '-d', '--wait', check=False)
    if done.returncode != 0:
        detail = (done.stderr or done.stdout).strip()
        if REQUIRE_LIVE:
            raise RuntimeError(f'rig failed to start: {detail}')
        pytest.skip(f'rig failed to start: {detail}')

    with socket.socket() as probe:
        if probe.connect_ex((HOST, PORT)) != 0:
            pytest.fail(f'rig reports healthy but nothing answers on {HOST}:{PORT}')

    return str(SSH_CONFIG)
