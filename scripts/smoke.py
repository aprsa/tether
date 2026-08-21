#!/usr/bin/env python3
"""Test-drive tether against a real cluster.

    python scripts/smoke.py [server]        # default: terra

Exercises everything currently implemented, in the order you would meet it.
Read-only apart from one small file written under the server's `workdir` and
removed again; nothing is submitted, cancelled, or installed.

Each step reports on its own line and a failure does not abort the run, so one
broken thing does not hide the state of everything after it.
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path
from tempfile import TemporaryDirectory

import tether

FAILURES: list[str] = []


def step(label: str, fn):
    """Run one probe, report it, and keep going if it fails."""
    start = time.perf_counter()
    try:
        value = fn()
    except Exception as exc:  # noqa: BLE001 - a smoke test reports, it does not raise
        FAILURES.append(label)
        print(f'  {label:<24} FAILED  {type(exc).__name__}: {exc}')
        if '-v' in sys.argv:
            traceback.print_exc()
        return None
    elapsed = time.perf_counter() - start
    if value is not None:
        print(f'  {label:<24} {value}   ({elapsed:.2f}s)')
    return value


def main(name: str) -> int:
    print(f'tether {tether.__version__} -> {name}\n')

    # Construction does no I/O, so a bad config fails here rather than later.
    srv = tether.server(name)
    print(f'  {"configured":<24} {srv!r} workdir={srv.workdir} timeout={srv.timeout}')
    print(f'  {"connected (lazy)":<24} {srv.connected}\n')

    with srv:
        print('link')
        step('info', srv.info)
        step('ping', lambda: f'{srv.ping() * 1000:.0f} ms')
        step('run', lambda: srv.run('echo hello', check=True).stdout.strip())
        step('run (nonzero is ok)', lambda: f'rc={srv.run("exit 3").returncode}')
        step('whoami', srv.whoami)
        step('home', lambda: srv.home)
        step('path()', lambda: srv.path('smoke'))

        print('\nscheduler')
        step('slurm version', lambda: srv.slurm_version)
        step('partitions', lambda: ', '.join(
            f'{p.name}({p.nodes_idle}/{p.nodes_total} idle)' for p in srv.partitions()
        ) or 'none reported')
        step('queue (all users)', lambda: f'{len(srv.queue())} jobs')
        step('queue (mine)', lambda: f'{len(srv.queue(user=srv.whoami()))} jobs')

        print('\ntransfer')

        def roundtrip() -> str:
            payload = 'tether smoke test\n'
            with TemporaryDirectory() as tmp:
                local = Path(tmp) / 'up.txt'
                local.write_text(payload)
                remote = srv.path('smoke', 'up.txt')
                srv.put(local, remote)          # creates workdir/smoke/ as needed
                back = Path(tmp) / 'down' / 'back.txt'
                srv.get(remote, back)
                ok = back.read_text() == payload
            srv.run(f'rm -rf {srv.path("smoke")}')
            return 'round trip ok' if ok else 'MISMATCH'

        step('put/get', roundtrip)

        print('\nenvironment')
        preamble = srv.preamble
        step('preamble', lambda: repr(preamble) if preamble else '(none configured)')
        step('verify', srv.verify_environment)

    print(f'\n  {"closed":<24} connected={srv.connected}')

    if FAILURES:
        print(f'\n{len(FAILURES)} failed: {", ".join(FAILURES)}')
        return 1
    print('\nall good')
    return 0


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    raise SystemExit(main(args[0] if args else 'terra'))
