# tether

**T**ask **E**xecution and **T**ransfer **H**andler for **E**xternal **R**esources.

A synchronous Python bridge to remote compute over a single reused SSH
connection. No remote listener, no callback channel, no agent to install on the
cluster. Intended to replace `crimpl` as PHOEBE's route to a Slurm HPC, while
staying independent of PHOEBE.

Requires Python 3.12+. Runtime dependency: `asyncssh`.

## Design axiom

> The remote filesystem is the source of truth. The SSH connection is a
> disposable, reconnectable link.

A submitted Slurm job routinely outlives the local Python session, so the
connection cannot carry job identity. A job is fully described by
`(server, remote_dir, jobid)`, and any fresh `tether` instance with the same
credentials can reattach and query it.

This makes reconnection a non-event rather than a feature: crash resilience
falls out for free, and the honest restatement of "keep the connection alive"
is **make reconnection invisible**.

## Layers

| Layer | Module | Knows about |
|---|---|---|
| 1 | `link.py` | SSH only: `run`, `put`, `get`, reconnect, timeouts |
| 2 | `slurm.py` | Slurm formats and parsers. Pure — no I/O, unit-testable |
| 3 | `server.py` | `Server` / `SlurmServer`: the API PHOEBE talks to |

`Server.run()` is a deliberate escape hatch at every level: tether should never
be the reason something is impossible.

## Usage

```python
import tether

with tether.server("terra") as terra:
    print(terra.info())            # terra.villanova.edu (6.8.0-51-generic, 128 cpus)
    print(terra.slurm_version)     # slurm 23.02.7

    for p in terra.partitions():   # how busy is the cluster?
        print(p.name, p.nodes_idle, "/", p.nodes_total, "idle")

    for job in terra.queue(user=terra.whoami()):
        print(job.jobid, job.state, job.elapsed, job.name)
```

No config file is needed — `Server("terra")` falls through to `~/.ssh/config`,
which asyncssh reads natively (`Hostname`, `User`, `Port`, `IdentityFile`,
`ProxyJump`, `ProxyCommand`, `Match`, `Include` all honoured). Let ssh own SSH.

Pass `ssh_config="/path/to/config"` to read a specific file instead, akin
to `ssh -F`. The test rig uses it to describe a throwaway server without
touching `~/.ssh`.

## Configuration

Optional, in `~/.tether/servers.toml`:

```toml
[server.terra]
kind    = "slurm"                  # slurm | plain   (default: slurm)
host    = "terra.villanova.edu"
user    = "andrej"
workdir = "~/.tether"
default_environment = "phoebe"

[environment.phoebe]
kind    = "conda"                  # conda | venv | none
name    = "phoebe-dev"
modules = ["openmpi/4.1.5"]
prelude = []                       # raw shell lines, sourced last
env     = { OMP_NUM_THREADS = "1" }
mpirun  = "mpirun"
```

Servers and environments are sibling tables, not nested, so one environment
definition is reusable across servers. Unknown keys are a hard error — a typo
in a config file should be loud.

Environments are parsed and validated but inert until job submission lands.

## Decisions worth knowing

**One connection, many channels.** SSH multiplexes: each command is a channel on
an already-authenticated link, so 50 commands cost one authentication.
This is what keeps sshd's `MaxStartups` (and fail2ban) out of the picture.
The SFTP client is held open for the same reason — each one spawns a subsystem
channel and an `sftp-server` process remotely.

**Keepalive's real scope.** `keepalive_interval=30` only fires while the event
loop is running, so it protects long single operations, not long idle periods.
Idle drops are handled by reconnect-and-retry-once instead.

**Timeouts terminate the remote process.** `conn.run(timeout=...)` in asyncssh
raises but leaves the remote process running and the channel open; tether wraps
`create_process` in a context manager and calls `terminate()` explicitly.
Default 60s for commands, and `None` for transfers — a multi-GB `put` legitimately
exceeds a minute.

**Host keys are validated, with no opt-out.** A `known_hosts=None` switch would
be a security footgun in a library whose job is running commands on someone
else's machine. Failures carry a hint pointing at `~/.ssh/known_hosts`.

**No global state.** Each `Server` owns one connection and one event loop; there
is no shared registry. Construct one and pass it around.

**Lazy connect.** Constructing a `Server` performs no I/O and raises only on bad
configuration, so PHOEBE can build them during setup. Call `connect()` to fail
fast.

**Slurm's absence is a hard error.** If you name a `SlurmServer`, tether trusts
you: `sinfo --version` is probed once per connection and a failure raises
`SlurmError` rather than degrading silently.

**Never parse default Slurm output.** Every command uses an explicit `-o` format
with `-h`. Job name is placed last so a `|` inside it cannot shift other fields.
`--json` may be more robust on newer Slurm but its cross-version schema
stability isn't something to assume; `slurm_version` is recorded so that call
can later be made on evidence.

**`queue()` defaults to all users** — the honest picture of cluster load.
Filter with `queue(user=...)`; `whoami()` exists because `user` may legitimately
be `None` when `~/.ssh/config` supplies it.

## Exceptions

```
TetherError
├── ConfigError          bad or missing configuration
├── LinkError            cannot connect, or lost and not recovered
├── RemoteCommandError   nonzero exit where success was required
└── SlurmError           Slurm absent, or present and disagreed
```

`run()` returns a `Result` and never raises on nonzero exit; pass `check=True`
to opt in. Probing commands legitimately fail, and asking a question shouldn't
require a `try`/`except`.

`LinkError` rather than `ConnectionError` to avoid shadowing the builtin,
which is caught internally.

## Not yet implemented

Milestone 2, deliberately deferred:

- `submit()`, environment activation, file staging into per-job directories
- `sacct` for finished jobs, plus an exit-code sentinel written into the job
  directory so completion survives `sacct` retention policy
- `cancel()`, log streaming, reattach-by-directory

`job()` returning `None` is currently ambiguous — "finished" and "never existed"
are indistinguishable until `sacct` and sentinels land.

## Tests

```bash
bash tests/rig.sh start     # local fake cluster: sshd + Slurm shims on :2222
python -m pytest tests/
bash tests/rig.sh stop
```

`tests/rig.sh` needs **no root** and writes nothing outside `/tmp/tether-rig`.
It generates its own host key, client key, `authorized_keys`
and `ssh_config`; the live tests connect to the alias `tether-rig` through that
file, which also exercises the ssh_config fall-through. `sshd` needs no
privileges here because the only account it ever authenticates is the one
running it.

`tests/test_units.py` (parsers, config) needs nothing. `tests/test_live.py`
covers authentication, channel reuse, timeout-with-terminate, reconnect after a
dropped link, and SFTP round trips, and skips itself when the rig is down.
Layer 3 has no other coverage, so set `TETHER_REQUIRE_LIVE=1` in CI to turn that
skip into a hard error.
