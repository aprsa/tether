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
| 2 | `environment.py` | Shell activation lines. Pure — no I/O, unit-testable |
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
workdir = "~/.tether"             # remote scratch root; ~ resolved on use
timeout = 3600                     # seconds per command (default: 3600)
default_environment = "phoebe"

[environment.phoebe]
kind            = "conda"          # conda | venv | none
name            = "phoebe-dev"     # conda env name, or venv path
conda_base      = "/opt/conda"     # conda only; source its hook directly
modules         = ["openmpi/4.1.6"]
pre_activation  = []               # raw shell lines, run first
post_activation = []               # raw shell lines, run last
env             = { OMP_NUM_THREADS = "1" }
mpirun          = "mpirun"
```

Servers and environments are sibling tables, not nested, so one environment
definition is reusable across servers. Unknown keys are a hard error — a typo
in a config file should be loud.

## Environments

Three kinds are supported: `none` (bare metal), `venv`, and `conda`. The
generated shell runs in a fixed order, and every slot earns its place:

| Slot | What it is for |
|---|---|
| `pre_activation` | Verbatim lines, first. Makes the machinery available. |
| `modules` | `module load` each entry, in order |
| `env` | `export` each variable — before activation, so vars that *configure* activation take effect |
| activation | conda or venv; its `PATH` wins over everything above |
| `post_activation` | Verbatim lines, last: the final word after activation |

`pre_activation` exists because of a specific cluster reality: `module` is
normally a shell *function* sourced from `/etc/profile.d`, and `conda activate`
needs its hook sourced, so a non-interactive shell often cannot run either until
something makes them available. That is what this slot is for:

```toml
pre_activation = ["source /etc/profile.d/modules.sh"]
```

Both verbatim slots run *before* the payload; they differ only in which side of
activation they land on. Nothing here runs *after* the payload — that belongs to
the wrapper script and its exit-code trap.

**Failures are loud.** `module load` and activation are emitted with an explicit
guard that aborts rather than continues. A `conda activate` that quietly fails
would otherwise run the payload against the wrong interpreter and surface much
later as an unrelated `ImportError`.

**Verify before you depend on it.** `verify_environment()` activates and reports
what came back, so a broken environment is caught before a job is built on it:

```python
with tether.server("terra") as terra:
    print(terra.preamble)              # exactly what runs ahead of the payload
    print(terra.verify_environment())  # phoebe (conda): /opt/conda/envs/phoebe-dev
```

It raises `EnvActivationError` when a step fails, *and* when activation reports
success but `$VIRTUAL_ENV`/`$CONDA_PREFIX` is unset — that is how "the activate
script was a no-op" is caught rather than trusted.

`Server.run()` stays raw by default so scheduler queries are unaffected; pass
`environment=True` to run a command under the activation lines.

**Quoting has one exception worth knowing.** Everything interpolated is
`shlex.quote`d, *except* a leading `~`, which becomes `"$HOME/..."` — quoting a
path suppresses tilde expansion, so `source '~/venv/bin/activate'` would look
correct and silently fail.

Environment variable *values* are quoted, so they are literal: `env` cannot
reference another variable. Use `pre_activation`/`post_activation` when a value
has to be computed by the shell.

`mpirun` is parsed but inert until job submission lands.

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
`create_process` in a context manager and calls `terminate()` explicitly. The
default is deliberately generous (3600s) because installing conda or compiling a
package remotely takes minutes, and a timeout that interrupts real work is worse
than one that lets a wedged command hang; set `timeout` per server in
`servers.toml`, per call, or `None` for no limit. Transfers are untimed by
default — a multi-GB `put` legitimately exceeds any of this.

**Remote paths are resolved, not quoted.** `Server.path()` builds absolute paths
under `workdir`, expanding `~` against the remote `$HOME`. This is not cosmetic:
SFTP never expands `~`, so `put("~/.tether/x")` fails outright — and `~/.tether`
is the default workdir. `put`/`get` also create missing parent directories,
since SFTP reports a missing parent as a bare "No such file" that reads like the
*source* is absent.

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
├── ConfigError           bad or missing configuration
├── EnvActivationError    environment did not activate, or activated nowhere
├── LinkError             cannot connect, or lost and not recovered
├── RemoteCommandError    nonzero exit where success was required
└── SlurmError            Slurm absent, or present and disagreed
```

`run()` returns a `Result` and never raises on nonzero exit; pass `check=True`
to opt in. Probing commands legitimately fail, and asking a question shouldn't
require a `try`/`except`.

`LinkError` rather than `ConnectionError` to avoid shadowing the builtin,
which is caught internally.

## Not yet implemented

Milestone 2, deliberately deferred:

- `submit()` and file staging into per-job directories
- `sacct` for finished jobs, plus an exit-code sentinel written into the job
  directory so completion survives `sacct` retention policy
- `cancel()`, log streaming, reattach-by-directory

`job()` returning `None` is currently ambiguous — "finished" and "never existed"
are indistinguishable until `sacct` and sentinels land.

## Tests

```bash
python -m pytest tests/
```

Unit tests (parsers, preamble generation, quoting) need nothing installed and
run in well under a second. The live tests run against a **real Slurm cluster in
a container** — `tests/conftest.py` brings it up on demand, so there is nothing
to start by hand:

```bash
docker compose -f tests/cluster/docker-compose.yml up -d --wait   # optional; done for you
docker compose -f tests/cluster/docker-compose.yml exec cluster sinfo
docker compose -f tests/cluster/docker-compose.yml down
```

Nothing in the container is mocked. `slurmctld`, `slurmd`, `slurmdbd` and
MariaDB actually run, so job states, exit codes, `sbatch` rejections and `sacct`
history are Slurm's own; conda is a real Miniforge install (pinned and
checksum-verified); `module` is real environment-modules. That last one matters:
`module` is a shell *function* from `/etc/profile.d`, absent from a
non-interactive SSH command, which is exactly the condition `pre_activation`
exists for — and the tests assert it rather than assume it.

Everything lives in **one container** on purpose. It emulates a single computing
resource rather than a microservice estate, which also means munge's shared
secret never crosses a container boundary.

The container is `privileged` because `slurmd` needs a writable cgroup2 tree;
`cgroup: private` keeps it in its own namespace. It binds only to `127.0.0.1`.
Key material is generated host-side into `tests/cluster/.rig/` (gitignored) and mounted
read-only, so `~/.ssh` is never touched.

Live tests skip when Docker is unavailable. Set `TETHER_REQUIRE_LIVE=1` in CI to
turn that skip into a hard error — layer 3 has no other coverage. The first run
builds the image, which takes a few minutes; after that a cold start is ~6s.
