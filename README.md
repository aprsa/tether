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
credentials can reattach and query it. It is crash-resilient because `tether` does not keep the connection alive; rather, it makes the reconnection opaque to the user.

<!-- ## Layers

| Layer | Module | Knows about |
|---|---|---|
| 0 | `event_loop.py` | The sync/async boundary. Knows nothing about SSH |
| 1 | `link.py` | SSH only: `run`, `put`, `get`, reconnect, timeouts |
| 2 | `shell.py` | Quoting and emission. Assumes bash. Pure — no I/O, unit-testable |
| 2 | `slurm.py` | Slurm formats and parsers. Pure — no I/O, unit-testable |
| 2 | `environment.py` | Activation lines. Pure — no I/O, unit-testable |
| 2 | `conda.py` | Finding and installing conda. Takes its transport as an argument, so it is unit-testable against a dict |
| 2 | `venv.py` | Interpreters and virtual environments. Same |
| 3 | `server.py` | `Server` / `SlurmServer`: the API PHOEBE talks to |

`Server.run()` is a deliberate escape hatch at every level: tether should never
be the reason something is impossible. -->

## Usage

```python
import tether

with tether.server('terra') as terra:
    print(terra.info())            # terra.villanova.edu (6.8.0-51-generic, 128 cpus)
    print(terra.slurm_version)     # slurm 23.02.7

    for p in terra.partitions():   # how busy is the cluster?
        print(p.name, p.nodes_idle, '/', p.nodes_total, 'idle')

    for job in terra.queue(user=terra.whoami()):
        print(job.jobid, job.state, job.elapsed, job.name)
```

Config files are optional: `tether.server('terra')` falls through to `~/.ssh/config`, which asyncssh reads natively (`Hostname`, `User`, `Port`, `IdentityFile`, `ProxyJump`, `ProxyCommand`, `Match`, `Include` all honored): `tether` lets ssh own SSH.

Custom ssh configurations are supported by passing `ssh_config='/path/to/config'` to read a specific file instead, akin to `ssh -F`.

## Configuration

Optional, one JSON file per server in `~/.tether/servers/`:

```json
// ~/.tether/servers/terra.json
{
  "tether": "0.1.0",
  "kind": "slurm",
  "host": "terra.villanova.edu",
  "user": "andrej",
  "workdir": "~/.tether",
  "default_environment": "phoebe",
  "environments": {
    "phoebe": {
      "kind": "conda",
      "name": "phoebe",
      "conda_base": "/opt/conda",
      "env": {"OMP_NUM_THREADS": "1"}
    }
  }
}
```

The filename corresponds to the server name, so renaming a filename is the same as renaming the server. Unknown server names (in `tether.server('server_name')`) raise an error.

**Note**: Environments are nested inside their server, not shared between servers. In practice they do not generalise: `modules`, `pre_activation_cmds` and `conda_base` each encode one cluster's assumptions, and only `kind` and `name` travel. Nesting also makes it impossible to point a server at an environment meant for a different machine.

Configuration can be saved straight from the `tether.server` object:

```python
srv = tether.server('terra', host='terra.villanova.edu', user='andrej')
srv.add_environment(tether.CondaEnvironment('phoebe', conda_base='/opt/conda'))
srv.add_environment(tether.VenvEnvironment('dev', path='~/.venvs/dev'))
srv.save()                         # -> ~/.tether/servers/terra.json

tether.list_servers()              # -> ["terra", ...]
tether.delete_server('terra')
```

Saving omits anything left at its default, so a file records what was actually chosen rather than every default in force when it was written. Runtime state (the connection, cached lookups) is never written.

## Environments

One class per kind: `SystemEnvironment` (bare metal), `VenvEnvironment`, `CondaEnvironment`, each owning its own fields, validation and activation lines.

```python
tether.CondaEnvironment('phoebe', conda_base='/opt/conda')
tether.VenvEnvironment('dev', path='~/venvs/dev', modules=['gcc'])
tether.SystemEnvironment('bare')

tether.env('dev', 'venv', path='~/venvs/dev')   # or by kind, mirroring server()
```

The first argument is the environment's **name**: tether's handle for it, and its key in the config file. It is not what gets activated — a venv named `dev` may live at `/scratch/venvs/phoebe-2.5`, and that name appears nowhere in the generated shell. What each kind activates is named for what it actually is: a venv has a `path` (there is no such thing as a venv *name*), and a conda environment has a `conda_env`, which defaults to the name as the two are usually the same word.

Fields belonging to another kind are refused because they are not fields on that class: `conda_base` on a venv is an error. Validation happens at construction, so an environment that cannot be loaded back cannot be built in the first place, let alone saved.

The generated shell runs in a fixed order:

| Slot | What it is for |
| --- | --- |
| `pre_activation_cmds` | Verbatim lines, first. Makes the machinery available. |
| `modules` | `module load` each entry, in order |
| `env` | `export` each variable — before activation, so vars that *configure* activation take effect |
| activation | conda or venv; its `PATH` wins over everything above |
| `post_activation_cmds` | Verbatim lines, last: the final word after activation |

`pre_activation_cmds` exists because of a specific cluster reality: `module` is
normally a shell *function* sourced from `/etc/profile.d`, and `conda activate`
needs its hook sourced, so a non-interactive shell often cannot run either until
something makes them available. That is what this slot is for:

```json
"pre_activation_cmds": ["source /etc/profile.d/modules.sh"]
```

**Failures are loud.** `module load` and activation are emitted with an explicit guard that aborts rather than continues on failure. A `conda activate` that quietly fails would otherwise run the payload against the wrong interpreter and surface much later as an unrelated `ImportError`.

**Finding conda.** `probe_conda()` reports the installations this account can actually reach, from three sources: `$PATH` — evaluated *after* `pre_activation_cmds` and `modules`, so a conda that only appears once a module is loaded counts — conda's own record in `~/.conda/environments.txt`, and `<workdir>/conda` where tether installs its own.

```python
for condas in srv.probe_conda():
    print(condas)
# /home/users/andrej/crimpl-conda (conda 25.5.1): no environments
# /home/users/andrej/miniconda3 (conda 25.5.1): crimpl
```

It deliberately does **not** search conventional directories. Such a list can
never be exhaustive, so "found nothing" would not mean "there is no conda" — and
a wrong answer that looks thorough is worse than no answer. An installation
somewhere else is reachable by naming it in `conda_base`, or by making it
reachable from `pre_activation_cmds`, which then also makes it discoverable.

Environment names include conda's per-user fallback `~/.conda/envs`, because a
named environment created against a site install lands there rather than under
the base — and `conda activate <name>` finds it either way.

**Provisioning.** `install_conda()` puts a Miniforge at `<workdir>/conda`, where
`probe_conda()` already looks. `probe_interpreters()` lists the bare-metal
pythons a venv can be built from — conda's are excluded, since conda
environments are conda's job — and `create_venv()` builds one.

```python
srv1.install_conda()                              # <workdir>/conda

srv2.probe_interpreters()
srv2.create_venv('analysis', python='3.12')       # raises if 3.12 is not on PATH
srv1.create_conda_env('phoebe', python='3.12')   # -> the prefix it landed in
srv2.probe_venvs('~/.venvs', include_broken=True)
srv2.install('phoebe')                            # pip, inside the activated env
```

Both installers are idempotent: an existing, *working* installation is adopted
rather than rebuilt, and one that exists but does not run raises rather than
being overwritten. That guard matters more for venvs than for conda —
`python -m venv` writes into an occupied directory and exits 0, so nothing but
tether stands between a mistyped path and someone's working directory.

`probe_venvs()` is told where to look rather than searching, because venv keeps
no registry to ask and the filesystem is the wrong place to guess: on terra,
`find $HOME` costs 22 seconds at the depth where venvs actually live.

`create_conda_env()` returns the prefix rather than letting you assume one:
conda puts a named environment under the base when the base is writable and in
`~/.conda/envs` when it is not, which is the usual outcome against a site-wide
installation.

`install()` uses pip for every kind, conda environments included. `conda
install` would be idiomatic there but can only offer what conda-forge carries,
and PHOEBE is published on PyPI alone. Bare metal refuses: installing into the
system python needs root, and `--user` leaks into every later job.

**Verify before you depend on it.** `verify_environment()` activates and reports
what came back, so a broken environment is caught before a job is built on it:

```python
with tether.server('terra') as terra:
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

Environment variable *values* are quoted, so they are literal: `env` cannot reference another variable. Use `pre_activation_cmds`/`post_activation_cmds` when a value has to be computed by the shell.

`mpirun` is parsed but inert until job submission lands.

## Jobs

`submit()` writes a batch script, sends it to `sbatch`, and gives back a record
of what was sent.

```python
s = srv.submit('python fit.py', name='fit', cpus=8, time='02:00:00',
               directives={'gres': 'gpu:1'})
# 626 'fit' in /home/u/.tether/jobs/fit.20260919-184816 (cpus=8, time=02:00:00)
```

Everything about one job lives in one directory — the script as submitted,
`stdout`, `stderr`, and the jobid. That directory is the handle: it is named
before submission (the jobid does not exist until `sbatch` has already been told
where to run), and it is what lets a later session pick the job up again.

`Submission` is a record of the *request*, not of the job's state, so it does not
go stale.

**Slurm is the source of truth for state**, and the filesystem for artifacts.
There is no exit-code sentinel: Slurm already records state and exit code, and
a second record would be a second truth.

`job()` asks two sources, depending on job status/availability:

| | `scontrol` | `sacct` |
| --- | --- | --- |
| pending or running, at any age | yes | yes |
| finished | for `MinJobAge` (300s) | durably |
| why a job is pending | yes | field exists, never filled |
| exit code | yes | yes |
| needs `slurmdbd` | no | **yes** |

Thus, `scontrol` answers everything during a poll loop, and the accounting
database is consulted only for a job that has both finished *and* aged out --
the ordinary case for a detached job someone comes back to hours later.

```python
job = srv.job(sub.jobid)
job.state        # 'FAILED'
job.exit_code    # 7
job.is_failed    # True
```

Read `state`, not `exit_code`, to decide whether a job succeeded: a *cancelled*
job reports exit code 0 on its allocation row, because only the `.batch` step
records the signal that killed it. And `signal` is kept separate from
`exit_code` for the same reason — a signalled job exits 0.

On a cluster with no accounting, a job that finished more than `MinJobAge` ago
is lost from Slurm. The job directory still proves it ran, and
its `stdout` is still there.

`cancel()` doesn't return a value; it raises if there was nothing to cancel. That is the whole signal, because `scancel` gives none: cancelling a running job, one that finished an hour ago, and an id that never existed are all silent and all exit 0, so a typo would quietly succeed.

It is also deliberately not synchronous. `scancel` returns immediately while the job moves through `COMPLETING` at its own pace. Ask `job()` to get the up-to-date status.

## Decisions worth knowing

**One connection, many channels.** SSH multiplexes: each command is a channel on an already-authenticated link, so 50 commands cost one authentication. The SFTP client is held open for the same reason — each one spawns a subsystem channel and an `sftp-server` process remotely.

**The event loop runs on its own thread.** tether is synchronous; asyncssh is
not. Rather than driving a loop in place with `run_until_complete` — which
cannot nest, so every call raised inside a Jupyter kernel — each `Link` owns a
loop on a thread and hands work to it. That makes the synchronous API work from
a script, a notebook, or inside someone else's async application, and it is
also what will let a held-open channel make progress *between* calls when
completion notification lands. The cost: a call from inside an async context
blocks that context until it returns.

**Keepalive therefore means what it says.** With the loop running continuously,
`keepalive_interval=30` fires during idle periods, not only for the duration of
a call. Reconnect-and-retry-once is the backstop for drops it does not catch.

**Timeouts terminate the remote process.** `conn.run(timeout=...)` in asyncssh
raises but leaves the remote process running and the channel open; tether wraps
`create_process` in a context manager and calls `terminate()` explicitly. The
default is deliberately generous (3600s) because installing conda or compiling a
package remotely takes minutes, and a timeout that interrupts real work is worse
than one that lets a wedged command hang; set `timeout` per server in
its config file, per call, or `None` for no limit. Transfers are untimed by
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

**Lazy connect, and `close()` is its counterpart.** Constructing a `Server`
performs no network I/O and raises only on bad configuration, so PHOEBE can
build them during setup. `connect()` fails fast; `close()` drops the connection
and stops the loop thread, leaving the object usable — a later call reconnects.
There is no separate `disconnect()`, because that is what `close()` already is.

An abandoned `Server` is reclaimed on garbage collection, which matters in a
notebook where re-running a cell rebinds the variable and orphans the previous
one. That is a safety net, not a substitute for `close()`.

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

```ascii
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

Environments are done; running things is not. Deliberately deferred:

- file staging into the job directory, and fetching results back out
- log streaming, and reattach-by-directory

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
non-interactive SSH command, which is exactly the condition `pre_activation_cmds`
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
