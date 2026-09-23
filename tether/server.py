"""User-facing server objects.

`Server` is a plain remote host.
`SlurmServer` adds scheduler queries.
"""

from __future__ import annotations

import os
import pathlib
import posixpath
import shlex
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, Self

from . import conda as _conda
from . import environment as _environment
from . import shell as _shell
from . import slurm as _slurm
from . import venv as _venv
from .config import DEFAULT_WORKDIR, ServerKind, _load_server, _save_server
from .conda import CondaInstallation
from .environment import Environment
from .venv import PythonInstallation, VenvInstallation
from .errors import ConfigError, EnvActivationError, SlurmError
from .shell import printf
from .slurm import Job, Partition, Submission
from .link import DEFAULT_KEEPALIVE, DEFAULT_TIMEOUT, Result, Link


_UNSET: Any = object()
"""Distinguishes "caller said nothing" from an explicit `timeout=None`, which
means no limit at all. Typed `Any` so the signatures using it stay honest."""


@dataclass(frozen=True)
class Host:
    """What the remote machine says about itself."""

    hostname: str
    kernel: str
    cpu_count: int | None

    def __str__(self) -> str:
        return f'{self.hostname} ({self.kernel}, {self.cpu_count} cpus)'


@dataclass(frozen=True)
class EnvironmentInfo:
    """What the remote shell reports *after* the environment was activated.

    The point is to answer "did I get the interpreter I asked for" before a job
    depends on the answer. `python` and `version` are empty when the remote has
    no python3 at all, which is a legitimate state for bare metal.
    """

    name: str
    kind: str
    python: str
    version: str
    prefix: str

    def __str__(self) -> str:
        where = self.prefix or 'no python'
        return f'{self.name or "bare metal"} ({self.kind}): {where}'


class Server:
    """A remote compute resource: the object PHOEBE holds and passes around.

    Despite the name, this is not merely a description of a host. One `Server`
    composes everything needed to work with a resource:

    - a `Link`, the single SSH connection, reconnected invisibly as needed;
    - an `Environment`, the environment its commands run under;
    - `workdir`, the remote scratch root that `path()` resolves against;
    - `timeout`, the default deadline for its commands.

    `SlurmServer` adds scheduler queries on top. If you are looking for where
    the connection lives, it is here -- there is no separate session object.

    Connects lazily: constructing a `Server` performs no I/O and raises only on
    bad configuration. Call `connect()` to fail fast, or just use it.

    Resolution order for every field is explicit argument, then
    `~/.tether/servers/<name>.json`, then `~/.ssh/config` (handled by
    asyncssh). A `name` with no config file is treated as a hostname or
    `ssh_config` alias, so `Server("terra")` works with no tether config at all.
    Pass `ssh_config=` to read a specific ssh_config file rather than
    `~/.ssh/config`, exactly like `ssh -F`.

    Each instance owns exactly one connection and one event loop; there is no
    shared registry and no module-level state. Construct one and pass it around
    rather than constructing many for the same host.
    """

    kind: ClassVar[ServerKind] = ServerKind.PLAIN
    """Which class a saved config maps back to. A class attribute rather than a
    field, so a stored kind can never disagree with the class holding it."""

    def __init__(
        self,
        name: str | None = None,
        *,
        host: str | None = None,
        user: str | None = None,
        port: int | None = None,
        workdir: str | None = None,
        environment: str | None = None,
        ssh_config: str | Path | None = None,
        config_dir: str | Path | None = None,
        timeout: float | None = None,
        link: Link | None = None,
        keepalive: int = DEFAULT_KEEPALIVE,
        connect_timeout: float = 30.0,
    ) -> None:
        saved = _load_server(name, config_dir) if name else None

        # A saved kind that disagrees with the class is the one mistake worth
        # catching: you would silently get a Server where the file says slurm,
        # and only notice when `queue()` turned out not to exist.
        if saved and saved['kind'] != type(self).kind:
            raise ConfigError(
                f"'{name}' is configured as kind '{saved['kind']}', not "
                f"'{type(self).kind}'; use tether.server('{name}')"
            )
        saved = saved or {}

        self.name = name
        self.host = host or saved.get('host') or name
        if not self.host:
            raise ConfigError('a server needs a name or an explicit host')

        self.user = user or saved.get('user')
        self.workdir = workdir or saved.get('workdir') or DEFAULT_WORKDIR
        self.timeout = timeout or saved.get('timeout') or DEFAULT_TIMEOUT

        self.environments: dict[str, Environment] = dict(
            saved.get('environments', {})
        )
        self.default_environment = environment or saved.get('default_environment')
        if self.default_environment and self.default_environment not in self.environments:
            raise ConfigError(
                f"unknown environment '{self.default_environment}' for server "
                f"'{name}'; defined here: {sorted(self.environments) or 'none'}"
            )

        self._username: str | None = None
        self._home: str | None = None

        # An injected link governs the transport outright: `host`, `user`,
        # `port` and `ssh_config` describe how a link *would* be built, and are
        # ignored when one is supplied. This is the seam a local (non-SSH)
        # transport plugs into for PHOEBE running on the login node.
        self._link = link or Link(
            self.host,
            self.user,
            port=port,
            ssh_config=ssh_config,
            keepalive=keepalive,
            connect_timeout=connect_timeout,
        )

    @property
    def environment(self) -> Environment | None:
        """The selected environment, or `None` when none is configured.

        `None` is not bare metal: bare metal is a `SystemEnvironment`, which
        still loads modules and exports variables. `None` means there is
        nothing to prepare at all.
        """
        if self.default_environment is None:
            return None
        return self.environments.get(self.default_environment)

    def add_environment(self, env: Environment, *, default: bool = False) -> Environment:
        """Declare an environment on this server. Returns it, for chaining."""
        self.environments[env.name] = env
        if default or self.default_environment is None:
            self.default_environment = env.name
        return env

    def to_dict(self) -> dict:
        """The JSON body `save()` writes. Runtime state -- the link, cached
        lookups -- is deliberately not part of it."""
        body: dict = {'kind': str(self.kind), 'host': self.host}
        if self.user:
            body['user'] = self.user
        if self.workdir != DEFAULT_WORKDIR:
            body['workdir'] = self.workdir
        if self.timeout != DEFAULT_TIMEOUT:
            body['timeout'] = self.timeout
        if self.default_environment:
            body['default_environment'] = self.default_environment
        if self.environments:
            body['environments'] = {
                name: env.to_dict() for name, env in self.environments.items()
            }
        return body

    def save(self, *, config_dir: str | Path | None = None,
             overwrite: bool = False) -> Path:
        """Write this server to `~/.tether/servers/<name>.json`."""
        if not self.name:
            raise ConfigError('a server needs a name before it can be saved')
        return _save_server(self.name, self.to_dict(),
                            config_dir=config_dir, overwrite=overwrite)

    @property
    def link(self) -> Link:
        """The underlying link, for anything this API does not cover."""
        return self._link

    @property
    def connected(self) -> bool:
        return self._link.connected

    def connect(self) -> Self:
        self._link.connect()
        return self

    def close(self) -> None:
        self._link.close()

    def __enter__(self) -> Self:
        return self.connect()

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __repr__(self) -> str:
        who = f'{self.user}@{self.host}' if self.user else self.host
        return f'<{type(self).__name__} {who}>'

    def run(
        self,
        command: str,
        *,
        check: bool = False,
        timeout: float | None = _UNSET,
        environment: bool = False,
    ) -> Result:
        """Run an arbitrary shell command. The escape hatch.

        Raw by default, so scheduler queries and probes are unaffected. Pass
        `environment=True` to prepend this server's activation lines, which is
        what a payload wants.

        `timeout` defaults to this server's; pass `None` for no limit.
        """
        if timeout is _UNSET:
            timeout = self.timeout
        if environment:
            command = _environment.with_preamble(self.environment, command)
        return self._link.run(command, check=check, timeout=timeout)

    @property
    def home(self) -> str:
        """The remote home directory. Resolved once, then cached."""
        if self._home is None:
            self._home = self.run(printf('"$HOME"'), check=True).stdout.strip()
        return self._home

    def path(self, *parts: str) -> str:
        """An absolute remote path under `workdir`.

        `workdir` defaults to `~/.tether`, and SFTP never expands `~` -- it
        fails outright rather than creating a literal `~` directory -- so the
        tilde is resolved here, once, against the remote `$HOME`. Use this for
        anything handed to `put`/`get`.
        """
        base = self.workdir
        if base == '~' or base.startswith('~/'):
            base = self.home + base[1:]
        # posixpath, not pathlib: this is a *remote* path, and the remote is
        # POSIX whatever the caller runs on.
        return posixpath.join(base, *parts)

    @property
    def preamble(self) -> str:
        """The shell lines that activate this server's environment.

        Empty when no environment is configured. Worth printing when a job
        misbehaves: it is exactly what runs ahead of the payload.
        """
        return _environment.preamble(self.environment)

    def probe_conda(self) -> list[CondaInstallation]:
        """Every conda installation this account can reach, and its environments.

        Delegates to `conda.probe()`, handing it this server's own `run`
        adapted to return stdout, and the environment's `pre_activation()` --
        so a conda that only appears once `module load anaconda` has run is
        found.
        See that function for what is searched, what is deliberately not, and
        why.
        """
        return _conda.probe(
            lambda command: self.run(command, check=True).stdout,
            self.path('conda'),
            setup=_environment.pre_activation(self.environment),
        )

    def install_conda(
        self,
        prefix: str | None = None,
        *,
        version: str | None = None,
        installer: str | None = None,
        sha256: str | None = None,
        adopt_if_exists: bool = True,
    ) -> CondaInstallation:
        """Install a conda that tether owns, and report what ended up there.

        Defaults to `<workdir>/conda`, which is also where `probe_conda()`
        looks -- so a default installation is discoverable afterwards
        without configuring anything.

        Calling this twice is harmless: the second call finds a working conda
        and adopts it. See `conda.install()` for further details.
        """
        return _conda.install(
            lambda command: self.run(command, check=True).stdout,
            prefix or self.path('conda'),
            version=version,
            installer=installer,
            sha256=sha256,
            adopt_if_exists=adopt_if_exists,
        )

    def probe_interpreters(self) -> list[PythonInstallation]:
        """Every bare-metal python on PATH that a venv could be built from.

        Runs the environment's `pre_activation()` first, so an interpreter that
        only appears once `module load python/3.12` has run is found -- which
        on a cluster is usually the only way a modern python appears at all.

        Conda interpreters are deliberately excluded; see `venv.py` for why.
        Use `probe_conda()` when a conda environment is what you want.
        """
        return _venv.probe_interpreters(
            lambda command: self.run(command, check=True).stdout,
            setup=_environment.pre_activation(self.environment),
        )

    def probe_venvs(
        self,
        venvs_base: str | None = None,
        *,
        include_broken: bool = False,
    ) -> list[VenvInstallation]:
        """The virtual environments tether can see here.

        `venvs_base` may name one venv or a directory of them; it defaults to
        `<workdir>/venvs`, where tether puts the ones it creates. An active
        `$VIRTUAL_ENV` is always included.

        The filesystem is not searched -- see `venv.probe_venvs()` for why, and
        for what `include_broken` costs.
        """
        return _venv.probe_venvs(
            lambda command: self.run(command, check=True).stdout,
            venvs_base or self.path('venvs'),
            setup=_environment.pre_activation(self.environment),
            include_broken=include_broken,
        )

    def create_venv(
        self,
        name: str,
        venvs_base: str | None = None,
        *,
        python: str | None = None,
        adopt_if_exists: bool = True,
    ) -> VenvInstallation:
        """Create a virtual environment, and report what ended up there.

        Lands at `<venvs_base>/<name>`, where `venvs_base` defaults to
        `<workdir>/venvs` -- the same place `probe_venvs()` looks by default,
        so anything created here is discoverable afterwards.

        The environment's `pre_activation()` runs first, which is how an
        interpreter that only exists after `module load python/3.12` can be
        used at all. See `venv.create()` for what `python` accepts and for
        what happens when the target is already occupied.
        """
        return _venv.create(
            lambda command: self.run(command, check=True).stdout,
            name,
            venvs_base or self.path('venvs'),
            python=python,
            setup=_environment.pre_activation(self.environment),
            adopt_if_exists=adopt_if_exists,
        )

    def install(self, *packages: str) -> None:
        """Install `packages` into this server's configured environment.

        Runs inside the activated environment, so the interpreter the preamble
        selected is the one that receives them. pip is used for every kind --
        see `Environment.install_command()` for why conda environments are no
        exception.

        Raises rather than installing anywhere surprising: bare metal and an
        unconfigured server both refuse, since neither has anything isolated
        to install into.
        """
        command = _environment.install_command(self.environment, packages)
        self.run(command, environment=True, check=True)

    def create_conda_env(
        self,
        name: str,
        *,
        base: str | None = None,
        python: str | None = None,
        packages: Iterable[str] = (),
        adopt_if_exists: bool = True,
    ) -> str:
        """Create a conda environment and report the prefix it landed in.

        `base` defaults to the conda tether installs, `<workdir>/conda`, so the
        common path needs no argument. Point it elsewhere to build against an
        installation `probe_conda()` found.

        The prefix is worth reading rather than assuming: conda relocates a
        named environment to `~/.conda/envs` when the base is not writable,
        which is the usual outcome against a site-wide install.
        """
        return _conda.create_env(
            lambda command: self.run(command, check=True).stdout,
            name,
            base or self.path('conda'),
            python=python,
            packages=packages,
            setup=_environment.pre_activation(self.environment),
            adopt_if_exists=adopt_if_exists,
        )

    def verify_environment(self) -> EnvironmentInfo:
        """Activate the environment and report what came back.

        Call this *before* a job depends on the environment. A failed
        `conda activate` inside a batch script surfaces later as an unrelated
        import error, which is a miserable thing to debug; here it is an
        `EnvActivationError` naming the step that failed.
        """
        env = self.environment
        kind = env.kind if env else 'none'
        label = env.name if env else ''

        env_probe = (
            printf('"${VIRTUAL_ENV-}"', '"${CONDA_PREFIX-}"') + '\n'
            + 'command -v python3 >/dev/null 2>&1 && python3 -c '
            + "'import sys; print(sys.executable); print(sys.version.split()[0]); "
            "print(sys.prefix)'"
        )

        result = self.run(env_probe, environment=True)
        if not result.ok:
            raise EnvActivationError(
                f'environment {label or "(none)"!r} failed to activate on '
                f'{self.host}: {result.stderr.strip() or "no output"}'
            )

        values = (result.stdout.splitlines() + [''] * 5)[:5]
        virtual_env, conda_prefix, python, version, prefix = (
            value.strip() for value in values
        )

        # Activation can "succeed" and do nothing -- an `activate` script that
        # is a no-op, or a hook that silently declined. Each kind knows its own
        # evidence; bare metal has none to give, and so cannot fail here.
        reported = {'VIRTUAL_ENV': virtual_env, 'CONDA_PREFIX': conda_prefix}
        if env and (why := env.activation_failure(reported)):
            raise EnvActivationError(
                f'environment {label!r} reported success on {self.host} but '
                f'{why}, so nothing was activated'
            )

        return EnvironmentInfo(
            name=label,
            kind=str(kind),
            python=python,
            version=version,
            prefix=prefix,
        )

    def put(
        self,
        local: str | os.PathLike[str],
        remote: str,
        *,
        recurse: bool = False,
    ) -> None:
        """Upload. `remote` is `str`: build it with `path()`, not `pathlib`."""
        self._link.put(local, remote, recurse=recurse)

    def get(
        self,
        remote: str,
        local: str | os.PathLike[str],
        *,
        recurse: bool = False,
    ) -> None:
        """Download."""
        self._link.get(remote, local, recurse=recurse)

    def ping(self) -> float:
        """Round-trip time in seconds."""
        return self._link.ping()

    def info(self) -> Host:
        """Basic facts about the remote machine, in one round trip."""
        result = self.run('printf "%s|%s|%s\\n" "$(uname -n)" "$(uname -r)" "$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN)"', check=True)
        parts = (result.stdout.strip().split('|') + ['', '', ''])[:3]
        hostname, kernel, cpus = (p.strip() for p in parts)
        return Host(hostname=hostname, kernel=kernel, cpu_count=_slurm._int(cpus))

    def whoami(self) -> str:
        """The remote username. Resolved once, then cached.

        Needed because `user` may legitimately be `None` when `~/.ssh/config`
        supplies it.
        """
        if self._username is None:
            self._username = self.run('id -un', check=True).stdout.strip()
        return self._username


class SlurmServer(Server):
    """A remote resource running Slurm.

    Slurm's presence is asserted once per connection, at `connect()` time or on
    first scheduler query, and its absence is a hard `SlurmError`: if you say a
    server runs Slurm, tether trusts you and complains loudly if you are wrong.
    """

    kind: ClassVar[ServerKind] = ServerKind.SLURM

    _slurm_version: str | None = None

    def connect(self) -> Self:
        super().connect()
        if self._slurm_version is None:
            result = self.run('sinfo --version', timeout=30)
            if not result.ok:
                self.close()
                stderr = result.stderr.strip() or 'no output'
                raise SlurmError(
                    f'no usable Slurm on {self.host}: '
                    f'`sinfo --version` exited {result.returncode} ({stderr})'
                )
            self._slurm_version = result.stdout.strip()
        return self

    @property
    def slurm_version(self) -> str:
        """e.g. 'slurm 23.02.7'. Triggers a connection if not yet known."""
        self.connect()
        if self._slurm_version is None:
            raise SlurmError(f'Slurm version unavailable on {self.host}')
        return self._slurm_version

    def partitions(self) -> list[Partition]:
        """Every partition, summarised one row each.

        Uses `sinfo -s`, so node counts come back as allocated/idle/other/total
        -- which is what you want in order to see how busy the cluster is.
        """
        self.connect()
        fmt = shlex.quote(_slurm.spec_format(_slurm.SINFO_SPEC))
        return _slurm.parse_sinfo(self.run(f'sinfo -h -s -o {fmt}', check=True).stdout)

    def queue(self, user: str | None = None) -> list[Job]:
        """Jobs in the queue.

        Defaults to *all* users, which is the honest picture of cluster load.
        Pass `user` to filter; `server.queue(user=server.whoami())` for your own.

        Note this reports pending and running jobs only -- that is what `squeue`
        knows. Completed jobs need `sacct`, which is a later addition.
        """
        self.connect()
        fmt = shlex.quote(_slurm.spec_format(_slurm.SQUEUE_SPEC))
        command = f'squeue -h -a -o {fmt}'
        if user:
            command += f' -u {shlex.quote(user)}'
        return _slurm.parse_squeue(self.run(command, check=True).stdout)

    def job(self, jobid: str | int) -> Job | None:
        """One job by id, at any point in its life, or `None` if no record of
        it survives anywhere.

        Two sources, depending on job status/timeline:

        - `scontrol`, which reads slurmctld's memory. It knows every pending
          and running job regardless of age, and every finished one until
          `MinJobAge` (300s by default) purges it. It carries the exit code,
          which `squeue` does not, and the pending `reason`, which `sacct`
          does not. No accounting service is required.
        - `sacct`, the accounting database, which is the only thing that still
          knows about a job that finished long ago -- the ordinary case for a
          detached job someone comes back to hours later. It needs `slurmdbd`;
          on a cluster without it, an old job is simply unknowable from Slurm,
          and the job directory is what proves it ran.

        Thus, the second call happens only for a job that has both finished *and*
        aged out, never during a poll loop.

        A job that finished is reported with
        its state and exit code; only one that Slurm has genuinely never heard
        of -- or has forgotten entirely -- comes back empty (`None`).
        """
        self.connect()

        live = self.run(_slurm.scontrol_command(jobid))
        if live.ok:
            found = _slurm.parse_scontrol(live.stdout)
            if found is not None:
                return found
        elif 'invalid job id' not in live.stderr.lower():
            raise SlurmError(
                f'scontrol failed for job {jobid}: {live.stderr.strip()}'
            )

        # Finished and purged from slurmctld, or never existed. Only the
        # accounting database can still tell the two apart.
        past = self.run(_slurm.sacct_command(jobid))
        if not past.ok:
            return None

        jobs = _slurm.parse_sacct(past.stdout)
        return jobs[0] if jobs else None

    def stdout(self, job: Submission | str) -> str:
        """What the job has written to stdout (thus far).

        Readable while the job is still running, which is how partial output
        is followed -- Slurm writes into the file as the job goes.

        Takes a `Submission` or a job directory, so a job picked up in a later
        session can be read without one. Empty means the job has not started
        writing; a missing directory raises an error, as that implies a manual
        removal of the job history.
        """
        return self._read_job_file(job, _slurm.STDOUT)

    def stderr(self, job: Submission | str) -> str:
        """What the job has written to stderr (thus far). See `stdout()`."""
        return self._read_job_file(job, _slurm.STDERR)

    def _read_job_file(self, job: Submission | str, filename: str) -> str:
        directory = job.directory if isinstance(job, Submission) else job
        at = _shell.remote_path(directory)

        # Two different absences: no directory is a problem, no file yet is
        # not. `cat` alone would report them identically.
        result = self.run(
            f'[ -d {at} ] || exit 2\ncat {at}/{filename} 2>/dev/null || true'
        )
        if result.returncode == 2:
            raise SlurmError(
                f'no job directory at {directory}: it was never created, or '
                f'it has been removed'
            )
        return result.stdout

    def cancel(self, jobid: str | int) -> None:
        """Cancel a job. No return value; raises if there is nothing to cancel.

        `scancel` reports nothing at all -- cancelling a running job, a job
        that finished an hour ago, and a job id that never existed are
        indistinguishable, all silent and all exit 0. So a typo would quietly
        "succeed". Tether checks first instead, and the exceptions carry what
        the return value otherwise would: silence means the job was running or
        queued and has now been told to stop.

        Deliberately not synchronous. `scancel` returns immediately while
        the job moves through `COMPLETING` on its own schedule -- cleanup and
        the epilog can take a while on a large job -- so waiting for a terminal
        state would block for an interval nothing can predict. Ask `job()` when
        you want to know where it got to.
        """
        found = self.job(jobid)

        if found is None:
            raise SlurmError(
                f'no record of job {jobid}, so there is nothing to cancel. It '
                f'may never have existed, or it finished long enough ago that '
                f'Slurm has forgotten it'
            )

        if found.is_finished:
            raise SlurmError(
                f'job {jobid} has already finished ({found.state}, exit '
                f'{found.exit_code}), so there is nothing to cancel'
            )

        self.run(_slurm.cancel_command(jobid), check=True)

    def submit(
        self,
        payload: str,
        *,
        name: str = 'job',
        partition: str | None = None,
        nodes: int | None = None,
        cpus: int | None = None,
        time: str | None = None,
        memory: str | None = None,
        directives: Mapping[str, str] | None = None,
        inputs: Iterable[str | os.PathLike[str]] = (),
        jobs_base: str | None = None,
    ) -> Submission:
        """Submit `payload` as a batch job, and report what was sent.

        Everything about the job lands in one directory, `<jobs_base>/<name>.
        <timestamp>`, defaulting to `<workdir>/jobs`. The script as submitted,
        `stdout`, `stderr` and the jobid are all there, which is what lets a
        later session pick the job up again -- and what keeps "finished long
        ago" distinguishable from "never submitted" on a cluster whose Slurm
        has since forgotten.

        The environment's full `preamble()` runs inside the script, so a job
        gets the environment an interactive `run()` would, guards included.

        `directives` passes anything tether has not anticipated straight
        through: `{'gres': 'gpu:1'}` becomes `#SBATCH --gres=gpu:1`.

        `inputs` are local files or directories copied into the job directory
        before submission. Because the job runs there, the payload refers to
        them by bare name -- `python fit.py data.csv`, not a path. They are
        checked before the directory is claimed, so a typo does not leave an
        empty job directory behind.

        Per-job inputs only. A large file shared by many jobs should be `put()`
        somewhere stable once and referred to by absolute path, rather than
        copied per submission.
        """
        staged = [pathlib.Path(source) for source in inputs]
        missing = [str(source) for source in staged if not source.exists()]
        if missing:
            raise ConfigError(
                f'cannot stage {", ".join(missing)}: no such file or directory'
            )

        base = jobs_base or self.path('jobs')
        # Naive local time on purpose: this is a label for a human reading
        # `ls`, never compared or converted. Uniqueness is enforced below, not
        # by the clock.
        directory = _slurm.job_directory(base, name, datetime.now())  # noqa: DTZ005
        script = _slurm.batch_script(
            payload,
            name=name,
            preamble=_environment.preamble(self.environment),
            partition=partition,
            nodes=nodes,
            cpus=cpus,
            time=time,
            memory=memory,
            directives=directives,
        )

        # The directory is claimed before anything is written into it, and
        # may come back with an index: two jobs of one name in one second are
        # the caller's business, not something to refuse or to conflate.
        directory = _slurm.parse_claim(
            self.run(_slurm.claim_command(directory), check=True).stdout
        )
        at = _shell.remote_path(directory)
        for source in staged:
            self.put(
                source,
                posixpath.join(directory, source.name),
                recurse=source.is_dir(),
            )
        self.run(
            f'printf %s {shlex.quote(script)} > {at}/{_slurm.SCRIPT}', check=True
        )

        jobid = _slurm.parse_submit(
            self.run(_slurm.submit_command(directory), check=True).stdout
        )
        # Written before anything can go wrong with it: the directory is how
        # tether knows this job existed once Slurm no longer does.
        self.run(f'printf %s {shlex.quote(jobid)} > {at}/{_slurm.JOBID}', check=True)

        return Submission(
            jobid=jobid,
            directory=directory,
            name=name,
            partition=partition,
            nodes=nodes,
            cpus=cpus,
            time=time,
            memory=memory,
            directives=tuple((directives or {}).items()),
            inputs=tuple(source.name for source in staged),
        )


SERVERS: dict[ServerKind, type[Server]] = {
    ServerKind.PLAIN: Server,
    ServerKind.SLURM: SlurmServer,
}


def server(name: str | None = None, *, kind: ServerKind | str | None = None, **kwargs: object) -> Server:
    """Build the right server class for `name`, per `~/.tether/servers.toml`.

    Defaults to `SlurmServer` when the config file says nothing, since that is
    the intended target. Pass `kind='plain'` for a bare remote shell.
    """

    if kind is None:
        # No explicit kind: try the config file, then fall back to Slurm.
        config_dir = kwargs.get('config_dir')
        if not isinstance(config_dir, (str, Path, type(None))):
            raise ConfigError(
                f'config_dir must be a str, Path or None, '
                f'not {type(config_dir).__name__}'
            )
        saved = _load_server(name, config_dir) if name else None
        kind = saved['kind'] if saved else ServerKind.SLURM

    try:
        kind = ServerKind(kind)
    except ValueError:
        raise ConfigError(
            f'unknown server kind "{kind}"; '
            f'expected one of {[k.value for k in ServerKind]}'
        ) from None

    return SERVERS[kind](name, **kwargs)  # type: ignore[arg-type]
