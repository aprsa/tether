"""User-facing server objects.

`Server` is a plain remote host.
`SlurmServer` adds scheduler queries.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from . import environment as _environment
from . import slurm as _slurm
from .config import EnvironmentConfig, EnvironmentKind, ServerConfig, ServerKind, load_config
from .errors import ConfigError, EnvActivationError, SlurmError
from .slurm import Job, Partition
from .link import DEFAULT_KEEPALIVE, DEFAULT_TIMEOUT, Result, Link


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

    label: str
    kind: str
    python: str
    version: str
    prefix: str

    def __str__(self) -> str:
        where = self.prefix or 'no python'
        return f'{self.label or "bare metal"} ({self.kind}): {where}'


class Server:
    """A remote resource reachable over SSH.

    Connects lazily: constructing a `Server` performs no I/O and raises only on
    bad configuration. Call `connect()` to fail fast, or just use it.

    Resolution order for every field is explicit argument, then
    `~/.tether/servers.toml`, then `~/.ssh/config` (handled by asyncssh). A
    `name` that is absent from the config file is treated as a hostname or
    `ssh_config` alias, so `Server("terra")` works with no tether config at all.
    Pass `ssh_config=` to read a specific ssh_config file rather than
    `~/.ssh/config`, exactly like `ssh -F`.

    Each instance owns exactly one connection and one event loop; there is no
    shared registry and no module-level state. Construct one and pass it around
    rather than constructing many for the same host.
    """

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
        keepalive: int = DEFAULT_KEEPALIVE,
        connect_timeout: float = 30.0,
    ) -> None:
        config = load_config(config_dir)
        cfg: ServerConfig | None = config.server(name) if name else None

        self.label = name
        self.host = host or (cfg.host if cfg else name)
        if not self.host:
            raise ConfigError('a server needs a name or an explicit host')

        self.user = user or (cfg.user if cfg else None)
        self.workdir = workdir or (cfg.workdir if cfg else '~/.tether')

        wanted = environment or (cfg.default_environment if cfg else None)
        if wanted and wanted not in config.environments:
            raise ConfigError(f"unknown environment '{wanted}'")
        self.environment: EnvironmentConfig | None = (
            config.environments.get(wanted) if wanted else None
        )

        self._username: str | None = None
        self._link = Link(
            self.host,
            self.user,
            port=port,
            ssh_config=ssh_config,
            keepalive=keepalive,
            connect_timeout=connect_timeout,
        )

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
        timeout: float | None = DEFAULT_TIMEOUT,
        environment: bool = False,
    ) -> Result:
        """Run an arbitrary shell command. The escape hatch.

        Raw by default, so scheduler queries and probes are unaffected. Pass
        `environment=True` to prepend this server's activation lines, which is
        what a payload wants.
        """
        if environment:
            command = _environment.wrap(self.environment, command)
        return self._link.run(command, check=check, timeout=timeout)

    @property
    def preamble(self) -> str:
        """The shell lines that activate this server's environment.

        Empty when no environment is configured. Worth printing when a job
        misbehaves: it is exactly what runs ahead of the payload.
        """
        return _environment.create_preamble(self.environment)

    def verify_environment(self) -> EnvironmentInfo:
        """Activate the environment and report what came back.

        Call this *before* a job depends on the environment. A failed
        `conda activate` inside a batch script surfaces later as an unrelated
        import error, which is a miserable thing to debug; here it is an
        `EnvActivationError` naming the step that failed.
        """
        kind = self.environment.kind if self.environment else EnvironmentKind.NONE
        label = self.environment.label if self.environment else ''

        result = self.run(_environment.PROBE, environment=True)
        if not result.ok:
            raise EnvActivationError(
                f'environment {label or "(none)"!r} failed to activate on '
                f'{self.host}: {result.stderr.strip() or "no output"}'
            )

        reported = (result.stdout.splitlines() + [''] * 5)[:5]
        virtual_env, conda_prefix, python, version, prefix = (
            value.strip() for value in reported
        )

        # Activation can "succeed" and do nothing -- an `activate` script that
        # is a no-op, or a hook that silently declined. The marker variables are
        # how we tell that apart from the real thing.
        expected = {
            EnvironmentKind.VENV: ('VIRTUAL_ENV', virtual_env),
            EnvironmentKind.CONDA: ('CONDA_PREFIX', conda_prefix),
        }.get(kind)  # type: ignore[arg-type]
        if expected and not expected[1]:
            raise EnvActivationError(
                f'environment {label!r} reported success on {self.host} but '
                f'${expected[0]} is unset, so nothing was activated'
            )

        return EnvironmentInfo(
            label=label,
            kind=str(kind),
            python=python,
            version=version,
            prefix=prefix,
        )

    def put(self, local: str, remote: str, *, recurse: bool = False) -> None:
        self._link.put(local, remote, recurse=recurse)

    def get(self, remote: str, local: str, *, recurse: bool = False) -> None:
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
        """One job by id, or `None` if the queue has never heard of it.

        `None` is ambiguous today: it means "not pending and not running", which
        covers both "finished" and "never existed". Disambiguating that is the
        job of `sacct` plus an exit-code sentinel in the job directory, and is
        deliberately deferred.
        """
        self.connect()
        fmt = shlex.quote(_slurm.spec_format(_slurm.SQUEUE_SPEC))
        command = f'squeue -h -a -o {fmt} --job={shlex.quote(str(jobid))}'
        result = self.run(command)

        if not result.ok:
            if 'invalid job id' in result.stderr.lower():
                return None
            raise SlurmError(
                f'squeue failed for job {jobid}: {result.stderr.strip()}'
            )

        jobs = _slurm.parse_squeue(result.stdout)
        return jobs[0] if jobs else None


server_dict: dict[ServerKind, type[Server]] = {
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
        cfg = load_config(config_dir).server(name) if name else None
        kind = cfg.kind if cfg else ServerKind.SLURM

    try:
        kind = ServerKind(kind)
    except ValueError:
        raise ConfigError(
            f'unknown server kind "{kind}"; '
            f'expected one of {[k.value for k in ServerKind]}'
        ) from None

    return server_dict[kind](name, **kwargs)  # type: ignore[arg-type]
