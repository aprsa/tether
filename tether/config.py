"""Configuration loading for tether.

Configuration lives in ``~/.tether/servers.toml`` and is optional: anything
omitted is resolved by asyncssh from ``~/.ssh/config``.

Layout::

    [server.terra]
    kind    = "slurm"                  # slurm | plain      (default: slurm)
    host    = "terra.villanova.edu"    # default: the table key
    user    = "andrej"                 # default: ssh_config / local user
    workdir = "~/.tether"              # remote scratch root
    default_environment = "phoebe"

    [environment.phoebe]
    kind            = "conda"          # conda | venv | none
    name            = "phoebe-dev"     # conda env name, or venv path
    conda_base      = "/opt/conda"     # conda only; source its hook directly
    modules         = ["openmpi/4.1.5"]   # module load, in order
    pre_activation  = []               # raw shell lines, run first
    post_activation = []               # raw shell lines, run last
    env             = { OMP_NUM_THREADS = "1" }
    mpirun          = "mpirun"

Servers and environments are sibling tables, so one environment definition can
be reused across servers.

`pre_activation` and `post_activation` are both verbatim shell lines,
distinguished only by where they land relative to conda/venv activation. Both
run *before* the payload. See `environment.py` for what belongs in each.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from enum import StrEnum

from .errors import ConfigError

DEFAULT_CONFIG_DIR = Path.home() / '.tether'
CONFIG_FILENAME = 'servers.toml'


class ServerKind(StrEnum):
    SLURM = 'slurm'
    PLAIN = 'plain'


class EnvironmentKind(StrEnum):
    CONDA = 'conda'
    VENV = 'venv'
    NONE = 'none'


@dataclass(frozen=True)
class EnvironmentConfig:
    """Prepare a shell before sending payload."""

    label: str
    kind: EnvironmentKind | str = EnvironmentKind.NONE
    name: str | None = None
    conda_base: str | None = None
    modules: tuple[str, ...] = ()
    pre_activation: tuple[str, ...] = ()
    post_activation: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    mpirun: str = 'mpirun'


@dataclass(frozen=True)
class ServerConfig:
    """Identity of a remote resource."""

    label: str
    kind: ServerKind | str = ServerKind.SLURM
    host: str | None = None
    user: str | None = None
    workdir: str = '~/.tether'
    default_environment: str | None = None


@dataclass(frozen=True)
class Config:
    servers: dict[str, ServerConfig] = field(default_factory=dict)
    environments: dict[str, EnvironmentConfig] = field(default_factory=dict)

    def server(self, label: str) -> ServerConfig | None:
        return self.servers.get(label)


def config_path(config_dir: str | Path | None = None) -> Path:
    base = Path(config_dir).expanduser() if config_dir else DEFAULT_CONFIG_DIR
    return base / CONFIG_FILENAME


def load_config(config_dir: str | Path | None = None) -> Config:
    """Read the config file. Note that config_dir can be None."""
    path = config_path(config_dir)
    if not path.exists():
        return Config()

    try:
        with path.open('rb') as fh:
            raw = tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f'{path}: {exc}') from exc
    except OSError as exc:
        raise ConfigError(f'cannot read {path}: {exc}') from exc

    servers = {
        label: _server(path, label, body)
        for label, body in _table(path, raw, 'server').items()
    }
    environments = {
        label: _environment(path, label, body)
        for label, body in _table(path, raw, 'environment').items()
    }

    for cfg in servers.values():
        wanted = cfg.default_environment
        if wanted and wanted not in environments:
            raise ConfigError(
                f"{path}: server '{cfg.label}' names unknown "
                f"environment '{wanted}'"
            )

    return Config(servers=servers, environments=environments)


def _table(path: Path, raw: dict, key: str) -> dict:
    body = raw.get(key, {})
    if not isinstance(body, dict):
        raise ConfigError(f'{path}: [{key}] must be a table')
    return body


def _server(path: Path, label: str, body: dict) -> ServerConfig:
    where = f'{path}: [server.{label}]'
    _reject_unknown(where, body, ServerConfig, skip={'label'})

    kind = body.get('kind', ServerKind.SLURM)
    if kind not in ServerKind:
        raise ConfigError(
            f'{where}: kind must be one of {[k.value for k in ServerKind]}, not {kind!r}'
        )

    return ServerConfig(
        label=label,
        kind=kind,
        host=body.get('host', label),
        user=body.get('user'),
        workdir=body.get('workdir', '~/.tether'),
        default_environment=body.get('default_environment'),
    )


def _environment(path: Path, label: str, body: dict) -> EnvironmentConfig:
    where = f'{path}: [environment.{label}]'
    _reject_unknown(where, body, EnvironmentConfig, skip={'label'})

    kind = body.get('kind', EnvironmentKind.NONE)
    if kind not in EnvironmentKind:
        raise ConfigError(
            f'{where}: kind must be one of {[k.value for k in EnvironmentKind]}, '
            f'not {kind!r}'
        )

    name = body.get('name')
    if kind in (EnvironmentKind.CONDA, EnvironmentKind.VENV) and not name:
        raise ConfigError(f"{where}: kind '{kind}' requires 'name'")

    conda_base = body.get('conda_base')
    if conda_base and kind != EnvironmentKind.CONDA:
        raise ConfigError(
            f"{where}: 'conda_base' is only meaningful for kind 'conda', not {kind!r}"
        )

    env = body.get('env', {})
    if not isinstance(env, dict):
        raise ConfigError(f"{where}: 'env' must be a table of strings")

    return EnvironmentConfig(
        label=label,
        kind=kind,
        name=name,
        conda_base=conda_base,
        modules=tuple(body.get('modules', ())),
        pre_activation=tuple(body.get('pre_activation', ())),
        post_activation=tuple(body.get('post_activation', ())),
        env={str(k): str(v) for k, v in env.items()},
        mpirun=body.get('mpirun', 'mpirun'),
    )


def _reject_unknown(where: str, body: dict, cls: type, skip: set[str]) -> None:
    """Typos in a config file should be loud, not silently ignored."""
    known = {f for f in cls.__dataclass_fields__ if f not in skip}
    unknown = set(body) - known
    if unknown:
        raise ConfigError(
            f'{where}: unknown key(s) {sorted(unknown)}; '
            f'expected any of {sorted(known)}'
        )
