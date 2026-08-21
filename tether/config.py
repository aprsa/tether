"""Configuration for tether.

One JSON file per server, under ``~/.tether/servers/``::

    ~/.tether/servers/terra.json

The filename *is* the server's name, so renaming a server is a `mv`, and
loading one reads one file rather than every file. Everything a server needs
lives in that file, environments included::

    {
      "tether": "0.1.0",
      "kind": "slurm",
      "host": "terra.villanova.edu",
      "user": "andrej",
      "workdir": "~/.tether",
      "default_environment": "phoebe",
      "environments": {
        "phoebe": {"kind": "conda", "name": "phoebe", "conda_base": "/opt/conda"}
      }
    }

Environments are nested rather than shared between servers because in practice
they do not generalise: `modules`, `pre_activation` and `conda_base` each encode
one cluster's assumptions, and only `kind` and `name` travel. A shared
definition could not say which cluster it was written against, and nesting
makes it impossible to point a server at an environment meant for another.

JSON rather than TOML because tether writes these files as well as reading
them, and the standard library can only read TOML. Configuration stays
optional: with no file at all, a server name falls through to ``~/.ssh/config``.
"""

from __future__ import annotations

import dataclasses
import json
import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from .errors import ConfigError

DEFAULT_CONFIG_DIR = Path.home() / '.tether'
SERVERS_DIRNAME = 'servers'
VERSION_KEY = 'tether'
# Stamped into every file tether writes, so a later version can recognise an
# older layout instead of guessing at it.

#: A server name becomes a filename, so it must not contain separators or
#: traverse. Leading character is alphanumeric, which also rules out `.`/`..`.
_NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]*\Z')


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
    """A remote resource, and the environments available on it."""

    label: str
    kind: ServerKind | str = ServerKind.SLURM
    host: str | None = None
    user: str | None = None
    workdir: str = '~/.tether'
    default_environment: str | None = None
    timeout: float | None = None
    """Seconds for remote commands; `None` falls back to `DEFAULT_TIMEOUT`."""
    environments: dict[str, EnvironmentConfig] = field(default_factory=dict)


# -- locations -------------------------------------------------------------


def servers_dir(config_dir: str | Path | None = None) -> Path:
    base = Path(config_dir).expanduser() if config_dir else DEFAULT_CONFIG_DIR
    return base / SERVERS_DIRNAME


def server_path(name: str, config_dir: str | Path | None = None) -> Path:
    """Where `name` is stored. Validates the name, since it is a filename."""
    if not _NAME.match(name):
        raise ConfigError(
            f'not a usable server name: {name!r} -- it becomes a filename, so '
            f'it must start with a letter or digit and contain only letters, '
            f'digits, dot, dash or underscore'
        )
    return servers_dir(config_dir) / f'{name}.json'


def list_servers(config_dir: str | Path | None = None) -> list[str]:
    """Every configured server name. Empty when nothing is configured."""
    directory = servers_dir(config_dir)
    if not directory.is_dir():
        return []
    return sorted(path.stem for path in directory.glob('*.json'))


# -- reading ---------------------------------------------------------------


def load_server(
    name: str, config_dir: str | Path | None = None
) -> ServerConfig | None:
    """One server's configuration, or `None` if it has none.

    `None` is not an error: an unconfigured name is treated as a hostname or an
    `ssh_config` alias.
    """
    path = server_path(name, config_dir)
    if not path.exists():
        return None

    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ConfigError(f'{path}: {exc}') from exc
    except OSError as exc:
        raise ConfigError(f'cannot read {path}: {exc}') from exc

    if not isinstance(raw, dict):
        raise ConfigError(f'{path}: top level must be an object')

    raw.pop(VERSION_KEY, None)   # written by tether; not a configuration field
    return _server(path, name, raw)


def _server(path: Path, label: str, body: dict) -> ServerConfig:
    where = str(path)
    _reject_unknown(where, body, ServerConfig, skip={'label'})

    kind = body.get('kind', ServerKind.SLURM)
    if kind not in ServerKind:
        raise ConfigError(
            f'{where}: kind must be one of {[k.value for k in ServerKind]}, '
            f'not {kind!r}'
        )

    raw_environments = body.get('environments', {})
    if not isinstance(raw_environments, dict):
        raise ConfigError(f"{where}: 'environments' must be an object")
    environments = {
        name: _environment(where, name, env_body)
        for name, env_body in raw_environments.items()
    }

    wanted = body.get('default_environment')
    if wanted and wanted not in environments:
        raise ConfigError(
            f"{where}: default_environment '{wanted}' is not among the "
            f'environments defined here ({sorted(environments) or "none"})'
        )

    return ServerConfig(
        label=label,
        kind=kind,
        host=body.get('host', label),
        user=body.get('user'),
        workdir=body.get('workdir', '~/.tether'),
        default_environment=wanted,
        timeout=_timeout(where, body.get('timeout')),
        environments=environments,
    )


def _environment(where: str, label: str, body: dict) -> EnvironmentConfig:
    where = f'{where}: environment {label!r}'
    if not isinstance(body, dict):
        raise ConfigError(f'{where}: must be an object')
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
            f"{where}: 'conda_base' is only meaningful for kind 'conda', "
            f'not {kind!r}'
        )

    env = body.get('env', {})
    if not isinstance(env, dict):
        raise ConfigError(f"{where}: 'env' must be an object of strings")

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


def _timeout(where: str, raw: object) -> float | None:
    """`timeout` may be a number of seconds, or absent."""
    if raw is None:
        return None
    if not isinstance(raw, (int, float)) or isinstance(raw, bool) or raw <= 0:
        raise ConfigError(
            f'{where}: timeout must be a positive number of seconds, not {raw!r}'
        )
    return float(raw)


def _reject_unknown(where: str, body: dict, cls: type, skip: set[str]) -> None:
    """Typos in a config file should be loud, not silently ignored."""
    known = {f for f in cls.__dataclass_fields__ if f not in skip}
    unknown = set(body) - known
    if unknown:
        raise ConfigError(
            f'{where}: unknown key(s) {sorted(unknown)}; '
            f'expected any of {sorted(known)}'
        )


# -- writing ---------------------------------------------------------------


def save_server(
    cfg: ServerConfig,
    *,
    config_dir: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Write `cfg` to `~/.tether/servers/<label>.json`; return the path."""
    # Deferred: `__init__` imports this module, so the version cannot be
    # imported at module scope without a cycle.
    from . import __version__

    path = server_path(cfg.label, config_dir)
    if path.exists() and not overwrite:
        raise ConfigError(
            f'{path} already exists; pass overwrite=True to replace it'
        )

    body = {VERSION_KEY: __version__, **_body(cfg)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + '\n')
    return path


def delete_server(name: str, *, config_dir: str | Path | None = None) -> bool:
    """Remove a saved server. `False` if there was nothing to remove."""
    path = server_path(name, config_dir)
    if not path.exists():
        return False
    path.unlink()
    return True


def _body(cfg: ServerConfig | EnvironmentConfig) -> dict:
    """Dataclass to JSON body.

    `label` is the filename (or the map key), and anything left at its default
    is dropped, so a saved file shows what was actually chosen rather than a
    transcript of every default in force on the day it was written.
    """
    default = type(cfg)(label=cfg.label)
    body: dict = {}
    for f in dataclasses.fields(cfg):
        if f.name == 'label':
            continue
        value = getattr(cfg, f.name)
        if f.name == 'environments':
            if value:
                body[f.name] = {name: _body(env) for name, env in value.items()}
            continue
        if value is None or value == getattr(default, f.name):
            continue
        body[f.name] = list(value) if isinstance(value, tuple) else value
    return body
