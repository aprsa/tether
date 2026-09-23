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
they do not generalize: `modules`, `pre_activation_cmds` and `conda_base` each encode
one cluster's assumptions, and only `kind` and `name` travel. A shared
definition could not say which cluster it was written against, and nesting
makes it impossible to point a server at an environment meant for another.

JSON rather than TOML because tether writes these files as well as reading
them, and the standard library can only read TOML. Configuration stays
optional: with no file at all, a server name falls through to ``~/.ssh/config``.
"""

from __future__ import annotations

import json
import re
from enum import StrEnum
from pathlib import Path

from .environment import Environment, EnvironmentKind, env
from .errors import ConfigError

DEFAULT_CONFIG_DIR = Path.home() / '.tether'
SERVERS_DIRNAME = 'servers'
DEFAULT_WORKDIR = '~/.tether'
# The remote scratch root, unless a server says otherwise.
VERSION_KEY = 'tether'
# Stamped into every file tether writes, so a later version can recognise an
# older layout instead of guessing at it.

#: A server name becomes a filename, so it must not contain separators or
#: traverse. Leading character is alphanumeric, which also rules out `.`/`..`.


class ServerKind(StrEnum):
    SLURM = 'slurm'
    PLAIN = 'plain'


# -- locations -------------------------------------------------------------


def servers_dir(config_dir: str | Path | None = None) -> Path:
    base = Path(config_dir).expanduser() if config_dir else DEFAULT_CONFIG_DIR
    return base / SERVERS_DIRNAME


def server_path(name: str, config_dir: str | Path | None = None) -> Path:
    """Where `name` is stored. Validates the name, since it is a filename."""
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', name):
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


def _load_server(name: str, config_dir: str | Path | None = None) -> dict | None:
    """The validated body of `<name>.json`, or `None` if there is no such file.

    Private: the body is a raw dict whose `environments` values are live
    objects, which is a shape only `Server` should have to know. Callers want
    `tether.server(name)`.

    A dict rather than an object: `Server` builds itself from this, and a
    function here returning a `Server` would have to construct one to read one.

    `None` is not an error -- an unconfigured name is treated as a hostname or
    an `ssh_config` alias.
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
    where = str(path)

    known = {'kind', 'host', 'user', 'workdir', 'default_environment',
             'timeout', 'environments'}
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(
            f'{where}: unknown key(s) {sorted(unknown)}; '
            f'expected any of {sorted(known)}'
        )

    kind = raw.get('kind', ServerKind.SLURM)
    if kind not in ServerKind:
        raise ConfigError(
            f'{where}: kind must be one of {[k.value for k in ServerKind]}, '
            f'not {kind!r}'
        )
    raw['kind'] = ServerKind(kind)
    raw['timeout'] = _timeout(where, raw.get('timeout'))

    raw_environments = raw.get('environments', {})
    if not isinstance(raw_environments, dict):
        raise ConfigError(f"{where}: 'environments' must be an object")
    raw['environments'] = {
        label: _environment(where, label, body)
        for label, body in raw_environments.items()
    }

    wanted = raw.get('default_environment')
    if wanted and wanted not in raw['environments']:
        raise ConfigError(
            f"{where}: default_environment '{wanted}' is not among the "
            f'environments defined here ({sorted(raw["environments"]) or "none"})'
        )
    return raw


def _save_server(
    name: str,
    body: dict,
    *,
    config_dir: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Write `body` to `<name>.json`; return the path.

    Private because it writes whatever it is handed. `Server.save()` builds the
    body from a validated object; a public writer taking a raw dict would let
    you produce a file that `_load_server` then refuses -- the round-trip hole
    the environment classes exist to close.
    """
    # Deferred: `__init__` imports this module, so the version cannot be
    # imported at module scope without a cycle.
    from . import __version__

    path = server_path(name, config_dir)
    if path.exists() and not overwrite:
        raise ConfigError(
            f'{path} already exists; pass overwrite=True to replace it'
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({VERSION_KEY: __version__, **body}, indent=2, sort_keys=True)
        + '\n'
    )
    return path


def _environment(where: str, label: str, body: dict) -> Environment:
    """Build one environment from its JSON body.

    Validation lives in the classes, so this only has to pick the right one and
    hand over the rest. `kind` is the discriminator and is consumed here, which
    is why it is popped before the unknown-key check.
    """
    if not isinstance(body, dict):
        raise ConfigError(f'{where}: environment {label!r} must be an object')

    body = dict(body)
    kind = body.pop('kind', EnvironmentKind.NONE)
    try:
        return env(label, kind, **body)
    except ConfigError as exc:
        raise ConfigError(f'{where}: {exc}') from None
    except TypeError as exc:                       # wrong type for a field
        raise ConfigError(f'{where}: environment {label!r}: {exc}') from None


def _timeout(where: str, raw: object) -> float | None:
    """`timeout` may be a number of seconds, or absent."""
    if raw is None:
        return None
    if not isinstance(raw, (int, float)) or isinstance(raw, bool) or raw <= 0:
        raise ConfigError(
            f'{where}: timeout must be a positive number of seconds, not {raw!r}'
        )
    return float(raw)


def delete_server(name: str, *, config_dir: str | Path | None = None) -> bool:
    """Remove a saved server. `False` if there was nothing to remove."""
    path = server_path(name, config_dir)
    if not path.exists():
        return False
    path.unlink()
    return True
