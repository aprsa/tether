"""Environment activation.

Turns an `EnvironmentConfig` into the shell lines that prepare a shell before
the payload runs.

Note that activation order is fixed:

================  ==============================================================
pre_activation    Verbatim lines, first. `module` is normally a shell function
                  sourced from /etc/profile.d, and conda's `activate` needs its
                  hook sourced, so a non-interactive shell frequently cannot run
                  either until something makes them available. This is that
                  slot.
modules           `module load` each entry, in order.
env               `export` each variable before activation so that variables
                  which configure activation (`CONDA_ENVS_PATH`,
                  `PYTHONNOUSERSITE`) take effect.
activation        conda or venv. Last of the machinery, so the PATH it prepends
                  wins over everything above it.
post_activation   Verbatim lines, last: the final word, after activation.
================  ==============================================================

Notes:

1. Failures are loud. `module load` and activation are emitted with an
   explicit guard, because a `conda activate` that quietly fails would run the
   payload against the wrong interpreter and produce a baffling error much
   later, far from its cause.

2. Tilde is not expanded in quoted paths; use $HOME instead.
"""

from __future__ import annotations

import re
import shlex

from .config import EnvironmentConfig, EnvironmentKind
from .errors import ConfigError

PROBE = (
    'printf \'%s\\n%s\\n\' "${VIRTUAL_ENV-}" "${CONDA_PREFIX-}"\n'
    'command -v python3 >/dev/null 2>&1 && python3 -c '
    "'import sys; print(sys.executable); print(sys.version.split()[0]); "
    "print(sys.prefix)'"
)
"""Reports the activated environment: VIRTUAL_ENV, CONDA_PREFIX, then -- if a
python3 exists at all -- its executable, version and prefix. Bare metal without
python is a legitimate outcome, so the interpreter is probed, not assumed."""


def remote_path(path: str) -> str:
    """Quote a path for the remote shell, preserving `~` expansion.

    `shlex.quote('~/env')` returns `'~/env'`, and those quotes stop the shell
    from expanding the tilde, so the path silently resolves to a literal `~`
    directory that does not exist. A leading `~` therefore becomes `$HOME`
    inside double quotes, which still protects spaces.

    `~user` is left to `shlex.quote`: it has no `$HOME` equivalent, so it is
    better to quote it and fail loudly than to expand it wrongly.
    """

    #: Characters that keep their special meaning inside double quotes.
    _IN_DQUOTES = re.compile(r'([\\"$`])')

    if path == '~':
        return '"$HOME"'
    if path.startswith('~/'):
        return f'"$HOME/{_IN_DQUOTES.sub(r"\\\1", path[2:])}"'
    return shlex.quote(path)


def _guard(command: str, message: str) -> str:
    """Run `command`, and abort loudly rather than continue if it fails."""
    complaint = shlex.quote(f'tether: {message}')
    return f'{command} || {{ echo {complaint} >&2; exit 1; }}'


def _exports(env: dict[str, str]) -> list[str]:
    """`export` lines. Values are quoted, so they are literal by design.

    A value cannot reference another variable -- `export B='$A'` stays a dollar
    sign. Use `pre_activation` or `post_activation` when a value has to be
    evaluated by the shell.
    """

    #: A shell variable name. Anything else could smuggle code into an `export`.
    _IDENTIFIER = re.compile(r'[A-Za-z_][A-Za-z0-9_]*\Z')

    lines = []
    for key, value in env.items():
        if not _IDENTIFIER.match(key):
            raise ConfigError(
                f'not a usable shell variable name: {key!r} '
                f'(expected letters, digits and underscore, not starting '
                f'with a digit)'
            )
        lines.append(f'export {key}={shlex.quote(value)}')
    return lines


def _activation(cfg: EnvironmentConfig) -> list[str]:
    """The conda or venv activation lines. Empty for bare metal."""
    if cfg.kind == EnvironmentKind.NONE:
        return []

    # Kind before name, so an unrecognised kind is not misreported as a
    # missing name.
    if cfg.kind not in (EnvironmentKind.VENV, EnvironmentKind.CONDA):
        raise ConfigError(
            f"environment '{cfg.label}': unsupported kind {cfg.kind!r}"
        )

    if not cfg.name:
        raise ConfigError(
            f"environment '{cfg.label}': kind '{cfg.kind}' requires 'name'"
        )

    if cfg.kind == EnvironmentKind.VENV:
        script = f'{cfg.name.rstrip("/")}/bin/activate'
        return [
            _guard(
                f'source {remote_path(script)}',
                f'could not activate venv {cfg.name}',
            )
        ]

    # Only conda is left. `conda activate` is a shell function, so its hook has
    # to be in scope first -- plain `conda` on PATH is not enough.
    if cfg.conda_base:
        hook = f'{cfg.conda_base.rstrip("/")}/etc/profile.d/conda.sh'
        enable = [
            _guard(
                f'source {remote_path(hook)}',
                f'could not source conda hook at {hook}',
            )
        ]
    else:
        # The presence check is a separate line on purpose. `eval "$(conda
        # ...)"` reports the exit status of the *evaluated string*, so when
        # conda is missing the substitution fails, `eval ""` succeeds, and a
        # guard on the eval never fires -- the failure then surfaces a line
        # later blaming the environment rather than the missing conda.
        enable = [
            _guard(
                'command -v conda >/dev/null 2>&1',
                'conda is not on PATH; set conda_base, load a module, '
                'or source its hook in pre_activation',
            ),
            _guard('eval "$(conda shell.bash hook)"', 'conda shell hook failed'),
        ]

    return [
        *enable,
        _guard(
            f'conda activate {shlex.quote(cfg.name)}',
            f'could not activate conda environment {cfg.name}',
        ),
    ]


def create_preamble(cfg: EnvironmentConfig | None) -> str:
    """
    Shell script that needs to be executed before running a command in the environment.
    """

    if cfg is None:
        return ''
    lines = [
        *cfg.pre_activation,
        *(_guard(f'module load {shlex.quote(m)}', f'module load {m} failed') for m in cfg.modules),
        *_exports(cfg.env),
        *_activation(cfg),
        *cfg.post_activation,
    ]

    return '\n'.join(lines)


def wrap(cfg: EnvironmentConfig | None, command: str) -> str:
    """
    Wrap a payload command with the environment preamble, if any.
    """

    preamble = create_preamble(cfg)
    return f'{preamble}\n{command}' if preamble else command
