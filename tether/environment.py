"""Environments: prepare a shell before the payload runs.

One class per kind. Each knows its own fields, its own validation, and its own
activation lines, so adding a kind is adding a corresponding class.

Activation order:

================  ==============================================================
pre_activation    Verbatim lines first. `module` is normally a shell function
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
post_activation   Verbatim lines last: the final word, after activation.
================  ==============================================================

Failures raise exceptions. `module load` and activation are emitted with an explicit
guard, because a `conda activate` that quietly fails would run the payload
against the wrong interpreter and produce a baffling error much later, far
from its cause.

Tilde is not expanded in quoted paths; use $HOME instead. See `remote_path`.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import MISSING, dataclass, field, fields
from enum import StrEnum
from typing import ClassVar

from .errors import ConfigError


class EnvironmentKind(StrEnum):
    CONDA = 'conda'
    VENV = 'venv'
    NONE = 'none'


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


def guard(command: str, message: str) -> str:
    """Run `command`, and abort loudly rather than continue if it fails."""
    complaint = shlex.quote(f'tether: {message}')
    return f'{command} || {{ echo {complaint} >&2; exit 1; }}'


#: A shell variable name. Anything else could smuggle code into an `export`.
_IDENTIFIER = re.compile(r'[A-Za-z_][A-Za-z0-9_]*\Z')


@dataclass(frozen=True)
class Environment:
    """
    Base environment class. Instantiable, with kind NONE (bare metal). Typically
    subclassed for venv, conda, ..., with their own fields and activation lines.
    """

    name: str
    modules: tuple[str, ...] = ()
    pre_activation: tuple[str, ...] = ()
    post_activation: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    mpirun: str = 'mpirun'

    #: The discriminator, in config and on the class. A ClassVar rather than a
    #: field so a saved kind can never disagree with the class holding it.
    kind: ClassVar[EnvironmentKind] = EnvironmentKind.NONE

    #: The variable that proves activation actually happened. `None` for bare
    #: metal, where there is nothing to prove.
    marker: ClassVar[str | None] = None

    def __post_init__(self) -> None:
        # JSON gives lists; the fields are tuples. Normalize so a constructed
        # environment compares equal to the same one loaded back from disk.
        for name in ('modules', 'pre_activation', 'post_activation'):
            object.__setattr__(self, name, tuple(getattr(self, name)))

    # to be (optionally) overloaded by subclasses:

    def activation(self) -> list[str]:
        """The lines that enter the environment. Bare metal enters nothing."""
        return []

    def lines(self) -> list[str]:
        """Every line that prepares this environment, in order."""
        return [
            *self.pre_activation,
            *(guard(f'module load {shlex.quote(m)}', f'module load {m} failed')
              for m in self.modules),
            *self.exports(),
            *self.activation(),
            *self.post_activation,
        ]

    def preamble(self) -> str:
        """`lines()` as one shell script -- every slot, not just one of them."""
        return '\n'.join(self.lines())

    def wrap(self, command: str) -> str:
        """`command`, preceded by whatever prepares this environment."""
        preamble = self.preamble()
        return f'{preamble}\n{command}' if preamble else command

    def to_dict(self) -> dict:
        """JSON body. `name` is the map key, so it is not repeated, and any
        field left at its default is dropped -- a saved file should record what
        was chosen, not every default in force the day it was written.

        `kind` is a ClassVar rather than a field, but it is the discriminator
        that picks the class on the way back in, so it is written explicitly.
        """
        body: dict = {'kind': str(self.kind)}
        for f in fields(self):
            if f.name == 'name':
                continue
            value = getattr(self, f.name)
            default = (f.default_factory() if f.default_factory is not MISSING
                       else f.default)
            if value is None or value == default:
                continue
            body[f.name] = list(value) if isinstance(value, tuple) else value
        return body

    def exports(self) -> list[str]:
        """`export` lines. Values are quoted, so they are literal by design.

        A value cannot reference another variable -- `export B='$A'` stays a
        dollar sign. Use `pre_activation` or `post_activation` when a value has
        to be evaluated by the shell.
        """
        lines = []
        for key, value in self.env.items():
            if not _IDENTIFIER.match(key):
                raise ConfigError(
                    f'not a usable shell variable name: {key!r} '
                    f'(expected letters, digits and underscore, not starting '
                    f'with a digit)'
                )
            lines.append(f'export {key}={shlex.quote(value)}')
        return lines


@dataclass(frozen=True)
class SystemEnvironment(Environment):
    """Bare metal: whatever the login shell already provides."""

    kind: ClassVar[EnvironmentKind] = EnvironmentKind.NONE


@dataclass(frozen=True)
class VenvEnvironment(Environment):
    """A virtualenv, activated by sourcing its `activate` script.

    `path` is exactly that -- a path -- and is never guessed from the name,
    because a guessed one comes out relative and fails somewhere confusing.
    """

    path: str | None = None

    kind: ClassVar[EnvironmentKind] = EnvironmentKind.VENV
    marker: ClassVar[str | None] = 'VIRTUAL_ENV'

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.path:
            raise ConfigError(
                f"environment '{self.name}': a venv needs 'path'"
            )

    def activation(self) -> list[str]:
        script = f'{self.path.rstrip("/")}/bin/activate'
        return [
            guard(f'source {remote_path(script)}',
                  f'could not activate venv {self.path}')
        ]


@dataclass(frozen=True)
class CondaEnvironment(Environment):
    """A conda environment, entered through conda's shell hook.

    `conda_env` defaults to the environment's own name, since for conda the two
    are usually the same word. `conda_base` points at an existing installation;
    without it, conda has to already be on PATH.
    """

    conda_env: str | None = None
    conda_base: str | None = None

    kind: ClassVar[EnvironmentKind] = EnvironmentKind.CONDA
    marker: ClassVar[str | None] = 'CONDA_PREFIX'

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.conda_env:
            object.__setattr__(self, 'conda_env', self.name)

    def activation(self) -> list[str]:
        # `conda activate` is a shell function, so its hook has to be in scope
        # first -- plain `conda` on PATH is not enough.
        if self.conda_base:
            hook = f'{self.conda_base.rstrip("/")}/etc/profile.d/conda.sh'
            enable = [
                guard(f'source {remote_path(hook)}',
                      f'could not source conda hook at {hook}')
            ]
        else:
            # The presence check is a separate line on purpose. `eval "$(conda
            # ...)"` reports the exit status of the *evaluated string*, so when
            # conda is missing the substitution fails, `eval ""` succeeds, and a
            # guard on the eval never fires -- the failure then surfaces a line
            # later blaming the environment rather than the missing conda.
            enable = [
                guard('command -v conda >/dev/null 2>&1',
                      'conda is not on PATH; set conda_base, load a module, '
                      'or source its hook in pre_activation'),
                guard('eval "$(conda shell.bash hook)"', 'conda shell hook failed'),
            ]

        return [
            *enable,
            guard(f'conda activate {shlex.quote(self.conda_env)}',
                  f'could not activate conda environment {self.conda_env}'),
        ]


#: kind -> class. Adding a kind is one line here plus the class itself.
ENVIRONMENTS: dict[EnvironmentKind, type[Environment]] = {
    EnvironmentKind.NONE: SystemEnvironment,
    EnvironmentKind.VENV: VenvEnvironment,
    EnvironmentKind.CONDA: CondaEnvironment,
}


def env(name: str, kind: EnvironmentKind | str = EnvironmentKind.NONE,
        **fields_: object) -> Environment:
    """
    Environment factory. It builds the right `Environment` subclass for `kind`.
    See `tether.server()` for the analogous implementation on the server side.
    """
    try:
        cls = ENVIRONMENTS[EnvironmentKind(kind)]
    except ValueError:
        raise ConfigError(
            f"environment '{name}': unsupported kind {kind!r}; expected one "
            f'of {[k.value for k in EnvironmentKind]}'
        ) from None

    allowed = {f.name for f in fields(cls)}
    unknown = set(fields_) - allowed
    if unknown:
        raise ConfigError(
            f"environment '{name}': {sorted(unknown)} "
            f"{'is' if len(unknown) == 1 else 'are'} not valid for kind "
            f"'{cls.kind}'; expected any of {sorted(allowed - {'name'})}"
        )
    return cls(name=name, **fields_)   # type: ignore[arg-type]


# -- None-tolerant helpers -------------------------------------------------
#
# A server may have no environment at all, which is not the same as bare metal:
# there is nothing to prepare. These keep that case out of every caller.


def create_preamble(environment: Environment | None) -> str:
    return environment.preamble() if environment else ''


def wrap(environment: Environment | None, command: str) -> str:
    return environment.wrap(command) if environment else command
