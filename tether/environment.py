"""Environments: prepare a shell before the payload runs.

One class per kind. Each knows its own fields, its own validation, and its own
activation lines, so adding a kind is adding a corresponding class.

Activation order:

pre_activation_cmds
                  Verbatim lines first. `module` is normally a shell function
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
post_activation_cmds
                  Verbatim lines last: the final word, after activation.

Failures raise exceptions. `module load` and activation are emitted with an explicit
guard, because a `conda activate` that quietly fails would run the payload
against the wrong interpreter and produce a baffling error much later, far
from its cause.
"""

from __future__ import annotations

import shlex
from collections.abc import Iterable
from dataclasses import MISSING, dataclass, field, fields
from enum import StrEnum
from typing import ClassVar

from .errors import ConfigError
from .shell import run_or_abort, is_identifier, remote_path


class EnvironmentKind(StrEnum):
    CONDA = 'conda'
    VENV = 'venv'
    NONE = 'none'


@dataclass(frozen=True)
class Environment:
    """
    Base environment class. Instantiable, with kind NONE (bare metal). Typically
    subclassed for venv, conda, ..., with their own fields and activation lines.
    """

    name: str
    modules: tuple[str, ...] = ()
    pre_activation_cmds: tuple[str, ...] = ()
    post_activation_cmds: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    mpirun: str = 'mpirun'

    # placeholder for subclass's kind:
    kind: ClassVar[EnvironmentKind]

    def __post_init__(self) -> None:
        # JSON gives lists; the fields are tuples. Normalize so a constructed
        # environment compares equal to the same one loaded back from disk.
        for name in ('modules', 'pre_activation_cmds', 'post_activation_cmds'):
            object.__setattr__(self, name, tuple(getattr(self, name)))

    # overloadable methods:

    def activation(self) -> list[str]:
        """The lines that enter the environment. Bare metal enters nothing."""
        return []

    def activation_failure(self, reported: dict[str, str]) -> str | None:
        """Why activation did not take effect, or `None` if it did.

        Activation can exit 0 and accomplish nothing -- an `activate` script
        that is empty, a hook that declined quietly -- so exit status alone
        cannot be trusted. `reported` maps variable name to the value the
        remote shell had once activation was supposed to have happened, and
        each kind knows which of those is its own evidence.

        Bare metal has nothing to prove, so nothing can fail.
        """
        return None

    def pre_activation(self) -> list[str]:
        """Everything up to, but not including, activation.

        The split exists because two callers need the machinery made available
        without entering the environment: discovery, which runs before there is
        anything to enter, and creation, which is what brings it into being.
        Both would fail on the activation step -- and both need
        `pre_activation_cmds` and `modules`, since that is how conda or a newer
        python reaches PATH in the first place.

        Not to be confused with the field it starts from: `pre_activation_cmds`
        is the verbatim lines a user wrote, while this is *everything* that runs
        before activation, module loads and exports included.
        """
        return [
            *self.pre_activation_cmds,
            *(run_or_abort(
                f'module load {shlex.quote(m)}', f'module load {m} failed'
            ) for m in self.modules),
            *self.exports(),
        ]

    def post_activation(self) -> list[str]:
        """`post_activation_cmds`, verbatim.

        An identity wrapper, and deliberately so: it exists to make the three
        stages read alike in `preamble()`. Nothing transforms these lines --
        unlike modules and exports, they are emitted exactly as written.
        """
        return list(self.post_activation_cmds)

    def install_command(self, packages: Iterable[str]) -> str:
        """Command installing `packages` into this environment.

        pip for every kind, including conda's. `conda install` would be the
        idiomatic choice for a conda environment, but it can only offer what
        conda-forge carries -- and PHOEBE, the reason this library exists, is
        published on PyPI alone. One mechanism that always works beats two that
        each sometimes do.

        `python -m pip` rather than `pip`, so the interpreter the preamble
        activated is the one that installs. A stale `pip` shim earlier on PATH
        would otherwise install somewhere nobody asked for.

        Specifiers are quoted, which matters more than it looks: unquoted,
        `numpy>=1.20` is a redirection and creates a file called `=1.20`.
        """
        wanted = [p for p in packages if p]
        if not wanted:
            raise ConfigError('install() needs at least one package')

        return run_or_abort(
            'python -m pip install ' + ' '.join(shlex.quote(p) for p in wanted),
            f'could not install {", ".join(wanted)}',
        )

    def preamble(self) -> str:
        """Every slot, in order, as one shell script: the whole of what runs
        ahead of the payload."""
        return '\n'.join([
            *self.pre_activation(),
            *self.activation(),
            *self.post_activation(),
        ])

    def with_preamble(self, command: str) -> str:
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
        dollar sign. Use `pre_activation_cmds` or `post_activation_cmds` when a value has
        to be evaluated by the shell.
        """
        lines = []
        for key, value in self.env.items():
            if not is_identifier(key):
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

    def install_command(self, packages: Iterable[str]) -> str:
        """Refused. Bare metal has nothing isolated to install into.

        The alternatives are both wrong by default: installing into the system
        python needs root and changes what every other user gets, and `--user`
        quietly puts packages somewhere that leaks into every later job. Make
        a venv or a conda environment, or use `run()` and own the consequences.
        """
        raise ConfigError(
            f"environment '{self.name}' is bare metal, so there is nothing to "
            f'install into. Create a venv or a conda environment first'
        )


@dataclass(frozen=True)
class VenvEnvironment(Environment):
    """A virtualenv, activated by sourcing its `activate` script.

    `path` is exactly that -- a path -- and is never guessed from the name,
    because a guessed one comes out relative and fails somewhere confusing.
    """

    path: str | None = None

    kind: ClassVar[EnvironmentKind] = EnvironmentKind.VENV

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.path:
            raise ConfigError(
                f"environment '{self.name}': a venv needs 'path'"
            )

    def activation(self) -> list[str]:
        assert self.path is not None  # mypy cannot see the __post_init__ above
        script = f'{self.path.rstrip("/")}/bin/activate'
        return [
            run_or_abort(
                f'source {remote_path(script)}',
                f'could not activate venv {self.path}'
            )
        ]

    def activation_failure(self, reported: dict[str, str]) -> str | None:
        """`activate` sets `VIRTUAL_ENV`; an empty or truncated one does not."""
        return None if reported.get('VIRTUAL_ENV') else '$VIRTUAL_ENV is unset'


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
                run_or_abort(
                    f'source {remote_path(hook)}',
                    f'could not source conda hook at {hook}'
                )
            ]
        else:
            # The presence check is a separate line on purpose. `eval "$(conda
            # ...)"` reports the exit status of the *evaluated string*, so when
            # conda is missing the substitution fails, `eval ""` succeeds, and a
            # guard on the eval never fires -- the failure then surfaces a line
            # later blaming the environment rather than the missing conda.
            enable = [
                run_or_abort(
                    'command -v conda >/dev/null 2>&1',
                    'conda is not on PATH; set conda_base, load a module, '
                    'or source its hook in pre_activation_cmds'
                ),
                run_or_abort(
                    'eval "$(conda shell.bash hook)"',
                    'conda shell hook failed'
                ),
            ]

        assert self.conda_env is not None  # mypy cannot see the __post_init__ above
        return [
            *enable,
            run_or_abort(
                f'conda activate {shlex.quote(self.conda_env)}',
                f'could not activate conda environment {self.conda_env}'
            ),
        ]

    def activation_failure(self, reported: dict[str, str]) -> str | None:
        """`conda activate` sets `CONDA_PREFIX`; a hook that declined does not."""
        return None if reported.get('CONDA_PREFIX') else '$CONDA_PREFIX is unset'


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


def pre_activation(environment: Environment | None) -> str:
    """`pre_activation()` as a script, or nothing if unconfigured. The class keeps
    only the list form, since this is the one caller that wants it joined."""
    return '\n'.join(environment.pre_activation()) if environment else ''


def preamble(environment: Environment | None) -> str:
    return environment.preamble() if environment else ''


def with_preamble(environment: Environment | None, command: str) -> str:
    return environment.with_preamble(command) if environment else command


def install_command(environment: Environment | None, packages: Iterable[str]) -> str:
    """`install_command()`, or a refusal if no environment is configured."""
    if environment is None:
        raise ConfigError(
            'no environment is configured on this server, so there is nothing '
            'to install into'
        )
    return environment.install_command(packages)
