"""Virtual environments, and the interpreters they are built from.

A venv is `python -m venv` and nothing more, so this module is about the two
halves of that: finding an interpreter worth pointing it at, and running it.

Conda interpreters are deliberately not offered. Building a venv on top of a
conda python is legal and occasionally done, but it produces an environment
whose base is managed by one tool and whose packages are managed by another --
and tether already has a first-class way to get a specific Python version from
conda. An interpreter is asked about its own provenance rather than guessed at
from its path: `sys.prefix` holding a `conda-meta` directory is conda's own
definition of a prefix, so the interpreter is the authority.

As in `conda.py`, the shell only gathers. Which paths are interpreters, which
are the same file twice, and which belong to conda are all decided in Python,
where they are testable without a machine that has any.
"""

from __future__ import annotations

import posixpath
import re
import shlex
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from .shell import printf, remote_path

#: Names an interpreter is plausibly reachable by: `python`, `python3`,
#: `python3.12`, and the free-threaded `python3.13t`. Deliberately not
#: `python3-config`, `python-dotenv` or anything else a package drops into the
#: same directory -- those would be *executed* by the identity probe otherwise.
#: What a version is when nothing would say.
UNKNOWN = 'unknown'

_INTERPRETER = re.compile(r'/python[0-9]*(?:\.[0-9]+)?t?\Z')

#: What each candidate is asked about itself. `sys.executable` because the
#: interpreter knows where it lives, `sys.version` because a filename can lie
#: about its version, and the last two because provenance is not guessable
#: from a path.
_IDENTITY = (
    'import os, sys; '
    'print(sys.executable, sys.version.split()[0], '
    "os.path.isdir(os.path.join(sys.prefix, 'conda-meta')), "
    "sys.prefix != sys.base_prefix, sep='\\t')"
)

#: Every executable `python*` on PATH, as `<name you could type>\t<what it is>`.
#:
#: Both halves are reported because they answer different questions: the typed
#: name says whether this looks like an interpreter, and the resolved one says
#: whether two names are the same file. Filtering on the resolved name instead
#: would be wrong -- `readlink -f` can rename entirely, as `/usr/bin/python3-config`
#: resolving to `x86_64-linux-gnu-python3.8-config` shows.
#:
#: The trailing `|| true` is not decoration: a `for` loop reports its last
#: iteration's status, so a final PATH entry with no `python*` would fail the
#: whole command.
_CANDIDATES = r"""for dir in $(printf '%s' "$PATH" | tr : ' '); do
    for candidate in "$dir"/python*; do
        [ -x "$candidate" ] && [ ! -d "$candidate" ] &&
            printf '%s\t%s\n' "$candidate" "$(readlink -f "$candidate")"
    done
done 2>/dev/null || true"""


@dataclass(frozen=True)
class PythonInstallation:
    """Functional bare-metal, in-PATH python interpreter.

    `path` is resolved, so symbolic links resolve to the same file, which is
    then listed only once. This is what `python -m venv` has to be invoked
    as, which is why discovery returns interpreters rather than version
    strings: a version alone would need a second lookup, and that lookup can
    disagree with this one.
    """

    path: str
    version: str

    def __str__(self) -> str:
        return f'{self.path} (python {self.version})'


def probe_interpreters(
    run: Callable[[str], str],
    *,
    setup: str = '',
) -> list[PythonInstallation]:
    """Every bare-metal interpreter on PATH.

    `run` takes a command and returns its stdout
    `setup` is what runs before the probe; for example, `module load python/3.12`

    Conventional directories are not searched because no such list can be
    exhaustive. An interpreter somewhere else is reached by putting it on
    PATH in `pre_activation_cmds`.

    Two round trips, and the second is skipped when the first found nothing.
    """
    paths = _parse_candidates(run('\n'.join(filter(None, (setup, _CANDIDATES)))))
    if not paths:
        return []

    return _parse_identity(run(_identity_query(paths)))


def _parse_candidates(stdout: str) -> tuple[str, ...]:
    """Resolved paths worth asking, in the order PATH offers them.

    Judged by the typed name, deduplicated by what it resolves to. Anything
    that does not look like an interpreter is dropped here rather than later,
    because the next step *runs* what survives, and running `python-dotenv` to
    see what it says is not a thing to do casually.
    """
    seen: dict[str, None] = {}

    for line in stdout.splitlines():
        parts = line.split('\t')
        if len(parts) != 2:
            continue
        typed, resolved = parts
        if resolved and _INTERPRETER.search(typed):
            seen.setdefault(resolved, None)

    return tuple(seen)


def _identity_query(paths: Iterable[str]) -> str:
    """Command asking each interpreter to describe itself.

    One line per path rather than a shell loop: the repetition is generated in
    Python, so the shell does nothing but execute. An entry that is not a
    working interpreter prints nothing and is skipped, which is the same
    contract every other parser here keeps.
    """
    # `|| true` per line, not decoration: a candidate that is not a working
    # interpreter exits non-zero, and the last command's status is the whole
    # script's -- so one dud at the end would fail a probe that succeeded.
    return '\n'.join(
        f'{remote_path(path)} -c {shlex.quote(_IDENTITY)} 2>/dev/null || true'
        for path in paths
    )


def _parse_identity(stdout: str) -> list[PythonInstallation]:
    """Bare-metal interpreters, dropping anything conda or venv owns.

    A conda interpreter is excluded because conda environments are conda's job;
    an interpreter already inside a venv is excluded because a venv built from
    a venv inherits a base nobody asked for.
    """
    found = []

    for line in stdout.splitlines():
        fields = line.split('\t')
        if len(fields) != 4:
            continue
        path, version, in_conda, in_venv = fields
        # Tab-separated and validated, because a four-word login banner splits
        # into exactly four fields, and a path with a space in it does not.
        if not path.startswith('/') or {in_conda, in_venv} - {'True', 'False'}:
            continue
        if in_conda == 'True' or in_venv == 'True':
            continue
        found.append(PythonInstallation(path=path, version=version))

    return found


#: A venv's own record of itself. Used only as a cheap marker that a directory
#: is a venv, and -- for one that will not run -- as the only remaining source
#: of the version it was built against.
#:
#: Its fields vary by Python version: `executable` and `command` arrived in
#: 3.11, so terra's 3.8-era venvs record only `home`, `version` and
#: `include-system-site-packages`. Anything read from here must tolerate that,
#: which is the reason a venv is judged by running it rather than by parsing
#: this.
_PYVENV = 'pyvenv.cfg'

#: What a venv is asked about itself. `sys.prefix` rather than `sys.executable`
#: because the prefix *is* the venv, and `sys.base_prefix` because the
#: interpreter behind it is what breaks when a system is upgraded or a pyenv
#: version is removed.
_VENV_IDENTITY = (
    "import sys; print(sys.prefix, sys.version.split()[0], sys.base_prefix, sep='\\t')"
)


@dataclass(frozen=True)
class VenvInstallation:
    """A virtual environment, and whether it still works.

    `base` is the interpreter behind it. A venv is a thin shell over another
    Python, so when that Python goes away -- a system upgrade, a removed pyenv
    version -- the venv stays on disk and stops working. `base` is what names
    the culprit.

    `broken` means its own `bin/python` would not run. Such a venv still
    reports a `version`, read from `pyvenv.cfg` rather than from the
    interpreter that refused, so the answer stays actionable: this wanted
    3.9.2 and cannot have it.
    """

    path: str
    version: str
    base: str
    broken: bool = False

    def __str__(self) -> str:
        state = ' (broken)' if self.broken else ''
        return f'{self.path} (python {self.version}){state}'


def probe_venvs(
    run: Callable[[str], str],
    venvs_base: str | None = None,
    *,
    setup: str = '',
    include_broken: bool = False,
) -> list[VenvInstallation]:
    """The virtual environments tether can see, and whether they still run.

    Two sources, because venv keeps no registry. That is the crucial
    difference from conda, which records every prefix it touches in
    `~/.conda/environments.txt` and so can be asked. `python -m venv` creates
    a directory and forgets, leaving nothing to consult:

    - `$VIRTUAL_ENV`, when one is active;
    - `venvs_base`, which may be a venv itself or a directory of them -- both
      are matched, so pointing at `~/.venvs` or at `~/.venvs/phoebe` works.

    The filesystem is deliberately not searched. On terra, `find $HOME` costs
    57ms at depth 1, 1.5s at depth 2 and **22 seconds** at depth 3 -- which is
    where the venvs actually are, on a NAS shared by everyone. A probe cannot
    cost that, and a shallower one would silently miss things, so tether asks
    to be told where to look. `venvs_base` is to venvs what `conda_base` is to
    conda: the answer to "I keep mine somewhere you would never guess".

    Each candidate is judged by running its own `bin/python`, never by reading
    `pyvenv.cfg`. That file's fields differ across Python versions, and a venv
    whose base interpreter has vanished still has a perfectly well-formed one.
    Only the interpreter knows whether it can start.

    `include_broken` adds the ones that would not run, at the cost of one more
    round trip to read the versions they were built against.
    """
    candidates = _parse_venvs(run(_venvs_query(venvs_base, setup=setup)))
    if not candidates:
        return []

    working = _parse_inspect(run(_inspect_query(candidates)))
    if not include_broken:
        return working

    broken = [path for path in candidates if path not in {v.path for v in working}]
    if not broken:
        return working

    config = _parse_config(run(_config_query(broken)))
    found = list(working)
    found += [
        VenvInstallation(path=path,
                         version=config.get(path, {}).get('version', UNKNOWN),
                         base=config.get(path, {}).get('home', UNKNOWN),
                         broken=True)
        for path in broken
    ]
    return found


def _venvs_query(venvs_base: str | None = None, *, setup: str = '') -> str:
    """Command printing directories that might be virtual environments.

    The glob matches `venvs_base` itself and its children, so the caller need
    not know whether they are naming one venv or a shelf of them. `ls` exits
    non-zero when a glob matches nothing, which here is ordinary.
    """
    lines = [printf('"${VIRTUAL_ENV:-}"')]
    if venvs_base:
        at = remote_path(venvs_base)
        lines.append(f'ls -d {at}/{_PYVENV} {at}/*/{_PYVENV} 2>/dev/null || true')

    return '\n'.join(filter(None, (setup, *lines)))


def _parse_venvs(stdout: str) -> tuple[str, ...]:
    """Candidate venv directories, in the order the sources offered them.

    Two shapes arrive: `$VIRTUAL_ENV` gives a directory, the glob gives the
    `pyvenv.cfg` inside one. Both reduce to the directory.
    """
    seen: dict[str, None] = {}

    for line in stdout.splitlines():
        path = line.strip().rstrip('/')
        if posixpath.basename(path) == _PYVENV:
            path = posixpath.dirname(path)
        if path:
            seen.setdefault(path, None)

    return tuple(seen)


def _inspect_query(paths: Iterable[str]) -> str:
    """Command asking each candidate's own python to describe the venv."""
    # A broken venv's `bin/python` dangles and exits 127. That is the answer,
    # not a failure, so it must not take the whole script down with it.
    return '\n'.join(
        f'{remote_path(posixpath.join(path, "bin", "python"))} '
        f'-c {shlex.quote(_VENV_IDENTITY)} 2>/dev/null || true'
        for path in paths
    )


def _parse_inspect(stdout: str) -> list[VenvInstallation]:
    """Venvs that ran. Anything that stayed silent could not start.

    A candidate whose prefix equals its base prefix is a Python installation
    rather than a venv -- someone pointed `venvs_base` at one -- and is not
    reported as a venv it is not.
    """
    found = []

    for line in stdout.splitlines():
        fields = line.split('\t')
        if len(fields) != 3:
            continue
        prefix, version, base = fields
        if not prefix.startswith('/') or prefix == base:
            continue
        found.append(VenvInstallation(path=prefix, version=version, base=base))

    return found


def _config_query(paths: Iterable[str]) -> str:
    """Command dumping each venv's `pyvenv.cfg`, tagged with whose it is.

    The whole file rather than one grepped field, so the parsing happens where
    the format differences between Python versions can be handled in the open.
    """
    return '\n'.join(
        f'{printf(remote_path(path))}\n'
        f'cat {remote_path(posixpath.join(path, _PYVENV))} 2>/dev/null || true'
        for path in paths
    )


def _parse_config(stdout: str) -> dict[str, dict[str, str]]:
    """`{venv path: {key: value}}` from the dumped config files.

    A bare line starting a block is a path; `key = value` lines belong to it.
    Unknown keys are kept rather than filtered, because which keys exist
    depends on the Python that wrote the file.
    """
    blocks: dict[str, dict[str, str]] = {}
    current: dict[str, str] | None = None

    for line in stdout.splitlines():
        if '=' in line and current is not None:
            key, _, value = line.partition('=')
            current[key.strip()] = value.strip()
        elif line.strip().startswith('/'):
            current = blocks.setdefault(line.strip(), {})

    return blocks
