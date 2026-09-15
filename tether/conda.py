"""Finding conda on a remote account.

Conda's location is genuinely unguessable -- terra has two installations,
neither on PATH -- so this asks, rather than assumes. Three sources, each
authoritative about something the others miss, and no search of conventional
directories: such a list can never be exhaustive, so "found nothing" would not
mean "there is no conda", and a wrong answer that looks thorough is worse than
no answer.

The shell built here only gathers. Every decision -- what a path means, whether
a directory is an installation, which environments belong to it, whether the
expensive lookup is worth making -- happens in Python, the same division
`slurm.py` keeps between a command and its parser. That is what makes the
awkward cases testable: a half-installed conda, an environment carrying its own
`conda-meta`, a login banner in the output, a base that will not name its
version. None of those need a machine that has one.

`probe()` takes the transport as an argument rather than reaching for a
connection, so the whole algorithm -- round trips included -- is exercisable
with a dictionary.
"""

from __future__ import annotations

import posixpath
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace

from .errors import RemoteCommandError
from .shell import printf, remote_path

#: What a version is when the installation would not say.
UNKNOWN = 'unknown'

#: Where to look. `$CONDA_EXE` is exported once conda's hook has been sourced;
#: `type -P` searches PATH regardless of the shell *function* that hook defines
#: -- `command -v conda` would answer `conda`, not a path; and conda's own
#: `environments.txt` records installations that no directory search would find,
#: because they are not conventionally named.
_SOURCES = (
    printf('"${CONDA_EXE:-}"'),
    'type -P conda 2>/dev/null',
    'cat "$HOME/.conda/environments.txt" 2>/dev/null',
)

#: The activation hook. Its presence is what separates a base from an
#: environment, both of which otherwise look alike.
_HOOK = '/etc/profile.d/conda.sh'

#: Conda records its own version in a `conda-meta` filename, so a directory
#: listing answers the question that `conda --version` would spend a Python
#: interpreter launch on.
_META = re.compile(r'(?P<base>.*)/conda-meta/conda-(?P<version>[0-9][^-]*)-[^/]*\.json')

#: An environment directory, under a base or under the per-user fallback. The
#: name may not contain a separator, which is what keeps a path like
#: `<base>/envs/<name>/conda-meta/conda-*.json` from being read as one.
_ENV = re.compile(r'(?P<parent>.*)/envs/(?P<name>[^/]+)')


@dataclass(frozen=True)
class CondaInstallation:
    """A conda installation found on the remote, and what it can activate.

    `environments` are names, not paths, because that is how tether addresses
    them -- and the list includes conda's per-user fallback directory, since a
    named environment created against a site install lands there rather than
    under the base.
    """

    base: str
    version: str
    environments: tuple[str, ...] = ()

    def __str__(self) -> str:
        envs = ', '.join(self.environments) or 'no environments'
        return f'{self.base} (conda {self.version}): {envs}'


def probe(
    run: Callable[[str], str],
    workdir_conda: str | None = None,
    *,
    setup: str = '',
) -> list[CondaInstallation]:
    """Every conda installation `run` can reach, and its environments.

    `run` takes a command and returns its stdout, raising `RemoteCommandError`
    if it fails -- `Server.probe_conda()` passes its own. `workdir_conda` is
    where tether installs its own.

    `setup` is shell that runs before the lookups -- typically an
    `Environment`'s `pre_activation()`, which is what makes a conda visible when
    it only appears once `module load anaconda` has run. That is the whole
    reason the `pre_activation_cmds` and `modules` slots exist.

    Three round trips at worst, and the later two are skipped when they have
    nothing to do: gather candidates; list what they contain; and, only for a
    base that would not name its version in `conda-meta`, ask that conda
    directly -- which costs an interpreter launch, roughly a second apiece.

    Returns installations in the order found. Empty means there is no conda
    here to point `conda_base` at; one that is both off PATH and unrecorded is
    what `conda_base` and `pre_activation_cmds` are for.
    """
    bases = _parse_search(run(_search(workdir_conda, setup=setup)))
    if not bases:
        return []

    found = _parse_listing(run(_listing(bases)))

    for index, installation in enumerate(found):
        if installation.version != UNKNOWN:
            continue
        try:
            version = _parse_version(run(_version_query(installation.base)))
        except RemoteCommandError:
            continue        # already known to be odd; unknown is the answer
        found[index] = replace(installation, version=version)

    return found


def _search(workdir_conda: str | None = None, *, setup: str = '') -> str:
    """Shell that prints paths which might belong to a conda installation.

    This only gathers; it decides nothing. What the paths mean is
    `_parse_search()`'s problem, and whether they exist is `_listing()`'s.
    """
    sources = [*_SOURCES]
    if workdir_conda:
        sources.append(printf(remote_path(workdir_conda)))

    return '\n'.join(filter(None, (setup, *sources)))


def _parse_search(stdout: str) -> tuple[str, ...]:
    """Reduce candidate paths to the installation directories they imply.

    A lookup lands on `<base>/bin/conda` or `<base>/condabin/conda`, and
    `environments.txt` lists environments alongside bases. Both are mapped back
    to the directory that would hold the activation hook. Whether it actually
    does is settled by looking, not guessed here -- so this may well name
    directories that turn out to be neither.

    Order is preserved and duplicates dropped, which keeps the command built
    from the result stable, and therefore its output diffable.
    """
    bases: dict[str, None] = {}

    for line in stdout.splitlines():
        path = line.strip().rstrip('/')
        head, tail = posixpath.split(path)
        if tail == 'conda' and posixpath.basename(head) in ('bin', 'condabin'):
            path = posixpath.dirname(head)
        if path:
            bases[path] = None

    return tuple(bases)


def _listing(bases: Iterable[str]) -> str:
    """Shell that dumps the facts about `bases` that identify installations.

    One `ls` over a set of globs -- the activation hook, the `conda-meta` entry
    naming conda's version, and the environment directories -- rather than a
    loop that reasons about them remotely. Everything it finds is reported;
    what any of it means is decided in `_parse_listing()`.

    `ls` exits non-zero when a glob matches nothing, which here is the ordinary
    case rather than a failure, so its status is discarded.
    """
    globs = []
    for base in bases:
        quoted = remote_path(base)
        globs += [
            f'{quoted}{_HOOK}',
            f'{quoted}/conda-meta/conda-*.json',
            f'{quoted}/envs/*/',
        ]

    # Where conda puts a named environment when the base is not writable, which
    # is the usual outcome of creating one against a site-wide installation.
    globs.append('"$HOME"/.conda/envs/*/')

    return f'ls -d {" ".join(globs)} 2>/dev/null || true'


def _parse_listing(stdout: str) -> list[CondaInstallation]:
    """Assemble `_listing()` output into the installations it describes.

    Bases are collected first, from the activation hook alone, so that an
    environment is never mistaken for one. Environments under `~/.conda` are
    reported against every base, since conda will activate one by name from any
    of them; environments under a candidate that turned out not to be a base
    belong to nothing and are dropped, rather than credited to every base.
    """
    paths = [line.rstrip('/') for line in stdout.splitlines() if line.strip()]

    environments: dict[str, list[str]] = {
        path[: -len(_HOOK)]: [] for path in paths if path.endswith(_HOOK)
    }
    versions: dict[str, str] = {}
    fallback: list[str] = []

    for path in paths:
        if meta := _META.fullmatch(path):
            if meta['base'] in environments:
                versions.setdefault(meta['base'], meta['version'])
        elif env := _ENV.fullmatch(path):
            if (known := environments.get(env['parent'])) is not None:
                known.append(env['name'])
            elif posixpath.basename(env['parent']) == '.conda':
                fallback.append(env['name'])

    return [
        CondaInstallation(base=base,
                          version=versions.get(base, UNKNOWN),
                          environments=tuple(sorted(set(names) | set(fallback))))
        for base, names in environments.items()
    ]


def _version_query(base: str) -> str:
    """Shell that asks a conda for its own version -- the expensive way.

    Worth running only when `_listing()` could not read the version off a
    `conda-meta` filename, since this launches a Python interpreter.
    """
    return f'{remote_path(base)}/bin/conda --version'


def _parse_version(stdout: str) -> str:
    """`conda --version` prints `conda <version>`; anything else is unknown."""
    parts = stdout.split()
    return parts[1] if len(parts) == 2 and parts[0] == 'conda' else UNKNOWN
