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

`probe()` and `install()` take the transport as an argument -- `run`, which
answers a command with its stdout -- rather than reaching for a connection, so the whole algorithm -- round trips included -- is exercisable
with a dictionary.
"""

from __future__ import annotations

import json
import posixpath
import re
import shlex
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace

from .errors import CondaError, RemoteCommandError
from .shell import printf, remote_path, run_or_abort

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
    if it fails -- note that this is stdout, not the `Result` that `Server.run`
    and `Link.run` hand back; `Server.probe_conda()` passes an adapter.
    `workdir_conda` is where tether installs its own.

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
    bases = _parse_search(run(_search_query(workdir_conda, setup=setup)))
    if not bases:
        return []

    return _resolve_versions(run, _parse_listing(run(_listing_query(bases))))


def _resolve_versions(
    run: Callable[[str], str],
    found: list[CondaInstallation],
) -> list[CondaInstallation]:
    """Fill in versions that `conda-meta` would not give up, the slow way.

    Only a base that failed to name itself pays for an interpreter launch, and
    one that cannot even run `conda --version` keeps `unknown` rather than
    raising -- it is already known to be odd, and that is the answer.
    """
    for index, installation in enumerate(found):
        if installation.version != UNKNOWN:
            continue
        try:
            version = _parse_version(run(_version_query(installation.base)))
        except RemoteCommandError:
            continue
        found[index] = replace(installation, version=version)

    return found


#: Miniforge rather than Miniconda: the `defaults` channel carries licence
#: terms that nobody can meaningfully accept on a shared cluster's behalf,
#: and conda-forge has none. The API is used rather than a download URL so
#: that assets are *discovered* -- see `_parse_release()`.
_MINIFORGE = 'https://api.github.com/repos/conda-forge/miniforge'

#: Long enough for 124MB over a slow link, short enough to fail before a
#: batch allocation expires.
_DOWNLOAD_TIMEOUT = 600


def install(
    run: Callable[[str], str],
    prefix: str,
    *,
    version: str | None = None,
    installer: str | None = None,
    sha256: str | None = None,
    adopt_if_exists: bool = True,
) -> CondaInstallation:
    """Put a working conda at `prefix`, and report what ended up there.

    `run` is the same transport `probe()` takes: a command in, its stdout back.

    `prefix` must not already exist -- the installer refuses to write into a
    directory that does, and tether never removes anything. What happens when
    something *is* there depends on whether it works:

    - it activates, and `adopt_if_exists`: adopted and returned, nothing
      installed. This is what makes calling `install()` twice harmless.
    - it activates, and not `adopt_if_exists`: `CondaError`, since the
      alternative would be to destroy a working installation.
    - it does not activate: `CondaError`. A half-extracted tree is exactly
      what you do not want silently overwritten, and only a person can know
      whether it is safe to delete.

    `version` is a Miniforge release tag such as `'26.7.2-0'`, not a conda
    version -- they usually coincide, but the returned `CondaInstallation`
    reports what `conda-meta` says, which is conda's own. Omit it for the
    latest release.

    `installer` is a path to an installer already on the remote, skipping the
    download entirely. It is trusted as given unless `sha256` accompanies it,
    which is the escape hatch for a site that mirrors the installer behind a
    firewall and publishes its own hash.

    A downloaded installer lands beside `prefix`, and is removed once it has
    run. It is deliberately kept when anything fails, so a bad download can
    be examined rather than silently fetched again.
    """
    present, active = _parse_state(run(_state_query(prefix)))

    if active:
        if not adopt_if_exists:
            raise CondaError(
                f'a working conda is already installed at {prefix}; pass '
                f'adopt_if_exists=True to use it, or choose another prefix'
            )
        return _describe(run, prefix)

    if present:
        raise CondaError(
            f'{prefix} exists but does not activate, so it is not a usable '
            f'conda installation. tether will not remove it: inspect it, '
            f'delete it by hand, or install somewhere else'
        )

    downloaded = None
    if installer is None:
        arch = run('uname -m').strip()
        url, checksum_url = _parse_release(run(_release_query(version)), arch)
        downloaded = posixpath.join(_parent(prefix), posixpath.basename(url))
        run(_download(url, downloaded))
        expected = _parse_sha256(run(_fetch(checksum_url)))
        installer = downloaded
    else:
        expected = sha256

    if expected:
        actual = _parse_sha256(run(_sha256_query(installer)))
        if actual != expected:
            raise CondaError(
                f'checksum mismatch for {installer}: expected {expected}, got '
                f'{actual or "nothing"}. The file has been left in place'
            )

    run(_install_command(installer, prefix))

    if downloaded:                      # only ever remove what we fetched
        run(f'rm -f {remote_path(downloaded)}')

    return _describe(run, prefix)


def _parent(path: str) -> str:
    """The directory `path` will be created in."""
    return posixpath.dirname(path.rstrip('/')) or '.'


def _release_query(version: str | None) -> str:
    """Command fetching one release's metadata, latest unless `version` says."""
    url = (f'{_MINIFORGE}/releases/latest' if version is None
           else f'{_MINIFORGE}/releases/tags/{version}')
    return _fetch(url)


def _fetch(url: str) -> str:
    """Command printing what is at `url`, failing loudly on a bad status."""
    return f'curl -fsSLm 60 {shlex.quote(url)}'


def _parse_release(stdout: str, arch: str) -> tuple[str, str]:
    """The installer URL for `arch`, and the URL of its checksum.

    Discovery, not construction. The release lists its own assets, so a change
    in Miniforge's naming surfaces here as "no asset matched" rather than as a
    404 on a URL we invented -- and the two are found together or not at all,
    because only the version-stamped filename is checksummed. The unversioned
    convenience alias `Miniforge3-Linux-x86_64.sh` has no `.sha256` at any
    URL, which is why it is never chosen.
    """
    try:
        release = json.loads(stdout)
        tag = release['tag_name']
        assets = {a['name']: a['browser_download_url'] for a in release['assets']}
    except (TypeError, KeyError, ValueError) as exc:
        raise CondaError(
            f'could not read the Miniforge release listing ({exc}); '
            f'pass installer= to skip the download entirely'
        ) from None

    suffix = f'-Linux-{arch}.sh'
    for name in sorted(assets):
        if name.endswith(suffix) and tag in name and f'{name}.sha256' in assets:
            return assets[name], assets[f'{name}.sha256']

    offered = ', '.join(sorted(n for n in assets if n.endswith('.sh'))) or 'none'
    raise CondaError(
        f'Miniforge {tag} publishes no checksummed installer for Linux-{arch}. '
        f'Installers offered: {offered}'
    )


def _download(url: str, dest: str) -> str:
    """Command fetching `url` to `dest`, creating the directory it needs."""
    return '\n'.join((
        run_or_abort(f'mkdir -p {remote_path(_parent(dest))}',
                     f'could not create {_parent(dest)}'),
        run_or_abort(
            f'curl -fsSLm {_DOWNLOAD_TIMEOUT} -o {remote_path(dest)} {shlex.quote(url)}',
            f'could not download {url}',
        ),
    ))


def _sha256_query(path: str) -> str:
    """Command printing the checksum of a file already on the remote."""
    return f'sha256sum {remote_path(path)}'


def _parse_sha256(stdout: str) -> str:
    """The hex digest out of `<hex>  <name>`.

    Both `sha256sum` and a published `.sha256` file use that shape, so one
    parser reads the expected value and the actual one -- and the comparison
    happens in Python, where a mismatch can say what it expected and what it
    got. `sha256sum -c` would only say FAILED, and would additionally require
    the download to carry the exact filename the checksum file names.
    """
    fields = stdout.split()
    return fields[0] if fields else ''


def _install_command(installer: str, prefix: str) -> str:
    """Command running the installer non-interactively into `prefix`.

    `-b` is batch (no prompts, no licence pager), `-p` is the prefix. The
    installer refuses a prefix that already exists, which is the behaviour
    tether wants, so only the *parent* is created here.
    """
    return '\n'.join((
        run_or_abort(f'mkdir -p {remote_path(_parent(prefix))}',
                     f'could not create {_parent(prefix)}'),
        run_or_abort(f'bash {remote_path(installer)} -b -p {remote_path(prefix)}',
                     f'the Miniforge installer failed for {prefix}'),
    ))


def _state_query(prefix: str) -> str:
    """Command reporting whether anything is at `prefix`, and whether it works.

    Activation is the test rather than a directory listing, because a tree can
    look exactly like conda and still not run. What the caller needs to know
    is not "does this resemble conda" but "can this be used", and only trying
    it answers that.
    """
    at = remote_path(prefix)
    return (
        f'[ -e {at} ] && echo present || echo absent\n'
        f'if . {at}/etc/profile.d/conda.sh >/dev/null 2>&1 && '
        f'conda activate base >/dev/null 2>&1; then echo active; fi'
    )


def _parse_state(stdout: str) -> tuple[bool, bool]:
    """(something is at the prefix, it activates)."""
    words = stdout.split()
    return 'present' in words, 'active' in words


def _describe(run: Callable[[str], str], prefix: str) -> CondaInstallation:
    """What is at `prefix`, described the same way `probe()` describes things."""
    found = _resolve_versions(run, _parse_listing(run(_listing_query([prefix]))))
    if not found:
        raise CondaError(
            f'the installer reported success but nothing that looks like a '
            f'conda installation is at {prefix}'
        )
    return found[0]


def _search_query(workdir_conda: str | None = None, *, setup: str = '') -> str:
    """Shell that prints paths which might belong to a conda installation.

    This only gathers; it decides nothing. What the paths mean is
    `_parse_search()`'s problem, and whether they exist is `_listing_query()`'s.
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


def _listing_query(bases: Iterable[str]) -> str:
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
    """Assemble `_listing_query()` output into the installations it describes.

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

    Worth running only when `_listing_query()` could not read the version off a
    `conda-meta` filename, since this launches a Python interpreter.
    """
    return f'{remote_path(base)}/bin/conda --version'


def _parse_version(stdout: str) -> str:
    """`conda --version` prints `conda <version>`; anything else is unknown."""
    parts = stdout.split()
    return parts[1] if len(parts) == 2 and parts[0] == 'conda' else UNKNOWN

def create_env(
    run: Callable[[str], str],
    name: str,
    base: str,
    *,
    python: str | None = None,
    packages: Iterable[str] = (),
    setup: str = '',
    adopt_if_exists: bool = True,
) -> str:
    """Create the conda environment `name` under `base`, and say where it landed.

    Where it lands is not always where you would guess: conda puts a named
    environment in `<base>/envs/<name>` when the base is writable, and in
    `~/.conda/envs/<name>` when it is not -- which is the usual outcome against
    a site-wide installation. The prefix is therefore returned rather than
    assumed, by asking the environment itself once it exists.

    - it already activates, and `adopt_if_exists`: adopted and returned;
    - it already activates and is not wanted: `CondaError`;
    - otherwise it is created.

    `python` is a version for conda to solve for (`'3.12'`), not a path: conda
    fetches an interpreter rather than building on one that is already here,
    which is the whole reason it can offer a version the machine does not have.
    """
    prefix = _parse_env_prefix(run(_env_state_query(base, name, setup=setup)))

    if prefix:
        if not adopt_if_exists:
            raise CondaError(
                f"conda environment '{name}' already exists at {prefix}; pass "
                f'adopt_if_exists=True to use it, or choose another name'
            )
        return prefix

    run(_create_env_command(base, name, python=python, packages=packages, setup=setup))

    prefix = _parse_env_prefix(run(_env_state_query(base, name, setup=setup)))
    if not prefix:
        raise CondaError(
            f"conda reported success but environment '{name}' does not "
            f'activate under {base}'
        )
    return prefix


def _hook(base: str) -> str:
    """Sourcing conda's shell hook, which `conda activate` does not exist without."""
    return f'. {remote_path(base)}/etc/profile.d/conda.sh'


def _env_state_query(base: str, name: str, *, setup: str = '') -> str:
    """Command printing an environment's prefix, or nothing if it does not work.

    Activation is the test, as it is everywhere else here: a directory under
    `envs/` can exist and be half-written, and what the caller needs to know is
    whether the environment can be entered.
    """
    activate = f'conda activate {shlex.quote(name)}'
    return '\n'.join(filter(None, (
        setup,
        f'if {_hook(base)} >/dev/null 2>&1 && {activate} >/dev/null 2>&1; then',
        '    printf \'%s\\n\' "$CONDA_PREFIX"',
        'fi',
    )))


def _parse_env_prefix(stdout: str) -> str:
    """The prefix an environment reported, or `''` if it never got that far."""
    for line in stdout.splitlines():
        candidate = line.strip()
        if candidate.startswith('/'):
            return candidate
    return ''


def _create_env_command(
    base: str,
    name: str,
    *,
    python: str | None = None,
    packages: Iterable[str] = (),
    setup: str = '',
) -> str:
    """Command creating the environment.

    `-y` because there is no one at the prompt, and `conda create` asks before
    it does anything. Specifications are quoted: `numpy>=1.20` is a redirection
    to a shell that has not been told otherwise.
    """
    wanted = [f'python={python}'] if python else []
    wanted += [p for p in packages if p]

    create = (
        f'conda create -y -n {shlex.quote(name)} '
        + ' '.join(shlex.quote(spec) for spec in wanted)
    ).rstrip()

    return '\n'.join(filter(None, (
        setup,
        run_or_abort(_hook(base), f'could not source the conda hook at {base}'),
        run_or_abort(create, f"could not create conda environment '{name}'"),
    )))
