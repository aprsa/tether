"""Wrappers for the remote shell commands.

Quoting and emission primitives, to be used by every module that runs a command.

Note that tether only speaks bash at the moment; tcsh or csh will not work.
"""

from __future__ import annotations

import re
import shlex


def remote_path(path: str) -> str:
    """Convert path string into a shell command, expanding `~` and quoting if
    necessary. The actual workhorse afterwards is `shlex.quote`.

    `~user` is left to `shlex.quote`: it has no `$HOME` equivalent, so it is
    better to quote it and fail loudly than to expand it wrongly.
    """
    if path == '~':
        return '"$HOME"'
    if path.startswith('~/'):
        # The four characters that keep their meaning inside double quotes.
        escaped = re.sub(r'([\\"$`])', r'\\\1', path[2:])
        return f'"$HOME/{escaped}"'
    return shlex.quote(path)


def run_or_abort(command: str, abort_message: str) -> str:
    """Run `command`, and abort loudly rather than continue if it fails."""
    complaint = shlex.quote(f'tether: {abort_message}')
    return f'{command} || {{ echo {complaint} >&2; exit 1; }}'


def printf(*words: str) -> str:
    """Combines shell words into a joint, executable printf command.

    This exists so that the calling function doesn't have to write the
    format string itself.
    """
    if not words:
        raise ValueError('printf() needs at least one word to print')

    return "printf '" + r'%s\n' * len(words) + "' " + ' '.join(words)


def is_identifier(name: str) -> bool:
    """Whether `name` is safe to use as a shell variable name.

    Anything else could smuggle code into an `export`.
    """
    return re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', name) is not None
