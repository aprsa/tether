"""Exception taxonomy for tether.

Flat and small on purpose: callers should be able to catch something
meaningful without importing a hierarchy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .link import Result


class TetherError(Exception):
    """Base class for every error raised by tether."""


class ConfigError(TetherError):
    """Malformed or missing configuration."""


class LinkError(TetherError):
    """Connection could not be established, or was lost and not recovered."""


class RemoteCommandError(TetherError):
    """A remote command returned nonzero where success was required."""

    def __init__(self, result: Result) -> None:
        self.result = result
        stderr = result.stderr.strip() or '<empty>'
        super().__init__(
            f'command failed (rc={result.returncode}): {result.command}\n'
            f'stderr: {stderr}'
        )


class CondaError(TetherError):
    """Conda could not be installed, or was found in an unusable state."""


class VenvError(TetherError):
    """A virtual environment could not be created, or was found in an
    unusable state."""


class SlurmError(TetherError):
    """Slurm is absent, or present and disagreed with us."""


class EnvActivationError(TetherError):
    """An environment did not activate, or activated into the wrong place."""
