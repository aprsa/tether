"""SSH link.

Exposes a synchronous API over asyncssh.

Two design points worth stating, because everything above depends on them:

1. *One connection, many channels.* SSH multiplexes: each command is a channel
   on an already-authenticated link, so N commands cost one authentication.
   The connection and the SFTP client are both held open and reused.

2. *Reconnection is invisible, not prevented.* The remote filesystem is the
   source of truth, so a dropped connection carries no state and is a non-event.
   Every operation is attempted, and on a connection-level failure the
   connection is discarded, re-established, and the operation retried exactly
   once. A second failure is raised.

`keepalive_interval` is set, but note its scope: it only fires while the event
loop is running, so it protects long single operations (a large transfer, a
streamed log) rather than long idle periods. Idle drops are handled by (2).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import asyncssh

from .errors import RemoteCommandError, TetherError, LinkError

DEFAULT_TIMEOUT = 60.0
"""Seconds. Applies to commands, not transfers."""

DEFAULT_KEEPALIVE = 30
"""Seconds between keepalive probes while the loop is running."""

_RETRYABLE = (
    asyncssh.ConnectionLost,
    asyncssh.DisconnectError,
    asyncssh.ChannelOpenError,
    ConnectionError,  # builtin: reset, aborted, broken pipe
    EOFError,
)

_CONNECT_FAILED = (
    asyncssh.PermissionDenied,
    asyncssh.Error,
    OSError,
    TimeoutError,
)


@dataclass(frozen=True)
class Result:
    """Outcome of a remote command."""

    command: str
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def check(self) -> Self:
        """Raise `RemoteCommandError` unless the command succeeded."""
        if not self.ok:
            raise RemoteCommandError(self)
        return self


class Link:
    """A reusable SSH connection to one host.

    Connects lazily on first use. Not thread-safe and not usable from inside a
    running event loop: it owns its own loop and drives it synchronously.

    `ssh_config` names an ssh_config file to read *instead of* `~/.ssh/config`,
    exactly like `ssh -F`. It is not a security relaxation: host keys are still
    validated strictly, against whatever `UserKnownHostsFile` that config
    names. It exists so a caller can be hermetic -- the test rig uses it to
    describe a throwaway server without touching the user's `~/.ssh`.
    """

    def __init__(
        self,
        host: str,
        user: str | None = None,
        *,
        port: int | None = None,
        ssh_config: str | Path | None = None,
        keepalive: int = DEFAULT_KEEPALIVE,
        connect_timeout: float = 30.0,
    ) -> None:
        self.host = host
        self.user = user
        self.port = port
        self.ssh_config = ssh_config
        self.keepalive = keepalive
        self.connect_timeout = connect_timeout

        self._loop: asyncio.AbstractEventLoop | None = None
        self._conn: asyncssh.SSHClientConnection | None = None
        self._sftp: asyncssh.SFTPClient | None = None

    @property
    def connected(self) -> bool:
        return self._conn is not None

    def connect(self) -> Self:
        """Establish the connection now instead of on first use."""
        self._call(self._noop)
        return self

    def close(self) -> None:
        """Close the connection and release the event loop.

        Shuts the SFTP client and connection down in order, so no asyncio tasks
        are orphaned. The object remains usable; a later call reconnects.
        """
        if self._conn is not None:
            try:
                self._sync(self._shutdown())
            except Exception:  # noqa: BLE001 - fall back to an abrupt teardown
                self._discard()

        self._conn = None
        self._sftp = None

        if self._loop is not None and not self._loop.is_closed():
            try:
                self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            except Exception:  # noqa: BLE001
                pass
            self._loop.close()
        self._loop = None

    def __enter__(self) -> Self:
        return self.connect()

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __repr__(self) -> str:
        target = f'{self.user}@{self.host}' if self.user else self.host
        state = 'connected' if self.connected else 'disconnected'
        return f'<Link {target} ({state})>'

    def run(
        self,
        command: str,
        *,
        check: bool = False,
        timeout: float | None = DEFAULT_TIMEOUT,
    ) -> Result:
        """Run a shell command remotely.

        Returns a `Result` regardless of exit status; pass `check=True` to
        raise `RemoteCommandError` on failure. Probing commands legitimately
        fail, so asking a question should not require a try/except.
        """
        result = self._call(lambda: self._run(command, timeout))
        return result.check() if check else result

    def put(
        self,
        local: str,
        remote: str,
        *,
        recurse: bool = False,
        timeout: float | None = None,
    ) -> None:
        """Upload. No timeout by default: large transfers legitimately take time."""
        self._call(lambda: self._put(local, remote, recurse, timeout))

    def get(
        self,
        remote: str,
        local: str,
        *,
        recurse: bool = False,
        timeout: float | None = None,
    ) -> None:
        """Download. No timeout by default."""
        self._call(lambda: self._get(remote, local, recurse, timeout))

    def ping(self) -> float:
        """Round-trip time in seconds for a trivial remote command."""
        start = time.perf_counter()
        self.run('true', check=True, timeout=DEFAULT_TIMEOUT)
        return time.perf_counter() - start

    async def _noop(self) -> None:
        await self._ensure()

    async def _shutdown(self) -> None:
        """Orderly teardown, in dependency order: SFTP first, then the link."""
        if self._sftp is not None:
            self._sftp.exit()
            await self._sftp.wait_closed()
            self._sftp = None
        if self._conn is not None:
            self._conn.close()
            await self._conn.wait_closed()
            self._conn = None

    async def _ensure(self) -> asyncssh.SSHClientConnection:
        """Return a live connection, opening one if needed."""
        if self._conn is not None:
            return self._conn

        options: dict[str, Any] = {'keepalive_interval': self.keepalive}
        if self.user:
            options['username'] = self.user
        if self.port:
            options['port'] = self.port
        if self.ssh_config:
            options['config'] = [str(self.ssh_config)]

        try:
            self._conn = await asyncio.wait_for(
                asyncssh.connect(self.host, **options),
                timeout=self.connect_timeout,
            )
        except _CONNECT_FAILED as exc:
            self._conn = None
            raise LinkError(
                f'cannot connect to {self.host}: {exc}{_hint(exc)}'
            ) from exc

        return self._conn

    async def _sftp_client(self) -> asyncssh.SFTPClient:
        """Return a live SFTP client, starting one if needed.

        Held open rather than created per transfer: each client spawns a
        subsystem channel and an `sftp-server` process on the remote side.
        """
        conn = await self._ensure()
        if self._sftp is None:
            self._sftp = await conn.start_sftp_client()
        return self._sftp

    async def _run(self, command: str, timeout: float | None) -> Result:
        conn = await self._ensure()

        # Deliberately not conn.run(timeout=...): that leaves the remote
        # process running and the channel open when the timeout fires.
        async with conn.create_process(command) as proc:
            try:
                completed = await asyncio.wait_for(proc.wait(), timeout=timeout)
            except TimeoutError as exc:
                proc.terminate()
                raise LinkError(
                    f'timed out after {timeout}s on {self.host}: {command}'
                ) from exc

        return Result(
            command=command,
            returncode=completed.exit_status if completed.exit_status is not None else -1,
            stdout=_text(completed.stdout),
            stderr=_text(completed.stderr),
        )

    async def _put(
        self, local: str, remote: str, recurse: bool, timeout: float | None
    ) -> None:
        sftp = await self._sftp_client()
        await asyncio.wait_for(
            sftp.put(local, remote, recurse=recurse), timeout=timeout
        )

    async def _get(
        self, remote: str, local: str, recurse: bool, timeout: float | None
    ) -> None:
        sftp = await self._sftp_client()
        await asyncio.wait_for(
            sftp.get(remote, local, recurse=recurse), timeout=timeout
        )

    # -- sync bridge -------------------------------------------------------

    def _call(self, factory: Callable[[], Coroutine[Any, Any, Any]]) -> Any:
        """Run a coroutine, reconnecting and retrying once if the link drops.

        `factory` must be a callable, not a coroutine: a coroutine cannot be
        awaited twice, and the retry needs a fresh one.
        """
        try:
            return self._sync(factory())
        except _RETRYABLE:
            self._discard()

        try:
            return self._sync(factory())
        except _RETRYABLE as exc:
            self._discard()
            # Not necessarily a lost link: a ChannelOpenError also covers a
            # permanent refusal (missing sftp subsystem, MaxSessions reached).
            # Report what happened rather than guessing why.
            raise LinkError(
                f'{self.host}: operation failed after one reconnect '
                f'attempt: {type(exc).__name__}: {exc}'
            ) from exc

    def _sync(self, coro: Coroutine[Any, Any, Any]) -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            coro.close()
            raise TetherError(
                "tether's synchronous API cannot be called from inside a "
                "running event loop; await the coroutines directly instead."
            )

        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop.run_until_complete(coro)

    def _discard(self) -> None:
        """Forget the connection without assuming it is still usable."""
        conn, self._conn, self._sftp = self._conn, None, None
        if conn is None:
            return
        try:
            conn.abort()
        except Exception:  # noqa: BLE001 - teardown must not mask the real error
            pass


def _hint(exc: BaseException) -> str:
    """Turn asyncssh's two most common opaque failures into actionable advice.

    Host keys are validated strictly, and deliberately not made optional:
    a `known_hosts=None` switch would be a security footgun in a library whose
    whole job is running commands on someone else's machine.
    """
    text = str(exc).lower()
    if 'host key' in text:
        return (
            '\nhint: the host is not in ~/.ssh/known_hosts. Connect once with '
            '`ssh` to record it.'
        )
    if 'permission denied' in text or 'authentication' in text:
        return (
            '\nhint: key-based authentication failed. Check that `ssh` alone '
            'succeeds, and that the key is loaded or named in ~/.ssh/config.'
        )
    return ''


def _text(raw: object) -> str:
    if raw is None:
        return ''
    if isinstance(raw, bytes):
        return raw.decode('utf-8', errors='replace')
    return str(raw)
