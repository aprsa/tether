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

`keepalive_interval` now does what it says: the loop runs continuously on its
own thread (see `event_loop.py`), so probes fire during idle periods too, not
only for the duration of a call. Reconnect-and-retry-once remains the backstop
for drops the keepalive does not catch.
"""

from __future__ import annotations

import asyncio
import os
import posixpath
import time
import weakref
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import asyncssh

from .event_loop import LoopThread
from .errors import RemoteCommandError, LinkError

DEFAULT_TIMEOUT = 3600.0
# In seconds. Applies to commands, not transfers. Override per call,
# or per server with `timeout` in `servers.toml`. `None` means no limit
# at all.

DEFAULT_KEEPALIVE = 30
# Seconds between keepalive probes. The loop runs continuously, so these
# fire during idle periods too.

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


class _Teardown:
    """What has to happen when a `Link` is abandoned rather than closed.

    Deliberately holds no reference back to the `Link`: a finalizer that did
    would keep its object alive forever, which is the opposite of the point.
    The connection lives here so the finalizer can still see the current one.
    """

    def __init__(self, loop: LoopThread) -> None:
        self.loop = loop
        self.conn: asyncssh.SSHClientConnection | None = None

    def __call__(self) -> None:
        if self.conn is not None:
            # abort(), not a graceful close: there is no one left to await a
            # clean disconnect, and an abandoned session on a shared login node
            # is worse than an abrupt one.
            self.loop.call_soon(self.conn.abort)
            self.conn = None
        self.loop.stop()


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

    Connects lazily on first use. Its event loop runs on a thread of its own
    (see `event_loop.py`), so the synchronous API works from a script, a
    notebook, or inside someone else's async application alike. Not thread-safe
    for *callers*: one `Link` expects one caller at a time.

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

        self._loop = LoopThread(name=f'tether-loop-{host}')
        self._sftp: asyncssh.SFTPClient | None = None

        # Rebinding a variable is the most ordinary thing in a notebook, and an
        # abandoned Link would otherwise keep both a thread and a live SSH
        # session until the kernel died. A safety net, not a substitute for
        # close(): collection is prompt in CPython but not guaranteed.
        #
        # `atexit` is switched off deliberately: at interpreter shutdown the
        # daemon thread is already going away and the OS reclaims the socket,
        # so running this then risks hanging for no benefit. (It is an
        # attribute on the finalizer -- passing it to the constructor forwards
        # it to the callback instead.)
        self._teardown = _Teardown(self._loop)
        finalizer = weakref.finalize(self, self._teardown)
        finalizer.atexit = False

    @property
    def _conn(self) -> asyncssh.SSHClientConnection | None:
        """Stored on `_Teardown` so the finalizer sees the current connection.

        A property rather than a plain attribute purely to keep one copy of it;
        every `self._conn = ...` below reads naturally and still works.
        """
        return self._teardown.conn

    @_conn.setter
    def _conn(self, conn: asyncssh.SSHClientConnection | None) -> None:
        self._teardown.conn = conn

    @property
    def connected(self) -> bool:
        return self._conn is not None

    def connect(self) -> Self:
        """Establish the connection now instead of on first use.

        `close()` is the counterpart: it drops the connection and stops the
        loop thread, and the object stays usable -- a later call reconnects.
        """
        self._call(self._noop)
        return self

    def close(self) -> None:
        """Close the connection and stop the event loop thread.

        Shuts the SFTP client and connection down in order, so no asyncio tasks
        are orphaned, then joins the thread. The object remains usable; a later
        call starts a fresh thread and reconnects.
        """
        if self._conn is not None:
            try:
                self._sync(self._shutdown())
            except Exception:  # noqa: BLE001 - fall back to an abrupt teardown
                self._discard()

        self._conn = None
        self._sftp = None
        self._loop.stop()

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
        local: str | os.PathLike[str],
        remote: str,
        *,
        recurse: bool = False,
        timeout: float | None = None,
    ) -> None:
        """Upload. No timeout by default: large transfers legitimately take time.

        `local` accepts a `Path`; `remote` is `str` only, since a local path
        object carries local separators and has no business describing a path
        on the far end.
        """
        self._call(lambda: self._put(local, remote, recurse, timeout))

    def get(
        self,
        remote: str,
        local: str | os.PathLike[str],
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
        self,
        local: str | os.PathLike[str],
        remote: str,
        recurse: bool,
        timeout: float | None,
    ) -> None:
        sftp = await self._sftp_client()
        # SFTP creates no parents and reports a missing one as a bare "No such
        # file", which reads like the *source* is absent. makedirs is `mkdir -p`.
        #
        # posixpath rather than pathlib: the remote is POSIX whatever the client
        # runs, so `Path` would emit backslashes on Windows. It also yields ''
        # for a bare filename, where `PurePosixPath(...).parent` yields a truthy
        # '.' and would cost a needless round trip on every relative put.
        parent = posixpath.dirname(remote)
        if parent:
            await sftp.makedirs(parent, exist_ok=True)
        # asyncssh takes bytes | str | PurePath, not PathLike in general, so
        # normalise here rather than advertise something it cannot honour.
        await asyncio.wait_for(
            sftp.put(os.fspath(local), remote, recurse=recurse), timeout=timeout
        )

    async def _get(
        self,
        remote: str,
        local: str | os.PathLike[str],
        recurse: bool,
        timeout: float | None,
    ) -> None:
        sftp = await self._sftp_client()
        # pathlib here, deliberately: this end is the *local* filesystem, so
        # host semantics are what you want.
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        await asyncio.wait_for(
            sftp.get(remote, os.fspath(local), recurse=recurse), timeout=timeout
        )

    # -- sync bridge -------------------------------------------------------

    def _call(self, factory: Callable[[], Coroutine[Any, Any, Any]]) -> Any:
        """Run a coroutine, reconnecting and retrying once if the link drops.

        `factory` must be a callable, not a coroutine: a coroutine cannot be
        awaited twice, and the retry needs a fresh one.
        """
        try:
            try:
                return self._sync(factory())
            except _RETRYABLE:
                self._discard()

            try:
                return self._sync(factory())
            except _RETRYABLE as exc:
                self._discard()
                # Not necessarily a lost link: a ChannelOpenError also covers a
                # permanent refusal (missing sftp subsystem, MaxSessions
                # reached). Report what happened rather than guessing why.
                raise LinkError(
                    f'{self.host}: operation failed after one reconnect '
                    f'attempt: {type(exc).__name__}: {exc}'
                ) from exc
        except BaseException:
            # Nothing connected, so the loop thread has nothing left to serve.
            # Without this a failed connect() leaves a thread behind, and code
            # that retries several unreachable hosts accumulates one each time.
            if self._conn is None:
                self._loop.stop()
            raise

    def _sync(self, coro: Coroutine[Any, Any, Any]) -> Any:
        """Hand a coroutine to the loop thread and wait for it.

        No check for a running loop in the caller: that is the whole point of
        owning a thread. tether works from a script, a notebook, or inside
        someone else's async application -- though in the last case it blocks
        that application until the call returns.
        """
        return self._loop.submit(coro)

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
