"""The boundary between tether's synchronous API and asyncio.

asyncssh is async; tether's callers are not, and some of them live inside
someone else's event loop already -- a Jupyter kernel, most importantly. So
tether owns an event loop running on a thread of its own, and every synchronous
call hands work to that thread and waits for the answer.

This is deliberately the only place that boundary is crossed. Reading this file
should tell you everything about how a blocking `srv.run(...)` becomes an
awaited coroutine, without any SSH in the way.

Why a thread at all, rather than driving the loop in place with
`run_until_complete`:

1. `run_until_complete` cannot nest. Called from inside a running loop it
   raises, which made every tether call fail in a notebook.
2. A loop that only runs during a call cannot make progress *between* calls.
   Keepalive could not fire while idle, and a held-open channel -- the basis of
   completion notification -- could not be watched at all.

The cost is that a call from inside an async context blocks that context until
it returns. For a notebook cell that is what you want; for a genuine async
application it is not, and the answer there would be awaitable variants of the
public API rather than this bridge.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable
from typing import Any

from .errors import LinkError, TetherError


class LoopThread:
    """An event loop running on a thread of its own.

    Started on first use and restartable after `stop()`, so a closed link can
    be reopened. One of these per `Link`: the loop owns the connection, and a
    shared loop would mean module-level state the rest of tether goes out of
    its way to avoid.
    """

    def __init__(self, name: str = 'tether-loop') -> None:
        self._name = name
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def submit(self, work: Awaitable[Any]) -> Any:
        """Run `work` on the loop thread and block until it finishes.

        Exceptions propagate to the caller as if the call had been synchronous.
        """
        try:
            if threading.current_thread() is self._thread:
                # Waiting here would wait for a result the loop is blocked
                # from producing. A deadlock is worse than an error.
                raise TetherError(
                    'the synchronous API cannot be called from the event loop '
                    'thread itself'
                )
            self._start()
        except BaseException:
            # Nothing will await `work` now, and an un-awaited coroutine warns
            # from the garbage collector, far from the code that made it.
            close = getattr(work, 'close', None)
            if close is not None:
                close()
            raise

        async def runner() -> Any:
            # Not `run_coroutine_threadsafe(work, ...)` directly: that insists
            # on a true coroutine, and `asyncssh.connect()` returns an
            # awaitable that is not one.
            return await work

        assert self._loop is not None
        future = asyncio.run_coroutine_threadsafe(runner(), self._loop)
        try:
            # No timeout here on purpose. The coroutines carry their own
            # deadlines, and a second one at this boundary could fire first,
            # leaving work running on a thread nobody is watching.
            return future.result()
        except BaseException:
            # Includes KeyboardInterrupt: a Ctrl-C in a notebook should not
            # leave the remote command running behind the user's back.
            future.cancel()
            raise

    def call_soon(self, fn: Any) -> None:
        """Schedule `fn` on the loop thread without waiting for it.

        For teardown paths that must not block -- notably a finalizer, which
        has no business waiting on anything.
        """
        if self.running and self._loop is not None:
            self._loop.call_soon_threadsafe(fn)

    def stop(self) -> None:
        """Stop the loop and join the thread. Safe to call more than once."""
        thread, loop = self._thread, self._loop
        self._thread, self._loop = None, None
        if thread is None or loop is None:
            return

        if thread.is_alive():
            try:
                loop.call_soon_threadsafe(loop.stop)
            except RuntimeError:            # loop already closed
                pass
            # Bounded: at interpreter shutdown the thread may already be gone,
            # and hanging on exit is worse than leaving a daemon thread to die.
            thread.join(timeout=5)
        # Only after the loop has stopped, or close() raises.
        if not loop.is_closed():
            loop.close()

    def _start(self) -> None:
        if self._thread is not None:
            if not self._thread.is_alive():
                # A fork leaves the child holding a thread that no longer
                # exists, and an unhandled error in the loop would look the
                # same. Either way, scheduling onto that loop would block
                # forever, so say so instead.
                raise LinkError(
                    'the event loop thread is gone (a fork, or it crashed); '
                    'this link cannot be reused -- build a new one'
                )
            return

        ready = threading.Event()

        def serve() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            ready.set()
            try:
                loop.run_forever()
            finally:
                # Let anything still pending unwind before stop() closes the
                # loop, so no async generator is left hanging -- and join the
                # default executor, which asyncio spawns for getaddrinfo and
                # which loop.close() does *not* clean up. Without this a thread
                # named asyncio_N survives every link.
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.run_until_complete(loop.shutdown_default_executor())

        # daemon: a link nobody closed must not keep the interpreter alive.
        self._thread = threading.Thread(target=serve, name=self._name, daemon=True)
        self._thread.start()
        ready.wait()
