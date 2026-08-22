"""Unit tests for the sync/async boundary. No cluster, no SSH, no container.

These are the threading edge cases: they run in milliseconds, which is the
point -- the live suite is where SSH is exercised, and none of this needs it.
"""

import asyncio
import contextlib
import threading

import pytest

import tether
from tether.event_loop import LoopThread


async def answer(value=42):
    return value


def test_submit_returns_the_result():
    lt = LoopThread()
    try:
        assert lt.submit(answer()) == 42
    finally:
        lt.stop()


def test_exceptions_propagate_as_if_synchronous():
    async def boom():
        raise ValueError('from the loop thread')

    lt = LoopThread()
    try:
        with pytest.raises(ValueError, match='from the loop thread'):
            lt.submit(boom())
    finally:
        lt.stop()


def test_works_from_inside_a_running_loop():
    """The reason this class exists: a Jupyter cell is a running loop, and
    `run_until_complete` cannot nest inside one."""

    async def caller():
        lt = LoopThread()
        try:
            return lt.submit(answer('nested'))
        finally:
            lt.stop()

    assert asyncio.run(caller()) == 'nested'


def test_accepts_awaitables_that_are_not_coroutines():
    """`asyncssh.connect()` returns one of these, and
    `run_coroutine_threadsafe` refuses it outright."""

    class Awaitable:
        def __await__(self):
            return answer('awaited').__await__()

    lt = LoopThread()
    try:
        assert lt.submit(Awaitable()) == 'awaited'
    finally:
        lt.stop()


def test_lazy_start_and_clean_stop():
    lt = LoopThread(name='probe-loop')
    assert not lt.running                      # constructing starts nothing
    lt.submit(answer())
    assert lt.running
    lt.stop()
    assert not lt.running
    assert 'probe-loop' not in [t.name for t in threading.enumerate()]


def test_restartable_after_stop():
    """`close()` then reuse is documented behaviour, and a Thread cannot be
    restarted -- so stop() has to leave room for a fresh one."""
    lt = LoopThread()
    assert lt.submit(answer(1)) == 1
    lt.stop()
    assert lt.submit(answer(2)) == 2           # new thread, transparently
    lt.stop()


def test_no_thread_leak_across_many_cycles():
    before = threading.active_count()
    for _ in range(20):
        lt = LoopThread()
        lt.submit(answer())
        lt.stop()
    assert threading.active_count() == before


def test_a_dead_thread_is_reported_not_waited_on():
    """After a fork the child holds a thread that no longer exists; scheduling
    onto its loop would block forever, so it has to complain instead."""
    lt = LoopThread()
    lt.submit(answer())
    lt._loop.call_soon_threadsafe(lt._loop.stop)   # kill it behind lt's back
    lt._thread.join()

    with pytest.raises(tether.LinkError, match='event loop thread is gone'):
        lt.submit(answer())


def test_calling_from_the_loop_thread_is_an_error_not_a_deadlock():
    lt = LoopThread()
    try:
        async def reenter():
            return lt.submit(answer())         # from the loop thread itself

        with pytest.raises(tether.TetherError, match='from the event loop thread'):
            lt.submit(reenter())
    finally:
        lt.stop()


def test_interrupting_a_call_cancels_the_work():
    """A Ctrl-C in a notebook must not leave the remote command running.

    `submit` cancels its future on any BaseException, KeyboardInterrupt
    included; this drives that cancellation directly, since raising a real
    signal inside a test is its own kind of unreliable.
    """
    started, cancelled = threading.Event(), []

    async def slow():
        started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.append(True)
            raise

    lt = LoopThread()
    try:
        def wait_and_be_cancelled():
            # submit() re-raises after cancelling, which is the contract; the
            # assertion below is about the coroutine, not this thread.
            with contextlib.suppress(BaseException):
                lt.submit(slow())

        waiter = threading.Thread(target=wait_and_be_cancelled, daemon=True)
        waiter.start()
        assert started.wait(timeout=5)

        # What submit()'s except-clause does when the wait is interrupted.
        for task in asyncio.all_tasks(lt._loop):
            lt._loop.call_soon_threadsafe(task.cancel)

        for _ in range(100):
            if cancelled:
                break
            threading.Event().wait(0.02)
        assert cancelled, 'the coroutine never saw the cancellation'
    finally:
        lt.stop()
