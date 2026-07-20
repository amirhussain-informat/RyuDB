"""The single engine worker — serialization + pending-request cancellation.

The ``Engine`` is single-session / one-thread (``executor.py:152``): its
``_txn`` / ``_commit_ts`` / ``_snapshots`` are unprotected mutable state, and a
query runs to completion on the calling thread. So the server may NOT run two
engine calls concurrently. This module is the serialization boundary: a single
dedicated thread owns the ``Engine`` (and the ``Catalog`` — the engine reads it
mid-query, so catalog mutations must serialize with queries too) and pulls work
items from a FIFO queue one at a time.

It also implements cancellation. ``ThreadPoolExecutor`` does not expose its
internal queue, so a hand-rolled worker is used instead: each submitted item
carries a ``threading.Event`` cancel flag. The worker checks it *before*
starting the call and drops the item if set (pending cancel). For a request
already in flight, the same event is published to ``engine.cancel_event`` for
the duration of the call, so the executor raises ``CancelledByUser`` at the
next plan-node boundary (cooperative in-flight cancel). A single long cuDF
call (a big groupby, a cold parquet read) is still not interruptible mid-call —
cancel fires at the next node boundary; that is the one remaining limit.

Results/exceptions are delivered back to the asyncio loop with
``loop.call_soon_threadsafe`` (an asyncio.Future may only be resolved on the
loop thread).
"""

from __future__ import annotations

import asyncio
import queue
import threading
from typing import Any, Callable


class Cancelled(Exception):
    """Raised into a request's future when it was dropped before starting."""


class EngineWorker:
    """Owns the single Engine; processes requests strictly one at a time on a
    dedicated thread. Supports cancelling a request that has not started."""

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self._q: queue.Queue = queue.Queue()
        self._thread = threading.Thread(
            target=self._run, name="ryudb-engine", daemon=True
        )
        self._thread.start()

    def submit(
        self, loop: asyncio.AbstractEventLoop, fn: Callable[..., Any], *args: Any
    ) -> tuple[asyncio.Future, threading.Event]:
        """Enqueue ``fn(*args)`` on the worker thread.

        Returns ``(future, cancel_event)``: await the future for the result (or
        ``Cancelled`` / the raised exception); set the cancel event to drop the
        request if it has not started yet.
        """
        fut: asyncio.Future = loop.create_future()
        cancel = threading.Event()
        self._q.put((fn, args, loop, fut, cancel))
        return fut, cancel

    def shutdown(self) -> None:
        """Signal the worker to stop and wait briefly for it to drain."""
        self._q.put(None)
        self._thread.join(timeout=5.0)

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return  # shutdown sentinel
            fn, args, loop, fut, cancel = item
            if cancel.is_set():
                loop.call_soon_threadsafe(self._resolve_cancelled, fut)
                continue
            # Expose this request's cancel event to the engine for the duration
            # of the call so a cooperative in-flight cancel (set by ``cancel``
            # while the request runs) raises ``CancelledByUser`` at the next
            # plan-node boundary. Cleared afterwards so a non-server caller
            # (CLI / in-process) sees no stale event. The engine is single-thread
            # (only this thread calls it), so the assignment needs no lock.
            self.engine.cancel_event = cancel
            try:
                result = fn(*args)
            except BaseException as exc:  # noqa: BLE001 -- surface every fault
                loop.call_soon_threadsafe(self._resolve_exc, fut, exc)
            else:
                loop.call_soon_threadsafe(self._resolve_ok, fut, result)
            finally:
                self.engine.cancel_event = None

    # --- future resolution (run on the loop thread) ---
    @staticmethod
    def _resolve_ok(fut: asyncio.Future, val: Any) -> None:
        if not fut.done():
            fut.set_result(val)

    @staticmethod
    def _resolve_exc(fut: asyncio.Future, exc: BaseException) -> None:
        if not fut.done():
            fut.set_exception(exc)

    @staticmethod
    def _resolve_cancelled(fut: asyncio.Future) -> None:
        if not fut.done():
            fut.set_exception(Cancelled())