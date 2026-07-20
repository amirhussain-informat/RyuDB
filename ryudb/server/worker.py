"""The single engine worker — serialization + pending-request cancellation.

The ``Engine`` is single-session / one-thread (``executor.py:152``): its
``_txn`` / ``_commit_ts`` / ``_snapshots`` are unprotected mutable state, and a
query runs to completion on the calling thread. So the server may NOT run two
engine calls concurrently. This module is the serialization boundary: a single
dedicated thread owns the ``Engine`` (and the ``Catalog`` — the engine reads it
mid-query, so catalog mutations must serialize with queries too) and pulls work
items from a FIFO queue one at a time.

It also implements *pending* request cancellation. ``ThreadPoolExecutor`` does
not expose its internal queue, so a hand-rolled worker is used instead: each
submitted item carries a ``threading.Event`` cancel flag; the worker checks it
*before* starting the call and drops the item if set. A request already in
flight cannot be interrupted (the engine has no cancel hook) — that is a
documented Phase 1 limitation; the cancel flag is a no-op for it and the
request runs to completion, resolving normally.

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
            try:
                result = fn(*args)
            except BaseException as exc:  # noqa: BLE001 -- surface every fault
                loop.call_soon_threadsafe(self._resolve_exc, fut, exc)
            else:
                loop.call_soon_threadsafe(self._resolve_ok, fut, result)

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