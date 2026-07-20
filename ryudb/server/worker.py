"""The engine worker pool — dispatch + pending-request cancellation.

The ``Engine``'s shared mutable state (catalog, delta, ``_commit_ts``,
``_snapshots``, ``_wal``, the scan/code/pk caches) is guarded by the Engine's
read/write lock (``_RWLock`` in ``executor.py``): SELECTs take the shared read
side and may run concurrently; writes and the global admin ops take the
exclusive write side. Per-request state that the executor reads mid-query --
the current transaction (``engine._txn``) and the in-flight cancel Event
(``engine.cancel_event``) -- is kept in per-thread slots (``threading.local``)
so each worker sees its own request's state, not a single clobbered slot.

So the server MAY run several engine calls concurrently, up to the pool size.
This module is the dispatch boundary: a pool of N dedicated threads (default 1,
preserving the old single-worker semantics; ``--workers N`` enlarges it) shares
one FIFO queue and pulls work items one at a time. Submission order within the
queue is still FIFO, but with N>1 a later item may be picked up by a free worker
before an earlier (still-running) item finishes -- so response ordering becomes
per-request, not global (match by ``id``). Correctness is unchanged: the RW lock
serializes writes and the per-thread slots isolate per-request state.

It also implements cancellation. ``ThreadPoolExecutor`` does not expose its
internal queue, so a hand-rolled worker is used instead: each submitted item
carries a ``threading.Event`` cancel flag. A worker checks it *before* starting
the call and drops the item if set (pending cancel -- only reachable when every
worker is busy, i.e. the pool is full). For a request already in flight, the
same event is published to ``engine.cancel_event`` (this worker's per-thread
slot) for the duration of the call, so the executor raises ``CancelledByUser``
at the next plan-node boundary (cooperative in-flight cancel) AND, for finer
granularity, at the top of the inner loops of long nodes (multi-aggregate,
multi-window-function, the per-row window-running host loop, per-table
checkpoint) -- narrowing the un-interruptible window from a whole node down to a
single cuDF call for those cases. A *single* long cuDF/CUDA call (a cold
``cudf.read_parquet``, one big ``groupby.agg``, ``sort_values``, a join
``merge``, or a fused C-extension kernel) is still not interruptible mid-call --
cancel fires after the current kernel returns; that is the one remaining limit
(full preemption of a single kernel would need subprocess isolation).

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
    """Owns the Engine; processes requests on a pool of N dedicated threads
    sharing one FIFO queue. Supports cancelling a request that has not started."""

    def __init__(self, engine: Any, n_workers: int = 1) -> None:
        if n_workers < 1:
            raise ValueError("n_workers must be >= 1")
        self.engine = engine
        self.n_workers = n_workers
        self._q: queue.Queue = queue.Queue()
        self._threads: list[threading.Thread] = [
            threading.Thread(target=self._run, name=f"ryudb-engine-{i}", daemon=True)
            for i in range(n_workers)
        ]
        for t in self._threads:
            t.start()

    def submit(
        self, loop: asyncio.AbstractEventLoop, fn: Callable[..., Any], *args: Any
    ) -> tuple[asyncio.Future, threading.Event]:
        """Enqueue ``fn(*args)`` on a worker thread.

        Returns ``(future, cancel_event)``: await the future for the result (or
        ``Cancelled`` / the raised exception); set the cancel event to drop the
        request if it has not started yet.
        """
        fut: asyncio.Future = loop.create_future()
        cancel = threading.Event()
        self._q.put((fn, args, loop, fut, cancel))
        return fut, cancel

    def shutdown(self) -> None:
        """Signal every worker to stop and wait briefly for them to drain."""
        for _ in self._threads:
            self._q.put(None)  # one shutdown sentinel per worker
        for t in self._threads:
            t.join(timeout=5.0)

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
            # plan-node boundary. ``engine.cancel_event`` is a per-thread slot
            # (threading.local), so each worker publishes its own in-flight
            # request's Event -- a pool of workers no longer clobber a single
            # shared slot. Cleared afterwards so the slot is fresh for the next
            # request this worker picks up.
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