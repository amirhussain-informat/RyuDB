"""The RyuDB server — owns one Engine, accepts WebSocket clients.

Lifecycle: ``Server(data_dir, host, port)`` builds a ``Catalog`` + ``Engine``
and starts the single engine worker thread; ``await server.serve()`` binds the
WebSocket server and runs until cancelled. ``server.stop()`` closes the
listener and joins the worker.

Connection model: one persistent WebSocket per client. Text frames are JSON
requests; binary frames are rejected (clients send no binary). Each connection
tracks its pending cancellable requests so a ``cancel`` op (or a disconnect)
can drop the ones that have not started yet.

Per-connection transactions (MVCC isolation): there is one ``Engine`` (one
worker thread, so all engine access serializes), but each connection owns its
own in-flight transaction. The engine's ``_txn`` is a *transient per-request*
pointer, not a global: ``run_sql`` loads the connection's txn (or ``None``)
into ``engine._txn`` before the call and reads it back after (``BEGIN`` leaves
a new ``Transaction`` there; ``COMMIT``/``ROLLBACK`` leave ``None``; any other
statement leaves it unchanged). So ``BEGIN`` on connection B no longer fails
while connection A has an open txn, and a txn's frozen ``snapshot_ts`` gives
real cross-session snapshot isolation — the delta store is already
timestamp-versioned (``batches_at(snapshot_ts)``), so B's commit after A's
``BEGIN`` is invisible to A. The engine internals are unchanged; in-process
callers (CLI/tests) keep the old persistent-``_txn`` behaviour on their own
instances. A disconnect drops the connection's open txn (its buffered writes
were never flushed to the delta/WAL, so dropping the reference *is* the
rollback). Global admin ops that rewrite history (``checkpoint``,
``restore``/``restore_to``) are refused while any connection has an open txn —
they would invalidate live snapshots.
"""

from __future__ import annotations

import asyncio
import collections
import logging
import threading
from typing import Any, Callable

from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

from .. import Catalog, Engine
from .handlers import handle_request
from .protocol import ProtocolError, dumps_json, loads_request
from .worker import EngineWorker

log = logging.getLogger("ryudb.server")


class Server:
    def __init__(
        self,
        data_dir: str,
        host: str = "127.0.0.1",
        port: int = 5430,
        max_rows: int = 200_000,
        history_size: int = 500,
        n_workers: int = 1,
    ) -> None:
        self.data_dir = data_dir
        self.host = host
        self.port = port
        self.max_rows = max_rows
        self.catalog = Catalog(data_dir)
        self.engine = Engine(self.catalog)
        # Worker pool size: 1 preserves the original single-worker semantics
        # (the RW lock is uncontended, the per-thread _txn/cancel_event slots
        # behave as a single slot). n_workers > 1 lets SELECTs run concurrently
        # (the Engine's read/write lock keeps writes serialized); each worker
        # thread gets its own per-thread _txn / cancel_event slot.
        self.worker = EngineWorker(self.engine, n_workers=n_workers)
        # per-connection pending cancel events: id(ws) -> {request_id -> Event}
        self._pending: dict[int, dict[str, threading.Event]] = {}
        # per-connection in-flight request tasks (so cancel/disconnect can drop them)
        self._tasks: dict[int, set] = {}
        # live connections (id(ws) -> ws) so has_open_txn can enumerate the
        # per-connection txns stored on each ws (see run_sql). A disconnect
        # removes its entry; an in-flight request may still hold the ws, but
        # that txn is then being rolled back and is dropped when the ws is GC'd.
        self._conns: dict[int, Any] = {}
        self._history: collections.deque[dict[str, Any]] = collections.deque(
            maxlen=history_size
        )
        self._srv: Any = None

    # -- engine submission (called from the asyncio loop thread) --
    def submit(self, fn: Callable[..., Any], *args: Any):
        loop = asyncio.get_running_loop()
        return self.worker.submit(loop, fn, *args)

    # -- per-connection transaction context (called on the worker thread) --
    def run_sql(self, ws, sql: str):
        """Run ``engine.sql(sql)`` with THIS connection's transaction loaded as
        the engine's current txn, then read it back. ``engine._txn`` is a
        per-thread slot (threading.local), so on a worker pool each request
        touches only its own worker's slot -- concurrent requests no longer
        clobber a single shared _txn. The txn is held on the ws
        (``_ryudb_txn``) between requests and dropped with the ws on disconnect
        (its buffered writes were never flushed, so dropping the reference is
        the rollback).

        ``BEGIN`` leaves a new ``Transaction`` in ``engine._txn`` -> stored on
        the ws. ``COMMIT``/``ROLLBACK`` leave ``None`` -> the ws txn is cleared.
        Any other statement leaves ``engine._txn`` untouched -> the ws txn is
        unchanged. The engine's own ``_begin``/``_commit``/``_rollback`` raise
        on misuse (BEGIN inside a txn, COMMIT without one) exactly as before."""
        self.engine._txn = getattr(ws, "_ryudb_txn", None)
        try:
            return self.engine.sql(sql)
        finally:
            ws._ryudb_txn = self.engine._txn
            self.engine._txn = None

    def has_open_txn(self) -> bool:
        """True if any live connection has an open transaction. Used to refuse
        the global history-rewriting admin ops (``checkpoint``, ``restore``)
        while a snapshot is live — they would invalidate it. Read on the worker
        thread; a briefly-stale view of a disconnecting connection is safe (it
        only makes the guard refuse for a moment, never incorrectly allow)."""
        return any(
            getattr(w, "_ryudb_txn", None) is not None
            for w in list(self._conns.values())
        )

    # -- pending-request tracking (for cancel + disconnect) --
    def register_pending(self, ws, rid: Any, cancel: threading.Event) -> None:
        if rid is None:
            return
        self._pending.setdefault(id(ws), {})[str(rid)] = cancel

    def unregister_pending(self, ws, rid: Any) -> None:
        if rid is None:
            return
        self._pending.get(id(ws), {}).pop(str(rid), None)

    def cancel_pending(self, ws, targets: list[str]) -> tuple[list[str], list[str]]:
        """Set the cancel flag for each target on this connection. Returns
        (cancelled, not_found). A request stays registered in ``_pending`` from
        submit until it completes, so a still-pending request AND an in-flight
        request are both found here and both get their flag set:

        - pending (not started): the worker drops it on dequeue -> ``cancelled``.
        - in-flight (running): the engine raises ``CancelledByUser`` at the next
          plan-node boundary -> ``cancelled`` (unless it already passed its last
          node and finished, in which case it resolves normally — the inherent
          race of cooperative cancel).

        A target not found has already completed or never existed —
        indistinguishable here, which is honest: the original request's own
        response tells the client whether it was dropped (``cancelled``) or ran
        to completion."""
        pending = self._pending.get(id(ws), {})
        cancelled, not_found = [], []
        for t in targets:
            ev = pending.get(t)
            if ev is not None:
                ev.set()
                cancelled.append(t)
            else:
                not_found.append(t)
        return cancelled, not_found

    # -- history (loop-thread only: appended after await, read on read) --
    def record_history(
        self, rid: Any, sql: str, duration_ms: float, rows: int, kind: str
    ) -> None:
        self._history.append({
            "id": rid, "sql": sql, "duration_ms": round(duration_ms, 3),
            "rows": rows, "kind": kind,
        })

    @property
    def history(self) -> collections.deque[dict[str, Any]]:
        return self._history

    # -- serve --
    async def serve(self) -> None:
        async with serve(
            self._handler, self.host, self.port, max_size=None
        ) as srv:
            self._srv = srv
            log.info("ryudb-server listening on ws://%s:%d (data=%s)",
                     self.host, self.port, self.data_dir)
            log.info("per-connection transactions (MVCC isolation); cancel drops "
                     "pending requests and raises CancelledByUser at the next node "
                     "boundary of an in-flight request (a single long cuDF call is "
                     "not mid-call interruptible)")
            await asyncio.Future()  # run forever

    def stop(self) -> None:
        """Stop the server. Shuts down the engine worker; the listening socket
        is closed when the event loop stops (``serve()`` returns) — we do not
        call the websockets ``close()`` here because its internal close task can
        hang on a client that won't finish the handshake, and the loop is about
        to stop anyway."""
        self.worker.shutdown()

    async def _handler(self, ws) -> None:
        conn_id = id(ws)
        self._pending[conn_id] = {}
        self._tasks[conn_id] = set()
        self._conns[conn_id] = ws
        # Per-connection send lock: keeps a result's meta+binary frames atomic
        # when several requests are in flight on one connection (pipelined
        # clients). Attached to ``ws`` so the handlers can reach it without an
        # extra parameter.
        ws._ryudb_send_lock = asyncio.Lock()
        # Per-connection transaction (Transaction | None). Loaded into
        # engine._txn per request by run_sql; the engine is single-threaded so
        # the assignment needs no lock. Dropped with the ws on disconnect.
        ws._ryudb_txn = None
        peer = ws.remote_address if hasattr(ws, "remote_address") else None
        log.info("connection open from %s", peer)
        try:
            async for msg in ws:
                if isinstance(msg, (bytes, bytearray)):
                    await ws.close(code=1003, reason="binary frames not accepted")
                    return
                try:
                    req = loads_request(msg)
                except ProtocolError as exc:
                    async with ws._ryudb_send_lock:
                        await ws.send(dumps_json({"op": "error", "kind": "protocol",
                                                  "message": str(exc)}))
                    continue
                # Dispatch each request as its own task so a connection can have
                # several requests in flight (one running on the worker, the rest
                # pending in its queue) — that is what makes ``cancel`` of a
                # pending request reachable. The worker still serializes all
                # engine/catalog access, so correctness and FIFO submission order
                # are preserved; only response ordering becomes per-request.
                task = asyncio.create_task(handle_request(req, ws, self))
                self._tasks[conn_id].add(task)
                task.add_done_callback(self._tasks[conn_id].discard)
        except ConnectionClosed:
            pass
        except Exception as exc:  # noqa: BLE001 -- never let a handler fault kill the server
            log.exception("connection handler fault: %s", exc)
        finally:
            # Set every still-registered cancel event for this connection. This
            # drops pending requests (worker dequeues them set) AND raises
            # CancelledByUser at the next node boundary of an in-flight request
            # (it is still in _pending until it completes). Then cancel the
            # handler tasks (the await fut) so a disconnect aborts promptly; if
            # the engine is mid-node it finishes that node first — the one
            # remaining limit of cooperative cancel.
            for ev in self._pending.pop(conn_id, {}).values():
                ev.set()
            for task in self._tasks.pop(conn_id, set()):
                task.cancel()
            # Drop this connection's session: an open txn (in ws._ryudb_txn) is
            # rolled back by dropping the reference -- its buffered writes were
            # never flushed to the delta/WAL. An in-flight request may still
            # hold the ws and write ws._ryudb_txn back in run_sql's finally; the
            # txn is then GC'd with the ws once that request completes.
            self._conns.pop(conn_id, None)
            log.info("connection closed")