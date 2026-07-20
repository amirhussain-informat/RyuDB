"""The RyuDB server — owns one Engine, accepts WebSocket clients.

Lifecycle: ``Server(data_dir, host, port)`` builds a ``Catalog`` + ``Engine``
and starts the single engine worker thread; ``await server.serve()`` binds the
WebSocket server and runs until cancelled. ``server.stop()`` closes the
listener and joins the worker.

Connection model: one persistent WebSocket per client. Text frames are JSON
requests; binary frames are rejected (clients send no binary). Each connection
tracks its pending cancellable requests so a ``cancel`` op (or a disconnect)
can drop the ones that have not started yet.

Single logical session (documented v1 limitation): there is one ``Engine``, so
transaction state (``_txn`` / ``_commit_ts`` / ``_snapshots``) is shared across
*all* connections — ``BEGIN`` on one connection and ``COMMIT`` on another
interleave. Fine for a local single-user console; multi-session isolation is a
later phase.
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
    ) -> None:
        self.data_dir = data_dir
        self.host = host
        self.port = port
        self.max_rows = max_rows
        self.catalog = Catalog(data_dir)
        self.engine = Engine(self.catalog)
        self.worker = EngineWorker(self.engine)
        # per-connection pending cancel events: id(ws) -> {request_id -> Event}
        self._pending: dict[int, dict[str, threading.Event]] = {}
        # per-connection in-flight request tasks (so cancel/disconnect can drop them)
        self._tasks: dict[int, set] = {}
        self._history: collections.deque[dict[str, Any]] = collections.deque(
            maxlen=history_size
        )
        self._srv: Any = None

    # -- engine submission (called from the asyncio loop thread) --
    def submit(self, fn: Callable[..., Any], *args: Any):
        loop = asyncio.get_running_loop()
        return self.worker.submit(loop, fn, *args)

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
            log.info("single logical session; cancel drops pending requests and "
                     "raises CancelledByUser at the next node boundary of an "
                     "in-flight request (a single long cuDF call is not mid-call "
                     "interruptible)")
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
        # Per-connection send lock: keeps a result's meta+binary frames atomic
        # when several requests are in flight on one connection (pipelined
        # clients). Attached to ``ws`` so the handlers can reach it without an
        # extra parameter.
        ws._ryudb_send_lock = asyncio.Lock()
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
            log.info("connection closed")