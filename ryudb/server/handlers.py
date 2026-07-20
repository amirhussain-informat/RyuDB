"""Request dispatch — turns a parsed JSON request into response frames.

Every engine/catalog call is submitted to the single worker (``server.submit``)
so it serializes with any in-flight query and is cancellable while pending. The
``_cancellable`` helper wires the submit -> register-pending -> await ->
classify-fault loop shared by all ops that touch the engine or catalog.

Op set (Phase 1):
  sql      run a statement (SELECT -> Arrow result; write -> rows_affected;
           txn/snapshot/restore/CREATE-FROM -> ok). With ``cursor: true`` the
           SELECT's full result is frozen as a server-side cursor and the first
           page is returned (the rest via ``fetch``).
  fetch    page a result cursor (cursor_id + offset + limit -> Arrow slice)
  close    drop a result cursor (idempotent)
  explain  structured plan tree (parse+optimize -> JSON)
  catalog  list tables + row counts + columns
  table    one table's columns/types/constraints/paths/row_count
  sample   SELECT * FROM <t> LIMIT n  (validated table name)
  export   run a SELECT and stream the FULL result as a binary file (Parquet;
           no in-browser writer for apache-arrow 17, so serialized server-side)
  upload   ingest a parquet file sent over the wire (two-frame text+binary);
           the connection loop reads the binary payload, then this op persists
           it to <data>/uploads and registers the table (browser file-upload)
  admin    drop / rename / alter / register / checkpoint / snapshot / restore /
           clear_cache
  cancel   drop pending requests on this connection, or raise CancelledByUser
           at the next plan-node boundary of a request already in flight
  history  recent sql invocations (server-side ring buffer)
"""

from __future__ import annotations

import time
import uuid
from typing import Any

import cudf
import pandas as pd

from .errors import classify
from .explain import build_plan, plan_tree
from .protocol import (
    ProtocolError,
    column_meta,
    df_to_arrow,
    dumps_json,
    error_frame,
    table_to_ipc,
)
from .worker import Cancelled
from ..exec.executor import CancelledByUser


# --------------------------------------------------------------------------- #
# send helpers
# --------------------------------------------------------------------------- #

async def _send_json(ws, obj: dict[str, Any]) -> None:
    """Send one text frame under the per-connection send lock so it cannot
    interleave with another request's meta+binary pair."""
    lock = getattr(ws, "_ryudb_send_lock", None)
    if lock is not None:
        async with lock:
            await ws.send(dumps_json(obj))
    else:
        await ws.send(dumps_json(obj))


def _require(req: dict[str, Any], key: str, typ: type) -> Any:
    """Extract a required typed field; raise ProtocolError if missing/wrong."""
    val = req.get(key)
    if not isinstance(val, typ) or (typ is str and not val):
        raise ProtocolError(f"op {req.get('op')!r} requires a {typ.__name__} {key!r}")
    return val


async def _cancellable(server, ws, rid: Any, fn, *args) -> tuple[bool, Any, float]:
    """Submit ``fn(*args)`` to the worker as a cancellable request, await it,
    and on fault send the appropriate frame. Returns ``(ok, result, duration_ms)``;
    ``ok=False`` means a cancelled/error frame was already sent and the caller
    should return."""
    t0 = time.perf_counter()
    fut, cancel = server.submit(fn, *args)
    server.register_pending(ws, rid, cancel)
    try:
        result = await fut
    except (Cancelled, CancelledByUser):
        # Cancelled: dropped while pending (never started). CancelledByUser:
        # raised at a plan-node boundary of a request already in flight. Both
        # surface to the client as the same `cancelled` frame.
        await _send_json(ws, {"id": rid, "op": "cancelled"})
        return False, None, 0.0
    except BaseException as exc:  # noqa: BLE001 -- classify + surface
        await _send_json(ws, error_frame(rid, exc))
        return False, None, 0.0
    finally:
        server.unregister_pending(ws, rid)
    return True, result, (time.perf_counter() - t0) * 1000.0


async def _send_result(ws, rid: Any, df, max_rows: int, duration_ms: float,
                       sql: str, server) -> None:
    """Serialize a SELECT result (cuDF or pandas) as JSON meta + one Arrow IPC
    binary frame, capping rows at ``max_rows``. Records a history entry."""
    table = df_to_arrow(df)
    total = table.num_rows
    truncated = total > max_rows
    if truncated:
        table = table.slice(0, max_rows)
    meta = {
        "id": rid, "op": "result",
        "columns": column_meta(table),
        "row_count": total, "returned": table.num_rows,
        "truncated": truncated, "duration_ms": round(duration_ms, 3),
        "frame_count": 1,
    }
    # Hold the send lock across both frames so the binary IPC frame stays
    # glued to its meta frame (no other response may slip between them).
    lock = getattr(ws, "_ryudb_send_lock", None)
    if lock is not None:
        async with lock:
            await ws.send(dumps_json(meta))
            await ws.send(table_to_ipc(table))
    else:
        await ws.send(dumps_json(meta))
        await ws.send(table_to_ipc(table))
    server.record_history(rid, sql, duration_ms, total, "select")


async def _send_arrow(ws, rid: Any, table, total: int, offset: int,
                      truncated: bool, duration_ms: float, server,
                      cursor_id: str | None = None,
                      cursor_fallback: bool = False) -> None:
    """Serialize a pyarrow Table slice as a JSON ``result`` meta + one Arrow IPC
    binary frame under the per-connection send lock. ``row_count`` is the true
    total (the cursor's full size or the uncapped result size); ``returned`` is
    the rows in ``table`` (the slice actually sent). When ``cursor_id`` is set
    the meta also carries ``cursor_id`` + ``offset`` so the client can issue
    follow-up ``fetch`` ops. ``cursor_fallback`` marks a cursor request that
    exceeded ``--max-cursor-rows`` and was served as a plain truncated result
    (``cursor: false, reason: "too_large"``) so the client does not retry."""
    meta: dict[str, Any] = {
        "id": rid, "op": "result",
        "columns": column_meta(table),
        "row_count": total, "returned": table.num_rows,
        "truncated": truncated, "duration_ms": round(duration_ms, 3),
        "frame_count": 1,
    }
    if cursor_id is not None:
        meta["cursor_id"] = cursor_id
        meta["offset"] = offset
    elif cursor_fallback:
        meta["cursor"] = False
        meta["reason"] = "too_large"
    lock = getattr(ws, "_ryudb_send_lock", None)
    if lock is not None:
        async with lock:
            await ws.send(dumps_json(meta))
            await ws.send(table_to_ipc(table))
    else:
        await ws.send(dumps_json(meta))
        await ws.send(table_to_ipc(table))


async def _send_bytes(ws, rid: Any, fmt: str, data: bytes, row_count: int,
                      duration_ms: float, sql: str, server) -> None:
    """Serialize a raw binary payload (e.g. Parquet bytes) as a JSON ``export``
    meta + one binary frame under the per-connection send lock. Same meta-then-
    binary glue as ``_send_arrow`` but the frame is NOT Arrow IPC — the client
    keeps it as a blob (it cannot decode Parquet in-browser). Records a history
    entry tagged ``export``."""
    meta = {
        "id": rid, "op": "export", "format": fmt,
        "row_count": row_count, "byte_count": len(data),
        "duration_ms": round(duration_ms, 3),
    }
    lock = getattr(ws, "_ryudb_send_lock", None)
    if lock is not None:
        async with lock:
            await ws.send(dumps_json(meta))
            await ws.send(data)
    else:
        await ws.send(dumps_json(meta))
        await ws.send(data)
    server.record_history(rid, sql, duration_ms, row_count, "export")


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #

async def handle_request(req: dict[str, Any], ws, server) -> None:
    rid = req.get("id")
    op = req.get("op")
    try:
        if op == "sql":
            await _op_sql(req, ws, server)
        elif op == "fetch":
            await _op_fetch(req, ws, server)
        elif op == "close":
            await _op_close(req, ws, server)
        elif op == "explain":
            await _op_explain(req, ws, server)
        elif op == "catalog":
            await _op_catalog(req, ws, server)
        elif op == "table":
            await _op_table(req, ws, server)
        elif op == "profile":
            await _op_profile(req, ws, server)
        elif op == "sample":
            await _op_sample(req, ws, server)
        elif op == "export":
            await _op_export(req, ws, server)
        elif op == "admin":
            await _op_admin(req, ws, server)
        elif op == "upload":
            # upload is a two-frame request handled in the connection loop
            # (Server._handler reads the binary payload inline before
            # dispatching _op_upload); it never reaches this dispatch.
            raise ProtocolError("upload requires a binary payload frame; "
                                "handled by the connection loop")
        elif op == "cancel":
            await _op_cancel(req, ws, server)
        elif op == "history":
            await _op_history(req, ws, server)
        else:
            raise ProtocolError(f"unknown op {op!r}")
    except ProtocolError as exc:
        await _send_json(ws, {"id": rid, "op": "error", "kind": "protocol",
                             "message": str(exc)})
    except Exception as exc:  # noqa: BLE001 -- a handler-side fault (not engine)
        kind, message, _ = classify(exc)
        await _send_json(ws, {"id": rid, "op": "error", "kind": kind,
                             "message": message})


# --------------------------------------------------------------------------- #
# ops
# --------------------------------------------------------------------------- #

async def _op_sql(req, ws, server) -> None:
    rid = req.get("id")
    sql = _require(req, "sql", str)
    max_rows = req.get("max_rows", server.max_rows)
    if not isinstance(max_rows, int) or max_rows <= 0:
        raise ProtocolError("max_rows must be a positive int")
    want_cursor = bool(req.get("cursor", False))
    # Route through server.run_sql so the connection's transaction is loaded as
    # the engine's current txn for this call (per-connection MVCC isolation).
    def run():
        return server.run_sql(ws, sql)
    ok, result, dur = await _cancellable(server, ws, rid, run)
    if not ok:
        server.record_history(rid, sql, 0.0, 0, "cancelled")
        return
    if isinstance(result, (cudf.DataFrame, pd.DataFrame)):
        if want_cursor:
            # Freeze the full result as a host-memory pa.Table and register a
            # cursor; the first page (max_rows rows) is sent now, the rest via
            # ``fetch``. The GPU frame is freed once df_to_arrow copies to host.
            full = df_to_arrow(result)
            total = full.num_rows
            if total > server.max_cursor_rows:
                # Too large to hold as a cursor — serve as a plain truncated
                # result and tell the client not to retry with a cursor.
                page = full.slice(0, max_rows)
                await _send_arrow(ws, rid, page, total, 0, total > page.num_rows,
                                  dur, server, cursor_fallback=True)
                server.record_history(rid, sql, dur, total, "select")
                return
            cursor_id = server.open_cursor(ws, full, sql)
            page = full.slice(0, max_rows)
            await _send_arrow(ws, rid, page, total, 0, total > page.num_rows,
                              dur, server, cursor_id=cursor_id)
            server.record_history(rid, sql, dur, total, "select")
        else:
            await _send_result(ws, rid, result, max_rows, dur, sql, server)
    elif isinstance(result, int):
        await _send_json(ws, {"id": rid, "op": "write", "rows_affected": result,
                              "duration_ms": round(dur, 3)})
        server.record_history(rid, sql, dur, result, "write")
    elif result is None:
        await _send_json(ws, {"id": rid, "op": "ok", "duration_ms": round(dur, 3)})
        server.record_history(rid, sql, dur, 0, "control")
    else:  # defensive: a future return shape we don't model yet
        await _send_json(ws, {"id": rid, "op": "ok", "detail": str(result),
                              "duration_ms": round(dur, 3)})
        server.record_history(rid, sql, dur, 0, "other")


async def _op_fetch(req, ws, server) -> None:
    """Page a result cursor opened by a ``sql`` request with ``cursor: true``.
    Slices the frozen host-memory pa.Table at [offset, offset+limit) and sends it
    as a ``result`` meta (with ``cursor_id``/``offset``) + binary frame."""
    rid = req.get("id")
    cursor_id = _require(req, "cursor_id", str)
    offset = req.get("offset", 0)
    if not isinstance(offset, int) or offset < 0:
        raise ProtocolError("fetch 'offset' must be a non-negative int")
    limit = req.get("limit", server.max_rows)
    if not isinstance(limit, int) or limit <= 0:
        raise ProtocolError("fetch 'limit' must be a positive int")
    cur = server.get_cursor(ws, cursor_id)
    if cur is None:
        raise ProtocolError(f"unknown cursor {cursor_id!r}")
    total = cur["total"]
    end = min(offset + limit, total)
    page = cur["table"].slice(offset, end - offset) if end > offset else \
        cur["table"].slice(offset, 0)
    await _send_arrow(ws, rid, page, total, offset, end < total, 0.0, server,
                      cursor_id=cursor_id)
    server.record_history(rid, cur["sql"], 0.0, page.num_rows, "fetch")


async def _op_close(req, ws, server) -> None:
    """Drop a result cursor. Idempotent: an unknown cursor id still returns
    ``ok`` (it may have been closed already, or auto-closed on a prior
    disconnect)."""
    rid = req.get("id")
    cursor_id = _require(req, "cursor_id", str)
    server.close_cursor(ws, cursor_id)
    await _send_json(ws, {"id": rid, "op": "ok"})


async def _op_explain(req, ws, server) -> None:
    rid = req.get("id")
    sql = _require(req, "sql", str)

    def build():
        with server.engine.read_locked():
            plan = build_plan(sql, server.catalog)
            return plan_tree(plan, server.catalog.stats_dict())

    ok, tree, _ = await _cancellable(server, ws, rid, build)
    if not ok:
        return
    await _send_json(ws, {"id": rid, "op": "plan", "tree": tree})


async def _op_catalog(req, ws, server) -> None:
    rid = req.get("id")

    def build():
        with server.engine.read_locked():
            schema = server.catalog.schema_dict()
            stats = server.catalog.stats_dict()
            return [{"name": t, "row_count": stats.get(t, 0), "columns": list(schema[t])}
                    for t in schema]

    ok, tables, _ = await _cancellable(server, ws, rid, build)
    if not ok:
        return
    await _send_json(ws, {"id": rid, "op": "catalog", "tables": tables})


async def _op_table(req, ws, server) -> None:
    rid = req.get("id")
    name = _require(req, "name", str)

    def build():
        with server.engine.read_locked():
            info = server.catalog.get(name)  # KeyError -> runtime error frame
            return {
                "name": info.name,
                "columns": [{"name": c,
                             "type": str(info.types.get(c)),
                             "nullable": c not in info.constraints.not_null}
                            for c in info.columns],
                "constraints": info.constraints.to_dict(),
                "paths": list(info.paths),
                "row_count": info.row_count,
            }

    ok, info, _ = await _cancellable(server, ws, rid, build)
    if not ok:
        return
    await _send_json(ws, {"id": rid, "op": "table", **info})


def _prof_scalar(v):
    """Coerce a cuDF/numpy/pandas scalar to a JSON-safe Python value (the
    response goes through ``dumps_json``, but interval bounds and value_counts
    entries arrive as numpy/cuDF scalars that are not all auto-coerced)."""
    if v is None:
        return None
    try:
        if hasattr(v, "item"):
            v = v.item()
    except Exception:  # noqa: BLE001 -- some scalars raise on .item()
        pass
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float, str)):
        return v
    try:
        if hasattr(v, "isoformat"):
            return v.isoformat()
    except Exception:  # noqa: BLE001
        pass
    return str(v)


def _profile_column(s, row_count: int, top_k: int) -> dict:
    """Per-column statistics for a cuDF Series ``s`` computed on the GPU.
    Returns a JSON-ready dict. Each potentially-failing reduction is guarded so
    a dtype that doesn't support it (e.g. mean on a string column) just yields
    ``None`` rather than aborting the whole profile."""
    null_count = int(s.isnull().sum())
    distinct = int(s.nunique(dropna=True))
    null_pct = (null_count / row_count) if row_count else 0.0
    out: dict = {
        "type": str(s.dtype),
        "row_count": row_count,
        "null_count": null_count,
        "null_pct": round(null_pct, 6),
        "distinct": distinct,
    }
    # min/max work for numeric and string/categorical columns in cuDF.
    try:
        out["min"] = _prof_scalar(s.min())
        out["max"] = _prof_scalar(s.max())
    except Exception:  # noqa: BLE001
        out["min"] = None
        out["max"] = None
    # Numeric-only reductions + a 10-bucket equal-width histogram. These raise
    # on non-numeric dtypes -> caught -> omitted.
    if row_count and distinct:
        try:
            out["mean"] = float(s.mean())
        except Exception:  # noqa: BLE001
            out["mean"] = None
        try:
            out["stddev"] = float(s.std(ddof=0))
        except Exception:  # noqa: BLE001
            out["stddev"] = None
        # 10-bucket equal-width histogram, computed on the GPU via an arithmetic
        # bucket id + value_counts (NOT cudf.cut — this cuDF build's interval
        # categories raise NotImplementedError when iterated/converted).
        try:
            snn = s.dropna()
            mn = float(snn.min())
            mx = float(snn.max())
            width = (mx - mn) or 1.0
            # Values are >= mn so the division is non-negative; int64 truncation
            # is floor here. clip handles the max value landing exactly on the
            # top edge (bucket 10 -> 9).
            bid = ((snn - mn) / width).astype("int64").clip(0, 9)
            vc = bid.value_counts()
            counts = {int(k): int(v) for k, v in
                      zip(vc.index.to_pandas().tolist(), vc.to_pandas().tolist())}
            out["histogram"] = [
                {"lo": mn + i * width, "hi": mn + (i + 1) * width,
                 "count": counts.get(i, 0)}
                for i in range(10)
            ]
        except Exception:  # noqa: BLE001 -- non-numeric / empty -> skip
            out["histogram"] = None
    else:
        out["mean"] = None
        out["stddev"] = None
        out["histogram"] = None
    # Top-K most frequent values. Only for low-NDV columns: a full value_counts
    # groupby on a continuous column is wasteful and the histogram already
    # covers its distribution.
    if distinct and distinct <= 1000 and top_k > 0:
        try:
            vc = s.value_counts(dropna=True).head(top_k)
            vals = vc.index.to_pandas().tolist()
            cnts = vc.to_pandas().tolist()
            out["top"] = [{"value": _prof_scalar(v), "count": int(c)}
                          for v, c in zip(vals, cnts)]
        except Exception:  # noqa: BLE001
            out["top"] = None
    else:
        out["top"] = None
    return out


async def _op_profile(req, ws, server) -> None:
    rid = req.get("id")
    name = _require(req, "name", str)
    top_k = req.get("top_k", 10)
    if not isinstance(top_k, int) or top_k <= 0:
        raise ProtocolError("top_k must be a positive int")

    def build():
        with server.engine.read_locked():
            info = server.catalog.get(name)  # KeyError -> runtime error frame
            # One full scan (profiling is inherently a full-column read); index
            # each column off the resulting cuDF DataFrame for the GPU stats.
            df = server.engine._scan(name, None)
            row_count = len(df)
            cols = []
            for c in info.columns:
                s = df[c] if c in df.columns else df.iloc[:, 0][:0]
                cols.append({"name": c, **_profile_column(s, row_count, top_k)})
            return {"name": info.name, "row_count": row_count, "columns": cols}

    ok, profile, _ = await _cancellable(server, ws, rid, build)
    if not ok:
        return
    await _send_json(ws, {"id": rid, "op": "profile", **profile})


async def _op_sample(req, ws, server) -> None:
    rid = req.get("id")
    name = _require(req, "name", str)
    n = req.get("n", 100)
    if not isinstance(n, int) or n <= 0:
        raise ProtocolError("n must be a positive int")
    # Validate the table name (allowlist) before interpolating it into SQL —
    # this is a local single-user server, but never build SQL from untrusted input.
    if not name.isidentifier():
        raise ProtocolError(f"invalid table name {name!r}")
    if name not in server.catalog.schema_dict():
        await _send_json(ws, {"id": rid, "op": "error", "kind": "runtime",
                             "message": f"unknown table {name!r}"})
        return
    sql = f"SELECT * FROM {name} LIMIT {n}"
    max_rows = server.max_rows
    # A sample is a read; route through run_sql so it respects an open txn on
    # this connection (sees the txn's snapshot + read-your-writes buffer).
    def run():
        return server.run_sql(ws, sql)
    ok, result, dur = await _cancellable(server, ws, rid, run)
    if not ok:
        return
    if isinstance(result, (cudf.DataFrame, pd.DataFrame)):
        await _send_result(ws, rid, result, max_rows, dur, sql, server)
    else:
        await _send_json(ws, {"id": rid, "op": "ok", "duration_ms": round(dur, 3)})


async def _op_export(req, ws, server) -> None:
    """Run a SELECT and stream the FULL result back as a binary file in a
    client-chosen format. v1 supports only Parquet (CSV/JSON/Arrow are already
    produced in-browser from the paged result; Parquet has no in-browser writer
    for apache-arrow 17, so it is serialized server-side with pyarrow.parquet).

    Runs the query on the worker thread (cancellable, serialized with other
    engine work), copies the cuDF frame to a host pyarrow.Table, writes it to a
    Parquet buffer, and sends one ``export`` meta + one binary frame. The result
    is NOT capped at ``max_rows`` — the point is to export everything — but
    ``--max-export-rows`` bounds host RAM (a larger result errors out instead of
    OOMing). Non-SELECT statements and bad formats surface as error frames."""
    rid = req.get("id")
    sql = _require(req, "sql", str)
    fmt = req.get("format", "parquet")
    if fmt != "parquet":
        raise ProtocolError("export 'format' must be 'parquet'")

    def run():
        result = server.run_sql(ws, sql)
        if not isinstance(result, (cudf.DataFrame, pd.DataFrame)):
            raise ProtocolError("export requires a SELECT statement")
        table = df_to_arrow(result)
        if table.num_rows > server.max_export_rows:
            raise RuntimeError(
                f"export too large ({table.num_rows} rows; limit "
                f"{server.max_export_rows}). Narrow the query or raise "
                f"--max-export-rows.")
        import io
        import pyarrow.parquet as pq
        buf = io.BytesIO()
        pq.write_table(table, buf)
        return buf.getvalue(), table.num_rows

    ok, payload, dur = await _cancellable(server, ws, rid, run)
    if not ok:
        server.record_history(rid, sql, 0.0, 0, "cancelled")
        return
    data, row_count = payload
    await _send_bytes(ws, rid, fmt, data, row_count, dur, sql, server)


async def _op_admin(req, ws, server) -> None:
    rid = req.get("id")
    action = _require(req, "action", str)
    args = req.get("args", {})
    if not isinstance(args, dict):
        raise ProtocolError("admin 'args' must be an object")

    def run():
        return _run_admin(server, action, args)

    ok, detail, _ = await _cancellable(server, ws, rid, run)
    if not ok:
        return
    await _send_json(ws, {"id": rid, "op": "ok", "detail": detail})


def _run_admin(server, action: str, args: dict[str, Any]) -> dict[str, Any]:
    """Execute an admin action against the catalog/engine. Runs on a worker
    thread under the engine's exclusive write lock (every admin op mutates
    shared state -- catalog register/drop/rename/alter, engine checkpoint/
    snapshot/restore, or the caches -- so it must serialize against reads and
    other writes). Raises on bad action/args (-> error frame)."""
    with server.engine.write_locked():
        cat = server.catalog
        eng = server.engine
        if action == "register":
            return {"registered": cat.register(_arg(args, "table", str),
                                               _arg(args, "path", str)).name}
        if action == "drop":
            cat.drop_table(_arg(args, "table", str))
            return {"dropped": True}
        if action == "rename":
            cat.rename_table(_arg(args, "old", str), _arg(args, "new", str))
            return {"renamed": True}
        if action == "alter":
            return _run_alter(cat, args)
        if action == "checkpoint":
            # Checkpoint folds the committed delta into the base and clears the
            # delta; base rows are snapshot-agnostic (_ins_ts=0), so a live txn
            # whose snapshot predates the checkpointed commits would suddenly see
            # them. Refuse while any connection has an open txn. The write lock
            # also waits for any in-flight SELECTs to finish before rewriting
            # the base + clearing the delta, so a concurrent read never sees a
            # half-checkpointed state.
            if server.has_open_txn():
                raise RuntimeError("cannot checkpoint while a transaction is open "
                                   "(COMMIT/ROLLBACK on all connections first)")
            return {"tables": eng.checkpoint()}
        if action == "snapshot":
            eng.snapshot(_arg(args, "name", str))
            return {"snapshot": True}
        if action == "restore":
            # Restore rewinds the global commit counter and discards delta tail;
            # a live txn's snapshot would be invalidated. Refuse while any txn is
            # open (the engine's own _txn check can't see other connections' txns
            # in the server, where _txn is a per-thread, per-request pointer).
            if server.has_open_txn():
                raise RuntimeError("cannot restore while a transaction is open "
                                   "(COMMIT/ROLLBACK on all connections first)")
            if "name" in args:
                eng.restore(_arg(args, "name", str))
                return {"restored": True}
            ts = args.get("ts")
            if not isinstance(ts, int):
                raise ProtocolError("restore requires 'name' or int 'ts'")
            eng.restore_to(ts)
            return {"restored_to": ts}
        if action == "clear_cache":
            which = args.get("which", "both")
            if which in ("scan", "both"):
                eng.clear_scan_cache()
            if which in ("code", "both"):
                eng.clear_code_cache()
            return {"cleared": which}
        raise ProtocolError(f"unknown admin action {action!r}")


async def _op_upload(req, payload: bytes, ws, server) -> None:
    """Handle a two-frame ``upload`` request: the binary parquet payload was
    already read by ``Server._handler`` and passed in as ``payload``. Validate
    the meta + size, then persist + register on the worker thread (mirroring
    ``_op_admin``). Wrapped in the same try/except as ``handle_request`` so a
    ``ProtocolError`` surfaces as kind ``protocol`` (``_cancellable`` would
    otherwise classify it as ``runtime``)."""
    rid = req.get("id")
    try:
        name = _require(req, "name", str)
        fmt = req.get("format", "parquet")
        if fmt != "parquet":
            raise ProtocolError("upload 'format' must be 'parquet'")
        if not payload:
            raise ProtocolError("upload payload is empty")
        if len(payload) > server.max_upload_bytes:
            raise RuntimeError(
                f"upload too large ({len(payload)} bytes; "
                f"limit {server.max_upload_bytes})")

        def run():
            return _run_upload(server, name, payload)

        ok, detail, dur = await _cancellable(server, ws, rid, run)
        if not ok:
            return
        await _send_json(ws, {"id": rid, "op": "ok", "detail": detail,
                             "duration_ms": round(dur, 3)})
    except ProtocolError as exc:
        await _send_json(ws, {"id": rid, "op": "error", "kind": "protocol",
                             "message": str(exc)})
    except Exception as exc:  # noqa: BLE001 -- a handler-side fault
        kind, message, _ = classify(exc)
        await _send_json(ws, {"id": rid, "op": "error", "kind": kind,
                             "message": message})


def _run_upload(server, name: str, payload: bytes) -> dict[str, Any]:
    """Persist the uploaded parquet bytes to the catalog's ``uploads`` dir and
    register the table. Runs on a worker thread; the file write is filesystem
    I/O done before taking the engine's write lock, then ``register`` mutates
    the catalog under the lock (serializing against reads/writes, like
    ``_run_admin``). On register failure the file is removed so a bad upload
    leaves no orphan. Raises on a bad payload (-> error frame)."""
    import os
    base = server.catalog.data_dir or "."
    uploads_dir = os.path.join(base, "uploads")
    os.makedirs(uploads_dir, exist_ok=True)
    path = os.path.join(uploads_dir, f"{uuid.uuid4().hex}.parquet")
    with open(path, "wb") as f:
        f.write(payload)
    try:
        with server.engine.write_locked():
            info = server.catalog.register(name, path)
    except Exception:
        try:
            os.remove(path)
        except OSError:
            pass
        raise
    return {"registered": info.name, "row_count": info.row_count,
            "columns": len(info.columns), "path": path}


def _run_alter(cat, args: dict[str, Any]) -> dict[str, Any]:
    table = _arg(args, "table", str)
    kind = _arg(args, "kind", str)
    if kind == "pk":
        cat.set_primary_key(table, _arg_list(args, "cols"))
        return {"set": "primary_key"}
    if kind == "notnull":
        cat.set_not_null(table, _arg(args, "col", str), True)
        return {"set": "not_null"}
    if kind == "notnull_off":
        cat.set_not_null(table, _arg(args, "col", str), False)
        return {"set": "not_null_off"}
    if kind == "unique":
        cat.set_unique(table, _arg_list(args, "cols"))
        return {"set": "unique"}
    if kind == "default":
        cat.set_default(table, _arg(args, "col", str), args.get("value"))
        return {"set": "default"}
    if kind == "drop_default":
        cat.drop_default(table, _arg(args, "col", str))
        return {"set": "drop_default"}
    raise ProtocolError(f"unknown alter kind {kind!r}")


def _arg(args: dict[str, Any], key: str, typ: type) -> Any:
    val = args.get(key)
    if not isinstance(val, typ) or (typ is str and not val):
        raise ProtocolError(f"admin action requires {typ.__name__} arg {key!r}")
    return val


def _arg_list(args: dict[str, Any], key: str) -> list[str]:
    val = args.get(key)
    if not isinstance(val, list) or not all(isinstance(x, str) for x in val) or not val:
        raise ProtocolError(f"admin action requires a non-empty list-of-str arg {key!r}")
    return val


async def _op_cancel(req, ws, server) -> None:
    rid = req.get("id")
    targets = req.get("targets", [])
    if not isinstance(targets, list) or not all(isinstance(t, str) for t in targets):
        raise ProtocolError("cancel 'targets' must be a list of request ids")
    cancelled, not_found = server.cancel_pending(ws, targets)
    await _send_json(ws, {
        "id": rid, "op": "ok",
        "detail": {"cancelled": cancelled, "not_found": not_found},
        "note": "pending requests are dropped; in-flight requests raise "
                "CancelledByUser at the next plan-node boundary",
    })


async def _op_history(req, ws, server) -> None:
    rid = req.get("id")
    await _send_json(ws, {"id": rid, "op": "history", "entries": list(server.history)})