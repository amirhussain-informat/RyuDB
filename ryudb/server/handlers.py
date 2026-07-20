"""Request dispatch — turns a parsed JSON request into response frames.

Every engine/catalog call is submitted to the single worker (``server.submit``)
so it serializes with any in-flight query and is cancellable while pending. The
``_cancellable`` helper wires the submit -> register-pending -> await ->
classify-fault loop shared by all ops that touch the engine or catalog.

Op set (Phase 1):
  sql      run a statement (SELECT -> Arrow result; write -> rows_affected;
           txn/snapshot/restore/CREATE-FROM -> ok)
  explain  structured plan tree (parse+optimize -> JSON)
  catalog  list tables + row counts + columns
  table    one table's columns/types/constraints/paths/row_count
  sample   SELECT * FROM <t> LIMIT n  (validated table name)
  admin    drop / rename / alter / register / checkpoint / snapshot / restore /
           clear_cache
  cancel   drop pending requests on this connection, or raise CancelledByUser
           at the next plan-node boundary of a request already in flight
  history  recent sql invocations (server-side ring buffer)
"""

from __future__ import annotations

import time
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


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #

async def handle_request(req: dict[str, Any], ws, server) -> None:
    rid = req.get("id")
    op = req.get("op")
    try:
        if op == "sql":
            await _op_sql(req, ws, server)
        elif op == "explain":
            await _op_explain(req, ws, server)
        elif op == "catalog":
            await _op_catalog(req, ws, server)
        elif op == "table":
            await _op_table(req, ws, server)
        elif op == "sample":
            await _op_sample(req, ws, server)
        elif op == "admin":
            await _op_admin(req, ws, server)
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
    ok, result, dur = await _cancellable(server, ws, rid, server.engine.sql, sql)
    if not ok:
        server.record_history(rid, sql, 0.0, 0, "cancelled")
        return
    if isinstance(result, (cudf.DataFrame, pd.DataFrame)):
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


async def _op_explain(req, ws, server) -> None:
    rid = req.get("id")
    sql = _require(req, "sql", str)

    def build():
        plan = build_plan(sql, server.catalog)
        return plan_tree(plan, server.catalog.stats_dict())

    ok, tree, _ = await _cancellable(server, ws, rid, build)
    if not ok:
        return
    await _send_json(ws, {"id": rid, "op": "plan", "tree": tree})


async def _op_catalog(req, ws, server) -> None:
    rid = req.get("id")

    def build():
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
    ok, result, dur = await _cancellable(server, ws, rid, server.engine.sql, sql)
    if not ok:
        return
    if isinstance(result, (cudf.DataFrame, pd.DataFrame)):
        await _send_result(ws, rid, result, max_rows, dur, sql, server)
    else:
        await _send_json(ws, {"id": rid, "op": "ok", "duration_ms": round(dur, 3)})


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
    """Execute an admin action against the catalog/engine. Runs on the worker
    thread (serializes with queries). Raises on bad action/args (-> error frame)."""
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
        return {"tables": eng.checkpoint()}
    if action == "snapshot":
        eng.snapshot(_arg(args, "name", str))
        return {"snapshot": True}
    if action == "restore":
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