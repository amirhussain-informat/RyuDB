"""End-to-end tests for ``ryudb.server`` — the WebSocket + Arrow IPC front.

These start a real ``Server`` on an ephemeral port (its own asyncio loop on a
background thread) and drive it with a ``websockets`` client, then assert:

  * round-trip correctness — a SELECT over the wire matches the in-process
    ``Engine.sql()`` result (same rows, same dtypes);
  * protocol shapes — result/write/ok/plan/catalog/table/error frames carry the
    documented fields;
  * serialization — the single worker serializes all engine access, so two
    concurrent connections each running several queries all succeed and run in
    submission order;
  * errors — sqlglot parse errors carry a ``position``; runtime errors are
    classified ``runtime``; protocol mistakes (bad JSON, unknown op, binary
    frame) are rejected;
  * cancel — a request still pending in the worker queue is dropped on cancel;
    a request already in flight raises ``CancelledByUser`` at the next
    plan-node boundary and resolves ``cancelled`` (cooperative: a single long
    cuDF call is not mid-call interruptible);
  * explain — the structured plan tree marks an Aggregate-over-Join as
    ``fused=True`` and Aggregate-over-Scan as ``fused=False``;
  * admin / catalog / table / sample / history ops.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from io import BytesIO
from typing import Any

import cudf
import numpy as np
import pandas as pd
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
import pytest
import websockets

from ryudb import Catalog, Engine
from ryudb.server import Server


# --------------------------------------------------------------------------- #
# server fixture: own loop on a background thread, ephemeral port
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def srv_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("ryudb_server")
    (d / "lineitem").mkdir()
    rng = np.random.default_rng(7)
    n = 500
    rows = {
        "l_orderkey": rng.integers(1, 200, size=n).astype(np.int64),
        "l_quantity": rng.integers(1, 50, size=n).astype(np.int64),
        "l_extendedprice": (rng.random(n) * 1000).astype("float64"),
        "l_returnflag": rng.choice(["N", "R", "A"], size=n).astype(object),
    }
    cudf.DataFrame(rows).to_pandas().to_parquet(d / "lineitem" / "0.parquet")
    return d


@pytest.fixture(scope="module")
def srv(srv_dir):
    """Start a Server on port 0; yield (port, server, ref_engine, data_path)."""
    srv_loop = asyncio.new_event_loop()
    server = Server(str(srv_dir), "127.0.0.1", 0, max_rows=100)
    ready = threading.Event()
    port_box: dict[str, int] = {}

    async def _start():
        from websockets.asyncio.server import serve
        ctx = serve(server._handler, server.host, 0, max_size=None)
        ws_srv = await ctx.__aenter__()
        server._srv = ws_srv
        server._srv_ctx = ctx  # keep the serve context alive for the loop's lifetime
        port_box["p"] = ws_srv.sockets[0].getsockname()[1]
        ready.set()
        # return (do NOT await a never-completing Future here — that would make
        # run_until_complete raise "loop stopped before Future completed" on
        # teardown; run_forever below keeps the loop alive and stops cleanly)

    def _run():
        asyncio.set_event_loop(srv_loop)
        srv_loop.run_until_complete(_start())
        srv_loop.run_forever()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    assert ready.wait(5), "server did not start"
    port = port_box["p"]

    # Register the table on the server's catalog. No query is in flight during
    # module setup, so a direct (non-worker) call is safe; routing it through
    # the worker would need a running loop on this thread, which we don't have.
    server.catalog.register("lineitem", str(srv_dir / "lineitem"))

    # a standalone in-process engine over the same data, for round-trip checks
    ref_cat = Catalog(str(srv_dir))
    ref_cat.register("lineitem", str(srv_dir / "lineitem"))
    ref_engine = Engine(ref_cat)

    yield port, server, ref_engine, str(srv_dir / "lineitem")

    # Sync shutdown: stop the worker, then stop the loop (which closes the
    # listening socket). Both are scheduled on the server loop's thread.
    srv_loop.call_soon_threadsafe(server.stop)
    srv_loop.call_soon_threadsafe(srv_loop.stop)
    t.join(timeout=5)


@pytest.fixture(scope="module")
def srv_pool(srv_dir):
    """Start a Server with a 4-worker engine pool on port 0; yield
    (port, server, ref_engine, data_path). Same data as ``srv`` but the engine
    runs on a pool of 4 threads so SELECTs can run concurrently (the RW lock
    keeps writes/admin ops serialized). For the concurrency tests below."""
    srv_loop = asyncio.new_event_loop()
    server = Server(str(srv_dir), "127.0.0.1", 0, max_rows=100, n_workers=4)
    ready = threading.Event()
    port_box: dict[str, int] = {}

    async def _start():
        from websockets.asyncio.server import serve
        ctx = serve(server._handler, server.host, 0, max_size=None)
        ws_srv = await ctx.__aenter__()
        server._srv = ws_srv
        server._srv_ctx = ctx  # keep the serve context alive for the loop's lifetime
        port_box["p"] = ws_srv.sockets[0].getsockname()[1]
        ready.set()

    def _run():
        asyncio.set_event_loop(srv_loop)
        srv_loop.run_until_complete(_start())
        srv_loop.run_forever()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    assert ready.wait(5), "pool server did not start"
    port = port_box["p"]

    # Register the table on the server's catalog (no query in flight during
    # module setup, same as the srv fixture).
    server.catalog.register("lineitem", str(srv_dir / "lineitem"))

    ref_cat = Catalog(str(srv_dir))
    ref_cat.register("lineitem", str(srv_dir / "lineitem"))
    ref_engine = Engine(ref_cat)

    yield port, server, ref_engine, str(srv_dir / "lineitem")

    srv_loop.call_soon_threadsafe(server.stop)
    srv_loop.call_soon_threadsafe(srv_loop.stop)
    t.join(timeout=5)


# --------------------------------------------------------------------------- #
# client helpers (run in the test's own event loop via asyncio.run)
# --------------------------------------------------------------------------- #

async def _call(ws, obj: dict[str, Any], want_bin: bool = False):
    """Send a request; return (meta_frame, [binary_frames]).

    A binary IPC frame follows *only* a ``result`` meta frame — error/cancelled/
    ok/plan frames carry no binary, so we must not wait for one (else we hang).
    """
    await ws.send(json.dumps(obj))
    meta: dict[str, Any] | None = None
    bins: list[bytes] = []
    while True:
        msg = await asyncio.wait_for(ws.recv(), timeout=20)
        if isinstance(msg, (bytes, bytearray)):
            bins.append(bytes(msg))
            if meta is not None:
                return meta, bins
            continue
        data = json.loads(msg)
        if data.get("id") == obj.get("id"):
            meta = data
            # a successful `result` (Arrow IPC) or `export` (Parquet blob) meta
            # is followed by exactly one binary frame.
            if want_bin and meta.get("op") in ("result", "export"):
                nxt = await asyncio.wait_for(ws.recv(), timeout=20)
                if isinstance(nxt, (bytes, bytearray)):
                    bins.append(bytes(nxt))
            return meta, bins


async def _recv(ws):
    """Receive exactly one text frame (for unsolicited / no-id responses)."""
    msg = await asyncio.wait_for(ws.recv(), timeout=20)
    assert isinstance(msg, str), f"expected text frame, got {type(msg)}"
    return json.loads(msg)


def _ipc_table(bins: list[bytes]) -> pd.DataFrame:
    return ipc.open_stream(bins[0]).read_all().to_pandas()


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------- #
# round-trip correctness
# --------------------------------------------------------------------------- #

def test_select_roundtrip_matches_engine(srv):
    port, server, ref, _ = srv
    sql = ("SELECT l_returnflag, count(*) AS c, sum(l_extendedprice) AS s "
           "FROM lineitem GROUP BY l_returnflag ORDER BY l_returnflag")

    async def client():
        async with websockets.connect(f"ws://127.0.0.1:{port}", max_size=None) as ws:
            meta, bins = await _call(ws, {"id": "q1", "op": "sql", "sql": sql}, want_bin=True)
        return meta, bins
    meta, bins = _run(client())

    assert meta["op"] == "result"
    assert meta["row_count"] == meta["returned"]  # 3 groups, under max_rows
    assert meta["truncated"] is False
    assert meta["frame_count"] == 1
    got = _ipc_table(bins)
    exp = ref.sql(sql).to_pandas()
    pd.testing.assert_frame_equal(
        got.reset_index(drop=True), exp.reset_index(drop=True), check_like=True,
    )


def test_select_truncation_caps_rows(srv):
    port, server, ref, _ = srv
    # max_rows=100 (fixture); lineitem has 500 rows
    async def client():
        async with websockets.connect(f"ws://127.0.0.1:{port}", max_size=None) as ws:
            meta, bins = await _call(
                ws, {"id": "q", "op": "sql", "sql": "SELECT * FROM lineitem"}, want_bin=True)
        return meta, bins
    meta, bins = _run(client())
    assert meta["op"] == "result"
    assert meta["row_count"] >= 500
    assert meta["returned"] == 100
    assert meta["truncated"] is True
    assert _ipc_table(bins).shape[0] == 100


# --------------------------------------------------------------------------- #
# result cursors (paged browsing)
# --------------------------------------------------------------------------- #

_CURSOR_SQL = "SELECT l_orderkey, l_quantity, l_returnflag FROM lineitem"


def _sorted_df(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by all columns for order-insensitive comparison (mirrors the
    conftest assert_same idea)."""
    return df.sort_values(list(df.columns)).reset_index(drop=True)


def test_sql_cursor_first_page(srv):
    """sql with cursor:true freezes the full result; the first page carries a
    cursor_id + offset:0 and is capped at max_rows."""
    port, server, ref, _ = srv
    async def client():
        async with websockets.connect(f"ws://127.0.0.1:{port}", max_size=None) as ws:
            meta, bins = await _call(
                ws, {"id": "q", "op": "sql", "sql": _CURSOR_SQL,
                     "max_rows": 100, "cursor": True}, want_bin=True)
        return meta, bins
    meta, bins = _run(client())
    assert meta["op"] == "result"
    assert "cursor_id" in meta and isinstance(meta["cursor_id"], str)
    assert meta["offset"] == 0
    assert meta["row_count"] >= 500
    assert meta["returned"] == 100
    assert meta["truncated"] is True
    assert _ipc_table(bins).shape[0] == 100


def test_fetch_pages_reassemble(srv):
    """First page + fetch pages reassemble to the full result, matching the
    in-process engine."""
    port, server, ref, _ = srv
    async def client():
        async with websockets.connect(f"ws://127.0.0.1:{port}", max_size=None) as ws:
            first, bins0 = await _call(
                ws, {"id": "q", "op": "sql", "sql": _CURSOR_SQL,
                     "max_rows": 100, "cursor": True}, want_bin=True)
            cid = first["cursor_id"]
            frames = [ipc.open_stream(bins0[0]).read_all()]
            off = first["returned"]
            total = first["row_count"]
            i = 0
            while off < total:
                m, b = await _call(
                    ws, {"id": f"f{i}", "op": "fetch", "cursor_id": cid,
                         "offset": off, "limit": 100}, want_bin=True)
                assert m["op"] == "result" and m["cursor_id"] == cid
                assert m["offset"] == off
                frames.append(ipc.open_stream(b[0]).read_all())
                off += m["returned"]
                i += 1
                assert m["returned"] > 0  # progress
            assert off == total
            return frames
    frames = _run(client())
    import pyarrow as pa
    full = pa.concat_tables(frames).to_pandas()
    exp = ref.sql(_CURSOR_SQL).to_pandas()
    pd.testing.assert_frame_equal(_sorted_df(full), _sorted_df(exp), check_like=True)


def test_fetch_offset_window(srv):
    """fetch offset 250 limit 50 returns exactly rows [250:300] of the cursor."""
    port, server, ref, _ = srv
    async def client():
        async with websockets.connect(f"ws://127.0.0.1:{port}", max_size=None) as ws:
            first, bins0 = await _call(
                ws, {"id": "q", "op": "sql", "sql": _CURSOR_SQL,
                     "max_rows": 500, "cursor": True}, want_bin=True)
            cid = first["cursor_id"]
            m, b = await _call(
                ws, {"id": "f", "op": "fetch", "cursor_id": cid,
                     "offset": 250, "limit": 50}, want_bin=True)
            full = ipc.open_stream(bins0[0]).read_all()
            window = ipc.open_stream(b[0]).read_all()
        return first, m, full, window
    first, m, full, window = _run(client())
    assert first["row_count"] >= 500
    assert m["returned"] == 50
    assert m["offset"] == 250
    pd.testing.assert_frame_equal(
        window.to_pandas().reset_index(drop=True),
        full.slice(250, 50).to_pandas().reset_index(drop=True),
        check_like=True,
    )


def test_cursor_per_connection_isolation(srv):
    """A cursor opened on conn A is unknown on conn B (cursors are
    per-connection)."""
    port, server, ref, _ = srv
    async def client():
        async with websockets.connect(f"ws://127.0.0.1:{port}", max_size=None) as a, \
                   websockets.connect(f"ws://127.0.0.1:{port}", max_size=None) as b:
            first, _ = await _call(
                a, {"id": "q", "op": "sql", "sql": _CURSOR_SQL,
                    "max_rows": 50, "cursor": True}, want_bin=True)
            err, _ = await _call(
                b, {"id": "f", "op": "fetch", "cursor_id": first["cursor_id"],
                    "offset": 0, "limit": 10}, want_bin=False)
        return first, err
    first, err = _run(client())
    assert first["op"] == "result" and "cursor_id" in first
    assert err["op"] == "error"
    assert err["kind"] == "protocol"
    assert "unknown cursor" in err["message"]


def test_close_cursor_then_fetch_errors(srv):
    port, server, ref, _ = srv
    async def client():
        async with websockets.connect(f"ws://127.0.0.1:{port}", max_size=None) as ws:
            first, _ = await _call(
                ws, {"id": "q", "op": "sql", "sql": _CURSOR_SQL,
                     "max_rows": 50, "cursor": True}, want_bin=True)
            cid = first["cursor_id"]
            ok, _ = await _call(ws, {"id": "c", "op": "close", "cursor_id": cid})
            err, _ = await _call(
                ws, {"id": "f", "op": "fetch", "cursor_id": cid,
                     "offset": 0, "limit": 10}, want_bin=False)
        return ok, err
    ok, err = _run(client())
    assert ok["op"] == "ok"
    assert err["op"] == "error" and "unknown cursor" in err["message"]


def test_close_unknown_cursor_is_ok(srv):
    """close is idempotent: an unknown cursor id still returns ok."""
    port, *_ = srv
    async def client():
        async with websockets.connect(f"ws://127.0.0.1:{port}", max_size=None) as ws:
            ok, _ = await _call(ws, {"id": "c", "op": "close",
                                     "cursor_id": "never-existed"})
        return ok
    ok = _run(client())
    assert ok["op"] == "ok"


def test_fetch_unknown_cursor_errors(srv):
    port, *_ = srv
    async def client():
        async with websockets.connect(f"ws://127.0.0.1:{port}", max_size=None) as ws:
            err, _ = await _call(
                ws, {"id": "f", "op": "fetch", "cursor_id": "nope",
                     "offset": 0, "limit": 10}, want_bin=False)
        return err
    err = _run(client())
    assert err["op"] == "error" and "unknown cursor" in err["message"]


def test_non_cursor_sql_unchanged(srv):
    """sql without cursor has no cursor_id and behaves exactly as before."""
    port, server, ref, _ = srv
    async def client():
        async with websockets.connect(f"ws://127.0.0.1:{port}", max_size=None) as ws:
            meta, bins = await _call(
                ws, {"id": "q", "op": "sql", "sql": "SELECT * FROM lineitem"},
                want_bin=True)
        return meta, bins
    meta, bins = _run(client())
    assert meta["op"] == "result"
    assert "cursor_id" not in meta
    assert "offset" not in meta
    assert meta["returned"] == 100
    assert meta["truncated"] is True
    assert _ipc_table(bins).shape[0] == 100


def test_cursor_max_per_conn(srv):
    """Opening more than max_cursors_per_conn cursors on one connection errors."""
    port, server, ref, _ = srv
    limit = server.max_cursors_per_conn
    async def client():
        async with websockets.connect(f"ws://127.0.0.1:{port}", max_size=None) as ws:
            ids = []
            for i in range(limit):
                m, _ = await _call(
                    ws, {"id": f"o{i}", "op": "sql", "sql": _CURSOR_SQL,
                         "max_rows": 1, "cursor": True}, want_bin=True)
                assert m["op"] == "result"
                ids.append(m["cursor_id"])
            err, _ = await _call(
                ws, {"id": "oX", "op": "sql", "sql": _CURSOR_SQL,
                     "max_rows": 1, "cursor": True}, want_bin=False)
        return ids, err
    ids, err = _run(client())
    assert len(ids) == limit and len(set(ids)) == limit
    assert err["op"] == "error" and "too many open cursors" in err["message"]


def test_cursor_too_large_falls_back(srv):
    """A cursor request whose result exceeds max_cursor_rows is served as a
    plain truncated result with cursor:false, reason:too_large (no cursor)."""
    port, server, ref, _ = srv
    server.max_cursor_rows = 50  # lineitem has 500 rows -> exceeds the cap
    try:
        async def client():
            async with websockets.connect(f"ws://127.0.0.1:{port}", max_size=None) as ws:
                meta, bins = await _call(
                    ws, {"id": "q", "op": "sql", "sql": "SELECT * FROM lineitem",
                         "max_rows": 100, "cursor": True}, want_bin=True)
            return meta, bins
        meta, bins = _run(client())
    finally:
        server.max_cursor_rows = 1_000_000
    assert meta["op"] == "result"
    assert "cursor_id" not in meta
    assert meta.get("cursor") is False
    assert meta.get("reason") == "too_large"
    assert meta["row_count"] >= 500
    assert meta["returned"] == 100
    assert meta["truncated"] is True


def test_insert_write_then_count(srv, tmp_path):
    """INSERT over the wire returns rows_affected, and the row lands. Uses a
    throwaway table registered on the fly so the shared ``lineitem`` row count
    (asserted by other tests) is untouched."""
    port, server, ref, _ = srv
    wdir = tmp_path / "writetest"
    wdir.mkdir()
    cudf.DataFrame({"k": np.array([1, 2, 3], dtype=np.int64),
                    "v": np.array([10.0, 20.0, 30.0])}).to_pandas().to_parquet(wdir / "0.parquet")
    ins = "INSERT INTO writetest (k, v) VALUES (999, 99.0)"

    async def client():
        async with websockets.connect(f"ws://127.0.0.1:{port}", max_size=None) as ws:
            reg, _ = await _call(ws, {"id": "reg", "op": "admin", "action": "register",
                                      "args": {"table": "writetest", "path": str(wdir)}})
            w, _ = await _call(ws, {"id": "i", "op": "sql", "sql": ins})
            c, bins = await _call(ws, {"id": "c", "op": "sql",
                                       "sql": "SELECT count(*) AS c FROM writetest"}, want_bin=True)
        return reg, w, c, bins
    reg, w, c, bins = _run(client())
    assert reg["op"] == "ok" and reg["detail"]["registered"] == "writetest"
    assert w["op"] == "write" and w["rows_affected"] == 1
    assert c["op"] == "result"
    assert _ipc_table(bins)["c"].iloc[0] == 4  # 3 base + 1 inserted


def test_control_statement_returns_ok(srv):
    port, server, ref, _ = srv
    async def client():
        async with websockets.connect(f"ws://127.0.0.1:{port}", max_size=None) as ws:
            b, _ = await _call(ws, {"id": "b", "op": "sql", "sql": "BEGIN"})
            c, _ = await _call(ws, {"id": "c", "op": "sql", "sql": "COMMIT"})
        return b, c
    b, c = _run(client())
    assert b["op"] == "ok"
    assert c["op"] == "ok"


# --------------------------------------------------------------------------- #
# protocol shapes
# --------------------------------------------------------------------------- #

def test_result_frame_shape(srv):
    port, *_ = srv
    async def client():
        async with websockets.connect(f"ws://127.0.0.1:{port}", max_size=None) as ws:
            meta, _ = await _call(ws, {"id": "q", "op": "sql",
                                       "sql": "SELECT l_returnflag FROM lineitem LIMIT 1"}, want_bin=True)
        return meta
    meta = _run(client())
    for k in ("id", "op", "columns", "row_count", "returned", "truncated",
              "duration_ms", "frame_count"):
        assert k in meta, f"missing {k}"
    assert isinstance(meta["columns"], list) and meta["columns"]
    assert meta["columns"][0]["name"] == "l_returnflag"


def test_explain_tree_shapes(srv):
    port, *_ = srv
    async def client():
        async with websockets.connect(f"ws://127.0.0.1:{port}", max_size=None) as ws:
            scan, _ = await _call(ws, {"id": "e1", "op": "explain",
                                       "sql": "SELECT count(*) AS c FROM lineitem"})
            join, _ = await _call(ws, {"id": "e2", "op": "explain",
                                       "sql": "SELECT l_returnflag, count(*) AS c FROM lineitem "
                                              "JOIN lineitem l2 ON l_orderkey=l2.l_orderkey "
                                              "GROUP BY l_returnflag"})
        return scan, join
    scan, join = _run(client())
    assert scan["op"] == "plan" and scan["tree"]["op"] == "Aggregate"
    assert scan["tree"]["fused"] is False           # Aggregate-over-Scan
    assert scan["tree"]["children"][0]["op"] == "Scan"
    assert scan["tree"]["children"][0]["est_rows"] >= 500  # catalog row count

    assert join["op"] == "plan" and join["tree"]["op"] == "Aggregate"
    assert join["tree"]["fused"] is True            # Aggregate-over-Join (eligible)
    assert join["tree"]["children"][0]["op"] == "Join"


# --------------------------------------------------------------------------- #
# errors + protocol rejection
# --------------------------------------------------------------------------- #

def test_parse_error_has_position(srv):
    port, *_ = srv
    async def client():
        async with websockets.connect(f"ws://127.0.0.1:{port}", max_size=None) as ws:
            r, _ = await _call(ws, {"id": "p", "op": "sql", "sql": "SELECT * FROM"})
        return r
    r = _run(client())
    assert r["op"] == "error" and r["kind"] == "parse"
    assert isinstance(r["position"], dict)
    assert {"line", "col"} <= set(r["position"])


def test_runtime_error_classified(srv):
    port, *_ = srv
    async def client():
        async with websockets.connect(f"ws://127.0.0.1:{port}", max_size=None) as ws:
            r, _ = await _call(ws, {"id": "r", "op": "sql",
                                    "sql": "SELECT * FROM no_such_table"})
        return r
    r = _run(client())
    assert r["op"] == "error" and r["kind"] == "runtime"
    assert "no_such_table" in r["message"] or "unknown" in r["message"].lower()


def test_bad_json_rejected(srv):
    port, *_ = srv
    async def client():
        async with websockets.connect(f"ws://127.0.0.1:{port}", max_size=None) as ws:
            await ws.send("not json")
            return await _recv(ws)
    r = _run(client())
    assert r["op"] == "error" and r["kind"] == "protocol"


def test_unknown_op_rejected(srv):
    port, *_ = srv
    async def client():
        async with websockets.connect(f"ws://127.0.0.1:{port}", max_size=None) as ws:
            r, _ = await _call(ws, {"id": "u", "op": "frobnicate"})
        return r
    r = _run(client())
    assert r["op"] == "error" and r["kind"] == "protocol"


def test_binary_frame_rejected(srv):
    port, *_ = srv
    async def client():
        async with websockets.connect(f"ws://127.0.0.1:{port}", max_size=None) as ws:
            await ws.send(b"\x00\x01\x02")
            with pytest.raises(websockets.ConnectionClosed):
                await ws.recv()
    _run(client())


# --------------------------------------------------------------------------- #
# serialization + cancel
# --------------------------------------------------------------------------- #

def test_concurrent_connections_all_succeed(srv):
    port, *_ = srv
    async def client():
        async def burst(prefix):
            async with websockets.connect(f"ws://127.0.0.1:{port}", max_size=None) as ws:
                for k in range(4):
                    m, _ = await _call(ws, {"id": f"{prefix}{k}", "op": "sql",
                                            "sql": "SELECT count(*) AS c FROM lineitem"},
                                       want_bin=True)
                    assert m["op"] == "result", m
        await asyncio.gather(burst("A"), burst("B"), burst("C"))
    _run(client())  # 3 connections x 4 queries, all must succeed


def test_serialization_preserves_order(srv):
    """The single worker runs requests in submission order. Patch engine.sql to
    record the order queries actually execute; submit A then B on one connection
    and confirm A runs before B."""
    port, server, ref, _ = srv
    order: list[str] = []
    orig = server.engine.sql

    def recording_sql(sql):
        # tag by a trailing comment the test injects
        if "/*A*/" in sql:
            order.append("A")
        elif "/*B*/" in sql:
            order.append("B")
        return orig(sql)
    server.engine.sql = recording_sql
    try:
        async def client():
            async with websockets.connect(f"ws://127.0.0.1:{port}", max_size=None) as ws:
                await _call(ws, {"id": "a", "op": "sql",
                                 "sql": "SELECT count(*) AS c FROM lineitem /*A*/"})
                await _call(ws, {"id": "b", "op": "sql",
                                 "sql": "SELECT count(*) AS c FROM lineitem /*B*/"})
        _run(client())
    finally:
        server.engine.sql = orig
    assert order == ["A", "B"]


def test_cancel_pending_request(srv):
    """Block the worker on the first query; a second query sits pending in the
    queue; cancel it before releasing the first -> the second returns
    ``cancelled`` (it never started). Uses raw sends + a single recv stream
    (two concurrent ``ws.recv()`` calls on one connection are not allowed)."""
    port, server, ref, _ = srv
    block = threading.Event()
    first_started = threading.Event()
    orig = server.engine.sql

    def blocking_sql(sql):
        if not first_started.is_set():
            first_started.set()
            block.wait(5)  # hold the worker until the test releases it
        return orig(sql)
    server.engine.sql = blocking_sql
    try:
        async def client():
            async with websockets.connect(f"ws://127.0.0.1:{port}", max_size=None) as ws:
                await ws.send(json.dumps({"id": "q1", "op": "sql",
                                          "sql": "SELECT count(*) AS c FROM lineitem"}))
                assert first_started.wait(2), "q1 did not start"
                await ws.send(json.dumps({"id": "q2", "op": "sql",
                                          "sql": "SELECT count(*) AS c FROM lineitem"}))
                await asyncio.sleep(0.05)  # let q2 register as pending
                await ws.send(json.dumps({"id": "cx", "op": "cancel", "targets": ["q2"]}))
                got: dict[str, dict] = {}
                block_set = False
                # Collect all three. The cx response is sent only after
                # cancel_pending() has set q2's event, so when cx arrives it is
                # safe to release q1 (q2 will be dropped on dequeue). q1 cannot
                # produce output before then — it is blocked on `block`.
                while len(got) < 3:
                    m = await asyncio.wait_for(ws.recv(), timeout=20)
                    if isinstance(m, (bytes, bytearray)):
                        continue  # q1's result binary frame; not needed here
                    d = json.loads(m)
                    rid = d.get("id")
                    if rid in ("q1", "q2", "cx"):
                        got[rid] = d
                    if rid == "cx" and not block_set:
                        block.set()
                        block_set = True
                return got
        got = _run(client())
    finally:
        server.engine.sql = orig
    assert got["q1"]["op"] == "result"          # ran to completion
    assert got["q2"]["op"] == "cancelled"       # dropped while pending
    assert "q2" in got["cx"]["detail"]["cancelled"]


def test_cancel_inflight_request(srv):
    """Block the worker INSIDE the first scan of a running query; cancel it
    mid-flight; release the scan -> the next plan-node boundary (the join's
    right-input _exec) sees the set cancel event and raises CancelledByUser, so
    the in-flight request resolves ``cancelled`` (cooperative in-flight cancel).
    Uses raw sends + a single recv stream (two concurrent ws.recv() calls on one
    connection are not allowed)."""
    port, server, ref, _ = srv
    hold = threading.Event()
    started = threading.Event()
    orig = server.engine._scan  # bound method (class _scan bound to the instance)

    first = [True]

    def blocking_scan(table, columns):
        # Patched onto the instance dict, so self._scan(a, b) calls this with no
        # bound self -> signature is (table, columns); orig is the bound original.
        if first[0]:
            first[0] = False
            started.set()
            hold.wait(5)  # hold the worker mid-query until the test releases it
        return orig(table, columns)
    server.engine._scan = blocking_scan
    try:
        async def client():
            async with websockets.connect(f"ws://127.0.0.1:{port}", max_size=None) as ws:
                # Self-join on lineitem: two scans. The plan is
                # Limit -> Project -> Join -> Scan(a), Scan(b). The first scan
                # blocks; after release the join's right-input _exec hits the
                # cancel check and raises CancelledByUser before the join runs.
                await ws.send(json.dumps({"id": "q1", "op": "sql", "sql":
                    "SELECT a.l_orderkey FROM lineitem a JOIN lineitem b "
                    "ON a.l_orderkey = b.l_orderkey LIMIT 5"}))
                assert started.wait(2), "q1 did not start (first scan never ran)"
                await ws.send(json.dumps({"id": "cx", "op": "cancel",
                                          "targets": ["q1"]}))
                got: dict[str, dict] = {}
                released = False
                while len(got) < 2:
                    m = await asyncio.wait_for(ws.recv(), timeout=20)
                    if isinstance(m, (bytes, bytearray)):
                        continue  # a result would be a bug here; ignore defensively
                    d = json.loads(m)
                    rid = d.get("id")
                    if rid in ("q1", "cx"):
                        got[rid] = d
                    # The cx ok frame is sent only after cancel_pending() set
                    # q1's event; once it arrives it is safe to release the
                    # blocked scan -- the engine's next _exec boundary will see
                    # the set event and raise CancelledByUser.
                    if rid == "cx" and not released:
                        hold.set()
                        released = True
                return got
        got = _run(client())
    finally:
        server.engine._scan = orig
    assert got["cx"]["op"] == "ok"
    assert "q1" in got["cx"]["detail"]["cancelled"]
    assert got["q1"]["op"] == "cancelled"   # raised CancelledByUser mid-flight


def test_cancel_unknown_target(srv):
    port, *_ = srv
    async def client():
        async with websockets.connect(f"ws://127.0.0.1:{port}", max_size=None) as ws:
            r, _ = await _call(ws, {"id": "c", "op": "cancel", "targets": ["nope"]})
        return r
    r = _run(client())
    assert r["op"] == "ok"
    assert r["detail"]["cancelled"] == []
    assert r["detail"]["not_found"] == ["nope"]


# --------------------------------------------------------------------------- #
# metadata + admin ops
# --------------------------------------------------------------------------- #

def test_catalog_and_table_ops(srv):
    port, server, ref, _ = srv
    async def client():
        async with websockets.connect(f"ws://127.0.0.1:{port}", max_size=None) as ws:
            cat, _ = await _call(ws, {"id": "cat", "op": "catalog"})
            tbl, _ = await _call(ws, {"id": "tbl", "op": "table", "name": "lineitem"})
        return cat, tbl
    cat, tbl = _run(client())
    assert cat["op"] == "catalog"
    names = [t["name"] for t in cat["tables"]]
    assert "lineitem" in names
    assert tbl["op"] == "table" and tbl["name"] == "lineitem"
    assert tbl["row_count"] >= 500
    col_names = [c["name"] for c in tbl["columns"]]
    assert col_names == ["l_orderkey", "l_quantity", "l_extendedprice", "l_returnflag"]


def test_profile_op(srv):
    port, server, ref, _ = srv
    async def client():
        async with websockets.connect(f"ws://127.0.0.1:{port}", max_size=None) as ws:
            prof, _ = await _call(ws, {"id": "pr", "op": "profile", "name": "lineitem"})
        return prof
    prof = _run(client())
    assert prof["op"] == "profile" and prof["name"] == "lineitem"
    assert prof["row_count"] == 500
    by = {c["name"]: c for c in prof["columns"]}
    assert set(by) == {"l_orderkey", "l_quantity", "l_extendedprice", "l_returnflag"}
    # common stats on every column
    for c in prof["columns"]:
        assert c["row_count"] == 500
        assert c["null_count"] == 0 and c["null_pct"] == 0.0
        assert c["distinct"] >= 1
        assert c["min"] is not None and c["max"] is not None
    # numeric columns carry mean/stddev + a 10-bucket histogram
    for nm in ("l_orderkey", "l_quantity", "l_extendedprice"):
        c = by[nm]
        assert c["mean"] is not None and c["stddev"] is not None
        assert c["histogram"] is not None and len(c["histogram"]) == 10
        assert sum(b["count"] for b in c["histogram"]) == 500
    # low-NDV categorical column has top-K value frequencies; high-NDV numeric
    # columns skip top (distinct > 1000 not the case here, but orderkey's
    # distinct is <= 199 so top IS present — verify the categorical one closely)
    flag = by["l_returnflag"]
    assert flag["distinct"] == 3
    assert flag["top"] is not None and len(flag["top"]) <= 10
    assert sum(t["count"] for t in flag["top"]) == 500
    values = {t["value"] for t in flag["top"]}
    assert values <= {"N", "R", "A"}
    # min/max of the categorical column are real flag values
    assert flag["min"] in {"N", "R", "A"} and flag["max"] in {"N", "R", "A"}


def test_profile_unknown_table_errors(srv):
    port, *_ = srv
    async def client():
        async with websockets.connect(f"ws://127.0.0.1:{port}", max_size=None) as ws:
            r, _ = await _call(ws, {"id": "pr", "op": "profile", "name": "nope"})
        return r
    r = _run(client())
    assert r["op"] == "error"  # catalog.get KeyError -> runtime error frame


def test_profile_rejects_bad_top_k(srv):
    port, *_ = srv
    async def client():
        async with websockets.connect(f"ws://127.0.0.1:{port}", max_size=None) as ws:
            r, _ = await _call(ws, {"id": "pr", "op": "profile",
                                    "name": "lineitem", "top_k": 0})
        return r
    r = _run(client())
    assert r["op"] == "error" and r["kind"] == "protocol"


def test_export_parquet(srv):
    port, server, ref, _ = srv
    sql = "SELECT l_orderkey, l_quantity, l_returnflag FROM lineitem"
    async def client():
        async with websockets.connect(f"ws://127.0.0.1:{port}", max_size=None) as ws:
            meta, bins = await _call(ws, {"id": "x", "op": "export", "sql": sql,
                                          "format": "parquet"}, want_bin=True)
        return meta, bins
    meta, bins = _run(client())
    assert meta["op"] == "export" and meta["format"] == "parquet"
    assert meta["row_count"] == 500
    assert meta["byte_count"] == len(bins[0])
    # Parquet files begin and end with the b"PAR1" magic.
    assert bins[0][:4] == b"PAR1" and bins[0][-4:] == b"PAR1"
    # Decode the Parquet blob and compare to the in-process reference result
    # (sorted by all columns, mirroring assert_same).
    got = pq.read_table(BytesIO(bins[0])).to_pandas()
    exp = ref.sql(sql).to_pandas()
    cols = list(exp.columns)
    got_s = got.sort_values(cols).reset_index(drop=True)
    exp_s = exp.sort_values(cols).reset_index(drop=True)
    pd.testing.assert_frame_equal(got_s, exp_s)


def test_export_default_format_is_parquet(srv):
    port, *_ = srv
    async def client():
        async with websockets.connect(f"ws://127.0.0.1:{port}", max_size=None) as ws:
            meta, bins = await _call(ws, {"id": "x", "op": "export",
                                          "sql": "SELECT l_orderkey FROM lineitem LIMIT 1"},
                                     want_bin=True)
        return meta, bins
    meta, bins = _run(client())
    assert meta["op"] == "export" and meta["format"] == "parquet"
    assert bins[0][:4] == b"PAR1" and meta["row_count"] == 1


def test_export_rejects_bad_format(srv):
    port, *_ = srv
    async def client():
        async with websockets.connect(f"ws://127.0.0.1:{port}", max_size=None) as ws:
            r, _ = await _call(ws, {"id": "x", "op": "export",
                                    "sql": "SELECT l_orderkey FROM lineitem LIMIT 1",
                                    "format": "csv"})
        return r
    r = _run(client())
    assert r["op"] == "error" and r["kind"] == "protocol"


def test_export_non_select_errors(srv):
    port, *_ = srv
    async def client():
        async with websockets.connect(f"ws://127.0.0.1:{port}", max_size=None) as ws:
            r, _ = await _call(ws, {"id": "x", "op": "export",
                                    "sql": "DROP TABLE nope_x"}, want_bin=False)
        return r
    r = _run(client())
    assert r["op"] == "error"  # non-SELECT -> ProtocolError inside run()


def test_export_too_large_errors(srv):
    port, server, *_ = srv
    orig = server.max_export_rows
    server.max_export_rows = 50  # the 500-row lineitem exceeds this
    try:
        async def client():
            async with websockets.connect(f"ws://127.0.0.1:{port}", max_size=None) as ws:
                r, _ = await _call(ws, {"id": "x", "op": "export",
                                        "sql": "SELECT * FROM lineitem"}, want_bin=False)
            return r
        r = _run(client())
    finally:
        server.max_export_rows = orig
    assert r["op"] == "error" and r["kind"] == "runtime"
    assert "too large" in r["message"]


def test_sample_op(srv):
    port, *_ = srv
    async def client():
        async with websockets.connect(f"ws://127.0.0.1:{port}", max_size=None) as ws:
            meta, bins = await _call(ws, {"id": "s", "op": "sample",
                                          "name": "lineitem", "n": 5}, want_bin=True)
        return meta, bins
    meta, bins = _run(client())
    assert meta["op"] == "result" and meta["returned"] == 5
    assert _ipc_table(bins).shape[0] == 5


def test_sample_rejects_bad_table_name(srv):
    port, *_ = srv
    async def client():
        async with websockets.connect(f"ws://127.0.0.1:{port}", max_size=None) as ws:
            r, _ = await _call(ws, {"id": "s", "op": "sample",
                                    "name": "lineitem; DROP", "n": 5})
        return r
    r = _run(client())
    assert r["op"] == "error" and r["kind"] == "protocol"  # not an identifier


def test_admin_alter_and_checkpoint(srv):
    port, server, ref, _ = srv
    async def client():
        async with websockets.connect(f"ws://127.0.0.1:{port}", max_size=None) as ws:
            nn, _ = await _call(ws, {"id": "a1", "op": "admin", "action": "alter",
                                     "args": {"table": "lineitem", "kind": "notnull",
                                              "col": "l_quantity"}})
            cp, _ = await _call(ws, {"id": "a2", "op": "admin", "action": "checkpoint"})
        return nn, cp
    nn, cp = _run(client())
    assert nn["op"] == "ok" and nn["detail"]["set"] == "not_null"
    assert cp["op"] == "ok" and "tables" in cp["detail"]


def test_history_records_invocations(srv):
    port, *_ = srv
    async def client():
        async with websockets.connect(f"ws://127.0.0.1:{port}", max_size=None) as ws:
            await _call(ws, {"id": "h1", "op": "sql",
                             "sql": "SELECT count(*) AS c FROM lineitem"}, want_bin=True)
            r, _ = await _call(ws, {"id": "h2", "op": "history"})
        return r
    r = _run(client())
    assert r["op"] == "history"
    assert any(e["sql"].startswith("SELECT count(*)") for e in r["entries"])


# --------------------------------------------------------------------------- #
# multi-session isolation (per-connection transactions / MVCC)
# --------------------------------------------------------------------------- #

def _register_iso(port, tmp_path, name, n_base=3):
    """Register a throwaway table ``name`` (k int64, v float64) with ``n_base``
    base rows on the shared server, over the wire. Returns the parquet dir."""
    wdir = tmp_path / name
    wdir.mkdir()
    cudf.DataFrame({
        "k": np.arange(1, n_base + 1, dtype=np.int64),
        "v": np.arange(1.0, n_base + 1, dtype="float64"),
    }).to_pandas().to_parquet(wdir / "0.parquet")

    async def client():
        async with websockets.connect(f"ws://127.0.0.1:{port}", max_size=None) as ws:
            r, _ = await _call(ws, {"id": "reg", "op": "admin", "action": "register",
                                    "args": {"table": name, "path": str(wdir)}})
        return r
    r = _run(client())
    assert r["op"] == "ok" and r["detail"]["registered"] == name
    return wdir


def test_two_connections_can_each_hold_open_txn(srv):
    """Per-connection transactions: BEGIN on conn B no longer fails while conn
    A has an open txn (the old single-``_txn``-slot limit is gone)."""
    port, *_ = srv

    async def client():
        async with websockets.connect(f"ws://127.0.0.1:{port}", max_size=None) as a, \
                   websockets.connect(f"ws://127.0.0.1:{port}", max_size=None) as b:
            ba, _ = await _call(a, {"id": "ba", "op": "sql", "sql": "BEGIN"})
            bb, _ = await _call(b, {"id": "bb", "op": "sql", "sql": "BEGIN"})
            ca, _ = await _call(a, {"id": "ca", "op": "sql", "sql": "COMMIT"})
            cb, _ = await _call(b, {"id": "cb", "op": "sql", "sql": "COMMIT"})
        return ba, bb, ca, cb
    ba, bb, ca, cb = _run(client())
    assert ba["op"] == "ok" and bb["op"] == "ok"   # both BEGINs succeed
    assert ca["op"] == "ok" and cb["op"] == "ok"


def test_snapshot_isolation_across_connections(srv, tmp_path):
    """A txn on conn A (BEGIN at S0) does not see conn B's commit (at S0+1):
    real cross-session MVCC via the timestamp-versioned delta store. After A
    commits, a fresh read sees B's commit."""
    port, *_ = srv
    _register_iso(port, tmp_path, "iso_snap", n_base=3)

    async def client():
        async with websockets.connect(f"ws://127.0.0.1:{port}", max_size=None) as a, \
                   websockets.connect(f"ws://127.0.0.1:{port}", max_size=None) as b:
            # A opens a txn (snapshot S0 = current commit counter).
            await _call(a, {"id": "ba", "op": "sql", "sql": "BEGIN"})
            # B inserts + commits (autocommit, new commit_ts > S0).
            ib, _ = await _call(b, {"id": "ib", "op": "sql",
                "sql": "INSERT INTO iso_snap (k, v) VALUES (999, 99.0)"})
            # A's txn read: snapshot S0 excludes B's commit -> 3 rows.
            ca, bins = await _call(a, {"id": "ca", "op": "sql",
                "sql": "SELECT count(*) AS c FROM iso_snap"}, want_bin=True)
            # A commits; a fresh autocommit read sees B's commit -> 4 rows.
            await _call(a, {"id": "xa", "op": "sql", "sql": "COMMIT"})
            da, bins2 = await _call(a, {"id": "da", "op": "sql",
                "sql": "SELECT count(*) AS c FROM iso_snap"}, want_bin=True)
        return ib, ca, bins, da, bins2
    ib, ca, bins, da, bins2 = _run(client())
    assert ib["op"] == "write" and ib["rows_affected"] == 1
    assert ca["op"] == "result"
    assert _ipc_table(bins)["c"].iloc[0] == 3       # A's snapshot excludes B's commit
    assert da["op"] == "result"
    assert _ipc_table(bins2)["c"].iloc[0] == 4      # after COMMIT, sees B's commit


def test_read_your_writes_then_rollback(srv, tmp_path):
    """A txn sees its own buffered INSERT (read-your-writes); ROLLBACK discards
    the buffer and the row is no longer visible."""
    port, *_ = srv
    _register_iso(port, tmp_path, "iso_ryw", n_base=3)

    async def client():
        async with websockets.connect(f"ws://127.0.0.1:{port}", max_size=None) as a:
            await _call(a, {"id": "b", "op": "sql", "sql": "BEGIN"})
            await _call(a, {"id": "i", "op": "sql",
                "sql": "INSERT INTO iso_ryw (k, v) VALUES (999, 99.0)"})
            c1, bins1 = await _call(a, {"id": "c1", "op": "sql",
                "sql": "SELECT count(*) AS c FROM iso_ryw"}, want_bin=True)
            await _call(a, {"id": "rb", "op": "sql", "sql": "ROLLBACK"})
            c2, bins2 = await _call(a, {"id": "c2", "op": "sql",
                "sql": "SELECT count(*) AS c FROM iso_ryw"}, want_bin=True)
        return c1, bins1, c2, bins2
    c1, bins1, c2, bins2 = _run(client())
    assert c1["op"] == "result"
    assert _ipc_table(bins1)["c"].iloc[0] == 4      # read-your-writes: 3 base + 1 buffered
    assert c2["op"] == "result"
    assert _ipc_table(bins2)["c"].iloc[0] == 3      # rollback discarded the buffer


def test_disconnect_rolls_back_open_txn(srv, tmp_path):
    """A connection that disconnects with an open txn has its buffered writes
    rolled back (they were never flushed to the delta) -- invisible to a later
    connection."""
    port, *_ = srv
    _register_iso(port, tmp_path, "iso_disc", n_base=3)

    async def client():
        # Conn A: BEGIN, INSERT (buffered), then disconnect WITHOUT COMMIT.
        async with websockets.connect(f"ws://127.0.0.1:{port}", max_size=None) as a:
            await _call(a, {"id": "b", "op": "sql", "sql": "BEGIN"})
            await _call(a, {"id": "i", "op": "sql",
                "sql": "INSERT INTO iso_disc (k, v) VALUES (999, 99.0)"})
        # `a` is now closed; its txn is abandoned (buffer discarded, never
        # flushed). Conn B: a fresh read sees only the 3 base rows.
        async with websockets.connect(f"ws://127.0.0.1:{port}", max_size=None) as b:
            c, bins = await _call(b, {"id": "c", "op": "sql",
                "sql": "SELECT count(*) AS c FROM iso_disc"}, want_bin=True)
        return c, bins
    c, bins = _run(client())
    assert c["op"] == "result"
    assert _ipc_table(bins)["c"].iloc[0] == 3       # A's buffered insert rolled back


def test_checkpoint_refused_during_open_txn(srv, tmp_path):
    """Checkpoint folds the committed delta into the base (snapshot-agnostic
    rows); it is refused while any connection has an open txn, then succeeds
    once the txn commits."""
    port, *_ = srv
    _register_iso(port, tmp_path, "iso_ckpt", n_base=3)

    async def client():
        async with websockets.connect(f"ws://127.0.0.1:{port}", max_size=None) as a, \
                   websockets.connect(f"ws://127.0.0.1:{port}", max_size=None) as b:
            await _call(a, {"id": "b", "op": "sql", "sql": "BEGIN"})
            # B tries to checkpoint while A has an open txn -> error.
            bad, _ = await _call(b, {"id": "cp1", "op": "admin", "action": "checkpoint"})
            # A commits, then B's checkpoint succeeds.
            await _call(a, {"id": "c", "op": "sql", "sql": "COMMIT"})
            good, _ = await _call(b, {"id": "cp2", "op": "admin", "action": "checkpoint"})
        return bad, good
    bad, good = _run(client())
    assert bad["op"] == "error" and bad["kind"] == "runtime"
    assert "transaction" in bad["message"].lower()
    assert good["op"] == "ok" and "tables" in good["detail"]


def test_restore_refused_during_open_txn(srv, tmp_path):
    """Restore rewinds the global commit counter and discards delta tail; it is
    refused while any connection has an open txn, then succeeds once it commits."""
    port, *_ = srv
    _register_iso(port, tmp_path, "iso_rst", n_base=3)

    async def client():
        async with websockets.connect(f"ws://127.0.0.1:{port}", max_size=None) as a, \
                   websockets.connect(f"ws://127.0.0.1:{port}", max_size=None) as b:
            # Capture a named snapshot at the base state, then advance past it.
            await _call(b, {"id": "s", "op": "sql", "sql": "CREATE SNAPSHOT base"})
            await _call(b, {"id": "i", "op": "sql",
                "sql": "INSERT INTO iso_rst (k, v) VALUES (999, 99.0)"})
            # A opens a txn (snapshot past `base`); B's restore is refused.
            await _call(a, {"id": "b", "op": "sql", "sql": "BEGIN"})
            bad, _ = await _call(b, {"id": "r1", "op": "admin", "action": "restore",
                                     "args": {"name": "base"}})
            # A commits; B's restore now succeeds.
            await _call(a, {"id": "c", "op": "sql", "sql": "COMMIT"})
            good, _ = await _call(b, {"id": "r2", "op": "admin", "action": "restore",
                                      "args": {"name": "base"}})
        return bad, good
    bad, good = _run(client())
    assert bad["op"] == "error" and bad["kind"] == "runtime"
    assert "transaction" in bad["message"].lower()
    assert good["op"] == "ok" and good["detail"].get("restored") is True


# --------------------------------------------------------------------------- #
# worker-pool concurrency (srv_pool: n_workers=4)
# --------------------------------------------------------------------------- #
#
# These prove the engine runs concurrently on a worker pool (Option 1: thread
# pool + MVCC). The Engine's read/write lock lets SELECTs overlap (shared read
# lock) while writes/admin ops serialize (exclusive write lock). Each worker
# thread has its own _txn / cancel_event slot (threading.local), so concurrent
# requests never clobber a single shared slot.


def test_pool_reads_run_concurrently(srv_pool):
    """With n_workers=4, two SELECTs dispatched together overlap INSIDE the
    engine -- proving the pool dispatches to separate worker threads AND the
    read lock is shared (not exclusive), so concurrent reads are allowed. Patch
    engine._scan (which runs under the read lock) to sleep and count how many
    scans are in flight at once; two pipelined SELECTs must reach
    max_in_flight >= 2. (On the single-worker ``srv`` fixture this would be 1 --
    the reads would serialize.)"""
    port, server, ref, _ = srv_pool
    orig = server.engine._scan
    state = {"in_flight": 0, "max": 0}
    lock = threading.Lock()

    def slow_scan(table, columns):
        with lock:
            state["in_flight"] += 1
            state["max"] = max(state["max"], state["in_flight"])
        try:
            time.sleep(0.05)
            return orig(table, columns)
        finally:
            with lock:
                state["in_flight"] -= 1
    server.engine._scan = slow_scan
    try:
        async def client():
            async with websockets.connect(f"ws://127.0.0.1:{port}", max_size=None) as ws:
                # Pipeline two SELECTs without awaiting between them so both
                # land in the worker queue before either completes.
                await ws.send(json.dumps({"id": "q1", "op": "sql",
                    "sql": "SELECT l_orderkey FROM lineitem LIMIT 5"}))
                await ws.send(json.dumps({"id": "q2", "op": "sql",
                    "sql": "SELECT l_orderkey FROM lineitem LIMIT 5"}))
                got: dict[str, dict] = {}
                while len(got) < 2:
                    m = await asyncio.wait_for(ws.recv(), timeout=20)
                    if isinstance(m, (bytes, bytearray)):
                        continue
                    d = json.loads(m)
                    if d.get("id") in ("q1", "q2"):
                        got[d["id"]] = d
            return got
        got = _run(client())
    finally:
        server.engine._scan = orig
    assert got["q1"]["op"] == "result"
    assert got["q2"]["op"] == "result"
    assert state["max"] >= 2, f"reads did not overlap (max_in_flight={state['max']})"


def test_pool_cancel_pending_when_full(srv_pool):
    """With n_workers=4, block all 4 workers on long scans; a 5th request sits
    pending in the queue (no free worker); cancel it before releasing the
    blockers -> it returns ``cancelled`` (dropped while pending, never started).
    This is the only way to reach a pending-cancel state with a pool: with 4
    workers, one in-flight request does NOT block a second, so we must fill the
    whole pool to force a pending request."""
    port, server, ref, _ = srv_pool
    block = threading.Event()
    orig = server.engine._scan
    n_started = [0]

    def blocking_scan(table, columns):
        n_started[0] += 1
        block.wait(5)  # hold this worker until the test releases it
        return orig(table, columns)
    server.engine._scan = blocking_scan
    try:
        async def client():
            async with websockets.connect(f"ws://127.0.0.1:{port}", max_size=None) as ws:
                # 4 queries fill the 4 workers.
                for i in range(4):
                    await ws.send(json.dumps({"id": f"q{i}", "op": "sql",
                        "sql": "SELECT l_orderkey FROM lineitem LIMIT 5"}))
                # Wait until all 4 workers are blocked inside _scan.
                deadline = time.time() + 5
                while n_started[0] < 4 and time.time() < deadline:
                    await asyncio.sleep(0.01)
                assert n_started[0] == 4, f"only {n_started[0]} workers started"
                # 5th request: no free worker -> sits pending.
                await ws.send(json.dumps({"id": "p5", "op": "sql",
                    "sql": "SELECT l_orderkey FROM lineitem LIMIT 5"}))
                await asyncio.sleep(0.05)  # let p5 register as pending
                await ws.send(json.dumps({"id": "cx", "op": "cancel",
                                          "targets": ["p5"]}))
                got: dict[str, dict] = {}
                block_set = False
                # Collect all 6: q0..q3 (result), p5 (cancelled), cx (ok). The cx
                # ok frame is sent only after cancel_pending() set p5's event, so
                # once cx arrives it is safe to release the 4 blockers -- p5 will
                # be dropped on dequeue.
                while len(got) < 6:
                    m = await asyncio.wait_for(ws.recv(), timeout=20)
                    if isinstance(m, (bytes, bytearray)):
                        continue  # a q result's binary frame; not needed here
                    d = json.loads(m)
                    rid = d.get("id")
                    if rid in ("q0", "q1", "q2", "q3", "p5", "cx"):
                        got[rid] = d
                    if rid == "cx" and not block_set:
                        block.set()
                        block_set = True
            return got
        got = _run(client())
    finally:
        server.engine._scan = orig
    for i in range(4):
        assert got[f"q{i}"]["op"] == "result"     # the 4 blockers ran to completion
    assert got["p5"]["op"] == "cancelled"        # dropped while pending
    assert "p5" in got["cx"]["detail"]["cancelled"]


def test_pool_checkpoint_waits_for_inflight_read(srv_pool):
    """An in-flight SELECT holds the shared read lock; a checkpoint (exclusive
    write lock) must WAIT for it to finish, then succeed -- it neither runs
    concurrently with the read (which would corrupt the base+delta rewrite)
    nor fails. Proves the RW lock coordinates checkpoint with concurrent reads
    on the pool (writer-preference: the waiting writer blocks until the reader
    releases)."""
    port, server, ref, _ = srv_pool
    hold = threading.Event()
    started = threading.Event()
    orig = server.engine._scan

    def slow_scan(table, columns):
        started.set()
        hold.wait(5)  # hold the read lock until the test releases it
        return orig(table, columns)
    server.engine._scan = slow_scan
    try:
        async def client():
            async with websockets.connect(f"ws://127.0.0.1:{port}", max_size=None) as a, \
                       websockets.connect(f"ws://127.0.0.1:{port}", max_size=None) as b:
                # a: start a slow read (holds the read lock).
                await a.send(json.dumps({"id": "q", "op": "sql",
                    "sql": "SELECT l_orderkey FROM lineitem LIMIT 5"}))
                assert started.wait(2), "read did not start"
                # b: checkpoint while the read is in flight. It must block on
                # the write lock until the read releases, then succeed.
                cp_done = asyncio.Event()
                cp_result: dict[str, Any] = {}

                async def do_ckpt():
                    r, _ = await _call(b, {"id": "cp", "op": "admin",
                                           "action": "checkpoint"})
                    cp_result["r"] = r
                    cp_done.set()
                asyncio.create_task(do_ckpt())
                await asyncio.sleep(0.1)  # give the checkpoint time to block
                assert not cp_done.is_set(), "checkpoint returned before the read finished"
                hold.set()  # release the read -> checkpoint acquires + runs
                await asyncio.wait_for(cp_done.wait(), timeout=5)
                # collect the read result
                rq: dict[str, dict] = {}
                while len(rq) < 1:
                    m = await asyncio.wait_for(a.recv(), timeout=20)
                    if isinstance(m, (bytes, bytearray)):
                        continue
                    d = json.loads(m)
                    if d.get("id") == "q":
                        rq["q"] = d
            return rq["q"], cp_result["r"]
        q, cp = _run(client())
    finally:
        server.engine._scan = orig
    assert q["op"] == "result"
    assert cp["op"] == "ok" and "tables" in cp["detail"]