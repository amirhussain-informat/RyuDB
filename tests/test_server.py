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
from typing import Any

import cudf
import numpy as np
import pandas as pd
import pyarrow.ipc as ipc
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
            # only a successful result is followed by a binary IPC frame
            if want_bin and meta.get("op") == "result":
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