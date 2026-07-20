"""End-to-end tests for ``ryudb.server.pgwire`` — the Postgres v3 wire front.

These start a real ``Server`` (engine + worker, per-connection transactions)
plus a ``PGServer`` on an ephemeral port (its own asyncio loop on a background
thread) and drive it two ways:

  * a **raw asyncio TCP client** that speaks the v3 protocol by hand
    (StartupMessage, Simple Query ``Q``, Extended Query ``P``/``B``/``D``/``E``/
    ``S``, ``Terminate``). These are driver-independent and always run — they
    pin the exact byte-level protocol shape (``R``/``K``/``Z``/``T``/``D``/``C``/
    ``E``/``I``/``n``/``t``/``1``/``2``/``3``) and the ``$N`` substitution path.
  * an **optional ``pg8000`` end-to-end test** (``importorskip``) proving a real
    Postgres driver can connect, run parameterized SQL, and read rows.

The raw tests are the contract; the pg8000 test is interoperability cover.
"""

from __future__ import annotations

import asyncio
import struct
import threading
import time
from typing import Any

import cudf
import numpy as np
import pytest

from ryudb import Catalog, Engine
from ryudb.server import Server
from ryudb.server.pgwire import CANCEL_REQUEST, PGServer


# --------------------------------------------------------------------------- #
# server fixture: own loop on a background thread, ephemeral PG port
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def pg_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("ryudb_pgwire")
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
def pg(pg_dir):
    """Start a Server + PGServer on ephemeral ports; yield (pg_port, server, ref)."""
    loop = asyncio.new_event_loop()
    server = Server(str(pg_dir), "127.0.0.1", 0, max_rows=100)
    ready = threading.Event()
    box: dict[str, Any] = {}

    async def _start():
        pg_srv = await asyncio.start_server(
            PGServer(server, "127.0.0.1", 0, max_rows=100)._handler,
            "127.0.0.1", 0,
        )
        box["pg_srv"] = pg_srv
        box["port"] = pg_srv.sockets[0].getsockname()[1]
        ready.set()
        await asyncio.Future()  # run forever

    def _run():
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(_start())
            loop.run_forever()
        except Exception:  # noqa: BLE001 -- teardown races
            pass

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    assert ready.wait(5), "pg server did not start"
    port = box["port"]

    # Register lineitem directly on the catalog (no query in flight during setup,
    # same as the ws fixture — safe because the worker only touches the catalog
    # while executing a submitted fn).
    server.catalog.register("lineitem", str(pg_dir / "lineitem"))

    ref_cat = Catalog(str(pg_dir))
    ref_cat.register("lineitem", str(pg_dir / "lineitem"))
    ref = Engine(ref_cat)

    yield port, server, ref

    pg_srv = box.get("pg_srv")
    if pg_srv is not None:
        loop.call_soon_threadsafe(pg_srv.close)
    loop.call_soon_threadsafe(server.stop)
    loop.call_soon_threadsafe(loop.stop)
    t.join(timeout=5)


# --------------------------------------------------------------------------- #
# throwaway-table helper (register directly on the catalog; no query in flight)
# --------------------------------------------------------------------------- #

def _register(server, tmp_path, name, n_base=3):
    wdir = tmp_path / name
    wdir.mkdir()
    cudf.DataFrame({
        "k": np.arange(1, n_base + 1, dtype=np.int64),
        "v": np.arange(1.0, n_base + 1, dtype="float64"),
    }).to_pandas().to_parquet(wdir / "0.parquet")
    server.catalog.register(name, str(wdir))
    return wdir


# --------------------------------------------------------------------------- #
# raw v3 protocol client
# --------------------------------------------------------------------------- #

def _fmsg(t: bytes, payload: bytes) -> bytes:
    """One typed frontend message: type + int32 length (incl. 4) + payload."""
    return t + struct.pack(">I", len(payload) + 4) + payload


async def _startup(reader, writer) -> tuple[str, list]:
    """Send a v3 StartupMessage; drain until ReadyForQuery. Return (status, msgs)."""
    body = struct.pack(">I", 196608) + b"user\x00ryudb\x00database\x00ryudb\x00\x00"
    writer.write(struct.pack(">I", len(body) + 4) + body)
    await writer.drain()
    return await _drain_until_ready(reader)


async def _recv_msg(reader) -> tuple[bytes, bytes]:
    t = await reader.readexactly(1)
    (length,) = struct.unpack(">I", await reader.readexactly(4))
    payload = await reader.readexactly(length - 4)
    return t, payload


async def _drain_until_ready(reader) -> tuple[str, list]:
    """Read backend messages until ReadyForQuery (``Z``); return (status, msgs)."""
    msgs: list[tuple[bytes, bytes]] = []
    while True:
        t, p = await _recv_msg(reader)
        if t == b"Z":
            return p.decode("ascii"), msgs
        msgs.append((t, p))


def _parse_row_description(payload: bytes) -> list[tuple[str, int]]:
    """``T`` -> list of (column name, type OID)."""
    (nfields,) = struct.unpack(">H", payload[:2])
    out, off = [], 2
    for _ in range(nfields):
        z = payload.find(b"\x00", off)
        name = payload[off:z].decode("utf-8")
        off = z + 1 + 4 + 2          # skip table OID (i32) + attnum (i16)
        (oid,) = struct.unpack(">I", payload[off:off + 4])
        off += 4 + 2 + 4 + 2         # skip typsize(i16) + typmod(i32) + fmt(i16)
        out.append((name, oid))
    return out


def _parse_data_row(payload: bytes) -> list[str | None]:
    """``D`` -> list of cell text values (None for NULL)."""
    (ncols,) = struct.unpack(">H", payload[:2])
    out, off = [], 2
    for _ in range(ncols):
        (plen,) = struct.unpack(">i", payload[off:off + 4])
        off += 4
        if plen == -1:
            out.append(None)
        else:
            out.append(payload[off:off + plen].decode("utf-8"))
            off += plen
    return out


def _parse_error(payload: bytes) -> dict[str, str]:
    """``E`` ErrorResponse -> {S, V, C, M, ...}."""
    out: dict[str, str] = {}
    off = 0
    while off < len(payload) and payload[off] != 0:
        tag = chr(payload[off])
        z = payload.find(b"\x00", off + 1)
        out[tag] = payload[off + 1:z].decode("utf-8", "replace")
        off = z + 1
    return out


def _parse_command_complete(payload: bytes) -> str:
    return payload.split(b"\x00", 1)[0].decode("utf-8")


def _collect(msgs: list) -> dict:
    """Walk backend messages (between two ReadyForQuery boundaries) into a
    structured dict: rows, columns, tag, error."""
    out: dict[str, Any] = {"columns": None, "rows": [], "tag": None,
                           "error": None, "msgs": msgs}
    for t, p in msgs:
        if t == b"T":
            out["columns"] = _parse_row_description(p)
        elif t == b"D":
            out["rows"].append(_parse_data_row(p))
        elif t == b"C":
            out["tag"] = _parse_command_complete(p)
        elif t == b"E":
            out["error"] = _parse_error(p)
        elif t == b"n":
            out["nodata"] = True
        elif t == b"t":
            out["param_oids"] = [struct.unpack(">I", p[2 + 4 * i:6 + 4 * i])[0]
                                 for i in range(struct.unpack(">H", p[:2])[0])]
    return out


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------- #
# startup + simple query
# --------------------------------------------------------------------------- #

def test_startup_handshake(pg):
    """StartupMessage -> AuthenticationOk (R 0) + BackendKeyData (K) +
    ReadyForQuery (Z 'I')."""
    port, *_ = pg

    async def client():
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        status, msgs = await _startup(reader, writer)
        writer.close()
        return status, msgs
    status, msgs = _run(client())
    assert status == "I"               # the Z (ReadyForQuery) is consumed as status
    types = [t for t, _ in msgs]
    assert types[0] == b"R"            # AuthenticationOk
    assert struct.unpack(">I", msgs[0][1])[0] == 0
    assert b"K" in types               # BackendKeyData (pid + secret)
    assert b"Z" not in types           # Z is the consumed ReadyForQuery status


def test_simple_select_matches_engine(pg):
    """Simple Query SELECT -> RowDescription + DataRow* + CommandComplete +
    ReadyForQuery, matching the in-process engine."""
    port, _, ref = pg
    sql = ("SELECT l_returnflag, count(*) AS c FROM lineitem "
           "GROUP BY l_returnflag ORDER BY l_returnflag")

    async def client():
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        await _startup(reader, writer)
        writer.write(_fmsg(b"Q", sql.encode() + b"\x00"))
        await writer.drain()
        status, msgs = await _drain_until_ready(reader)
        writer.close()
        return status, msgs
    status, msgs = _run(client())
    assert status == "I"
    res = _collect(msgs)
    assert res["columns"] is not None
    assert [c[0] for c in res["columns"]] == ["l_returnflag", "c"]
    # row count + values match the in-process engine
    exp = ref.sql(sql).to_pandas().sort_values("l_returnflag").reset_index(drop=True)
    assert len(res["rows"]) == len(exp)
    for got, (_, erow) in zip(res["rows"], exp.iterrows()):
        assert got[0] == erow["l_returnflag"]
        assert int(got[1]) == int(erow["c"])
    assert res["tag"].startswith("SELECT ")


def test_simple_insert_then_count(pg, tmp_path):
    """INSERT over the wire -> CommandComplete ``INSERT 0 1``; a follow-up
    SELECT count sees the row (autocommit)."""
    port, server, _ = pg
    _register(server, tmp_path, "pg_ins", n_base=3)

    async def client():
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        await _startup(reader, writer)
        writer.write(_fmsg(b"Q",
            b"INSERT INTO pg_ins (k, v) VALUES (999, 99.0)\x00"))
        await writer.drain()
        s1, m1 = await _drain_until_ready(reader)
        writer.write(_fmsg(b"Q", b"SELECT count(*) AS c FROM pg_ins\x00"))
        await writer.drain()
        s2, m2 = await _drain_until_ready(reader)
        writer.close()
        return s1, m1, s2, m2
    s1, m1, s2, m2 = _run(client())
    assert s1 == "I" and s2 == "I"
    assert _collect(m1)["tag"] == "INSERT 0 1"
    r2 = _collect(m2)
    assert r2["rows"] == [["4"]]                       # 3 base + 1 inserted


def test_simple_query_empty(pg):
    """An empty Simple Query -> EmptyQueryResponse (``I``) + ReadyForQuery."""
    port, *_ = pg

    async def client():
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        await _startup(reader, writer)
        writer.write(_fmsg(b"Q", b"\x00"))
        await writer.drain()
        status, msgs = await _drain_until_ready(reader)
        writer.close()
        return status, msgs
    status, msgs = _run(client())
    assert status == "I"
    assert b"I" in [t for t, _ in msgs]               # EmptyQueryResponse


# --------------------------------------------------------------------------- #
# transactions: BEGIN/COMMIT/ROLLBACK + aborted-txn gate
# --------------------------------------------------------------------------- #

def test_begin_commit_rollback_status(pg, tmp_path):
    """BEGIN -> ``T``; COMMIT/ROLLBACK -> ``I``. The ReadyForQuery status
    tracks the per-connection transaction."""
    port, server, _ = pg
    _register(server, tmp_path, "pg_txn", n_base=3)

    async def client():
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        await _startup(reader, writer)
        writer.write(_fmsg(b"Q", b"BEGIN\x00"))
        await writer.drain()
        s_b, m_b = await _drain_until_ready(reader)
        writer.write(_fmsg(b"Q", b"SELECT count(*) AS c FROM pg_txn\x00"))
        await writer.drain()
        s_q, m_q = await _drain_until_ready(reader)
        writer.write(_fmsg(b"Q", b"COMMIT\x00"))
        await writer.drain()
        s_c, m_c = await _drain_until_ready(reader)
        writer.write(_fmsg(b"Q", b"BEGIN\x00"))
        await writer.drain()
        _, _ = await _drain_until_ready(reader)
        writer.write(_fmsg(b"Q", b"ROLLBACK\x00"))
        await writer.drain()
        s_r, m_r = await _drain_until_ready(reader)
        writer.close()
        return s_b, m_b, s_q, m_q, s_c, m_c, s_r, m_r
    s_b, m_b, s_q, m_q, s_c, m_c, s_r, m_r = _run(client())
    assert s_b == "T" and _collect(m_b)["tag"] == "BEGIN"
    assert s_q == "T"                                  # still in txn
    assert s_c == "I" and _collect(m_c)["tag"] == "COMMIT"
    assert s_r == "I" and _collect(m_r)["tag"] == "ROLLBACK"


def test_aborted_txn_gate(pg, tmp_path):
    """An error inside a txn -> status ``E``; further statements are gated
    (25P02) until ROLLBACK clears the gate."""
    port, server, _ = pg
    _register(server, tmp_path, "pg_gate", n_base=3)

    async def client():
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        await _startup(reader, writer)
        writer.write(_fmsg(b"Q", b"BEGIN\x00"))
        await writer.drain()
        _, _ = await _drain_until_ready(reader)
        # runtime error: unknown table
        writer.write(_fmsg(b"Q", b"SELECT * FROM no_such_table\x00"))
        await writer.drain()
        s_e, m_e = await _drain_until_ready(reader)
        # gated: a good statement is refused while aborted
        writer.write(_fmsg(b"Q", b"SELECT count(*) AS c FROM pg_gate\x00"))
        await writer.drain()
        s_g, m_g = await _drain_until_ready(reader)
        # ROLLBACK clears the gate
        writer.write(_fmsg(b"Q", b"ROLLBACK\x00"))
        await writer.drain()
        s_r, m_r = await _drain_until_ready(reader)
        # now a good statement runs again
        writer.write(_fmsg(b"Q", b"SELECT count(*) AS c FROM pg_gate\x00"))
        await writer.drain()
        s_ok, m_ok = await _drain_until_ready(reader)
        writer.close()
        return s_e, m_e, s_g, m_g, s_r, m_r, s_ok, m_ok
    s_e, m_e, s_g, m_g, s_r, m_r, s_ok, m_ok = _run(client())
    assert s_e == "E"
    assert _collect(m_e)["error"] is not None
    assert s_g == "E"
    err = _collect(m_g)["error"]
    assert err is not None and err["C"] == "25P02"     # aborted txn gate
    assert s_r == "I" and _collect(m_r)["tag"] == "ROLLBACK"
    assert s_ok == "I"
    assert _collect(m_ok)["rows"] == [["3"]]           # gate cleared


def test_error_response_shape(pg):
    """A parse error -> ErrorResponse with S/V/C/M fields (SQLSTATE 42601)."""
    port, *_ = pg

    async def client():
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        await _startup(reader, writer)
        writer.write(_fmsg(b"Q", b"SELECT FROM\x00"))
        await writer.drain()
        status, msgs = await _drain_until_ready(reader)
        writer.close()
        return status, msgs
    status, msgs = _run(client())
    # autocommit error -> status I (the implicit txn rolled back); the error
    # frame is still emitted.
    assert status == "I"
    err = _collect(msgs)["error"]
    assert err is not None
    assert err["S"] in ("ERROR", "FATAL")
    assert err["C"] == "42601"                         # syntax_error
    assert err["M"]


# --------------------------------------------------------------------------- #
# extended query (Parse/Bind/Describe/Execute/Sync)
# --------------------------------------------------------------------------- #

def _parse_msg(name: bytes, sql: bytes, oids: list[int]) -> bytes:
    return _cstring(name) + _cstring(sql) + struct.pack(">H", len(oids)) + \
        b"".join(struct.pack(">I", o) for o in oids)


def _cstring(s: bytes) -> bytes:
    return s + b"\x00"


def _bind(portal: bytes, stmt: bytes, params: list[bytes | None],
          oids_unused=()) -> bytes:
    # formats: 0 (text) for all params; result format: 0 (text)
    out = _cstring(portal) + _cstring(stmt)
    out += struct.pack(">H", 0)                        # 0 param formats -> text
    out += struct.pack(">H", len(params))
    for p in params:
        if p is None:
            out += struct.pack(">i", -1)
        else:
            out += struct.pack(">i", len(p)) + p
    out += struct.pack(">H", 0)                        # 0 result formats -> text
    return out


def _describe(what: bytes, name: bytes) -> bytes:
    return what + _cstring(name)


def _execute(portal: bytes, maxrows: int = 0) -> bytes:
    return _cstring(portal) + struct.pack(">I", maxrows)


def test_extended_select_no_params(pg):
    """Parse/Bind/Describe(portal)/Execute/Sync on a 0-param SELECT fires the
    fused path's row stream: RowDescription (from Describe) + DataRow* +
    CommandComplete, without executing twice."""
    port, _, ref = pg
    sql = ("SELECT l_returnflag, count(*) AS c FROM lineitem "
           "GROUP BY l_returnflag ORDER BY l_returnflag")

    async def client():
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        await _startup(reader, writer)
        writer.write(_fmsg(b"P", _parse_msg(b"", sql.encode(), [])))
        writer.write(_fmsg(b"B", _bind(b"", b"", [])))
        writer.write(_fmsg(b"D", _describe(b"P", b"")))   # portal describe
        writer.write(_fmsg(b"E", _execute(b"", 0)))
        writer.write(_fmsg(b"S", b""))                    # Sync -> ReadyForQuery
        await writer.drain()
        status, msgs = await _drain_until_ready(reader)
        writer.close()
        return status, msgs
    status, msgs = _run(client())
    assert status == "I"
    res = _collect(msgs)
    assert [c[0] for c in res["columns"]] == ["l_returnflag", "c"]
    exp = ref.sql(sql).to_pandas().sort_values("l_returnflag").reset_index(drop=True)
    assert len(res["rows"]) == len(exp)
    assert res["tag"].startswith("SELECT ")


def test_extended_parameterized_int(pg, tmp_path):
    """A parameterized SELECT with an int8 ``$1``: the Bind value is substituted
    as a numeric literal and the filtered row returns."""
    port, server, _ = pg
    _register(server, tmp_path, "pg_pint", n_base=3)

    async def client():
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        await _startup(reader, writer)
        sql = b"SELECT k, v FROM pg_pint WHERE k = $1 ORDER BY k"
        writer.write(_fmsg(b"P", _parse_msg(b"ps", sql, [20])))  # PG_INT8
        writer.write(_fmsg(b"B", _bind(b"po", b"ps", [b"2"])))
        writer.write(_fmsg(b"E", _execute(b"po", 0)))
        writer.write(_fmsg(b"S", b""))
        await writer.drain()
        status, msgs = await _drain_until_ready(reader)
        writer.close()
        return status, msgs
    status, msgs = _run(client())
    assert status == "I"
    res = _collect(msgs)
    assert res["rows"] == [["2", "2.0"]]                # k=2, v=2.0


def test_extended_parameterized_string(pg):
    """A parameterized SELECT with a text ``$1`` over the shared lineitem: the
    Bind value is substituted as a quoted string literal."""
    port, _, ref = pg

    async def client():
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        await _startup(reader, writer)
        sql = b"SELECT count(*) AS c FROM lineitem WHERE l_returnflag = $1"
        writer.write(_fmsg(b"P", _parse_msg(b"ps", sql, [25])))  # PG_TEXT
        writer.write(_fmsg(b"B", _bind(b"po", b"ps", [b"N"])))
        writer.write(_fmsg(b"E", _execute(b"po", 0)))
        writer.write(_fmsg(b"S", b""))
        await writer.drain()
        status, msgs = await _drain_until_ready(reader)
        writer.close()
        return status, msgs
    status, msgs = _run(client())
    assert status == "I"
    res = _collect(msgs)
    exp = int(ref.sql("SELECT count(*) AS c FROM lineitem WHERE l_returnflag = 'N'")
               .to_pandas()["c"].iloc[0])
    assert res["rows"] == [[str(exp)]]


def test_extended_null_param(pg, tmp_path):
    """A NULL Bind parameter substitutes to SQL NULL (``v IS NULL`` style via a
    comparison that yields no rows)."""
    port, server, _ = pg
    _register(server, tmp_path, "pg_null", n_base=3)

    async def client():
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        await _startup(reader, writer)
        sql = b"SELECT count(*) AS c FROM pg_null WHERE k = $1"
        writer.write(_fmsg(b"P", _parse_msg(b"ps", sql, [20])))
        writer.write(_fmsg(b"B", _bind(b"po", b"ps", [None])))
        writer.write(_fmsg(b"E", _execute(b"po", 0)))
        writer.write(_fmsg(b"S", b""))
        await writer.drain()
        status, msgs = await _drain_until_ready(reader)
        writer.close()
        return status, msgs
    status, msgs = _run(client())
    assert status == "I"
    res = _collect(msgs)
    # k = NULL is never true -> 0 rows
    assert res["rows"] == [["0"]]


def test_statement_describe_emits_param_description(pg):
    """A statement-Describe (``D S``) emits ParameterDescription (``t``) +
    NoData (``n``) — no execution (params aren't bound yet)."""
    port, *_ = pg

    async def client():
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        await _startup(reader, writer)
        sql = b"SELECT count(*) AS c FROM lineitem WHERE l_quantity > $1"
        writer.write(_fmsg(b"P", _parse_msg(b"ps", sql, [23])))  # PG_INT4
        writer.write(_fmsg(b"D", _describe(b"S", b"ps")))
        writer.write(_fmsg(b"S", b""))
        await writer.drain()
        status, msgs = await _drain_until_ready(reader)
        writer.close()
        return status, msgs
    status, msgs = _run(client())
    assert status == "I"
    res = _collect(msgs)
    assert res.get("param_oids") == [23]
    assert res.get("nodata") is True
    assert res["rows"] == []                           # no execution on a describe


# --------------------------------------------------------------------------- #
# terminate
# --------------------------------------------------------------------------- #

def test_terminate_closes_connection(pg):
    """A Terminate (``X``) message ends the session; the server closes the
    TCP connection (reader hits EOF)."""
    port, *_ = pg

    async def client():
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        await _startup(reader, writer)
        writer.write(_fmsg(b"X", b""))
        await writer.drain()
        # server closes after Terminate -> readexactly raises IncompleteReadError
        with pytest.raises(asyncio.IncompleteReadError):
            await reader.readexactly(1)
        writer.close()
    _run(client())


# --------------------------------------------------------------------------- #
# CancelRequest (pid, secret) -> cooperative cancel of an in-flight statement
# --------------------------------------------------------------------------- #

def test_cancel_request_aborts_inflight(pg):
    """A CancelRequest on a SEPARATE connection carries the target's
    (pid, secret) (from the BackendKeyData we emit at startup) and sets the
    target session's in-flight cancel Event. We block the worker INSIDE the
    first scan of conn A's query; the scan self-releases once it observes the
    cancel Event set (the CancelRequest is sent on conn B), then the next
    plan-node boundary (the join's right-input _exec) raises CancelledByUser ->
    conn A gets an ErrorResponse 57014 + ReadyForQuery (the cooperative cancel
    from PR #58, now reachable over the PG wire via CancelRequest).

    ``engine.cancel_event`` is a per-thread slot (threading.local) under the
    worker pool, so the cancel-confirmation poll must run on the SAME worker
    thread that owns conn A's request -- i.e. inside ``blocking_scan`` -- not on
    the asyncio loop thread (which would see a different, empty slot). The scan
    reads ``engine.cancel_event`` (its own thread's slot = the request's Event)
    and returns once ``cancel_backend`` sets it; the next _exec boundary then
    sees the set Event and raises. No external ``hold`` is needed."""
    port, server, ref = pg
    started = threading.Event()
    orig = server.engine._scan  # bound method (class _scan bound to the instance)
    first = [True]

    def blocking_scan(table, columns):
        # Patched onto the instance dict, so self._scan(a, b) calls this with no
        # bound self -> signature is (table, columns); orig is the bound original.
        if first[0]:
            first[0] = False
            started.set()
            # Hold inside the first scan until the CancelRequest sets this
            # request's cancel Event. cancel_backend runs on the loop thread;
            # engine.cancel_event is a per-thread slot and this scan runs on the
            # SAME worker thread that owns the request, so reading it here sees
            # the request's Event. Then return -> the next _exec boundary (the
            # join's right scan) sees the set Event and raises CancelledByUser
            # -> ErrorResponse 57014.
            for _ in range(500):  # ~5s
                ev = server.engine.cancel_event
                if ev is not None and ev.is_set():
                    break
                time.sleep(0.01)
            else:
                raise AssertionError("cancel Event never set within 5s")
        return orig(table, columns)

    server.engine._scan = blocking_scan
    try:
        async def client():
            # conn A: a real session. Capture its (pid, secret) from BackendKeyData.
            ra, wa = await asyncio.open_connection("127.0.0.1", port)
            _status, msgs = await _startup(ra, wa)
            pid = secret = None
            for t, p in msgs:
                if t == b"K":
                    pid, secret = struct.unpack(">II", p)
            assert pid is not None and secret is not None, "no BackendKeyData"

            # Self-join on lineitem: two scans. Plan Limit -> Project -> Join ->
            # Scan(a), Scan(b). The first scan blocks (self-releasing on cancel);
            # after release the join's right-input _exec hits the cancel check and
            # raises CancelledByUser.
            qsql = (b"SELECT a.l_orderkey FROM lineitem a JOIN lineitem b "
                    b"ON a.l_orderkey = b.l_orderkey LIMIT 5")
            wa.write(_fmsg(b"Q", qsql + b"\x00"))
            await wa.drain()
            qtask = asyncio.create_task(_drain_until_ready(ra))

            # Wait for the worker to reach the blocking scan (conn A's request is
            # now in flight; its cancel Event is published to the worker's
            # per-thread slot).
            for _ in range(500):
                if started.is_set():
                    break
                await asyncio.sleep(0.01)
            assert started.is_set(), "conn A query did not reach the blocking scan"

            # conn B: a short-lived CancelRequest connection. It sends ONLY the
            # CancelRequest startup message (version 80877102 + pid + secret) and
            # closes; the server sets conn A's cancel Event (cancel_backend) and
            # closes conn B. blocking_scan on conn A's worker observes the set
            # Event and returns -> the next node boundary raises CancelledByUser.
            rb, wb = await asyncio.open_connection("127.0.0.1", port)
            body = struct.pack(">I", CANCEL_REQUEST) + struct.pack(">II", pid, secret)
            wb.write(struct.pack(">I", len(body) + 4) + body)
            await wb.drain()
            wb.close()
            await wb.wait_closed()

            # conn A's query resolves with ErrorResponse 57014 + ReadyForQuery.
            status, qmsgs = await asyncio.wait_for(qtask, timeout=5)
            err = None
            for t, p in qmsgs:
                if t == b"E":
                    err = _parse_error(p)
            assert err is not None, "expected an ErrorResponse (cancel)"
            assert err.get("C") == "57014", f"expected SQLSTATE 57014, got {err.get('C')}"
            assert status == "I"  # autocommit SELECT, not in a txn
            wa.close()
            await wa.wait_closed()

        _run(client())
    finally:
        server.engine._scan = orig


def test_cancel_request_unknown_backend_is_noop(pg):
    """A CancelRequest for a (pid, secret) that matches no live session is a
    no-op: the cancel connection just closes, and a concurrent real query on a
    DIFFERENT connection runs to completion (not cancelled)."""
    port, *_ = pg

    async def client():
        # conn A: a real session running a normal query to completion.
        ra, wa = await asyncio.open_connection("127.0.0.1", port)
        await _startup(ra, wa)
        wa.write(_fmsg(b"Q", b"SELECT count(*) FROM lineitem\x00"))
        await wa.drain()
        qtask = asyncio.create_task(_drain_until_ready(ra))

        # conn B: CancelRequest with a bogus (pid, secret) that matches nobody.
        rb, wb = await asyncio.open_connection("127.0.0.1", port)
        body = struct.pack(">I", CANCEL_REQUEST) + struct.pack(">II", 1, 999999)
        wb.write(struct.pack(">I", len(body) + 4) + body)
        await wb.drain()
        wb.close()
        await wb.wait_closed()

        # conn A's query must complete normally (a count), NOT be cancelled.
        status, qmsgs = await asyncio.wait_for(qtask, timeout=5)
        tag = None
        for t, p in qmsgs:
            if t == b"C":
                tag = _parse_command_complete(p)
        assert tag is not None, "conn A query did not complete normally"
        assert not any(t == b"E" for t, _ in qmsgs), "conn A was unexpectedly cancelled"
        wa.close()
        await wa.wait_closed()

    _run(client())


# --------------------------------------------------------------------------- #
# real-driver interop (optional; skips if pg8000 is not installed)
# --------------------------------------------------------------------------- #

def test_pg8000_end_to_end(pg, tmp_path):
    """A real Postgres driver (pg8000) connects, runs parameterized SQL, and
    reads rows through the wire front. Skips if pg8000 is not installed."""
    pytest.importorskip("pg8000")
    import pg8000
    port, server, _ = pg
    _register(server, tmp_path, "pg_drv", n_base=3)

    conn = pg8000.connect(user="ryudb", host="127.0.0.1", port=port,
                          database="ryudb", ssl_context=False, timeout=10)
    try:
        cur = conn.cursor()
        # pg8000 defaults to autocommit=False, so it implicitly begins a
        # transaction before the first statement (status 'T'). No explicit
        # BEGIN — that would nest and the engine refuses nested txns.
        cur.execute("SELECT k, v FROM pg_drv ORDER BY k")
        rows = cur.fetchall()
        assert list(rows) == [[1, 1.0], [2, 2.0], [3, 3.0]]
        # parameterized (extended protocol: Parse/Bind/Describe-Statement/
        # Execute/Sync) — exercises $N substitution + Execute-emitted
        # RowDescription (no portal-Describe precedes it).
        cur.execute("SELECT count(*) FROM pg_drv WHERE k = %s", (2,))
        assert cur.fetchone()[0] == 1
        # write + read-your-writes inside the implicit txn, then rollback.
        cur.execute("INSERT INTO pg_drv (k, v) VALUES (999, 99.0)")
        cur.execute("SELECT count(*) FROM pg_drv")
        assert cur.fetchone()[0] == 4           # read-your-writes: 3 base + 1 buffered
        conn.rollback()                         # drop the buffered insert
        cur.execute("SELECT count(*) FROM pg_drv")
        assert cur.fetchone()[0] == 3           # rollback discarded the buffer
    finally:
        conn.close()