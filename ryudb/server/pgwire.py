"""PostgreSQL v3 wire-protocol server front for the RyuDB engine.

Speaks enough of the frontend/backend protocol that real drivers (``psql``,
``psycopg``, ``pg8000``, ``asyncpg``, JDBC) can connect, run SQL, and receive
results — instead of the custom JSON+Arrow WebSocket protocol. Shares the
single ``Engine`` + worker + per-connection transaction model with the
WebSocket ``Server`` (``ryudb/server/app.py``): each PG connection is a session
with its own ``_ryudb_txn`` routed through ``Server.run_sql``, so the
cross-session MVCC isolation (PR #61) applies identically over both fronts.

Supported
  - Startup v3.0 + ``AuthenticationOk`` (no auth; the server binds
    ``127.0.0.1`` for a local single-user console). ``SSLRequest`` /
    ``GSSRequest`` are refused with a single ``N`` byte (the client then
    retries plain).
  - Simple Query (``Q``): ``RowDescription`` + ``DataRow``* (text format) +
    ``CommandComplete`` + ``ReadyForQuery``.
  - Extended Query (``P``/``B``/``D``/``E``/``S``/``C``): Parse/Bind/Describe/
    Execute/Sync/Close. ``$N`` placeholders are substituted with typed
    literals via a sqlglot AST walk (``exp.Parameter`` -> ``exp.Literal``),
    so the engine's string-SQL entry point (``engine.sql``) is reused as-is.
  - ``Terminate`` (``X``), ``ErrorResponse`` (``E``) with SQLSTATE,
    ``EmptyQueryResponse`` (``I``), ``NoData`` (``n``).
  - Per-connection transactions; ``ReadyForQuery`` status ``I``/``T``/``E``
    (idle / in-txn / aborted-txn — the latter gates further statements on
    ``ROLLBACK``, matching real Postgres).

Not supported (documented limits)
  - SSL/TLS, GSSAPI, SCRAM/password auth (no auth). ``CancelRequest`` is not
    wired to the cooperative cancel (PR #58) yet — a future PR can map
    ``(pid, secret)`` to the connection's pending cancel event.
  - ``COPY``, ``LISTEN``/``NOTIFY``, portal scroll, binary result format
    (text only), binary param format (text only). A client requesting binary
    results/params gets an error.
  - Result rows are capped at ``max_rows`` (the same cap as the WebSocket
    front); the PG protocol has no truncation signal, so the cap is silent
    (documented). Raise ``--max-rows`` / ``--pg-max-rows`` for full exports.
  - Parameter type inference uses the Parse param OIDs; an unknown OID (0)
    renders the param as a quoted string literal and lets the engine coerce.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import math
import os
import secrets
import struct
from decimal import Decimal
from typing import Any

import pyarrow as pa
import sqlglot
from sqlglot import exp

from .errors import classify
from .protocol import df_to_arrow
from .worker import Cancelled
from ..exec.executor import CancelledByUser

log = logging.getLogger("ryudb.server.pgwire")

# --------------------------------------------------------------------------- #
# Postgres type OIDs (only the ones we emit)
# --------------------------------------------------------------------------- #
PG_BOOL = 16
PG_INT8 = 20
PG_INT2 = 21
PG_INT4 = 23
PG_TEXT = 25
PG_FLOAT4 = 700
PG_FLOAT8 = 701
PG_VARCHAR = 1043
PG_DATE = 1082
PG_TIMESTAMP = 1114
PG_NUMERIC = 1700

# Protocol version codes carried in the StartupMessage's version field.
PROTOCOL_V3 = 196608
SSL_REQUEST = 80877103
CANCEL_REQUEST = 80877102
GSS_REQUEST = 80877104


def _arrow_type_oid(t: pa.DataType) -> int:
    """Map a pyarrow column type to a Postgres type OID (text-format results)."""
    if pa.types.is_int64(t):
        return PG_INT8
    if pa.types.is_int32(t):
        return PG_INT4
    if pa.types.is_int16(t) or pa.types.is_int8(t):
        return PG_INT2
    if pa.types.is_float64(t):
        return PG_FLOAT8
    if pa.types.is_float32(t):
        return PG_FLOAT4
    if pa.types.is_boolean(t):
        return PG_BOOL
    if pa.types.is_string(t) or pa.types.is_large_string(t):
        return PG_TEXT
    if pa.types.is_date32(t) or pa.types.is_date64(t):
        return PG_DATE
    if pa.types.is_timestamp(t):
        return PG_TIMESTAMP
    if pa.types.is_decimal(t):
        return PG_NUMERIC
    return PG_TEXT  # structs/lists/unknown -> text (stringified)


def _val_to_text(v: Any) -> str | None:
    """Render one Python value (from ``pyarrow.Table.to_pydict``) as a Postgres
    text-format cell. ``None`` -> ``None`` (caller emits a NULL)."""
    if v is None:
        return None
    if isinstance(v, bool):
        return "t" if v else "f"
    if isinstance(v, float):
        if math.isnan(v):
            return "NaN"
        if math.isinf(v):
            return "Infinity" if v > 0 else "-Infinity"
        return repr(v)
    if isinstance(v, datetime.datetime):
        # PG timestamp text uses a space (not 'T') between date and time.
        return v.isoformat(sep=" ")
    if isinstance(v, datetime.date):
        return v.isoformat()
    if isinstance(v, Decimal):
        return str(v)
    return str(v)


def _param_literal(value: bytes | None, oid: int) -> str:
    """Render one Bind parameter (text format) as a SQL literal for ``$N``
    substitution. ``value is None`` (NULL param) -> ``NULL``."""
    if value is None:
        return "NULL"
    s = value.decode("utf-8", errors="replace")
    if oid in (PG_INT2, PG_INT4, PG_INT8):
        # numeric literal; validate digits so a malformed text param can't
        # inject SQL (a real driver sends a parseable int here).
        try:
            return str(int(s))
        except ValueError:
            return "NULL"
    if oid in (PG_FLOAT4, PG_FLOAT8):
        try:
            return repr(float(s))
        except ValueError:
            return "NULL"
    if oid == PG_BOOL:
        return "TRUE" if s in ("t", "true", "TRUE", "1") else "FALSE"
    # text / varchar / date / timestamp / numeric / unknown(0): quoted string
    # literal with single-quotes doubled (the standard SQL escape).
    return "'" + s.replace("'", "''") + "'"


def _substitute_params(sql: str, params: list[bytes | None], oids: list[int]) -> str:
    """Replace ``$N`` placeholders in ``sql`` with the bound parameters rendered
    as typed literals, via a sqlglot AST walk (robust against ``$N`` inside
    string literals — only real placeholders are replaced). Returns SQL ready
    for ``engine.sql``."""
    if not params:
        return sql
    tree = sqlglot.parse_one(sql, read="postgres")
    by_index: dict[str, bytes | None] = {}
    for i, p in enumerate(params, start=1):
        by_index[str(i)] = p
    for node in list(tree.find_all(exp.Parameter)):
        idx = node.name or (node.this.name if isinstance(node.this, exp.Literal) else "")
        val = by_index.get(idx)
        oid = oids[i - 1] if i - 1 < len(oids) else 0
        node.replace(_param_literal_node(val, oid))
    return tree.sql(dialect="postgres")


def _param_literal_node(value: bytes | None, oid: int) -> exp.Expression:
    """Build a sqlglot literal/NULL node for one parameter (see
    ``_param_literal`` for the typing rules)."""
    if value is None:
        return exp.Null()
    s = value.decode("utf-8", errors="replace")
    if oid in (PG_INT2, PG_INT4, PG_INT8):
        try:
            return exp.Literal.number(str(int(s)))
        except ValueError:
            return exp.Null()
    if oid in (PG_FLOAT4, PG_FLOAT8):
        try:
            return exp.Literal.number(repr(float(s)))
        except ValueError:
            return exp.Null()
    if oid == PG_BOOL:
        return exp.Boolean(this=s in ("t", "true", "TRUE", "1"))
    return exp.Literal.string(s)


def _command_tag(sql: str, result: Any) -> str:
    """The ``CommandComplete`` tag (e.g. ``SELECT 3``, ``INSERT 0 1``,
    ``BEGIN``). The engine returns a DataFrame for SELECT, an int for writes,
    None for control/DDL."""
    if isinstance(result, int):
        kw = _first_keyword(sql)
        if kw == "INSERT":
            return f"INSERT 0 {result}"
        if kw == "DELETE":
            return f"DELETE {result}"
        if kw == "UPDATE":
            return f"UPDATE {result}"
        if kw == "MERGE":
            return f"MERGE {result}"
        return f"SELECT {result}"
    if hasattr(result, "num_rows"):  # pyarrow Table (SELECT)
        return f"SELECT {result.num_rows}"
    if result is None:
        kw = _first_keyword(sql)
        if kw == "BEGIN":
            return "BEGIN"
        if kw == "COMMIT":
            return "COMMIT"
        if kw == "ROLLBACK":
            return "ROLLBACK"
        if kw:
            return kw  # CREATE / ALTER / DROP / RESTORE / etc.
        return "OK"
    return "SELECT 0"


def _first_keyword(sql: str) -> str:
    """The first uppercase keyword of ``sql`` (for command-tag + txn-status
    classification). Skips leading whitespace and a possible ``/*...*/``."""
    s = sql.lstrip()
    if s.startswith("/*"):
        i = s.find("*/")
        s = s[i + 2:].lstrip() if i >= 0 else s
    out = []
    for ch in s:
        if ch.isalpha() or ch == "_":
            out.append(ch.upper())
        else:
            break
    return "".join(out)


def _sqlstate(kind: str) -> str:
    """Map a RyuDB error kind (from ``errors.classify``) to a SQLSTATE."""
    if kind == "parse":
        return "42601"  # syntax_error
    return "XX000"  # internal_error


# --------------------------------------------------------------------------- #
# binary message codec
# --------------------------------------------------------------------------- #

def _msg(t: bytes, payload: bytes) -> bytes:
    """One typed backend message: ``type`` + int32 length (incl. the 4 length
    bytes) + payload."""
    return t + struct.pack(">I", len(payload) + 4) + payload


def _cstring(s: str) -> bytes:
    return s.encode("utf-8") + b"\x00"


def _row_description(table: pa.Table) -> bytes:
    """``T`` RowDescription for a result table (text format, format code 0)."""
    fields = table.schema
    parts = [struct.pack(">H", len(fields))]
    for f in fields:
        parts.append(_cstring(f.name))
        parts.append(struct.pack(">I", 0))   # table OID (none)
        parts.append(struct.pack(">H", 0))   # column attnum (none)
        parts.append(struct.pack(">I", _arrow_type_oid(f.type)))
        parts.append(struct.pack(">h", -1))  # type size (variable -> -2; -1 ok)
        parts.append(struct.pack(">i", -1))  # type modifier (none)
        parts.append(struct.pack(">H", 0))   # format code: text
    return _msg(b"T", b"".join(parts))


def _data_rows(table: pa.Table) -> list[bytes]:
    """``D`` DataRow messages (text format) for every row of ``table``."""
    n = table.num_rows
    if n == 0:
        return []
    cols = table.to_pydict()  # column-major dict of python lists
    names = table.schema.names
    rows: list[bytes] = []
    for i in range(n):
        parts = [struct.pack(">H", len(names))]
        for name in names:
            v = cols[name][i]
            txt = _val_to_text(v)
            if txt is None:
                parts.append(struct.pack(">i", -1))  # NULL
            else:
                b = txt.encode("utf-8")
                parts.append(struct.pack(">i", len(b)))
                parts.append(b)
        rows.append(_msg(b"D", b"".join(parts)))
    return rows


def _command_complete(tag: str) -> bytes:
    return _msg(b"C", _cstring(tag))


def _ready(status: str) -> bytes:
    return _msg(b"Z", status.encode("ascii"))


def _error(severity: str, sqlstate: str, message: str) -> bytes:
    """``E`` ErrorResponse: ``S``everity, ``V``everity (PG >= 9.6), ``C``ode,
    ``M``essage, terminated by a ``\0`` field."""
    parts = [
        b"S" + _cstring(severity),
        b"V" + _cstring(severity),
        b"C" + _cstring(sqlstate),
        b"M" + _cstring(message),
        b"\x00",
    ]
    return _msg(b"E", b"".join(parts))


# --------------------------------------------------------------------------- #
# connection
# --------------------------------------------------------------------------- #

class PGConn:
    """One Postgres frontend connection. Holds the per-connection transaction
    (``_ryudb_txn``, managed by ``Server.run_sql``) and the extended-protocol
    prepared-statement / portal state."""

    def __init__(self, server, pg_server, reader, writer) -> None:
        self.server = server            # the shared WS Server (engine/worker/run_sql)
        self.pg = pg_server
        self.reader = reader
        self.writer = writer
        self._ryudb_txn: Any = None     # managed by Server.run_sql
        self.failed = False             # aborted-txn gate (ReadyForQuery 'E')
        self.pid = os.getpid()
        self.secret = secrets.randbits(32)
        self.statements: dict[str, str] = {}   # name -> SQL (Parse)
        self._oids: dict[str, list[int]] = {}  # name -> Parse param OIDs
        self.portals: dict[str, tuple[str, list, list]] = {}  # name -> (sql, params, oids)
        # A portal's result table from a portal-Describe, reused by the next
        # Execute so a Parse/Bind/Describe/Execute sequence doesn't run the
        # query twice (RyuDB has no describe-only path; SELECTs are read-only
        # and idempotent, so caching the Describe result is safe).
        self._portal_cache: dict[str, Any] = {}
        self._cancel: Any = None

    # -- low-level send --
    async def _send(self, data: bytes) -> None:
        self.writer.write(data)
        await self.writer.drain()

    @property
    def in_txn(self) -> bool:
        return self._ryudb_txn is not None

    # -- startup --
    async def startup(self) -> bool:
        """Read StartupMessages until a v3.0 startup (handling SSL/GSS refusal
        and CancelRequest). Returns False if the connection should close."""
        while True:
            hdr = await self.reader.readexactly(4)
            (length,) = struct.unpack(">I", hdr)
            if length < 4 or length > 1_000_000:
                return False
            body = await self.reader.readexactly(length - 4)
            (version,) = struct.unpack(">I", body[:4])
            if version == SSL_REQUEST or version == GSS_REQUEST:
                await self._send(b"N")  # no SSL/GSS; client retries plain
                continue
            if version == CANCEL_REQUEST:
                # int32 pid, int32 secret. Find the target connection and set its
                # cancel event (cooperative cancel, PR #58). Not wired yet: we
                # only parse and close. (A future PR maps pid+secret -> event.)
                if len(body) >= 12:
                    _pid, _secret = struct.unpack(">II", body[4:12])
                return False
            if version == PROTOCOL_V3:
                # key-value cstring pairs (user, database, ...); we ignore them
                # (no auth, local single-user). Terminated by a single \0.
                await self._send(_msg(b"R", struct.pack(">I", 0)))  # AuthenticationOk
                await self._send(_msg(b"K", struct.pack(">II", self.pid, self.secret)))
                await self._send(_ready("I"))
                return True
            # unknown protocol version
            await self._send(_error("FATAL", "08P01", f"unsupported protocol version {version}"))
            return False

    # -- main message loop --
    async def loop(self) -> None:
        while True:
            t = await self.reader.readexactly(1)
            if t == b"X":  # Terminate
                return
            (length,) = struct.unpack(">I", await self.reader.readexactly(4))
            payload = await self.reader.readexactly(length - 4)
            try:
                if t == b"Q":
                    await self._simple_query(payload)
                elif t == b"P":
                    await self._parse(payload)
                elif t == b"B":
                    await self._bind(payload)
                elif t == b"D":
                    await self._describe(payload)
                elif t == b"E":
                    await self._execute(payload)
                elif t == b"S":
                    await self._send(_ready(self._status()))
                elif t == b"C":
                    await self._close(payload)
                elif t == b"H":
                    await self.writer.drain()  # Flush: just drain
                else:
                    await self._send(_error("ERROR", "08P01",
                                            f"unsupported message type {t!r}"))
                    await self._send(_ready(self._status()))
            except asyncio.IncompleteReadError:
                return
            except (Cancelled, CancelledByUser):
                await self._send(_error("ERROR", "57014", "canceling statement due to user request"))
                if self.in_txn:
                    self.failed = True
                await self._send(_ready(self._status()))
            except Exception as exc:  # noqa: BLE001 -- surface as ErrorResponse
                kind, message, _ = classify(exc)
                await self._send(_error("ERROR", _sqlstate(kind), message))
                if self.in_txn:
                    self.failed = True
                await self._send(_ready(self._status()))

    def _status(self) -> str:
        if self.failed:
            return "E"
        return "T" if self.in_txn else "I"

    # -- simple query --
    async def _simple_query(self, payload: bytes) -> None:
        # cstring SQL (drop the trailing \0)
        sql = payload.split(b"\x00", 1)[0].decode("utf-8", errors="replace")
        if not sql.strip():
            await self._send(_msg(b"I", b""))  # EmptyQueryResponse
            await self._send(_ready(self._status()))
            return
        # A simple Query may contain multiple statements separated by ';'.
        # Run each in turn; emit CommandComplete per statement. RowDescription/
        # DataRow precede the CommandComplete of a SELECT.
        for stmt in _split_statements(sql):
            if not await self._gate(stmt):
                continue
            result = await self._exec(stmt)
            await self._emit_result(stmt, result, describe=True)
        await self._send(_ready(self._status()))

    # -- extended query --
    async def _parse(self, payload: bytes) -> None:
        name, rest = _cstring_and_rest(payload)
        sql, rest = _cstring_and_rest(rest)
        # int16 num_params, then int32[] param OIDs
        (nparams,) = struct.unpack(">H", rest[:2])
        oids = list(struct.unpack(">" + "i" * nparams, rest[2:2 + 4 * nparams])) if nparams else []
        self.statements[name] = sql
        self._oids[name] = oids
        await self._send(_msg(b"1", b""))  # ParseComplete

    async def _bind(self, payload: bytes) -> None:
        portal, rest = _cstring_and_rest(payload)
        stmt_name, rest = _cstring_and_rest(rest)
        (nfmt,) = struct.unpack(">H", rest[:2])
        fmts = list(struct.unpack(">" + "H" * nfmt, rest[2:2 + 2 * nfmt])) if nfmt else []
        off = 2 + 2 * nfmt
        (nparams,) = struct.unpack(">H", rest[off:off + 2])
        off += 2
        params: list[bytes | None] = []
        for _ in range(nparams):
            (plen,) = struct.unpack(">i", rest[off:off + 4])
            off += 4
            if plen == -1:
                params.append(None)
            else:
                params.append(bytes(rest[off:off + plen]))
                off += plen
        (nrfmt,) = struct.unpack(">H", rest[off:off + 2])
        off += 2
        rfmts = list(struct.unpack(">" + "H" * nrfmt, rest[off:off + 2 * nrfmt])) if nrfmt else []
        if any(f == 1 for f in rfmts):
            raise RuntimeError("binary result format is not supported (text only)")
        if any(f == 1 for f in fmts):
            raise RuntimeError("binary parameter format is not supported (text only)")
        sql = self.statements.get(stmt_name, "")
        oids = self._oids.get(stmt_name, [])
        self.portals[portal] = (sql, params, oids)
        await self._send(_msg(b"2", b""))  # BindComplete

    async def _describe(self, payload: bytes) -> None:
        what = payload[:1]
        name, _ = _cstring_and_rest(payload[1:])
        if what == b"S":
            # Statement-Describe: the params aren't bound yet, and RyuDB has no
            # schema-inference-without-execution path, so we can't produce a
            # RowDescription here. Send ParameterDescription + NoData; the
            # driver gets the output columns from a portal-Describe (after Bind)
            # or from the Execute path.
            oids = self._oids.get(name, [])
            await self._send(_msg(b"t", struct.pack(">H", len(oids)) +
                                  b"".join(struct.pack(">I", o) for o in oids)))
            await self._send(_msg(b"n", b""))  # NoData
        else:  # portal: params are bound -> execute to infer the output schema
            sql, params, oids = self.portals.get(name, ("", [], []))
            full = _substitute_params(sql, params, oids)
            await self._describe_output(full, name)

    async def _describe_output(self, sql: str, portal: str | None = None) -> None:
        """Send a RowDescription for ``sql``'s output schema, or NoData. We
        infer the schema by executing the query (the engine has no describe-
        only path). For a non-row-returning statement this is NoData. When
        ``portal`` is given the result table is cached for the next Execute
        (avoids running the query twice in a Describe-then-Execute sequence)."""
        if not sql.strip():
            await self._send(_msg(b"n", b""))  # NoData
            return
        kw = _first_keyword(sql)
        if kw not in ("SELECT", "WITH", "VALUES", "TABLE"):
            await self._send(_msg(b"n", b""))  # NoData (not a row-returning stmt)
            return
        try:
            result = await self._exec(sql)
        except Exception:
            await self._send(_msg(b"n", b""))
            return
        if hasattr(result, "num_rows"):
            await self._send(_row_description(result))
            if portal is not None:
                self._portal_cache[portal] = result
        else:
            await self._send(_msg(b"n", b""))

    async def _execute(self, payload: bytes) -> None:
        portal, rest = _cstring_and_rest(payload)
        (maxrows,) = struct.unpack(">I", rest[:4])
        entry = self.portals.get(portal)
        if entry is None:
            await self._send(_msg(b"I", b""))  # EmptyQueryResponse (no such portal)
            return
        sql, params, oids = entry
        # Pop the portal-Describe cache before the gate so a gated (aborted-txn)
        # Execute doesn't leave a stale result for a later post-ROLLBACK Execute.
        cached = self._portal_cache.pop(portal, None)
        if not await self._gate(sql):
            return
        if cached is not None:
            # A portal-Describe already ran the query and sent its
            # RowDescription; reuse the result so we don't execute twice, and
            # don't re-emit the RowDescription.
            result = cached
            emit_desc = False
        else:
            full = _substitute_params(sql, params, oids)
            if not full.strip():
                await self._send(_msg(b"I", b""))  # EmptyQueryResponse
                return
            result = await self._exec(full)
            # No portal-Describe preceded this Execute (e.g. a driver that
            # only described the statement, which answers NoData). Real Postgres
            # emits the RowDescription from Execute in that case, so a row-
            # returning statement's output schema reaches the driver. For a
            # non-row-returning statement ``describe`` is ignored (no T sent).
            emit_desc = True
        await self._emit_result(sql, result, describe=emit_desc, maxrows=maxrows)

    async def _close(self, payload: bytes) -> None:
        what = payload[:1]
        name, _ = _cstring_and_rest(payload[1:])
        if what == b"S":
            self.statements.pop(name, None)
        else:
            self.portals.pop(name, None)
        await self._send(_msg(b"3", b""))  # CloseComplete

    # -- emit a statement's result frames (no exec, no gate) --
    async def _emit_result(self, sql: str, result, describe: bool,
                           maxrows: int = 0) -> None:
        if hasattr(result, "num_rows"):
            table = result
            if maxrows and maxrows < table.num_rows:
                table = table.slice(0, maxrows)
            rows = table
            if self.pg.max_rows and rows.num_rows > self.pg.max_rows:
                rows = rows.slice(0, self.pg.max_rows)
            if describe:
                await self._send(_row_description(table))  # schema (row-count agnostic)
            for chunk in _data_rows(rows):
                await self._send(chunk)
            await self._send(_command_complete(f"SELECT {rows.num_rows}"))
        elif isinstance(result, int):
            await self._send(_command_complete(_command_tag(sql, result)))
        else:  # None: control / DDL
            await self._send(_command_complete(_command_tag(sql, result)))
        # ROLLBACK ends the (possibly aborted) txn -> clear the abort gate.
        if _first_keyword(sql) == "ROLLBACK":
            self.failed = False

    async def _gate(self, sql: str) -> bool:
        """Aborted-txn gate. Returns True if the statement may proceed; False
        if it was blocked (an ErrorResponse has already been sent). Only
        ROLLBACK proceeds while a txn is aborted."""
        if self.failed and _first_keyword(sql) != "ROLLBACK":
            await self._send(_error("ERROR", "25P02",
                                    "current transaction is aborted, commands "
                                    "ignored until end of transaction block"))
            return False
        return True

    async def _exec(self, sql: str):
        """Submit ``sql`` to the worker via ``Server.run_sql`` (per-connection
        txn) and await. Returns a pyarrow Table for SELECT, an int for writes,
        None for control/DDL — matching ``engine.sql``'s return shape, with a
        cuDF DataFrame converted to pyarrow for type mapping."""
        fut, cancel = self.server.submit(self.server.run_sql, self, sql)
        self._cancel = cancel
        try:
            res = await fut
        finally:
            self._cancel = None
        if res is None or isinstance(res, int):
            return res
        # cuDF / pandas DataFrame -> pyarrow Table (canonical path)
        return df_to_arrow(res)


def _split_statements(sql: str) -> list[str]:
    """Split a simple-Query string on top-level ``;`` (rough — drivers usually
    send one statement per Query; ``psql`` may send a block). A trailing ``;``
    does not produce an empty statement."""
    stmts: list[str] = []
    depth = 0
    cur = []
    i = 0
    quote: str | None = None
    while i < len(sql):
        ch = sql[i]
        if quote:
            cur.append(ch)
            if ch == quote:
                quote = None
            elif ch == "\\" and quote == "'":
                # backtick-style escape not standard SQL; keep next char
                if i + 1 < len(sql):
                    cur.append(sql[i + 1])
                    i += 1
        elif ch in ("'", '"'):
            quote = ch
            cur.append(ch)
        elif ch == "(":
            depth += 1
            cur.append(ch)
        elif ch == ")":
            depth -= 1
            cur.append(ch)
        elif ch == ";" and depth == 0:
            s = "".join(cur).strip()
            if s:
                stmts.append(s)
            cur = []
        else:
            cur.append(ch)
        i += 1
    s = "".join(cur).strip()
    if s:
        stmts.append(s)
    return stmts


def _cstring_and_rest(buf: bytes) -> tuple[str, bytes]:
    """Split ``buf`` at the first ``\\0`` into (cstring, rest-after-null)."""
    i = buf.find(b"\x00")
    if i < 0:
        return buf.decode("utf-8", errors="replace"), b""
    return buf[:i].decode("utf-8", errors="replace"), buf[i + 1:]


# --------------------------------------------------------------------------- #
# server
# --------------------------------------------------------------------------- #

class PGServer:
    """A Postgres-wire TCP front sharing one ``Server``'s engine + worker +
    per-connection transaction core. Construct with the WebSocket ``Server``
    (which owns the ``Engine``); ``serve()`` starts the TCP listener."""

    def __init__(self, server, host: str = "127.0.0.1", port: int = 5432,
                 max_rows: int = 200_000) -> None:
        self.server = server
        self.host = host
        self.port = port
        self.max_rows = max_rows
        self._srv: Any = None

    async def _handler(self, reader, writer) -> None:
        conn = PGConn(self.server, self, reader, writer)
        self.server._conns[id(conn)] = conn  # register for has_open_txn
        peer = writer.get_extra_info("peername") if hasattr(writer, "get_extra_info") else None
        log.info("pg connection from %s", peer)
        try:
            if not await conn.startup():
                return
            await conn.loop()
        except (asyncio.IncompleteReadError, ConnectionResetError, BrokenPipeError):
            pass
        except Exception as exc:  # noqa: BLE001 -- never let a conn fault kill the server
            log.exception("pg connection fault: %s", exc)
        finally:
            # Drop the session: an open txn is rolled back by dropping the
            # reference (buffered writes were never flushed). Mirrors the WS
            # disconnect path (PR #61).
            self.server._conns.pop(id(conn), None)
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:  # noqa: BLE001
                pass
            log.info("pg connection closed")

    async def serve(self) -> None:
        self._srv = await asyncio.start_server(self._handler, self.host, self.port)
        log.info("ryudb pgwire listening on pg://%s:%d (data shared with ws front)",
                 self.host, self.port)
        async with self._srv:
            await asyncio.Future()  # run forever

    def stop(self) -> None:
        if self._srv is not None:
            self._srv.close()