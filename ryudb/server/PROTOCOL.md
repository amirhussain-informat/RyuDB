# ryudb-server wire protocol

`ryudb-server` fronts one `Engine` over `data_dir` with **two** wire fronts
sharing the same engine, worker, and per-connection transaction model:

- a **WebSocket** front (custom JSON + Arrow IPC; this document), default
  `127.0.0.1:5430` (`--port` / `RYUDB_PORT`); and
- an optional **Postgres v3 wire-protocol** front (see *Postgres wire front*
  below), enabled with `--pg-port` / `RYUDB_PG_PORT` (default `0` = disabled).

Both fronts dispatch to the same engine worker pool (default **1** worker,
`--workers` / `RYUDB_WORKERS`; `N>1` lets SELECTs run concurrently). The
engine guards its shared state with a read/write lock — SELECTs take the shared
read lock (concurrent reads allowed), writes and admin ops take the exclusive
write lock (serialized) — so correctness is identical across both fronts; a PG
connection and a WS connection are just two sessions over the same engine
(per-connection MVCC isolation applies to both).

## WebSocket front

A client opens a single WebSocket connection to the server and exchanges
frames. The transport is plain WebSocket (no subprotocol). Two frame kinds:

- **text frames** — UTF-8 JSON. Used for every request and for every response
  *except* the binary result payload.
- **binary frames** — either an Arrow IPC stream (sent by the server only, as
  the second frame of a successful `sql`/`sample`/`fetch` result) or an uploaded
  parquet payload (sent by the client, as the second frame of an `upload`
  request — see *upload* below). An unexpected binary frame at any other point
  closes the connection with code 1003.

`ryudb-server` listens on `127.0.0.1:5430` by default (`--host`/`--port`, or the
`RYUDB_HOST`/`RYUDB_PORT` env vars). `max_size` is `--max-upload-bytes` (default
256MB) — it bounds every incoming frame (a larger upload is rejected with 1009;
the frontend pre-checks the file size). A `sql`/`sample`
result is sent as one binary frame capped at `--max-rows` rows (see `truncated`);
a `sql` request with `cursor: true` freezes the full result as a server-side
cursor and pages the rest via `fetch` (see *Result cursors* below).

## Request frame

Every request is a JSON object with an `op` and a client-chosen `id`:

```json
{"id": "q1", "op": "sql", "sql": "SELECT count(*) AS c FROM lineitem"}
```

`id` may be a string, number, or null; the server echoes it on the matching
response so a client can multiplex several in-flight requests on one connection
(see *Concurrency* below). `id: null` is allowed but cannot be cancelled.

## Response frames

The response `op` tells the client what to expect next:

| op          | follows?         | meaning |
|-------------|------------------|---------|
| `result`    | 1 binary frame   | a SELECT/sample returned rows (first page of a cursor, or a `fetch` page) |
| `write`     | nothing          | an INSERT/UPDATE/DELETE/MERGE; `rows_affected` |
| `ok`        | nothing          | a control statement (BEGIN/COMMIT/...), admin op, or `close` succeeded |
| `plan`      | nothing          | EXPLAIN tree (`tree`) |
| `catalog`   | nothing          | `tables` list |
| `table`     | nothing          | one table's schema (`columns`, `constraints`, ...) |
| `profile`   | nothing          | per-column GPU statistics (`columns` with null%/distinct/min/max/mean/stddev/histogram/top) |
| `export`    | 1 binary frame   | a SELECT serialized as a binary file (Parquet); `format`/`row_count`/`byte_count` |
| `history`   | nothing          | `entries` ring buffer |
| `py`        | nothing          | a Python notebook cell ran; `stdout`/`result`/`error` (+ `duration_ms`) |
| `cancelled` | nothing          | dropped while pending, or `CancelledByUser` at a node boundary |
| `error`     | nothing          | something failed; `kind` + `message` (+ `position`) |

Every response echoes the request `id` (except server-initiated protocol errors
on an unparseable request, which carry no `id`).

### `result` (text) + binary

```json
{"id": "q1", "op": "result",
 "columns": [{"name": "c", "type": "int64"}],
 "row_count": 3, "returned": 3, "truncated": false,
 "duration_ms": 12.4, "frame_count": 1}
```

Immediately followed by one binary frame: an Arrow IPC **stream**
(`pyarrow.ipc.new_stream`), schema metadata stripped. Read it with
`pyarrow.ipc.open_stream(bytes).read_all()`. `row_count` is the true total;
`returned` is the rows in the binary frame (≤ `max_rows`); `truncated` is true
when `row_count > returned`. `columns` describes the schema in the binary frame.

When the `result` is the first page of a cursor (a `sql` request with
`cursor: true`), the meta also carries:

```json
{"id": "q1", "op": "result", "...": "...",
 "cursor_id": "0e3f...hex", "offset": 0}
```

`cursor_id` identifies the server-side cursor holding the full result; `offset`
is the zero-based row offset of the first row in this binary frame. Page the
rest with `fetch`; drop the cursor with `close`. `truncated` doubles as "has
more" (`row_count > returned + offset`). A `sql` request with `cursor: true`
whose result exceeds `--max-cursor-rows` is **not** given a cursor — the server
falls back to a plain truncated result and adds `"cursor": false, "reason":
"too_large"` so the client does not retry with a cursor (the rest is not
pageable; use a narrower query or `LIMIT`).

### `write`

```json
{"id": "i1", "op": "write", "rows_affected": 1, "duration_ms": 211.1}
```

### `ok`

```json
{"id": "b1", "op": "ok", "duration_ms": 0.3}
```
Admin ops add a `detail` object, e.g. `{"registered": "lineitem"}`,
`{"set": "not_null"}`, `{"tables": {"lineitem": 500}}` (checkpoint).

### `error`

```json
{"id": "p1", "op": "error", "kind": "parse",
 "message": "Expected table name but got None",
 "position": {"line": 1, "col": 13}}
```

`kind` is one of:

- `parse` — SQL did not parse. Carries `position` `{line, col}` (1-based) when
  the underlying parser reports one.
- `runtime` — execution failed (unknown table, type error, …). The exception
  type prefixes `message` (e.g. `"KeyError: unknown table: 'lineitem'"`).
- `protocol` — the request itself was bad (unparseable JSON, unknown op, wrong
  argument types). Server-initiated protocol errors on an unparseable request
  carry no `id`.

## Ops

### `sql`

```json
{"id": "...", "op": "sql", "sql": "<statement>", "max_rows": 200000,
 "cursor": true}
```
`max_rows` is optional (defaults to the server's `--max-rows`); caps the rows in
the binary result. `cursor` is optional (default `false`); when `true` the
SELECT's full result is frozen as a server-side cursor (host memory) and the
first `max_rows` rows are returned as a `result` carrying `cursor_id` + `offset`
— page the rest with `fetch`, drop it with `close`. A result larger than
`--max-cursor-rows` falls back to a plain truncated result (`cursor: false,
reason: "too_large"`). `cursor` only affects SELECTs; it is ignored for
writes/DDL (which return `write`/`ok` as usual). Returns `result`+binary,
`write`, or `ok` (control / DDL / CREATE-FROM). Failed statements return
`error`.

### `fetch`

```json
{"id": "...", "op": "fetch", "cursor_id": "<id>", "offset": 1000,
 "limit": 1000}
```
Page a cursor opened by a `sql` request with `cursor: true`. Returns a
`result`+binary frame: `cursor_id` + `offset` (the offset of the first row in
this frame), `row_count` = the cursor's full size, `returned` = rows in this
slice, `truncated` = `offset + returned < row_count`. `offset` defaults to 0;
`limit` defaults to the server's `--max-rows`. An unknown or closed `cursor_id`
(a cursor belongs to its connection, so a different connection's id is unknown)
returns `error` (`kind: protocol`).

### `close`

```json
{"id": "...", "op": "close", "cursor_id": "<id>"}
```
Drop a result cursor, freeing its host-memory table. Returns `ok`. Idempotent:
an unknown `cursor_id` (already closed, or auto-closed on a prior disconnect)
still returns `ok`. Cursors are also auto-closed when their connection closes.

### `explain`

```json
{"id": "...", "op": "explain", "sql": "<select or write>"}
```
Returns `plan` with a `tree` — a recursive node:

```json
{"op": "Aggregate", "est_rows": null, "fused": true,
 "detail": {"group_keys": ["l_returnflag"], "aggs": ["c"]},
 "children": [{"op": "Join", "est_rows": null, "fused": false,
               "detail": {"how": "inner", "on_left": ["l_orderkey"],
                          "on_right": ["l_orderkey"]},
               "children": [...]}]}
```

- `op` — node type (`Scan`, `Filter`, `Project`, `Join`, `Aggregate`, `Window`,
  `Sort`, `Limit`, `Distinct`, `Derived`, `SetOp`, `Insert`, `Delete`, `Update`,
  `Merge`, `TxnControl`).
- `est_rows` — only `Scan` carries a real value (the catalog row count); other
  nodes are `null` in this phase.
- `fused` — `true` when an `Aggregate` sits directly over a `Join`: the
  star-join+aggregate shape *eligible* for the fused C++ kernel. This is
  eligibility, not a guarantee the fused path fired (that depends on runtime
  cardinality the static plan cannot see).
- `detail` — a few scalar fields per node (table, join how/keys, group keys,
  …). Rich expression rendering is a later phase.

### `catalog`

```json
{"id": "...", "op": "catalog"}
```
→ `{"op": "catalog", "tables": [{"name": "lineitem", "row_count": 500,
 "columns": ["l_orderkey", ...]}]}`

### `table`

```json
{"id": "...", "op": "table", "name": "lineitem"}
```
→ `{"op": "table", "name": ..., "columns": [{"name": ..., "type": ...,
 "nullable": bool}], "constraints": {...}, "paths": [...], "row_count": N}`

### `sample`

```json
{"id": "...", "op": "sample", "name": "lineitem", "n": 100}
```
Runs `SELECT * FROM <name> LIMIT <n>`. The table name is validated
(`str.isidentifier()` and present in the catalog) before it is ever interpolated
into SQL. Returns `result`+binary.

### `profile`

```json
{"id": "...", "op": "profile", "name": "lineitem", "top_k": 10}
```
Computes per-column statistics on the **GPU** (one full scan; cuDF reductions +
a value_counts) and returns them as a single JSON frame — no binary. `top_k`
(default 10) caps the most-frequent-values list, emitted only for low-NDV
columns (`distinct <= 1000`); continuous columns get a 10-bucket equal-width
histogram instead. Unknown table → `error` (runtime). `top_k` must be a
positive int → else `error` (protocol).

```json
{"op": "profile", "name": "lineitem", "row_count": 500,
 "columns": [{"name": "l_quantity", "type": "int64", "row_count": 500,
  "null_count": 0, "null_pct": 0.0, "distinct": 49, "min": 1, "max": 49,
  "mean": 25.1, "stddev": 14.2,
  "histogram": [{"lo": 1.0, "hi": 5.8, "count": 51}, ...],
  "top": [{"value": 7, "count": 14}, ...]}]}
```
`min`/`max` are numbers for numeric columns and strings for categorical ones;
`mean`/`stddev`/`histogram` are present only for numeric columns; `top` only
for low-NDV columns. Runs under the shared read lock (excludes concurrent
admin writes).

### `export`

```json
{"id": "...", "op": "export", "sql": "SELECT * FROM lineitem", "format": "parquet"}
```
Runs the SELECT on the worker thread (cancellable, serialized with other engine
work), copies the full cuDF frame to a host pyarrow.Table, and serializes it as
a binary file. v1 supports only `parquet` (the default); CSV/JSON/Arrow are
already produced in-browser from the paged result, and apache-arrow 17 has no
in-browser Parquet writer, so Parquet is written server-side with
`pyarrow.parquet`. The result is **not** capped at `--max-rows` — the point is
to export everything — but `--max-export-rows` (default 5M) bounds host RAM: a
larger result → `error` (runtime, "export too large") instead of OOMing. A
non-SELECT statement → `error` (protocol, "export requires a SELECT"). An
unsupported `format` → `error` (protocol).

```json
{"id": "...", "op": "export", "format": "parquet",
 "row_count": 600572, "byte_count": 842313, "duration_ms": 187.3}
```
followed by one binary frame: the raw Parquet bytes (NOT Arrow IPC — the client
keeps them as a blob for download). Records a history entry tagged `export`.

### `upload`

Ingest a parquet file sent over the wire (browser file-upload). A **two-frame
request**: one text meta frame, then immediately one binary frame with the
parquet bytes.

```json
{"id": "...", "op": "upload", "name": "<table>", "format": "parquet"}
```
followed by one binary frame: the raw parquet bytes. `format` must be
`"parquet"` (the only supported ingest format — the engine is parquet-only).

Response (text `ok`):
```json
{"id": "...", "op": "ok",
 "detail": {"registered": "<name>", "row_count": 600572, "columns": 16,
            "path": "<data>/uploads/<uuid>.parquet"},
 "duration_ms": 187.3}
```
The server writes the bytes to `<data_dir>/uploads/<uuid>.parquet` then
registers the table (same path as `admin register` / `CREATE TABLE FROM`). On a
register failure (e.g. not a valid parquet) the file is removed so a bad upload
leaves no orphan. The payload is bounded by `--max-upload-bytes` (default
256MB) — which is also the WebSocket `max_size`, so an oversized upload is
rejected by the transport with close code 1009 (the frontend pre-checks the
file size and refuses before sending). `upload` is handled in the connection
loop (the binary payload is read inline before dispatch), so it never reaches
the generic op dispatch.

### `admin`

```json
{"id": "...", "op": "admin", "action": "<action>", "args": {...}}
```

| action        | args                                       | detail |
|---------------|--------------------------------------------|--------|
| `register`    | `table`, `path`                            | `{"registered": <name>}` |
| `drop`        | `table`                                    | `{"dropped": true}` |
| `rename`      | `old`, `new`                               | `{"renamed": true}` |
| `alter`       | `table`, `kind`, … (below)                 | `{"set": <kind>}` |
| `checkpoint`  | —                                          | `{"tables": {<table>: <rows flushed>}}` (refused while any connection has an open txn) |
| `snapshot`    | `name`                                     | `{"snapshot": true}` |
| `restore`     | `name` *or* `ts` (int)                     | `{"restored": true}` / `{"restored_to": <ts>}` (refused while any connection has an open txn) |
| `clear_cache` | `which`: `"scan"`/`"code"`/`"both"`        | `{"cleared": <which>}` |

`alter` `kind`s: `pk` (`cols`), `notnull`/`notnull_off` (`col`), `unique`
(`cols`), `default` (`col`, `value`), `drop_default` (`col`).

### `cancel`

```json
{"id": "cx", "op": "cancel", "targets": ["q2"]}
```
Cancels requests on this connection. Returns `{"op": "ok", "detail":
{"cancelled": ["q2"], "not_found": [...]}}`.

- **Pending** (queued, not yet started by the worker): dropped on dequeue →
  the request resolves with a `cancelled` frame.
- **In-flight** (already running): the engine raises `CancelledByUser` at the
  next **plan-node boundary** → the request resolves with a `cancelled` frame.
  This is cooperative: a single long cuDF call (a big groupby, a cold parquet
  read) is *not* interruptible mid-call — cancel fires after the current node
  finishes. See *Limitations*.

A target in `not_found` had already completed or never existed —
indistinguishable, and honest: that request's own response tells the client
what happened. There is an inherent race: a target reported in `cancelled` may
have already passed its last node and finished normally (its own response is
then a normal result, not `cancelled`).

### `history`

```json
{"id": "...", "op": "history"}
```
→ `{"op": "history", "entries": [{"id": ..., "sql": ..., "duration_ms": ...,
"rows": N, "kind": "select|write|export|fetch|control|cancelled|other",
"ts": 1721554234.7}]}` — the server-side ring buffer (default 500 entries).
`ts` is the wall-clock epoch seconds when the entry was recorded (process-local,
not persisted — a session timestamp) so the UI can order entries by time and show
when each query ran.

### `py`

Run a Python notebook cell on the worker thread (so it serializes with other
ops and the engine's read/write lock applies to any `sql()` it makes).

```json
{"id": "...", "op": "py", "code": "<python source>"}
```
→ `{"op": "py", "stdout": "...", "result": "...|null", "error": "...|null",
"duration_ms": N}` (text-only — no binary frame follows).

The cell runs in a fresh namespace with a `sql(s)` helper plus `pd` and `cudf`
injected. `sql()` runs a statement through the connection's MVCC txn (exactly
like a `sql` op) and returns a **pandas** DataFrame for SELECTs (cuDF is copied
to host so the cell can use plain pandas/matplotlib-style APIs) or the int
rows-affected / `null` for writes/control. A cell is `eval`ed if it is a single
expression, else `exec`ed (chosen by `compile`); the expression's `repr()` is
returned as `result` (a statement block leaves `result` null). Captured stdout
is returned in `stdout`, capped at 64 KiB; `result` is capped at 4 KiB.

A cell that raises is **not** an `error` frame — the formatted traceback travels
in `error` (with `result` null) so the notebook UI can render it inline next to
the cell. `py` ops are not recorded in the `history` ring buffer (notebook cells
are not query-history entries; the `sql()` calls inside a cell go through the
engine but bypass `record_history`).

`py` is the execution backend for the web UI's **SQL+Python notebooks** (Phase 4):
each Python cell is a self-contained script that calls `sql()` to read/write the
database and `print()`s or returns a value. Cells are stateless across runs
(every run starts a fresh namespace), so a cell that needs a variable from an
earlier cell must re-derive it or re-run both.

## Concurrency

Each request is dispatched as its own task, so one connection may have several
requests in flight. They run on the engine worker pool (default 1 worker;
`--workers N` for a pool of `N`). The engine's read/write lock gives:

- **concurrent reads** — SELECTs (and `explain`/`catalog`/`table`) take the
  shared read lock, so with `--workers N>1` several SELECTs run on the engine
  at once. Each worker thread has its own `_txn` / `cancel_event` slot
  (`threading.local`), so concurrent requests never clobber a single shared
  slot. MVCC isolation is unchanged: a txn read still uses its frozen
  `snapshot_ts` (`batches_at(snapshot_ts)`); autocommit reads see all committed
  batches.
- **serialized writes** — INSERT/UPDATE/DELETE/MERGE and admin ops take the
  exclusive write lock, so no two writes (and no write alongside a read) ever
  overlap on the engine. This is what keeps the `_code_cache`/`_pk_cache`
  coherent without versioning: a write's cache invalidation runs with no
  concurrent reader, and a read's cache population runs with no concurrent
  writer.
- **submission order** is FIFO (the worker pool drains its queue in order), so
  with the default 1 worker, responses for sequentially-submitted requests
  arrive in order — identical to in-process `engine.sql()`. With `N>1` workers,
  submission is still FIFO but response ordering becomes per-request (a later
  query that finishes faster may return first) — match by `id`.
- a **pipelined** client (many requests sent without awaiting) may see responses
  arrive out of order — match by `id`.

A per-connection send lock keeps each `result`'s meta+binary frames atomic, so
a binary IPC frame is always immediately preceded by its own meta frame on the
wire. The same lock glues a `fetch` page's meta+binary pair.

## Result cursors (paged browsing)

A `sql` request with `cursor: true` runs the query once and freezes the full
result as a server-side **cursor** — a host-memory `pyarrow.Table` (the GPU
frame is freed once the result is copied to host). The first `max_rows` rows are
returned as a `result` carrying `cursor_id` + `offset: 0`; the rest are paged
with `fetch`, and the cursor is dropped with `close` (or auto-closed when the
connection closes). A cursor's frozen snapshot is independent of the engine and
GPU, so pages are always consistent with the first page (no row-shifting between
pages from a concurrent write) and paging re-executes nothing on the GPU.

Cursors are **per-connection**: a `cursor_id` is only valid on the connection
that opened it; a `fetch`/`close` from another connection reads as unknown.
Bounds: `--max-cursor-rows` (default `1_000_000`) — a larger result is not given
a cursor (served truncated with `cursor: false, reason: "too_large"`, the rest
not pageable); `--max-cursors-per-conn` (default `16`) — opening more raises an
`error` (close one first). Each cursor holds its full result in host RAM for its
lifetime, so prefer `close` when done and avoid cursors on results you only need
the head of.

## Limitations (Phase 1)

- **Per-connection transactions (MVCC isolation).** Each connection owns its
  own in-flight transaction. `BEGIN` captures a frozen `snapshot_ts` (the
  engine's commit counter); reads see committed batches at-or-before that ts
  plus the txn's own buffered writes (read-your-writes). A commit on another
  connection after this `BEGIN` is therefore invisible until this txn
  `COMMIT`s. The engine's `_txn` is a transient per-request pointer (loaded
  from the connection's txn before each `sql`/`sample` call, read back after),
  so `BEGIN` on connection B no longer fails while connection A has an open
  txn. A disconnect rolls back the connection's open txn (its buffered writes
  were never flushed to the delta/WAL, so dropping them is the rollback).
  Snapshots (`CREATE SNAPSHOT`) and the commit counter remain global
  (single-writer, serialized through the worker). `checkpoint` and
  `restore`/`restore_to` are refused while *any* connection has an open txn —
  they rewrite history globally and would invalidate live snapshots.
- **Cooperative in-flight cancel.** `cancel` drops pending requests and raises
  `CancelledByUser` at the next plan-node boundary of an in-flight request. A
  single long cuDF call (a big groupby, a cold parquet read) is *not*
  interruptible mid-call — cancel fires after the current node finishes. A
  `py` cell with a long pure-Python loop is likewise not interruptible mid-call
  (cancel fires after the cell — and any `sql()` it makes — returns). Full
  preemption would need subprocess isolation.
- **Local single-user.** Binds to `127.0.0.1` by default; no auth. The `sample`
  op interpolates a validated identifier into SQL; `sql`/`admin` trust the
  client, and `py` runs arbitrary Python (filesystem/subprocess access) in the
  server process (appropriate for a local console, not for exposing on a
  network).
- **One binary frame per result page.** A `sql`/`sample`/`fetch` result is one
  binary frame capped at `--max-rows` (or `limit`) rows; the true `row_count` is
  reported and `truncated` flags more. A `sql` request with `cursor: true` pages
  the rest via `fetch` (see *Result cursors*); without a cursor, rows beyond
  `max_rows` are simply truncated. An `export` response is also one binary frame,
  but raw Parquet bytes (not Arrow IPC). No chunked streaming of a single
  response as multiple Arrow batches (the `frame_count` field is reserved for
  that future option).

## Postgres wire front (`--pg-port`)

With `--pg-port` set, `ryudb-server` also binds a PostgreSQL v3 wire-protocol
front (`ryudb/server/pgwire.py`) so real drivers — `psql`, `psycopg`,
`pg8000`, `asyncpg`, JDBC — can connect and run SQL instead of speaking the
custom JSON+Arrow protocol. It shares the WebSocket `Server`'s engine, worker
pool, and per-connection transaction core: each PG connection is a session with
its own `_ryudb_txn` routed through `Server.run_sql`, so the cross-session MVCC
isolation above applies identically. `--pg-max-rows` / `RYUDB_PG_MAX_ROWS`
(default `200000`) caps the rows returned per SELECT over this front.

```
psql "host=127.0.0.1 port=5432 user=ryudb dbname=ryudb"
```

### Supported

- **Startup v3.0 + `AuthenticationOk`** (no auth; the server binds
  `127.0.0.1` for a local single-user console). `SSLRequest` / `GSSRequest`
  are refused with a single `N` byte (the client retries plain).
- **Simple Query (`Q`)**: `RowDescription` + `DataRow`* (text format) +
  `CommandComplete` + `ReadyForQuery`. A multi-statement Query is split on
  top-level `;` and run statement-by-statement.
- **Extended Query (`P`/`B`/`D`/`E`/`S`/`C`)**: Parse/Bind/Describe/Execute/
  Sync/Close. `$N` placeholders are substituted with typed literals via a
  sqlglot AST walk (`exp.Parameter` → `exp.Literal`), so the engine's
  string-SQL entry point is reused as-is. A statement-Describe answers
  `ParameterDescription` + `NoData`; a portal-Describe (or an Execute with no
  preceding portal-Describe) emits the `RowDescription` — matching real
  Postgres, which sends the output schema from Execute when no portal-Describe
  preceded it.
- **`Terminate` (`X`)**, **`ErrorResponse` (`E`)** with SQLSTATE,
  **`EmptyQueryResponse` (`I`)**, **`NoData` (`n`)**.
- **Per-connection transactions**; `ReadyForQuery` status `I`/`T`/`E` (idle /
  in-txn / aborted-txn — the latter gates further statements on `ROLLBACK`,
  matching real Postgres). `BEGIN`/`COMMIT`/`ROLLBACK` map to the engine's
  transaction control; a disconnect rolls back the open txn (buffered writes
  were never flushed).
- **SQL cursors (`DECLARE`/`FETCH`/`MOVE`/`CLOSE`)**, layered on the same
  per-connection frozen-result cursor store as the WebSocket `fetch` op.
  `DECLARE <name> CURSOR FOR <select>` runs the query once and freezes the
  full result as a `pa.Table`; `FETCH`/`MOVE` page it with the full Postgres
  scroll-direction grammar (`NEXT`/`PRIOR`/`FIRST`/`LAST`/`FORWARD`/`BACKWARD`
  `[n|ALL]`/`ABSOLUTE n`/`RELATIVE n`, bare count, `ALL`). `FETCH` emits
  `RowDescription` + `DataRow`* + `CommandComplete` (`FETCH n`) directly,
  **bypassing the `--pg-max-rows` per-Response cap**, so a driver can page a
  result larger than the cap — `psql`'s `\set FETCH_COUNT 100`, or a named
  cursor in a real driver. `MOVE` advances without emitting rows
  (`CommandComplete` `MOVE n`). `CLOSE <name>` / `CLOSE ALL` drop cursors
  (unknown name → error `34000`, matching Postgres). `WITH HOLD` is rejected
  (`0A000` — cursors would have to survive `COMMIT`, which RyuDB does not
  support); every cursor is `WITHOUT HOLD` and is closed at
  `COMMIT`/`ROLLBACK` and on disconnect. Cursors work over both the
  Simple-Query front and the Extended-Query front (an Execute's max-rows caps
  an explicit `FETCH` count). Errors carry real SQLSTATEs: `34000`
  (`invalid_cursor_name`), `42P03` (`duplicate_cursor`), `42P11`
  (`invalid_cursor_definition`), `54000` (`program_limit_exceeded` — the
  per-connection `--max-cursors-per-conn` limit), `0A000`, `42601`.
- **`CancelRequest`** (the v3 startup message with version `80877102`). At
  startup the server emits `BackendKeyData` (`K`) carrying the session's
  `(pid, secret)`. A *separate, short-lived* connection may send a
  `CancelRequest` with that `(pid, secret)`; the server looks up the target
  session and sets its in-flight cancel `Event` — the cooperative cancel (the
  same mechanism as the WS `cancel` op) then drops a pending request or raises
  `CancelledByUser` at the next plan-node boundary of an in-flight one,
  surfaced to the target as an `ErrorResponse` (`57014`) + `ReadyForQuery`.
  Cancelling an idle backend is a no-op. A `CancelRequest` gets **no response**
  (the client just closes the cancel connection); an unknown `(pid, secret)` is
  silently ignored.

### Not supported (documented limits)

- **SSL/TLS, GSSAPI, SCRAM/password auth** (no auth).
- **`COPY`, `LISTEN`/`NOTIFY`, binary result format (text only), binary
  parameter format (text only).** A client requesting binary results/params
  gets an error.
- **Result rows are capped at `--pg-max-rows`** (the same cap as the
  WebSocket front); the PG protocol has no truncation signal, so the cap is
  silent for a plain `SELECT`. Raise `--pg-max-rows` for full exports, or
  page past it with a **`DECLARE CURSOR`/`FETCH`** (see *Supported* above),
  which bypasses the cap per `FETCH`. `WITH HOLD` cursors are not supported.
- **Parameter type inference** uses the Parse param OIDs; an unknown OID (`0`)
  renders the param as a quoted string literal and lets the engine coerce.
- **`SELECT` without `FROM`** is not supported by the engine (a parameterized
  `SELECT $1` must reference a table, e.g. `SELECT $1 AS x FROM t LIMIT 1`).