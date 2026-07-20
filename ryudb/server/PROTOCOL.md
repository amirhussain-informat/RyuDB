# ryudb-server wire protocol

`ryudb-server` fronts one `Engine` over `data_dir` with **two** wire fronts
sharing the same engine, worker, and per-connection transaction model:

- a **WebSocket** front (custom JSON + Arrow IPC; this document), default
  `127.0.0.1:5430` (`--port` / `RYUDB_PORT`); and
- an optional **Postgres v3 wire-protocol** front (see *Postgres wire front*
  below), enabled with `--pg-port` / `RYUDB_PG_PORT` (default `0` = disabled).

Both fronts serialize all engine/catalog access through the single worker
thread, so correctness and FIFO submission order are identical across them; a
PG connection and a WS connection are just two sessions over the same engine
(per-connection MVCC isolation applies to both).

## WebSocket front

A client opens a single WebSocket connection to the server and exchanges
frames. The transport is plain WebSocket (no subprotocol). Two frame kinds:

- **text frames** — UTF-8 JSON. Used for every request and for every response
  *except* the binary result payload.
- **binary frames** — one Arrow IPC stream. Sent by the server only, only as
  the second frame of a successful `sql`/`sample` result. Clients never send
  binary frames (the server closes the connection with code 1003 if one
  arrives).

`ryudb-server` listens on `127.0.0.1:5430` by default (`--host`/`--port`, or the
`RYUDB_HOST`/`RYUDB_PORT` env vars). `max_size` is unbounded, so large result
sets are streamed in a single binary frame (capped at `--max-rows` rows; see
`truncated`).

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
| `result`    | 1 binary frame   | a SELECT/sample returned rows |
| `write`     | nothing          | an INSERT/UPDATE/DELETE/MERGE; `rows_affected` |
| `ok`        | nothing          | a control statement (BEGIN/COMMIT/...) or admin op succeeded |
| `plan`      | nothing          | EXPLAIN tree (`tree`) |
| `catalog`   | nothing          | `tables` list |
| `table`     | nothing          | one table's schema (`columns`, `constraints`, ...) |
| `history`   | nothing          | `entries` ring buffer |
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
{"id": "...", "op": "sql", "sql": "<statement>", "max_rows": 200000}
```
`max_rows` is optional (defaults to the server's `--max-rows`); caps the rows in
the binary result. Returns `result`+binary, `write`, or `ok` (control / DDL /
CREATE-FROM). Failed statements return `error`.

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
"rows": N, "kind": "select|write|control|cancelled|other"}]}` — the server-side
ring buffer (default 500 entries).

## Concurrency

Each request is dispatched as its own task, so one connection may have several
requests in flight. All engine/catalog access is serialized through a single
worker thread (one `Engine`), so:

- **correctness** is identical to in-process `engine.sql()` — no two statements
  ever run concurrently on the engine;
- **submission order** is FIFO (the worker drains its queue in order), so
  responses for sequentially-submitted requests arrive in order;
- a **pipelined** client (many requests sent without awaiting) may see responses
  arrive out of order — match by `id`.

A per-connection send lock keeps each `result`'s meta+binary frames atomic, so
a binary IPC frame is always immediately preceded by its own meta frame on the
wire.

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
  interruptible mid-call — cancel fires after the current node finishes. Full
  preemption would need subprocess isolation.
- **Local single-user.** Binds to `127.0.0.1` by default; no auth. The `sample`
  op interpolates a validated identifier into SQL; `sql`/`admin` trust the
  client (appropriate for a local console, not for exposing on a network).
- **One binary frame per result.** No chunked streaming of huge results; rows
  beyond `max_rows` are truncated (the true `row_count` is reported; a future
  export/fetch op can page them).

## Postgres wire front (`--pg-port`)

With `--pg-port` set, `ryudb-server` also binds a PostgreSQL v3 wire-protocol
front (`ryudb/server/pgwire.py`) so real drivers — `psql`, `psycopg`,
`pg8000`, `asyncpg`, JDBC — can connect and run SQL instead of speaking the
custom JSON+Arrow protocol. It shares the WebSocket `Server`'s engine, worker,
and per-connection transaction core: each PG connection is a session with its
own `_ryudb_txn` routed through `Server.run_sql`, so the cross-session MVCC
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

### Not supported (documented limits)

- **SSL/TLS, GSSAPI, SCRAM/password auth** (no auth). `CancelRequest` is
  parsed but not wired to the cooperative cancel (`cancel` op) yet — a future
  PR can map `(pid, secret)` to the connection's pending cancel event.
- **`COPY`, `LISTEN`/`NOTIFY`, portal scroll, binary result format (text
  only), binary parameter format (text only).** A client requesting binary
  results/params gets an error.
- **Result rows are capped at `--pg-max-rows`** (the same cap as the
  WebSocket front); the PG protocol has no truncation signal, so the cap is
  silent. Raise `--pg-max-rows` for full exports.
- **Parameter type inference** uses the Parse param OIDs; an unknown OID (`0`)
  renders the param as a quoted string literal and lets the engine coerce.
- **`SELECT` without `FROM`** is not supported by the engine (a parameterized
  `SELECT $1` must reference a table, e.g. `SELECT $1 AS x FROM t LIMIT 1`).