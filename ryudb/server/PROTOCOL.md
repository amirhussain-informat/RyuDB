# ryudb-server wire protocol

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
| `checkpoint`  | —                                          | `{"tables": {<table>: <rows flushed>}}` |
| `snapshot`    | `name`                                     | `{"snapshot": true}` |
| `restore`     | `name` *or* `ts` (int)                     | `{"restored": true}` / `{"restored_to": <ts>}` |
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

- **Single logical session.** There is one `Engine`, so transaction state
  (`BEGIN`/`COMMIT`, snapshots, restore points) is shared across *all*
  connections. `BEGIN` on one connection and `COMMIT` on another interleave.
  Fine for a local single-user console; multi-session isolation is a later
  phase.
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