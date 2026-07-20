# RyuDB — Architecture Design Document

**Status:** Living document (reflects merged state through PR #70)
**Platform:** WSL2 Ubuntu + NVIDIA GPU (RTX 3090), single-user local console

## 1. Overview

RyuDB is a GPU-powered HTAP RDBMS that executes analytical SQL on NVIDIA GPUs
via RAPIDS **cuDF**, with a CPU delta-store + WAL + MVCC write path layered
over an immutable Parquet base. It targets a single-user, local-console
workload (127.0.0.1, no auth) and is reachable through three interchangeable
fronts: a CLI, a custom WebSocket front (for the Snowsight-like web UI), and a
Postgres v3 wire front (for real drivers).

```
                ┌─────────── RyuDB ───────────┐
   CLI ───────▶ │  Parser (sqlglot)            │
   WS  front ──▶│    → Logical Plan            │
   PG  front ──▶│    → Rule-based Optimizer    │
                │    → Physical Plan           │
                │    → cuDF Executor (GPU)     │
                │    → Parquet base + delta    │
                └──────────────────────────────┘
```

## 2. Layered Architecture

### 2.1 SQL Frontend (`sqlglot`)
- `sqlglot` parses SQL into an AST, lowered to relational algebra.
- Broad SQL subset: SELECT/WHERE/JOIN/GROUP BY/ORDER BY/LIMIT; window functions
  (frames, running aggregates, QUALIFY, named windows, expression keys);
  IN/EXISTS/scalar/correlated subqueries; INSERT/UPDATE/DELETE/MERGE DML;
  GROUP BY ROLLUP/CUBE/GROUPING SETS + GROUPING(); aggregate FILTER;
  DISTINCT-qualified aggregates; statistical/logical/population aggregates
  (window + grouped); date/time, math, string, misc expressions;
  composite-equi-correlated subqueries.

### 2.2 Planner / Optimizer (custom)
- Rule-based: predicate pushdown, projection pruning, join reordering by
  estimated cardinality.
- `ORDER BY` scalar expressions, self-join pushdown, correlated-subquery
  inner-join key preservation.

### 2.3 Executor (`cuDF`, GPU)
- Lowers the physical plan to cuDF operations: scan, filter, join,
  groupby/aggregate, sort, limit, window.
- **Fused C++/CUDA kernels** (`ryudb/kernels/fused.cu`, built via
  `ryudb build`): a single GPU kernel for global aggregates
  (COUNT/SUM/AVG/MIN/MAX) and for star-join + aggregate. The kernel is
  **optional** — absent/stale → Numba/cuDF fallback (correctness never
  depends on it).
- Fused-join coverage: cross-table WHERE folding, group-key-in-fact,
  same-table multi-key stride-combine, full Q3 (keys spanning fact + dim),
  HASH accumulator for high-cardinality group keys + MIN/MAX/AVG.
- Per-plan-node cooperative cancel check (raises `CancelledByUser` at the
  next node boundary).

### 2.4 Storage — HTAP (immutable base + delta)
- **Base:** columnar Parquet (never mutated).
- **Delta store:** per-table list of timestamped cuDF batches + a tombstone
  channel (timestamp-aware PK anti-join → delete-then-reinsert-same-PK works).
- **MVCC:** `commit_ts`-versioned batches; `batches_at(snapshot_ts)` /
  `tombstones_at(ts)`.
- **WAL:** append-only `ryudb.wal`, one record per commit (CRC-validated,
  atomic fsync); `commit_ts` = WAL LSN; replay on startup recovers delta +
  counter.
- **Checkpoint:** flushes committed delta into a new base Parquet file,
  clears delta, truncates WAL, preserves DECIMAL/DATE type fidelity on disk.
- **Catalog:** typed + persistent (`ryudb.catalog.json`); Arrow schema +
  constraints (NOT NULL/PK/UNIQUE/DEFAULT). PK/UNIQUE enforced on INSERT;
  NOT NULL enforced.
- DML surface: INSERT / UPDATE / DELETE / MERGE.

### 2.5 Concurrency Model
- Single `Engine` behind a worker pool of N (default 1, opt-in `--workers`).
- **RW lock** (writer-preference): SELECTs concurrent under a shared read
  lock; writes/admin serialize under an exclusive write lock. Sidesteps the
  `_code_cache`/`_pk_cache` coherency race (keyed by `(table,col)`, not data
  version) without versioning.
- `_txn` and `cancel_event` are `threading.local`-backed (per worker thread).

## 3. Server Architecture (`ryudb/server`)

A single `Engine` + worker pool shared by two wire fronts:

### 3.1 WebSocket front (custom JSON + Arrow IPC)
- For the web frontend / any client.
- Ops: `sql`, `fetch`, `close` (result-cursor paging), `explain`, `catalog`,
  `table`, `sample`, `admin`, `cancel`, `history`.
- **Result cursors:** `sql` with `cursor: true` freezes the full result as a
  host-memory `pa.Table` (GPU freed after `df_to_arrow`); `fetch` pages it;
  `close` drops it. Frozen snapshot → consistent pages, no GPU re-scan.
  Bounded by `--max-cursor-rows` (1M) + `--max-cursors-per-conn` (16) +
  auto-close on disconnect.
- Per-connection send lock glues meta + binary frame atomicity.

### 3.2 Postgres v3 wire front (`--pg-port`, optional)
- For `psql`/`psycopg`/`pg8000`/`asyncpg`/JDBC against the same engine.
- Simple + Extended Query with `$N` substitution via a sqlglot AST walk;
  per-conn txn status I/T/E; `CancelRequest` (pid, secret) → cooperative
  cancel.

### 3.3 Per-connection MVCC isolation
- Each connection owns an in-flight transaction with a frozen snapshot; a
  commit on another connection is invisible until this one commits
  (read-your-writes within a txn; disconnect rolls back the open txn).
- `checkpoint` / `restore` refused while any connection has an open txn.

## 4. Frontend (`ryudb-frontend`)

React + TypeScript + Vite; Monaco SQL editor (bundled locally, offline);
Apache Arrow results in a virtualized grid.

- **Worksheets:** multi-tab, persisted to `localStorage` (names + SQL +
  active tab); per-tab in-memory view state (results/plan/message/cursor).
- **Results:** virtualized grid (true column count + horizontal scroll),
  cursor paging + Load more, Download CSV/Arrow.
- **Explain:** structured plan tree with a `fused` badge on the
  star-join+aggregate shape.
- **Catalog / History sidebars;** dark/light theme (persisted, OS-pref
  default).

## 5. Packaging & Deployment

- **Pip install:** `pip install . && ryudb build` (kernel sources ship in the
  wheel via package-data; pybind11 declared).
- **Docker (Tier C single-artifact):** `ubuntu:22.04` + conda env (CUDA 13.3,
  cuDF 26.06, libnvcomp 5.2) + built `fused.so` + `ryudb-server` entrypoint;
  host driver injected via `--gpus all`; `RYUDB_CUDA_ARCH` fat-binary knob.

## 6. Key Properties

| Property | Mechanism |
|---|---|
| GPU acceleration | cuDF executor + optional fused C++/CUDA kernels |
| HTAP | Immutable Parquet base + timestamped delta + WAL + MVCC |
| Durability | WAL (CRC, fsync) + checkpoint write-back |
| Isolation | Per-connection MVCC snapshots |
| Concurrency | RW lock + worker pool (default 1) |
| Cancel | Cooperative (pending-drop + next-node-boundary) |
| Interop | PG v3 wire + custom WS/Arrow front |
| Correctness | Fused kernel optional → cuDF/Numba fallback always correct |

## 7. Non-Goals (local single-user console)

Cloud/multi-tenant concerns are out of scope: warehouse/compute management,
RBAC/users/roles, data sharing & Marketplace, Snowpipe, Cortex cloud AI,
governance/masking, billing/budgets, replication.

## 8. Open Horizons

- **Mid-call cancel preemption** — a single long cuDF call is not mid-call
  interruptible (would need subprocess isolation).
- **Fused-join perf tail** — remaining narrower cases beyond landed #53–#55.
- **WebUI Snowsight-parity roadmap** — Phase 0 COMPLETE (multi-worksheet tabs
  + localStorage persistence, dark/light theme, command palette Cmd/Ctrl+K,
  keyboard-shortcuts help, global object search Cmd/Ctrl+Shift+F, per-worksheet
  version history Cmd/Ctrl+Shift+H). Phase 1 (results grid sort/filter/resize/
  reorder/copy + chart tab + GPU-instant column profiling + multi-result tabs +
  JSON/Parquet export), Phase 2 (object explorer + data-loading wizard +
  schema-aware autocomplete), Phase 3 (visual GPU-kernel-aware query profile +
  filterable history + dashboards), Phase 4 (NL→SQL assistant, notebooks).

---

*Summarizes the architecture as established in the project's merged PR
history; reconcile with the code when used as a binding reference. Full
wire-format detail lives in [`ryudb/server/PROTOCOL.md`](ryudb/server/PROTOCOL.md).*