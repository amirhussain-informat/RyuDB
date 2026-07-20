# ryudb-frontend

A Snowsight-like worksheet UI for [`ryudb-server`](../ryudb/server/PROTOCOL.md) —
React + TypeScript + Vite, Monaco SQL editor, Apache Arrow results. The server
does the GPU work; this is its client.

## Layout

```
┌─ toolbar: connection (ws://host:port) + Run / Explain / Cancel ─┐
├─ sidebar ──────┬─ main ──────────────────────────────────────────┤
│  Catalog       │  Monaco SQL editor  (Ctrl/Cmd+Enter = Run)      │
│  / History     ├─────────────────────────────────────────────────┤
│                │  Results  Explain  Message                      │
│                │  (Arrow grid | plan tree | error)               │
└────────────────┴─────────────────────────────────────────────────┘
```

- **Editor** — Monaco with SQL syntax. `Ctrl/Cmd+Enter` runs the current
  statement. A parse error from the server is underlined at the reported
  `position`. Multiple **worksheets** live as tabs above the editor: click to
  switch, double-click a name to rename, `×` to close, `+` to create a new one.
  Worksheets (names + SQL + the active tab) persist to `localStorage`, so they
  survive a reload; each tab keeps its own last results/plan/message during a
  session.
- **Results** — the server's Arrow IPC binary frame is decoded with
  `apache-arrow` and rendered as a virtualized grid (`react-window`). The true
  `row_count` is shown. Each run opens a server-side **cursor** (`sql` with
  `cursor: true`) that freezes the full result; the first page is shown and a
  **Load more** button pages the rest via the `fetch` op (growing the grid) up to
  `--max-cursor-rows` (default 1M) — above that the result is served truncated
  and not pageable. **Download CSV / Arrow** serializes the full set: a
  cursor-backed result is paged from the cursor; a non-cursor (too-large) result
  re-runs with `max_rows = row_count`. A result above 1M rows asks for
  confirmation first.
- **Explain** — the structured plan tree with a `fused` badge on an
  `Aggregate` over a `Join` (the star-join+aggregate shape eligible for the
  fused C++ kernel — eligibility, not a guarantee).
- **Catalog** — tables + row counts; expand a table for its columns; click a
  name to drop it into the editor; the `⤵` button runs `SELECT * … LIMIT 100`.
- **History** — the server-side query ring buffer; click an entry to reload
  its SQL.

## Develop

The frontend is a Vite app; Node lives in a separate conda env so it doesn't
touch the GPU `ryudb` env.

```bash
# one-time: create the node env
conda create -c conda-forge -n ryudb-fe nodejs -y

# install + run the dev server (http://localhost:5173)
conda activate ryudb-fe
cd frontend
npm install
npm run dev
```

In another shell, start the server (from the repo root, in the `ryudb` env):

```bash
conda activate ryudb
ryudb-server --data ./data --port 5430
# register a table to query, e.g.:
#   ryudb -e "CREATE TABLE lineitem FROM '/path/to/lineitem/*.parquet'"
```

Then open the worksheet UI, point it at `ws://127.0.0.1:5430`, and Connect.

## Build + smoke

```bash
npm run build      # tsc -b + vite build (type-checks + bundles to dist/)
npm run smoke      # node test/smoke.mjs — Arrow IPC round-trip vs a running server
node test/csv_check.mjs   # src/lib/csv.ts serializer (null-aware CSV) — no server needed
```

`npm run smoke` needs a running `ryudb-server` with a registered `lineitem`; set
`RYUDB_PORT` and `RYUDB_LINEITEM_PATH` (a directory containing a lineitem
parquet) and `RYUDB_N` (its row count). It verifies admin register, SELECT
round-trip + Arrow decode + value match, GROUP BY, EXPLAIN fused badge, and a
parse error with position — the same wire path the browser uses.

## Notes / Phase 1 limits

- Monaco is bundled locally from the `monaco-editor` package (a SQL-only subset:
  the editor API + editor contributions + the SQL tokenizer, wired in
  `src/monaco.ts`), so the worksheet works fully offline — no CDN fetch at
  runtime. `@monaco-editor/react`'s loader is configured with the local
  instance, which short-circuits its default CDN path.
- The server gives each connection its own transaction (per-connection MVCC
  isolation): `BEGIN` captures a frozen snapshot, so a commit on another
  connection is invisible until this one `COMMIT`s; a disconnect rolls back the
  open txn. `Cancel` drops pending requests and raises `CancelledByUser` at
  the next plan-node boundary of a running request — a single long cuDF call
  (a big groupby, a cold read) is not interruptible mid-call. See the server's
  `PROTOCOL.md` limits.