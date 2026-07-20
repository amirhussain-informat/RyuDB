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
│                │  (Arrow grid | profile graph | error)           │
└────────────────┴─────────────────────────────────────────────────┘
```

- **Editor** — Monaco with SQL syntax. `Ctrl/Cmd+Enter` runs the current
  statement. A parse error from the server is underlined at the reported
  `position`. **Schema-aware autocomplete** — type to get the catalog's tables +
  columns (plus common SQL keywords), ranked tables-first; after `table.` you
  get just that table's columns. The schema is fetched on connect (the `catalog`
  op + a per-table `table` op fan-out) and re-fetched after any write/DDL, then
  handed to a Monaco completion provider registered once on editor mount.
  Multiple **worksheets** live as tabs above the editor: click to switch,
  double-click a name to rename, `×` to close, `+` to create a new one.
  Worksheets (names + SQL + the active tab) persist to `localStorage`, so they
  survive a reload; each tab keeps its own last results/plan/message during a
  session. The **⮚** menu at the end of the tab strip (also command-palette
  entries) exports/imports worksheets as plain `.sql` for **Git-backed
  versioning**: export the active worksheet as a single `.sql` file, export all
  as one `.sql` bundle (`-- @@worksheet: <name>` separators), or import one or
  many `.sql` files (a bundle splits into sections; a plain file becomes one
  worksheet named from its stem). The format is dependency-free and
  git-diffable; round-trip preserves names + SQL (a line starting with
  `-- @@worksheet:` is always a section separator — see
  `test/worksheetTransfer_check.mjs`).
- **Results** — the server's Arrow IPC binary frame is decoded with
  `apache-arrow` and rendered as a virtualized grid (`react-window`). The true
  `row_count` is shown. Each run opens a server-side **cursor** (`sql` with
  `cursor: true`) that freezes the full result; the first page is shown and a
  **Load more** button pages the rest via the `fetch` op (growing the grid) up to
  `--max-cursor-rows` (default 1M) — above that the result is served truncated
  and not pageable. **Download CSV / JSON / Arrow** serializes the full set: a
  cursor-backed result is paged from the cursor; a non-cursor (too-large) result
  re-runs with `max_rows = row_count`. A result above 1M rows asks for
  confirmation first. **Download Parquet** uses a server-side `export` op
  (apache-arrow 17 has no in-browser Parquet writer): the server runs the query
  and streams the full result as one Parquet binary blob (bounded by
  `--max-export-rows`, default 5M). **Copy** writes the displayed rows as TSV to the
  clipboard (pastes into Excel / Sheets); **click a cell** to copy its value.
  **Sort + filter** run in-browser on the loaded page (no server round-trip):
  click a column header to cycle ascending → descending → clear (a `▲`/`▼`
  marks the active sort; NULLs sort last); toggle the **Filter** button for a
  per-column case-insensitive substring filter row. **Resize** a column by
  dragging the right edge of its header (floor 40px); **reorder** columns by
  dragging a header onto another header. Sort + filter keep operating on the
  original column, so reorder is a pure display remap (the grid and Copy follow
  the new order; Download still pulls the full server result). Widths and order
  reset with each new result.
  **Result history** — each successful SELECT run is kept as a tab in a strip
  above the results (newest first, capped at 10); click a past tab to view that
  result again, `×` to drop one, **Clear** to drop all. Download re-runs the
  statement that produced the viewed tab (not the editor's current text). A past
  tab shows its first page (its server cursor is freed on the next run); use
  Download for the full set.
- **Chart** — a **Chart** output tab (shown when a result is present) renders
  the loaded result rows as **bar / line / scatter** over hand-rolled SVG (no
  charting library — keeps the offline/no-CDN ethos). Pick the X and Y columns;
  bar plots one bar per row (capped at 60, ideal for an aggregated GROUP BY
  result), line/scatter need numeric X and Y. v1 charts the displayed page;
  GPU-accelerated inline aggregations are a later piece.
- **Explain** — the optimized plan, with two views (toggle in the panel header):
  - **Graph** (default) — a Snowsight-style **query-profile graph**: a
    left-to-right box-and-arrow tree where each node is a colored box (op label,
    `~rows` estimate on `Scan`, `fused` badge on an `Aggregate` over a `Join`) and
    edges connect parents to children. Nodes are colored by category — `Scan`
    (green), `Join` (blue), `Aggregate` (purple), `Filter` (orange), `Sort`/
    `Limit`/`Distinct` (teal), writes (red). Compact detail sits under the op
    label; the full per-node detail (table, join keys, group keys, …) is in the
    hover tooltip. **Wheel zooms toward the cursor; pointer-drag pans; the
    toolbar +/−/⟲ buttons zoom and reset.** Layout is computed by the pure
    `lib/planLayout.ts` (tidy leaf-counting tree) — unit-tested by
    `test/planLayout_check.mjs`.
  - **Tree** — the classic indented text tree (op + `fused`/`~rows` badges +
    full detail inline), for compact plans and reading detail without hovering.
  The `fused` badge marks the star-join+aggregate shape eligible for the fused
  C++ kernel — eligibility, not a guarantee it fired.
- **Catalog** — tables + row counts; expand a table for its columns; click a
  name to drop it into the editor; the `⤵` button runs `SELECT * … LIMIT 100`;
  the `▮` button opens the **column profile** panel for that table. The **+
  Load** button opens the **data-load wizard**, which has two modes:
  - **Upload file** (default) — pick a `.parquet` from your browser; it is sent
    over the wire via the `upload` op (a two-frame text+binary request), the
    server writes it to `<data>/uploads/<uuid>.parquet` and registers the table.
    The table name defaults to the file stem. The size is pre-checked against
    the server's `--max-upload-bytes` (default 256 MB) so an oversized file is
    refused before sending.
  - **From path** — give a table name + a parquet path on the server filesystem
    (a file, a directory expanded to `*.parquet`, or a glob) and it registers
    the table via `admin register` (the engine reads the server's FS directly).
  The per-table **✕** drops it via `admin drop` (unregisters; source files are
  kept). Both refresh the catalog + the autocomplete schema. A `Load data from
  parquet path` command-palette entry opens the wizard too.
- **Column profile** — the `▮` button (per table in the Catalog sidebar) opens
  a panel of per-column statistics computed on the **GPU** by the server's
  `profile` op (one full scan; cuDF reductions + value_counts). Each column
  shows null count/%, distinct count, min/max, and (for numeric columns) mean,
  stddev, a 10-bucket equal-width histogram, and the top-K most frequent
  values for low-NDV columns. The headline GPU differentiator — the stats come
  back in milliseconds, not by scanning in the browser.
- **Table details** — the `ℹ` button (per table in the Catalog) opens an
  object-detail panel: the columns with types + nullability, the constraints
  (primary key, NOT NULL, UNIQUE, defaults — stored, not enforced), the source
  parquet paths, a best-effort reconstructed registration DDL (copyable), and a
  rename control. All from the `table` op + `admin rename` — no server change.
- **History** — the server-side query ring buffer with Snowsight-style
  filtering: a **SQL text filter** (case-insensitive substring), a **kind**
  filter (all / select / write / export / fetch / control / cancelled / other),
  and a **sort** (newest / oldest / slowest / most rows). Each entry shows a
  timestamp, a colored kind badge, row count, and duration; click an entry to
  reload its SQL into the editor. The buffer is bounded (default 500) so
  filtering + sorting run client-side. The `history` op now carries a `ts`
  (epoch-seconds session timestamp) on every entry.
- **Dashboards** — a sidebar tab (Catalog / History / **Dashboards**) lists the
  saved dashboards (each with its widget count); click to open one, double-click
  a name to rename inline, `✕` to delete, `+ New` to create an empty one. A
  **dashboard** is a named, persisted grid of **chart widgets** — each widget is
  a saved SQL query + a chart spec (kind + X/Y column names). Opening a
  dashboard re-runs every widget's query against the live GPU engine and paints
  its chart via the headless `ChartView` (no picker controls); **⟳ Refresh**
  re-runs them all. Widgets are authored on the **Chart** tab: tune the chart
  (kind / X / Y), then **Pin to dashboard** picks a target dashboard (existing
  or "New dashboard") + a widget title, and saves the spec + the SQL that
  produced the charted result (so re-running reproduces that result). A stale
  spec (a column renamed/dropped, or the query reshaped) is healed to a sensible
  default rather than crashing. Dashboards persist to `localStorage` like
  worksheets; the format is client-side only (no server op beyond the widget
  SELECTs). The chart geometry is shared with the interactive Chart tab via the
  pure `lib/chartRender.ts` (unit-tested by `test/chartRender_check.mjs`).
  Command-palette entries: Open Dashboards sidebar, New dashboard, Pin current
  chart (jumps to the Chart tab), and one per saved dashboard.
- **Ask (NL → SQL)** — `Ctrl/Cmd+I` (or the command palette) opens an offline
  natural-language assistant. Type a question and a **schema-aware recognizer**
  (the pure `lib/nlSql.ts`, unit-tested by `test/nlSql_check.mjs`) resolves it
  against the live catalog to a best-effort `SELECT` in real time, showing which
  template matched + which table. Accept loads the SQL into the editor (the user
  runs/refines it — it is never auto-run). Recognized patterns: row count
  (`how many rows in <table>`), top/bottom N by a column (`top 10 lineitem by
  extendedprice`), avg/sum/min/max of a column (`average extendedprice in
  lineitem`), group-by (`group lineitem by l_returnflag`), and sample
  (`show me orders`). Table + column names match **fuzzily** (case-insensitive,
  substring, then a small edit distance, plus a dotted-name tail match so
  `extendedprice` finds `l_extendedprice`), so you don't need the exact
  identifier. An unrecognized question that still names a table falls back to a
  sample; one that names nothing shows a hint. **No network call** — the
  recognizer is a local heuristic, honest about its limits (it is not a general
  NL→SQL model); it covers the common analytical asks over a known schema with
  no CDN/hosted-LLM dependency, matching the offline ethos.
- **Compare SQL (diff)** — `Ctrl/Cmd+Shift+D` (or the command palette) opens a
  side-by-side SQL compare. Pick a left + right source from the dropdowns —
  every worksheet plus the active worksheet's saved version snapshots (the
  common "what did I change?" compare) — and a line-level LCS diff (the pure
  `lib/diff.ts`, unit-tested by `test/diff_check.mjs`) renders equal / added /
  deleted lines with gutter line numbers on each side: added lines are blank on
  the left and green on the right; deleted lines are red on the left and blank
  on the right. A header summary shows `+added / −removed`. Whitespace is
  significant (a re-indented line counts as a change) — right for SQL where
  indentation can matter, and you see exactly what moved. Pure client-side — no
  server op.

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
node test/autocomplete_check.mjs  # src/lib/autocomplete.ts suggestion logic — no server needed
node test/planLayout_check.mjs  # src/lib/planLayout.ts graph geometry — no server needed
node test/worksheetTransfer_check.mjs  # src/lib/worksheetTransfer.ts .sql bundle round-trip — no server needed
node test/chartRender_check.mjs  # src/lib/chartRender.ts bar/point geometry — no server needed
node test/nlSql_check.mjs  # src/lib/nlSql.ts NL->SQL recognizer — no server needed
node test/diff_check.mjs  # src/lib/diff.ts line-level LCS diff — no server needed
```

`npm run smoke` needs a running `ryudb-server` with a registered `lineitem`; set
`RYUDB_PORT` and `RYUDB_LINEITEM_PATH` (a directory containing a lineitem
parquet) and `RYUDB_N` (its row count). It verifies admin register, SELECT
round-trip + Arrow decode + value match, GROUP BY, EXPLAIN fused badge, a
parse error with position, cursor paging, column profile, Parquet export, and
the two-frame `upload` ingest (sends a real parquet's bytes, asserts the table
is registered + queryable) — the same wire path the browser uses.

## Notes / Phase 1 limits

- A **dark/light theme** toggle in the toolbar (☀/☾) switches the whole UI and
  the Monaco editor chrome; the choice persists to `localStorage` and defaults
  to the OS color-scheme preference on first run.
- A **command palette** (`Ctrl/Cmd+K`) fuzzy-filters every action — run /
  explain / cancel, connect / disconnect, new / close / switch worksheet,
  toggle theme, open a sidebar, show shortcuts. `?` opens a keyboard-shortcuts
  help modal.
- A **global object search** (`Ctrl/Cmd+Shift+F`) reaches across the whole
  catalog + query history in one place: type to fuzzy-filter tables, columns,
  and past queries; pick a table/column to drop it into the editor at the
  cursor, or pick a history entry to load its SQL into the active worksheet.
- **Worksheet version history** (`Ctrl/Cmd+Shift+H`) keeps a per-worksheet ring
  of saved SQL snapshots (newest first, capped at 30) in `localStorage`,
  independent of the server-side query history. A snapshot is captured each
  time you run a query (deduped against the latest) or on demand via “Save
  version”; restore loads a snapshot back into the editor, delete/clear prune
  the ring, and rings for closed worksheets are garbage-collected.
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