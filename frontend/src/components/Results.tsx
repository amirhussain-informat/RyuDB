import { useEffect, useMemo, useState } from "react";
import { FixedSizeList } from "react-window";
import type { Table, Vector } from "apache-arrow";
import type { ResultMeta } from "../lib/types";
import { copyText, tableToTSV, viewToTSV } from "../lib/csv";

interface Props {
  meta: ResultMeta;
  table: Table | null;
  onDownload: (format: "csv" | "json" | "arrow" | "parquet") => void;
  downloading: boolean;
  onLoadMore: () => void;
  hasMore: boolean;
  loadingMore: boolean;
}

const ROW_HEIGHT = 24;
type SortDir = "asc" | "desc";

function formatCell(v: unknown): string {
  if (v === null || v === undefined) return "NULL";
  if (v instanceof Date) return v.toISOString();
  if (typeof v === "bigint") return v.toString();
  if (typeof v === "object") {
    try {
      return JSON.stringify(v);
    } catch {
      return String(v);
    }
  }
  return String(v);
}

/** Total-order comparator on a raw Arrow cell value. Nulls sort last in both
 * directions (a NULL is "unknown", not smallest). bigint/number/Date compare
 * numerically; everything else falls back to string compare. The column is
 * homogeneous in Arrow, so the type branches don't have to mix. */
function compareValues(a: unknown, b: unknown, dir: SortDir): number {
  const na = a === null || a === undefined;
  const nb = b === null || b === undefined;
  if (na && nb) return 0;
  if (na) return 1; // nulls last regardless of dir
  if (nb) return -1;
  const mul = dir === "desc" ? -1 : 1;
  if (typeof a === "bigint" || typeof b === "bigint") {
    const x = BigInt(a as bigint | number);
    const y = BigInt(b as bigint | number);
    return x < y ? -mul : x > y ? mul : 0;
  }
  if (a instanceof Date || b instanceof Date) {
    const x = a instanceof Date ? a.getTime() : Number(a);
    const y = b instanceof Date ? b.getTime() : Number(b);
    return x < y ? -mul : x > y ? mul : 0;
  }
  if (typeof a === "number" && typeof b === "number") {
    return (a - b) * mul;
  }
  const sa = String(a);
  const sb = String(b);
  return sa < sb ? -mul : sa > sb ? mul : 0;
}

/** Renders an Arrow Table as a header + a virtualized row grid. The server caps
 * the first page at `max_rows`; `meta.row_count` is the true total. When
 * `hasMore` is set the result is cursor-backed and a "Load more" button pages
 * the next slice (growing `meta.returned`); otherwise `meta.truncated` marks a
 * non-pageable truncated result.
 *
 * Sort + filter operate on the LOADED page in-browser (no server round-trip):
 * click a header to cycle asc → desc → clear; toggle the Filter row to type a
 * per-column case-insensitive substring. The grid (and Copy) reflect the
 * resulting view; Download still pulls the full server result. */
export default function Results({ meta, table, onDownload, downloading,
                                  onLoadMore, hasMore, loadingMore }: Props) {
  const columns = meta.columns;
  // Keep both the Arrow Vectors (for null-aware `.get`/`.isValid`) and their
  // typed arrays (for fast indexed numeric reads in the sort comparator).
  // `.toArray()` drops null info — a NULL int renders as 0 — so the grid uses
  // the vector validity bitmap to distinguish NULL from a real 0/NaN.
  const colVecs = useMemo<(Vector | null)[]>(
    () => (table ? columns.map((c) => table.getChild(c.name) ?? null) : []),
    [table, columns],
  );
  const colArrays = useMemo<unknown[]>(
    () => colVecs.map((v) => (v ? v.toArray() : [])),
    [colVecs],
  );
  const cellVal = (ci: number, r: number): unknown => {
    const v = colVecs[ci];
    if (v && !v.isValid(r)) return null;
    return (colArrays[ci] as unknown[])[r];
  };

  const [sortCol, setSortCol] = useState<number | null>(null);
  const [sortDir, setSortDir] = useState<SortDir | null>(null);
  const [filters, setFilters] = useState<string[]>([]);
  const [showFilters, setShowFilters] = useState(false);
  const [copiedKey, setCopiedKey] = useState<string | null>(null);
  const [copyAllState, setCopyAllState] = useState<"idle" | "ok" | "err">("idle");

  // Reset sort + filter whenever a new result lands (new columns identity).
  useEffect(() => {
    setSortCol(null);
    setSortDir(null);
    setFilters([]);
    setShowFilters(false);
  }, [columns]);

  // `view` is the list of source-row indices after filtering + sorting; the
  // grid renders `view[index]` instead of `index` directly so a re-sort or
  // filter just rebuilds this array (the underlying Arrow Table is untouched).
  const view = useMemo(() => {
    const n = meta.returned;
    let idx = Array.from({ length: n }, (_, i) => i);
    const active = filters
      .map((f, i) => (f && f.trim() ? { i, q: f.trim().toLowerCase() } : null))
      .filter((x): x is { i: number; q: string } => x !== null);
    if (active.length) {
      idx = idx.filter((r) =>
        active.every(({ i, q }) => formatCell(cellVal(i, r)).toLowerCase().includes(q)),
      );
    }
    if (sortCol != null && sortDir && colVecs[sortCol]) {
      const arr = colArrays[sortCol] as unknown[];
      const vec = colVecs[sortCol]!;
      const dir = sortDir;
      // Precompute a null mask once (O(n) validity checks) so the n log n
      // sort comparator only does cheap array reads instead of per-compare
      // bitmap lookups.
      const nullMask = new Uint8Array(n);
      for (let i = 0; i < n; i++) nullMask[i] = vec.isValid(i) ? 0 : 1;
      idx = [...idx].sort((a, b) => {
        const na = nullMask[a] === 1;
        const nb = nullMask[b] === 1;
        if (na && nb) return 0;
        if (na) return 1; // nulls last regardless of dir
        if (nb) return -1;
        return compareValues(arr[a], arr[b], dir);
      });
    }
    return idx;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [colVecs, colArrays, meta.returned, filters, sortCol, sortDir]);

  const copyCell = (key: string, value: string) => {
    void copyText(value).then((ok) => {
      if (!ok) return;
      setCopiedKey(key);
      window.setTimeout(() => setCopiedKey((k) => (k === key ? null : k)), 800);
    });
  };
  const copyAll = () => {
    if (!table) return;
    const unfiltered = view.length === meta.returned && !filters.some((f) => f && f.trim()) && sortCol == null;
    const tsv = unfiltered
      ? tableToTSV(table.slice(0, meta.returned))
      : viewToTSV(
          columns.map((c) => c.name),
          colVecs.map((v) => (v ?? { get: () => null })),
          view,
        );
    void copyText(tsv).then((ok) => {
      setCopyAllState(ok ? "ok" : "err");
      window.setTimeout(() => setCopyAllState("idle"), 1200);
    });
  };

  const cycleSort = (ci: number) => {
    if (sortCol !== ci) {
      setSortCol(ci);
      setSortDir("asc");
    } else if (sortDir === "asc") {
      setSortDir("desc");
    } else if (sortDir === "desc") {
      setSortCol(null);
      setSortDir(null);
    } else {
      setSortDir("asc");
    }
  };

  // Fixed column width + a fixed inner grid width so the header, the optional
  // filter row, and the virtualized rows lay out with the TRUE column count and
  // stay aligned. The .grid-scroll wrapper horizontally scrolls them together.
  const COL_W = 120;
  const gridTemplate = `repeat(${columns.length}, ${COL_W}px)`;
  const gridWidth = columns.length * COL_W;
  const Row = ({ index, style }: { index: number; style: React.CSSProperties }) => {
    const r = view[index];
    return (
      <div className="grid-row" style={{ ...style, gridTemplateColumns: gridTemplate }}>
        {colArrays.map((_, ci) => {
          const key = `${r}:${ci}`;
          const val = formatCell(cellVal(ci, r));
          return (
            <div
              className={"grid-cell" + (copiedKey === key ? " copied" : "")}
              key={ci}
              title={`${val}  (click to copy)`}
              onClick={() => copyCell(key, val)}
            >
              {val}
            </div>
          );
        })}
      </div>
    );
  };

  const filtered = filters.some((f) => f && f.trim());
  const shown = view.length;

  return (
    <div className="results">
      <div className="results-meta">
        <span>{meta.row_count} row{meta.row_count === 1 ? "" : "s"}</span>
        {(filtered || sortCol != null) && (
          <span className="truncated">(view: {shown} of {meta.returned})</span>
        )}
        {(hasMore || meta.truncated) && !filtered && sortCol == null && (
          <span className="truncated">
            (showing {meta.returned} of {meta.row_count})
          </span>
        )}
        {hasMore && (
          <button className="dl" disabled={loadingMore} onClick={onLoadMore}>
            {loadingMore ? "Loading…" : "Load more"}
          </button>
        )}
        <span className="dur">{meta.duration_ms.toFixed(1)} ms</span>
        <span className="dl-group">
          <button
            className={"dl" + (showFilters ? " primary" : "")}
            disabled={!table || meta.returned === 0}
            onClick={() => setShowFilters((s) => !s)}
            title="Toggle per-column substring filter"
          >
            Filter
          </button>
          <button
            className="dl"
            disabled={!table || shown === 0 || copyAllState !== "idle"}
            onClick={copyAll}
            title="Copy the current view as TSV to the clipboard"
          >
            {copyAllState === "idle" ? "Copy" : copyAllState === "ok" ? "Copied" : "Failed"}
          </button>
          <button className="dl" disabled={downloading} onClick={() => onDownload("csv")}>
            {downloading ? "…" : "Download CSV"}
          </button>
          <button className="dl" disabled={downloading} onClick={() => onDownload("json")}>
            JSON
          </button>
          <button className="dl" disabled={downloading} onClick={() => onDownload("arrow")}>
            Arrow
          </button>
          <button
            className="dl"
            disabled={downloading}
            onClick={() => onDownload("parquet")}
            title="Download the full result as Parquet (serialized server-side)"
          >
            Parquet
          </button>
        </span>
      </div>
      <div className="grid-scroll">
        <div className="grid-header" style={{ gridTemplateColumns: gridTemplate, width: gridWidth }}>
          {columns.map((c, ci) => {
            const active = sortCol === ci;
            const arrow = active ? (sortDir === "asc" ? " ▲" : " ▼") : "";
            return (
              <div
                className={"grid-cell grid-head" + (active ? " sorted" : "")}
                key={c.name}
                title={`${c.type} — click to sort`}
                onClick={() => cycleSort(ci)}
              >
                <span className="col-name-h">{c.name}{arrow}</span>
                <span className="col-type">{c.type}</span>
              </div>
            );
          })}
        </div>
        {showFilters && table && meta.returned > 0 && (
          <div className="grid-filter" style={{ gridTemplateColumns: gridTemplate, width: gridWidth }}>
            {columns.map((c, ci) => (
              <input
                key={c.name}
                className="filter-input"
                type="text"
                value={filters[ci] ?? ""}
                onChange={(e) =>
                  setFilters((fs) => {
                    const next = fs.slice();
                    next[ci] = e.target.value;
                    return next;
                  })
                }
                placeholder={`filter ${c.name}`}
                title={`Case-insensitive substring filter on ${c.name}`}
              />
            ))}
          </div>
        )}
        {table && meta.returned > 0 ? (
          shown === 0 ? (
            <div className="empty">No rows match the filter.</div>
          ) : (
            <FixedSizeList
              height={Math.min(600, shown * ROW_HEIGHT)}
              itemCount={shown}
              itemSize={ROW_HEIGHT}
              width={gridWidth}
            >
              {Row}
            </FixedSizeList>
          )
        ) : (
          <div className="empty">No rows.</div>
        )}
      </div>
    </div>
  );
}