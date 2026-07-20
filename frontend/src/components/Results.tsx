import { useMemo, useState } from "react";
import { FixedSizeList } from "react-window";
import type { Table } from "apache-arrow";
import type { ResultMeta } from "../lib/types";
import { copyText, tableToTSV } from "../lib/csv";

interface Props {
  meta: ResultMeta;
  table: Table | null;
  onDownload: (format: "csv" | "json" | "arrow") => void;
  downloading: boolean;
  onLoadMore: () => void;
  hasMore: boolean;
  loadingMore: boolean;
}

const ROW_HEIGHT = 24;

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

/** Renders an Arrow Table as a header + a virtualized row grid. The server caps
 * the first page at `max_rows`; `meta.row_count` is the true total. When
 * `hasMore` is set the result is cursor-backed and a "Load more" button pages
 * the next slice (growing `meta.returned`); otherwise `meta.truncated` marks a
 * non-pageable truncated result. */
export default function Results({ meta, table, onDownload, downloading,
                                  onLoadMore, hasMore, loadingMore }: Props) {
  const columns = meta.columns;
  const colArrays = useMemo(() => {
    if (!table) return [];
    return columns.map((c) => table.getChild(c.name)?.toArray() ?? []);
  }, [table, columns]);

  // `copiedKey` flashes a "copied" tick on the last cell copied to the
  // clipboard; cleared after a short timeout so the indicator is transient.
  const [copiedKey, setCopiedKey] = useState<string | null>(null);
  const [copyAllState, setCopyAllState] = useState<"idle" | "ok" | "err">("idle");
  const copyCell = (key: string, value: string) => {
    void copyText(value).then((ok) => {
      if (!ok) return;
      setCopiedKey(key);
      window.setTimeout(() => setCopiedKey((k) => (k === key ? null : k)), 800);
    });
  };
  const copyAll = () => {
    if (!table) return;
    void copyText(tableToTSV(table.slice(0, meta.returned))).then((ok) => {
      setCopyAllState(ok ? "ok" : "err");
      window.setTimeout(() => setCopyAllState("idle"), 1200);
    });
  };

  // Fixed column width + a fixed inner grid width so the header and the
  // virtualized rows lay out with the TRUE column count and stay aligned.
  // The .grid-row CSS rule otherwise falls back to a hardcoded 4 columns via
  // an unset `--cols` var, which garbles wide results like SELECT *. The
  // outer .grid-scroll wrapper horizontally scrolls header + rows together
  // when the table is wider than the pane.
  const COL_W = 120;
  const gridTemplate = `repeat(${columns.length}, ${COL_W}px)`;
  const gridWidth = columns.length * COL_W;
  const Row = ({ index, style }: { index: number; style: React.CSSProperties }) => (
    <div className="grid-row" style={{ ...style, gridTemplateColumns: gridTemplate }}>
      {colArrays.map((arr, ci) => {
        const key = `${index}:${ci}`;
        const val = formatCell(arr[index]);
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

  return (
    <div className="results">
      <div className="results-meta">
        <span>{meta.row_count} row{meta.row_count === 1 ? "" : "s"}</span>
        {(hasMore || meta.truncated) && (
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
            className="dl"
            disabled={!table || meta.returned === 0 || copyAllState !== "idle"}
            onClick={copyAll}
            title="Copy the displayed rows as TSV to the clipboard"
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
        </span>
      </div>
      <div className="grid-scroll">
        <div className="grid-header" style={{ gridTemplateColumns: gridTemplate, width: gridWidth }}>
          {columns.map((c) => (
            <div className="grid-cell grid-head" key={c.name} title={c.type}>
              {c.name}
              <span className="col-type">{c.type}</span>
            </div>
          ))}
        </div>
        {table && meta.returned > 0 ? (
          <FixedSizeList
            height={Math.min(600, meta.returned * ROW_HEIGHT)}
            itemCount={meta.returned}
            itemSize={ROW_HEIGHT}
            width={gridWidth}
          >
            {Row}
          </FixedSizeList>
        ) : (
          <div className="empty">No rows.</div>
        )}
      </div>
    </div>
  );
}