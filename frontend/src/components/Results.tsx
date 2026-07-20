import { useMemo } from "react";
import { FixedSizeList } from "react-window";
import type { Table } from "apache-arrow";
import type { ResultMeta } from "../lib/types";

interface Props {
  meta: ResultMeta;
  table: Table | null;
  onDownload: (format: "csv" | "arrow") => void;
  downloading: boolean;
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
 * rows at `max_rows`; `meta.row_count` is the true total and `meta.truncated`
 * flags whether the displayed `meta.returned` rows are a subset. */
export default function Results({ meta, table, onDownload, downloading }: Props) {
  const columns = meta.columns;
  const colArrays = useMemo(() => {
    if (!table) return [];
    return columns.map((c) => table.getChild(c.name)?.toArray() ?? []);
  }, [table, columns]);

  const Row = ({ index, style }: { index: number; style: React.CSSProperties }) => (
    <div className="grid-row" style={style}>
      {colArrays.map((arr, ci) => (
        <div className="grid-cell" key={ci} title={formatCell(arr[index])}>
          {formatCell(arr[index])}
        </div>
      ))}
    </div>
  );

  return (
    <div className="results">
      <div className="results-meta">
        <span>{meta.row_count} row{meta.row_count === 1 ? "" : "s"}</span>
        {meta.truncated && (
          <span className="truncated">
            (showing first {meta.returned} of {meta.row_count})
          </span>
        )}
        <span className="dur">{meta.duration_ms.toFixed(1)} ms</span>
        <span className="dl-group">
          <button className="dl" disabled={downloading} onClick={() => onDownload("csv")}>
            {downloading ? "…" : "Download CSV"}
          </button>
          <button className="dl" disabled={downloading} onClick={() => onDownload("arrow")}>
            Arrow
          </button>
        </span>
      </div>
      <div className="grid-header" style={{ gridTemplateColumns: `repeat(${columns.length}, minmax(120px, 1fr))` }}>
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
          width="100%"
        >
          {Row}
        </FixedSizeList>
      ) : (
        <div className="empty">No rows.</div>
      )}
    </div>
  );
}