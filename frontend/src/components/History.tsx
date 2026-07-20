import { useEffect, useMemo, useState } from "react";
import type { HistoryEntry } from "../lib/types";

interface Props {
  fetchHistory: () => Promise<HistoryEntry[]>;
  onPick: (sql: string) => void;
  status: string;
}

type SortKey = "newest" | "oldest" | "slowest" | "rows";

const KINDS = ["all", "select", "write", "export", "fetch", "control", "cancelled", "other"];

/** Server-side query history ring buffer, with Snowsight-style filtering:
 *  - a SQL text filter (case-insensitive substring),
 *  - a kind filter (all / select / write / export / fetch / …),
 *  - a sort (newest / oldest / slowest / most rows),
 *  - a timestamp on each entry.
 *  The ring buffer is bounded (default 500), so filtering + sorting run
 *  client-side over the fetched entries. Click an entry to load its SQL back
 *  into the editor. Re-mounts on sidebar switch, so opening History refreshes. */
export default function History({ fetchHistory, onPick, status }: Props) {
  const [entries, setEntries] = useState<HistoryEntry[]>([]);
  const [filter, setFilter] = useState("");
  const [kind, setKind] = useState("all");
  const [sort, setSort] = useState<SortKey>("newest");

  const refresh = () => {
    fetchHistory().then(setEntries).catch(() => setEntries([]));
  };

  useEffect(() => {
    if (status === "open") refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [status]);

  const view = useMemo(() => {
    const q = filter.trim().toLowerCase();
    let out = entries.filter((e) => {
      if (kind !== "all" && e.kind !== kind) return false;
      if (q && !e.sql.toLowerCase().includes(q)) return false;
      return true;
    });
    const cmp = (a: HistoryEntry, b: HistoryEntry) => {
      switch (sort) {
        case "oldest": return (a.ts ?? 0) - (b.ts ?? 0);
        case "slowest": return b.duration_ms - a.duration_ms;
        case "rows": return b.rows - a.rows;
        case "newest":
        default: return (b.ts ?? 0) - (a.ts ?? 0);
      }
    };
    out = [...out].sort(cmp);
    return out;
  }, [entries, filter, kind, sort]);

  return (
    <div className="history">
      <div className="sidebar-head">
        <span>History</span>
        <button onClick={refresh} title="refresh">⟳</button>
      </div>
      <div className="hist-controls">
        <input
          className="hist-filter"
          type="text"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          placeholder="filter SQL…"
          disabled={entries.length === 0}
        />
        <select value={kind} onChange={(e) => setKind(e.target.value)} disabled={entries.length === 0}>
          {KINDS.map((k) => <option key={k} value={k}>{k}</option>)}
        </select>
        <select value={sort} onChange={(e) => setSort(e.target.value as SortKey)} disabled={entries.length === 0}>
          <option value="newest">newest</option>
          <option value="oldest">oldest</option>
          <option value="slowest">slowest</option>
          <option value="rows">most rows</option>
        </select>
      </div>
      <div className="hist-count">
        {entries.length === 0
          ? "No queries yet."
          : `${view.length}${view.length !== entries.length ? ` of ${entries.length}` : ""} entr${view.length === 1 ? "y" : "ies"}`}
      </div>
      {view.length === 0 && entries.length > 0 && <div className="empty">No matches.</div>}
      <ul className="history-list">
        {view.map((e, i) => (
          <li key={i} onClick={() => onPick(e.sql)} title={e.sql}>
            {e.ts !== undefined && <div className="hist-ts">{fmtTime(e.ts)}</div>}
            <div className="hist-sql">{e.sql}</div>
            <div className="hist-meta">
              <span className={`hist-kind k-${e.kind}`}>{e.kind}</span>
              <span>{e.rows.toLocaleString()} rows</span>
              <span>{e.duration_ms.toFixed(1)} ms</span>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}

/** Format an epoch-seconds timestamp as a local HH:MM:SS string. */
function fmtTime(ts: number): string {
  try {
    const d = new Date(ts * 1000);
    return d.toLocaleTimeString(undefined, { hour12: false });
  } catch {
    return "";
  }
}