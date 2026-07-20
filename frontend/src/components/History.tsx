import { useEffect, useState } from "react";
import type { HistoryEntry } from "../lib/types";

interface Props {
  fetchHistory: () => Promise<HistoryEntry[]>;
  onPick: (sql: string) => void;
  status: string;
}

/** Server-side query history ring buffer. Click an entry to load its SQL back
 * into the editor. */
export default function History({ fetchHistory, onPick, status }: Props) {
  const [entries, setEntries] = useState<HistoryEntry[]>([]);

  const refresh = () => {
    fetchHistory().then(setEntries).catch(() => setEntries([]));
  };

  useEffect(() => {
    if (status === "open") refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [status]);

  return (
    <div className="history">
      <div className="sidebar-head">
        <span>History</span>
        <button onClick={refresh} title="refresh">⟳</button>
      </div>
      {entries.length === 0 && <div className="empty">No queries yet.</div>}
      <ul className="history-list">
        {entries.map((e, i) => (
          <li key={i} onClick={() => onPick(e.sql)} title={e.sql}>
            <div className="hist-sql">{e.sql}</div>
            <div className="hist-meta">
              <span>{e.kind}</span>
              <span>{e.rows} rows</span>
              <span>{e.duration_ms.toFixed(1)} ms</span>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}