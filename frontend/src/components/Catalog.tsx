import { useEffect, useState } from "react";
import type { CatalogTable, TableResp } from "../lib/types";

interface Props {
  fetchCatalog: () => Promise<CatalogTable[]>;
  fetchTable: (name: string) => Promise<TableResp>;
  onInsert: (text: string) => void;
  onSample: (name: string) => void;
  onProfile: (name: string) => void;
  /** Open the table detail panel (columns/constraints/paths/DDL/rename). */
  onDetail: (name: string) => void;
  /** Open the data-load wizard (admin register). */
  onLoad: () => void;
  /** Drop a table (admin drop); the parent confirms + refreshes. */
  onDrop: (name: string) => void;
  status: string;
  /** Bumped by the parent after a DDL op so this list re-fetches. */
  nonce: number;
}

/** Sidebar catalog browser. Lists tables + row counts; expanding a table loads
 * its columns (click a column to drop the name into the editor; click the
 * table name to insert it). The refresh button re-fetches after DDL. The
 * histogram button opens the GPU column-profile panel for a table. The "Load
 * data" button opens the registration wizard; the per-table × drops a table. */
export default function Catalog({ fetchCatalog, fetchTable, onInsert, onSample, onProfile, onDetail, onLoad, onDrop, status, nonce }: Props) {
  const [tables, setTables] = useState<CatalogTable[]>([]);
  const [expanded, setExpanded] = useState<string | null>(null);
  const [cols, setCols] = useState<TableResp | null>(null);

  const refresh = () => {
    fetchCatalog().then(setTables).catch(() => setTables([]));
  };

  useEffect(() => {
    if (status === "open") refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [status, nonce]);

  const toggle = (name: string) => {
    if (expanded === name) {
      setExpanded(null);
      return;
    }
    setExpanded(name);
    setCols(null);
    fetchTable(name).then(setCols).catch(() => setCols(null));
  };

  return (
    <div className="catalog">
      <div className="sidebar-head">
        <span>Tables</span>
        <span className="head-actions">
          <button onClick={onLoad} title="Load data from a parquet path" disabled={status !== "open"}>+ Load</button>
          <button onClick={refresh} title="refresh">⟳</button>
        </span>
      </div>
      {tables.length === 0 && <div className="empty">No tables. Click + Load to register a parquet path.</div>}
      <ul className="table-list">
        {tables.map((t) => (
          <li key={t.name}>
            <div className="table-row">
              <button className="caret" onClick={() => toggle(t.name)}>
                {expanded === t.name ? "▾" : "▸"}
              </button>
              <span className="table-name" onClick={() => onInsert(t.name)} title="insert name">
                {t.name}
              </span>
              <span className="row-count">{t.row_count}</span>
              <button className="sample" onClick={() => onProfile(t.name)} title="Column profile (GPU)">▮</button>
              <button className="sample" onClick={() => onSample(t.name)} title="SELECT * LIMIT 100">⤵</button>
              <button className="sample" onClick={() => onDetail(t.name)} title="Table details (columns, constraints, DDL, rename)">ℹ</button>
              <button className="sample drop" onClick={() => onDrop(t.name)} title="Drop table">✕</button>
            </div>
            {expanded === t.name && (
              <ul className="col-list">
                {cols ? (
                  cols.columns.map((c) => (
                    <li key={c.name} onClick={() => onInsert(c.name)} title={`${c.type}${c.nullable ? "" : " NOT NULL"}`}>
                      <span className="col-name">{c.name}</span>
                      <span className="col-type">{c.type}</span>
                    </li>
                  ))
                ) : (
                  <li className="empty">loading…</li>
                )}
              </ul>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}