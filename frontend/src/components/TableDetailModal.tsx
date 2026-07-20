import { useEffect, useState } from "react";
import type { TableResp, TableColumn } from "../lib/types";
import { copyText } from "../lib/csv";

interface Props {
  open: boolean;
  name: string | null;
  fetchTable: (name: string) => Promise<TableResp>;
  onRename: (oldName: string, newName: string) => Promise<void>;
  onClose: () => void;
}

interface Constraints {
  not_null?: string[];
  primary_key?: string[] | null;
  unique?: string[][];
  defaults?: Record<string, unknown>;
}

/** Best-effort DDL reconstruction. The engine has no SHOW CREATE and stores
 *  only the RESOLVED parquet paths (not the original register arg), so for a
 *  single file we emit `CREATE TABLE <name> FROM '<path>'`; for many files we
 *  emit the common parent directory if they share one, else the path list. */
function buildDdl(name: string, paths: string[]): string {
  if (paths.length === 0) return `CREATE TABLE ${name} FROM '<unknown>'`;
  if (paths.length === 1) return `CREATE TABLE ${name} FROM '${paths[0]}'`;
  // common parent directory
  const split = paths.map((p) => p.split("/"));
  const n = Math.min(...split.map((s) => s.length));
  let common = 0;
  for (let i = 0; i < n; i++) {
    if (split.every((s) => s[i] === split[0][i])) common = i + 1;
    else break;
  }
  if (common > 0) {
    const dir = split[0].slice(0, common).join("/");
    return `CREATE TABLE ${name} FROM '${dir}/'  -- ${paths.length} parquet files`;
  }
  return `CREATE TABLE ${name} FROM '<${paths.length} files>'  -- ${paths.join(", ")}`;
}

/** Per-table object-detail panel: columns (name/type/nullable), constraints,
 *  source parquet paths, row count, a reconstructed registration DDL (copyable),
 *  and a rename control — all from the existing `table` op + `admin rename`.
 *  No server change. */
export default function TableDetailModal({ open, name, fetchTable, onRename, onClose }: Props) {
  const [info, setInfo] = useState<TableResp | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [renameVal, setRenameVal] = useState("");
  const [renaming, setRenaming] = useState(false);
  const [renameErr, setRenameErr] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (!open || !name) {
      setInfo(null);
      setErr(null);
      return;
    }
    setLoading(true);
    setErr(null);
    setRenameVal(name);
    setRenameErr(null);
    fetchTable(name)
      .then((t) => setInfo(t))
      .catch((e) => setErr((e as Error).message ?? "failed to load table"))
      .finally(() => setLoading(false));
  }, [open, name, fetchTable]);

  if (!open) return null;

  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === "Escape") {
      e.preventDefault();
      if (!renaming) onClose();
    }
  };

  const constraints: Constraints = (info?.constraints ?? {}) as Constraints;
  const ddl = info ? buildDdl(info.name, info.paths) : "";

  const copyDdl = () => {
    void copyText(ddl).then((ok) => {
      if (!ok) return;
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1200);
    });
  };

  const doRename = async () => {
    if (!name) return;
    const nn = renameVal.trim();
    if (!nn || nn === name) {
      setRenameErr(null);
      return;
    }
    setRenaming(true);
    setRenameErr(null);
    try {
      await onRename(name, nn);
      onClose();
    } catch (e) {
      setRenameErr((e as Error).message ?? "rename failed");
    } finally {
      setRenaming(false);
    }
  };

  return (
    <div className="palette-overlay" onClick={() => !renaming && onClose()}>
      <div className="detail-modal" onClick={(e) => e.stopPropagation()} onKeyDown={onKey}>
        <div className="sidebar-head">
          <span>{name ? `Table: ${name}` : "Table"}</span>
          <button onClick={onClose} disabled={renaming}>Close</button>
        </div>
        <div className="detail-body">
          {loading && <div className="empty">loading…</div>}
          {err && <div className="msg error">{err}</div>}
          {info && (
            <>
              <div className="detail-section">
                <div className="detail-section-head">
                  <span>Columns</span>
                  <span className="row-count">{info.columns.length} · {info.row_count.toLocaleString()} rows</span>
                </div>
                <table className="detail-cols">
                  <thead><tr><th>Name</th><th>Type</th><th>Nullable</th></tr></thead>
                  <tbody>
                    {info.columns.map((c: TableColumn) => (
                      <tr key={c.name}>
                        <td className="col-name">{c.name}</td>
                        <td className="col-type">{c.type}</td>
                        <td>{c.nullable ? "YES" : "NO"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="detail-section">
                <div className="detail-section-head"><span>Constraints</span></div>
                <div className="detail-constraints">
                  <div><label>Primary key:</label> {constraints.primary_key?.length ? constraints.primary_key.join(", ") : "—"}</div>
                  <div><label>Not null:</label> {constraints.not_null?.length ? constraints.not_null.join(", ") : "—"}</div>
                  <div><label>Unique:</label> {constraints.unique?.length ? constraints.unique.map((u) => u.join(",")).join(" | ") : "—"}</div>
                  <div><label>Defaults:</label> {constraints.defaults && Object.keys(constraints.defaults).length ? Object.entries(constraints.defaults).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(", ") : "—"}</div>
                </div>
              </div>

              <div className="detail-section">
                <div className="detail-section-head"><span>Source files</span></div>
                <ul className="detail-paths">
                  {info.paths.map((p) => <li key={p} title={p}>{p}</li>)}
                </ul>
              </div>

              <div className="detail-section">
                <div className="detail-section-head">
                  <span>DDL</span>
                  <button className="dl" onClick={copyDdl}>{copied ? "Copied" : "Copy"}</button>
                </div>
                <pre className="detail-ddl">{ddl}</pre>
              </div>

              <div className="detail-section">
                <div className="detail-section-head"><span>Rename table</span></div>
                <div className="detail-rename">
                  <input
                    className="load-input"
                    type="text"
                    value={renameVal}
                    onChange={(e) => setRenameVal(e.target.value)}
                    disabled={renaming}
                  />
                  <button className="primary" onClick={doRename} disabled={renaming || !renameVal.trim() || renameVal.trim() === name}>
                    {renaming ? "…" : "Rename"}
                  </button>
                </div>
                {renameErr && <div className="load-error">{renameErr}</div>}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}