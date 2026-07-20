import { useEffect, useMemo, useState } from "react";
import type { TableResp, TableColumn } from "../lib/types";
import { copyText } from "../lib/csv";

interface Props {
  open: boolean;
  name: string | null;
  fetchTable: (name: string) => Promise<TableResp>;
  onRename: (oldName: string, newName: string) => Promise<void>;
  /** Apply a constraint change via the `admin alter` op. Resolves on `ok`,
   *  rejects (Error) on an error frame so the panel can surface the message.
   *  `kind` is one of pk / pk_clear / notnull / notnull_off / unique. */
  onAlter: (args: Record<string, unknown>) => Promise<Record<string, unknown>>;
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

/** Per-table object-detail panel: columns (name/type/nullable + a NOT NULL
 *  toggle), constraints (editable PK / NOT NULL / UNIQUE), source parquet
 *  paths, row count, a reconstructed registration DDL (copyable), and a rename
 *  control — all from the existing `table` op + `admin alter`/`rename`. The
 *  engine enforces PK/UNIQUE on INSERT. */
export default function TableDetailModal({ open, name, fetchTable, onRename, onAlter, onClose }: Props) {
  const [info, setInfo] = useState<TableResp | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [renameVal, setRenameVal] = useState("");
  const [renaming, setRenaming] = useState(false);
  const [renameErr, setRenameErr] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [altering, setAltering] = useState(false);
  const [alterErr, setAlterErr] = useState<string | null>(null);

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
    setAlterErr(null);
    fetchTable(name)
      .then((t) => setInfo(t))
      .catch((e) => setErr((e as Error).message ?? "failed to load table"))
      .finally(() => setLoading(false));
  }, [open, name, fetchTable]);

  const constraints: Constraints = (info?.constraints ?? {}) as Constraints;
  const colNames = useMemo(() => (info ? info.columns.map((c) => c.name) : []), [info]);
  const notNullSet = useMemo(() => new Set(constraints.not_null ?? []), [constraints.not_null]);
  const pkList = constraints.primary_key ?? [];
  const uniqueList = constraints.unique ?? [];

  // Selection state for the PK / add-UNIQUE editors; reset when the table
  // (re)loads so it tracks the persisted constraints.
  const [pkSel, setPkSel] = useState<Set<string>>(new Set());
  const [unqSel, setUnqSel] = useState<Set<string>>(new Set());
  useEffect(() => { setPkSel(new Set(pkList)); setUnqSel(new Set()); /* eslint-disable-next-line react-hooks/exhaustive-deps */ }, [info?.name, JSON.stringify(pkList)]);

  if (!open) return null;

  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === "Escape") {
      e.preventDefault();
      if (!renaming && !altering) onClose();
    }
  };

  const ddl = info ? buildDdl(info.name, info.paths) : "";

  const copyDdl = () => {
    void copyText(ddl).then((ok) => {
      if (!ok) return;
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1200);
    });
  };

  const refresh = () => {
    if (!name) return;
    return fetchTable(name).then((t) => setInfo(t));
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

  const doAlter = async (args: Record<string, unknown>) => {
    if (!name || altering) return;
    setAltering(true);
    setAlterErr(null);
    try {
      await onAlter({ table: name, ...args });
      await refresh();
    } catch (e) {
      setAlterErr((e as Error).message ?? "alter failed");
    } finally {
      setAltering(false);
    }
  };

  const toggle = (sel: Set<string>, setSel: (s: Set<string>) => void, col: string) => {
    const next = new Set(sel);
    if (next.has(col)) next.delete(col); else next.add(col);
    setSel(next);
  };

  const pkChanged = pkSel.size > 0 && JSON.stringify([...pkSel].sort()) !== JSON.stringify([...pkList].sort());

  return (
    <div className="palette-overlay" onClick={() => !renaming && !altering && onClose()}>
      <div className="detail-modal" onClick={(e) => e.stopPropagation()} onKeyDown={onKey}>
        <div className="sidebar-head">
          <span>{name ? `Table: ${name}` : "Table"}</span>
          <button onClick={onClose} disabled={renaming || altering}>Close</button>
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
                  <thead><tr><th>Name</th><th>Type</th><th>Nullable</th><th>NOT NULL</th></tr></thead>
                  <tbody>
                    {info.columns.map((c: TableColumn) => (
                      <tr key={c.name}>
                        <td className="col-name">{c.name}</td>
                        <td className="col-type">{c.type}</td>
                        <td>{c.nullable ? "YES" : "NO"}</td>
                        <td>
                          <input
                            type="checkbox"
                            title={notNullSet.has(c.name) ? "NOT NULL — click to drop" : "click to set NOT NULL"}
                            checked={notNullSet.has(c.name)}
                            disabled={altering}
                            onChange={() =>
                              doAlter(notNullSet.has(c.name)
                                ? { kind: "notnull_off", col: c.name }
                                : { kind: "notnull", col: c.name })
                            }
                          />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="detail-section">
                <div className="detail-section-head"><span>Constraints</span></div>
                <div className="detail-constraints">
                  {/* Primary key: set over a column selection, or clear. */}
                  <div className="con-row">
                    <label>Primary key:</label>
                    <span className="con-val">{pkList.length ? pkList.join(", ") : "—"}</span>
                    {pkList.length > 0 && (
                      <button className="con-act" disabled={altering}
                              onClick={() => doAlter({ kind: "pk_clear" })}>Clear PK</button>
                    )}
                  </div>
                  <div className="con-pick">
                    {colNames.map((c) => (
                      <label key={c} className="con-chip" title={c}>
                        <input type="checkbox" checked={pkSel.has(c)} disabled={altering}
                               onChange={() => toggle(pkSel, setPkSel, c)} />
                        {c}
                      </label>
                    ))}
                    <button className="con-act primary" disabled={altering || !pkChanged}
                            onClick={() => doAlter({ kind: "pk", cols: [...pkSel] })}>Set PK</button>
                  </div>

                  {/* Unique: add-only (the catalog has no drop-unique). */}
                  <div className="con-row">
                    <label>Unique:</label>
                    <span className="con-val">{uniqueList.length ? uniqueList.map((u) => u.join(",")).join(" | ") : "—"}</span>
                  </div>
                  <div className="con-pick">
                    {colNames.map((c) => (
                      <label key={c} className="con-chip" title={c}>
                        <input type="checkbox" checked={unqSel.has(c)} disabled={altering}
                               onChange={() => toggle(unqSel, setUnqSel, c)} />
                        {c}
                      </label>
                    ))}
                    <button className="con-act primary" disabled={altering || unqSel.size === 0}
                            onClick={() => doAlter({ kind: "unique", cols: [...unqSel] })}>Add unique</button>
                  </div>

                  <div><label>Defaults:</label> {constraints.defaults && Object.keys(constraints.defaults).length ? Object.entries(constraints.defaults).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(", ") : "—"}</div>
                  {alterErr && <div className="load-error">{alterErr}</div>}
                  <div className="con-hint">PK / UNIQUE are enforced on INSERT. UNIQUE is add-only here.</div>
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