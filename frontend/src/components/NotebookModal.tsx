import { useCallback, useEffect, useState } from "react";
import type { Table } from "apache-arrow";
import type { Notebook, NotebookCell, NotebookCellType, PyResp, Result, ResultMeta } from "../lib/types";

interface Props {
  open: boolean;
  notebook: Notebook | null;
  connected: boolean;
  /** Run a SQL cell's statement and return the result (meta + decoded Arrow
   *  table), or null on a non-result / error response. */
  fetchSql: (sql: string) => Promise<Result | null>;
  /** Run a Python cell and return the `py` response, or null on error. */
  runPy: (code: string) => Promise<PyResp | null>;
  onAddCell: (type: NotebookCellType) => void;
  onUpdateCell: (cellId: string, patch: Partial<Pick<NotebookCell, "code" | "type">>) => void;
  onRemoveCell: (cellId: string) => void;
  onMoveCell: (cellId: string, delta: number) => void;
  onClose: () => void;
}

/** A full-overlay SQL+Python notebook. Each cell is either a SQL cell (runs
 *  through the `sql` op → a mini result table) or a Python cell (runs through
 *  the `py` op with an injected `sql()` helper → captured stdout + a result
 *  repr, or an inline traceback on error). Outputs are transient — re-running
 *  a cell overwrites its output; nothing is persisted. Run-all executes the
 *  cells top-to-bottom in order. This is the Snowsight "Worksheets / notebooks"
 *  parity — mixing SQL and Python against the live GPU engine. */
export default function NotebookModal({
  open, notebook, connected, fetchSql, runPy,
  onAddCell, onUpdateCell, onRemoveCell, onMoveCell, onClose,
}: Props) {
  // Per-cell output, keyed by cell id. Cleared when a different notebook opens.
  const [outs, setOuts] = useState<Record<string, CellOut>>({});
  const [runningAll, setRunningAll] = useState(false);

  useEffect(() => {
    if (open) setOuts({});
  }, [open, notebook?.id]);

  const runCell = useCallback(async (cell: NotebookCell) => {
    if (!connected) return;
    setOuts((o) => ({ ...o, [cell.id]: { tag: "loading" } }));
    try {
      if (cell.type === "sql") {
        const res = await fetchSql(cell.code);
        if (!res) { setOuts((o) => ({ ...o, [cell.id]: { tag: "error", message: "no response" } })); return; }
        if (res.meta.op === "result") {
          setOuts((o) => ({ ...o, [cell.id]: { tag: "sql-ok", meta: res.meta as ResultMeta, table: res.table } }));
        } else if (res.meta.op === "write") {
          const w = res.meta as { rows_affected?: number };
          setOuts((o) => ({ ...o, [cell.id]: { tag: "py-ok", resp: { op: "py", stdout: "", result: `${w.rows_affected ?? 0} rows affected`, error: null, duration_ms: 0 } as PyResp } }));
        } else if (res.meta.op === "ok") {
          setOuts((o) => ({ ...o, [cell.id]: { tag: "py-ok", resp: { op: "py", stdout: "", result: "ok", error: null, duration_ms: 0 } as PyResp } }));
        } else if (res.meta.op === "error") {
          setOuts((o) => ({ ...o, [cell.id]: { tag: "error", message: (res.meta as { message?: string }).message ?? "error" } }));
        } else {
          setOuts((o) => ({ ...o, [cell.id]: { tag: "error", message: `unexpected op: ${res.meta.op}` } }));
        }
      } else {
        const resp = await runPy(cell.code);
        if (!resp) { setOuts((o) => ({ ...o, [cell.id]: { tag: "error", message: "no response" } })); return; }
        setOuts((o) => ({ ...o, [cell.id]: { tag: "py-ok", resp } }));
      }
    } catch (e: unknown) {
      setOuts((o) => ({ ...o, [cell.id]: { tag: "error", message: (e as Error)?.message ?? String(e) } }));
    }
  }, [connected, fetchSql, runPy]);

  const runAll = async () => {
    if (!notebook || runningAll) return;
    setRunningAll(true);
    try {
      for (const cell of notebook.cells) {
        // re-fetch the latest code from the notebook each iteration (a prior
        // cell's run can't edit cells, but this keeps the loop honest).
        await runCell(cell);
      }
    } finally {
      setRunningAll(false);
    }
  };

  if (!open || !notebook) return null;

  return (
    <div className="palette-overlay diff-overlay nb-overlay" onClick={onClose}>
      <div className="nb-modal" onClick={(e) => e.stopPropagation()}>
        <div className="palette-input shortcuts-head vhist-head diff-head nb-head">
          <span className="dash-title" title={notebook.name}>{notebook.name}</span>
          <span className="vhist-actions">
            <button onClick={() => onAddCell("sql")} disabled={!connected} title="Append a SQL cell">+ SQL</button>
            <button onClick={() => onAddCell("python")} disabled={!connected} title="Append a Python cell">+ Python</button>
            <button
              onClick={runAll}
              disabled={!connected || notebook.cells.length === 0 || runningAll}
              title="Run every cell top-to-bottom"
            >{runningAll ? "Running…" : "Run all"}</button>
            <button className="vhist-save" onClick={onClose}>Close</button>
          </span>
        </div>
        <div className="nb-body">
          {!connected && <div className="empty">Connect to ryudb-server to run notebook cells.</div>}
          {connected && notebook.cells.length === 0 && (
            <div className="empty">No cells. “+ SQL” or “+ Python” to add one.</div>
          )}
          {connected && notebook.cells.map((cell, i) => (
            <Cell
              key={cell.id}
              cell={cell}
              index={i}
              last={notebook.cells.length - 1}
              out={outs[cell.id]}
              running={outs[cell.id]?.tag === "loading"}
              connected={connected}
              onChangeCode={(code) => onUpdateCell(cell.id, { code })}
              onChangeType={(type) => onUpdateCell(cell.id, { type })}
              onRun={() => runCell(cell)}
              onRemove={() => onRemoveCell(cell.id)}
              onMove={(d) => onMoveCell(cell.id, d)}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

type CellOut =
  | { tag: "loading" }
  | { tag: "error"; message: string }
  | { tag: "sql-ok"; meta: ResultMeta; table: Table | null }
  | { tag: "py-ok"; resp: PyResp };

function Cell({
  cell, index, last, out, running, connected,
  onChangeCode, onChangeType, onRun, onRemove, onMove,
}: {
  cell: NotebookCell;
  index: number;
  last: number;
  out: CellOut | undefined;
  running: boolean;
  connected: boolean;
  onChangeCode: (code: string) => void;
  onChangeType: (type: NotebookCellType) => void;
  onRun: () => void;
  onRemove: () => void;
  onMove: (delta: number) => void;
}) {
  return (
    <div className="nb-cell">
      <div className="nb-cell-head">
        <span className="nb-cell-index" title="cell order">[{index + 1}]</span>
        <select
          className="nb-cell-type"
          value={cell.type}
          onChange={(e) => onChangeType(e.target.value as NotebookCellType)}
          title="Cell type"
        >
          <option value="sql">SQL</option>
          <option value="python">Python</option>
        </select>
        <span className="nb-cell-spacer" />
        <button onClick={() => onMove(-1)} disabled={index === 0} title="Move up">↑</button>
        <button onClick={() => onMove(1)} disabled={index === last} title="Move down">↓</button>
        <button className="sample drop" onClick={onRemove} title="Delete cell">✕</button>
        <button className="nb-run" onClick={onRun} disabled={!connected || running} title="Run this cell">
          {running ? "…" : "▶ Run"}
        </button>
      </div>
      <textarea
        className={"nb-code " + (cell.type === "sql" ? "nb-code-sql" : "nb-code-py")}
        value={cell.code}
        spellCheck={false}
        placeholder={cell.type === "sql" ? "SELECT * FROM lineitem LIMIT 10" : "df = sql('SELECT * FROM lineitem LIMIT 10')\nprint(df.head())"}
        onChange={(e) => onChangeCode(e.target.value)}
        onKeyDown={(e) => {
          // Ctrl/Cmd+Enter runs this cell (matches the editor shortcut).
          if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); onRun(); }
        }}
      />
      <div className="nb-out">
        {out?.tag === "loading" && <div className="empty">running…</div>}
        {out?.tag === "error" && <pre className="msg error nb-err">{out.message}</pre>}
        {out?.tag === "sql-ok" && (
          out.table ? <CellTable meta={out.meta} table={out.table} /> : <div className="empty">no rows</div>
        )}
        {out?.tag === "py-ok" && <PyOutput resp={out.resp} />}
      </div>
    </div>
  );
}

/** A compact read-only HTML table from an Arrow result, capped at 100 rows.
 *  Intentionally not the virtualized Results grid — notebook cells show a small
 *  preview, and a plain table keeps the modal light. */
const PREVIEW_ROWS = 100;

function CellTable({ meta, table }: { meta: ResultMeta; table: Table }) {
  const cols = meta.columns;
  const colArrs = cols.map((c) => table.getChild(c.name)?.toArray() ?? []);
  const n = Math.min(meta.returned, PREVIEW_ROWS);
  return (
    <div className="nb-table-wrap">
      <div className="nb-table-meta">
        {meta.row_count.toLocaleString()} rows · {meta.duration_ms.toFixed(0)} ms
        {meta.returned > PREVIEW_ROWS && ` (showing first ${PREVIEW_ROWS})`}
      </div>
      <table className="nb-table">
        <thead>
          <tr>{cols.map((c) => <th key={c.name} title={c.type}>{c.name}</th>)}</tr>
        </thead>
        <tbody>
          {Array.from({ length: n }, (_, r) => (
            <tr key={r}>
              {colArrs.map((arr, ci) => {
                const v = arr[r];
                return <td key={ci}>{v === null || v === undefined ? <span className="null">NULL</span> : String(v)}</td>;
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function PyOutput({ resp }: { resp: PyResp }) {
  const hasErr = !!resp.error;
  return (
    <div className="nb-py-out">
      {resp.stdout && <pre className="nb-stdout">{resp.stdout}</pre>}
      {resp.result !== null && resp.result !== "" && (
        <pre className="nb-result" title="repr() of the cell's expression value">[{resp.result}]</pre>
      )}
      {hasErr && <pre className="msg error nb-err">{resp.error}</pre>}
      {!resp.stdout && resp.result === null && !hasErr && <div className="empty">(no output)</div>}
      {resp.duration_ms > 0 && <div className="nb-table-meta">{resp.duration_ms.toFixed(0)} ms</div>}
    </div>
  );
}