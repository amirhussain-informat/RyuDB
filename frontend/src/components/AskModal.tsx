import { useEffect, useMemo, useState } from "react";
import { nlToSql, type NlTable } from "../lib/nlSql";

interface Props {
  open: boolean;
  tables: NlTable[];
  onClose: () => void;
  /** Called with the recognized SQL (and template) when the user accepts. The
   *  caller loads it into the active worksheet's editor for refinement. */
  onAccept: (sql: string) => void;
}

/** The offline NL->SQL assistant. A small modal: type a question, the
 *  schema-aware recognizer (`lib/nlSql.ts`) resolves it against the live
 *  catalog to a best-effort SELECT in real time, showing which template matched
 *  + which table. Accept loads the SQL into the editor (the user runs/refines
 *  it — it is never auto-run). No network call: the recognizer is a local
 *  heuristic, honest about its limits. */
export default function AskModal({ open, tables, onClose, onAccept }: Props) {
  const [q, setQ] = useState("");

  useEffect(() => {
    if (open) setQ("");
  }, [open]);

  const result = useMemo(() => (q.trim() ? nlToSql(q, tables) : null), [q, tables]);

  if (!open) return null;

  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && result) {
      e.preventDefault();
      onAccept(result.sql);
      onClose();
    } else if (e.key === "Escape") {
      e.preventDefault();
      onClose();
    }
  };

  return (
    <div className="palette-overlay" onClick={onClose}>
      <div className="load-modal ask-modal" onClick={(e) => e.stopPropagation()} onKeyDown={onKey}>
        <div className="palette-input shortcuts-head vhist-head">
          <span>Ask (NL → SQL)</span>
          <button className="vhist-save" onClick={onClose}>Close</button>
        </div>
        <div className="load-body">
          <input
            className="load-input ask-input"
            type="text"
            value={q}
            autoFocus
            placeholder="e.g. top 10 lineitem by extendedprice"
            onChange={(e) => setQ(e.target.value)}
            onKeyDown={onKey}
          />
          <div className="load-hint">
            Offline schema-aware recognizer · {tables.length} table{tables.length === 1 ? "" : "s"} loaded.
            {" "}Count / top·bottom N by a column / avg·sum·min·max / group-by / show.
          </div>
          {q.trim() && (
            result ? (
              <div className="ask-result">
                <div className="ask-template" title="matched template">
                  interpreted as · {result.template} · table <code>{result.table}</code>
                </div>
                <pre className="ask-sql">{result.sql}</pre>
                <div className="load-actions">
                  <button onClick={() => { onAccept(result.sql); onClose(); }} disabled={!result}>
                    Load into editor
                  </button>
                </div>
              </div>
            ) : (
              <div className="empty ask-empty">
                Couldn't recognize that. Try naming a table (e.g. “count {tables[0]?.name ?? "mytable"}”).
              </div>
            )
          )}
        </div>
      </div>
    </div>
  );
}