import { useEffect, useMemo, useState } from "react";
import { diffLines, toSideBySide } from "../lib/diff";

export interface DiffSource {
  id: string;
  label: string;
  sql: string;
}

interface Props {
  open: boolean;
  sources: DiffSource[];
  onClose: () => void;
}

/** A side-by-side SQL compare view. Pick a left + right source (worksheets,
 *  version-history snapshots of the active worksheet) from the dropdowns; the
 *  line-level LCS diff (`lib/diff.ts`) renders equal / added / deleted lines
 *  with gutter line numbers on each side. Added lines are blank on the left and
 *  green on the right; deleted lines are red on the left and blank on the right.
 *  A summary line shows +added / -removed. Pure client-side — no server op. */
export default function DiffModal({ open, sources, onClose }: Props) {
  const [leftId, setLeftId] = useState("");
  const [rightId, setRightId] = useState("");

  // Default to the first two distinct sources when opened.
  useEffect(() => {
    if (open) {
      setLeftId(sources[0]?.id ?? "");
      setRightId(sources[1]?.id ?? sources[0]?.id ?? "");
    }
  }, [open, sources]);

  const left = sources.find((s) => s.id === leftId) ?? null;
  const right = sources.find((s) => s.id === rightId) ?? null;

  const rows = useMemo(
    () => (left && right ? toSideBySide(diffLines(left.sql, right.sql)) : []),
    [left, right],
  );
  // Count added (right-only) + removed (left-only) lines from the side-by-side
  // rows directly (a del shows as a left row; an add as a right row).
  const { added, removed } = useMemo(() => {
    let a = 0, r = 0;
    for (const row of rows) {
      if (row.aOp === "del") r++;
      else if (row.bOp === "add") a++;
    }
    return { added: a, removed: r };
  }, [rows]);

  if (!open) return null;

  return (
    <div className="palette-overlay diff-overlay" onClick={onClose}>
      <div className="diff-modal" onClick={(e) => e.stopPropagation()}>
        <div className="palette-input shortcuts-head vhist-head diff-head">
          <span>Compare SQL</span>
          <span className="vhist-actions">
            <span className="diff-summary" title="lines changed">
              <span className="diff-add">+{added}</span>
              <span className="diff-del">−{removed}</span>
            </span>
            <button className="vhist-save" onClick={onClose}>Close</button>
          </span>
        </div>
        <div className="diff-pickers">
          <label>left
            <select value={leftId} onChange={(e) => setLeftId(e.target.value)}>
              {sources.map((s) => <option key={s.id} value={s.id}>{s.label}</option>)}
            </select>
          </label>
          <label>right
            <select value={rightId} onChange={(e) => setRightId(e.target.value)}>
              {sources.map((s) => <option key={s.id} value={s.id}>{s.label}</option>)}
            </select>
          </label>
        </div>
        <div className="diff-body">
          {sources.length < 2 ? (
            <div className="empty">Need at least two SQL sources to compare (open another worksheet or save a version).</div>
          ) : (
            <div className="diff-cols">
              <div className="diff-col">
                {rows.map((r, i) => (
                  <div key={i} className={`diff-line ${r.aOp ?? "gap"}`}>
                    <span className="diff-gutter">{r.aLine ?? ""}</span>
                    <span className="diff-text">{r.aText ?? ""}</span>
                  </div>
                ))}
              </div>
              <div className="diff-col">
                {rows.map((r, i) => (
                  <div key={i} className={`diff-line ${r.bOp ?? "gap"}`}>
                    <span className="diff-gutter">{r.bLine ?? ""}</span>
                    <span className="diff-text">{r.bText ?? ""}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}