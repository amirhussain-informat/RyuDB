import { useEffect, useRef } from "react";
import type { Snapshot } from "../hooks/useVersions";

interface Props {
  open: boolean;
  worksheetName: string;
  snapshots: Snapshot[];
  onClose: () => void;
  /** Save a snapshot of the current editor SQL right now. */
  onSaveNow: () => void;
  /** Load a snapshot's SQL into the active worksheet. */
  onRestore: (snap: Snapshot) => void;
  /** Delete one snapshot. */
  onDelete: (snap: Snapshot) => void;
  /** Drop the whole ring for this worksheet. */
  onClear: () => void;
}

function relTime(ts: number): string {
  const s = Math.max(1, Math.round((Date.now() - ts) / 1000));
  if (s < 60) return `${s}s ago`;
  const m = Math.round(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.round(m / 60);
  if (h < 24) return `${h}h ago`;
  const d = Math.round(h / 24);
  if (d < 30) return `${d}d ago`;
  return new Date(ts).toLocaleDateString();
}

function absTime(ts: number): string {
  try {
    return new Date(ts).toLocaleString();
  } catch {
    return String(ts);
  }
}

function preview(sql: string): string {
  const first = sql.split("\n").find((l) => l.trim().length > 0) ?? "";
  return first.length > 70 ? first.slice(0, 70) + "…" : first;
}

/** A per-worksheet version-history modal. Lists saved SQL snapshots (newest
 * first), lets the user restore one into the editor, delete single snapshots,
 * save the current SQL as a version, or clear the lot. Reuses the palette
 * overlay styling. */
export default function VersionHistory({
  open, worksheetName, snapshots, onClose, onSaveNow, onRestore, onDelete, onClear,
}: Props) {
  const saveBtnRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (open) {
      const t = setTimeout(() => saveBtnRef.current?.focus(), 0);
      return () => clearTimeout(t);
    }
  }, [open]);

  if (!open) return null;

  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === "Escape") { e.preventDefault(); onClose(); }
  };

  return (
    <div className="palette-overlay" onClick={onClose}>
      <div className="palette" onClick={(e) => e.stopPropagation()} onKeyDown={onKey}>
        <div className="palette-input shortcuts-head vhist-head">
          <span>Version history — {worksheetName}</span>
          <span className="vhist-actions">
            <button ref={saveBtnRef} className="vhist-save" onClick={onSaveNow}>Save version</button>
            {snapshots.length > 0 && (
              <button className="vhist-clear" onClick={onClear}>Clear all</button>
            )}
          </span>
        </div>
        <div className="palette-list">
          {snapshots.length === 0 && (
            <div className="palette-empty">
              No versions yet. Run a query or “Save version” to snapshot the current SQL.
            </div>
          )}
          {snapshots.map((s) => (
            <div className="palette-item vhist-row" key={s.id}>
              <span className="palette-label vhist-sql" title={s.sql}>
                <span className="vhist-preview">{preview(s.sql) || "(empty)"}</span>
                <span className="vhist-time" title={absTime(s.ts)}>{relTime(s.ts)}</span>
              </span>
              <span className="palette-meta vhist-row-actions">
                <button className="vhist-restore" onClick={() => onRestore(s)}>Restore</button>
                <button className="vhist-del" onClick={() => onDelete(s)}>Delete</button>
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}