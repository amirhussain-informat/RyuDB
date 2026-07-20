import { useState } from "react";
import type { Worksheet } from "../hooks/useWorksheets";

interface Props {
  worksheets: Worksheet[];
  activeId: string;
  onSelect: (id: string) => void;
  onCreate: () => void;
  onClose: (id: string) => void;
  onRename: (id: string, name: string) => void;
  /** Export the active worksheet as a .sql file (Git-friendly). */
  onExportActive: () => void;
  /** Export all worksheets as one .sql bundle. */
  onExportAll: () => void;
  /** Open the .sql file picker to import worksheets. */
  onImport: () => void;
}

/** Tab bar for the worksheet list. Click to switch; double-click a name to
 * rename (Enter/blur commits, Escape cancels); × closes; + creates a new
 * worksheet after the active one. The trailing ⮚ group exports/imports
 * worksheets as plain .sql (single or bundle) for Git-backed versioning. */
export default function WorksheetTabs({
  worksheets, activeId, onSelect, onCreate, onClose, onRename,
  onExportActive, onExportAll, onImport,
}: Props) {
  const [menuOpen, setMenuOpen] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [draft, setDraft] = useState("");

  const beginEdit = (w: Worksheet) => {
    setEditingId(w.id);
    setDraft(w.name);
  };
  const commit = () => {
    if (editingId) onRename(editingId, draft.trim() || "Untitled");
    setEditingId(null);
  };

  return (
    <div className="worksheet-tabs">
      {worksheets.map((w) => (
        <div
          key={w.id}
          className={"ws-tab" + (w.id === activeId ? " active" : "")}
          onClick={() => onSelect(w.id)}
          onDoubleClick={() => beginEdit(w)}
          title={w.name}
        >
          {editingId === w.id ? (
            <input
              className="ws-tab-input"
              autoFocus
              value={draft}
              size={Math.max(8, draft.length + 1)}
              onChange={(e) => setDraft(e.target.value)}
              onBlur={commit}
              onKeyDown={(e) => {
                if (e.key === "Enter") commit();
                if (e.key === "Escape") setEditingId(null);
              }}
              onClick={(e) => e.stopPropagation()}
            />
          ) : (
            <>
              <span className="ws-tab-name">{w.name}</span>
              <button
                className="ws-tab-close"
                title="Close worksheet"
                onClick={(e) => {
                  e.stopPropagation();
                  onClose(w.id);
                }}
              >
                ×
              </button>
            </>
          )}
        </div>
      ))}
      <button className="ws-tab-add" title="New worksheet" onClick={onCreate}>
        +
      </button>
      <div className="ws-actions">
        <button
          className="ws-actions-btn"
          title="Export / import worksheets (.sql)"
          onClick={(e) => { e.stopPropagation(); setMenuOpen((o) => !o); }}
        >
          ⮚
        </button>
        {menuOpen && (
          <div className="ws-actions-menu" onClick={(e) => e.stopPropagation()}>
            <button onClick={() => { setMenuOpen(false); onExportActive(); }}>
              Export active as .sql
            </button>
            <button onClick={() => { setMenuOpen(false); onExportAll(); }}>
              Export all as .sql bundle
            </button>
            <button onClick={() => { setMenuOpen(false); onImport(); }}>
              Import from .sql…
            </button>
          </div>
        )}
      </div>
    </div>
  );
}