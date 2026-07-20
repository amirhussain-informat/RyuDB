import { useState } from "react";
import type { Worksheet } from "../hooks/useWorksheets";

interface Props {
  worksheets: Worksheet[];
  activeId: string;
  onSelect: (id: string) => void;
  onCreate: () => void;
  onClose: (id: string) => void;
  onRename: (id: string, name: string) => void;
}

/** Tab bar for the worksheet list. Click to switch; double-click a name to
 * rename (Enter/blur commits, Escape cancels); × closes; + creates a new
 * worksheet after the active one. */
export default function WorksheetTabs({
  worksheets, activeId, onSelect, onCreate, onClose, onRename,
}: Props) {
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
    </div>
  );
}