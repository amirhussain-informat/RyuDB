import { useState } from "react";
import type { Notebook } from "../lib/types";

interface Props {
  notebooks: Notebook[];
  onOpen: (id: string) => void;
  onCreate: () => void;
  onRename: (id: string, name: string) => void;
  onRemove: (id: string) => void;
}

/** Sidebar notebook browser. Lists the saved SQL+Python notebooks (each with
 *  its cell count); click to open one in the notebook modal, double-click a
 *  name to rename inline, ✕ to delete. "+ New" creates an empty notebook.
 *  Re-mounts on sidebar switch (conditionally rendered by App). */
export default function Notebooks({ notebooks, onOpen, onCreate, onRename, onRemove }: Props) {
  const [editing, setEditing] = useState<string | null>(null);
  const [draft, setDraft] = useState("");

  const beginRename = (n: Notebook) => {
    setEditing(n.id);
    setDraft(n.name);
  };
  const commitRename = () => {
    if (editing) {
      const name = draft.trim();
      if (name) onRename(editing, name);
    }
    setEditing(null);
  };

  return (
    <div className="dashboards">
      <div className="sidebar-head">
        <span>Notebooks</span>
        <span className="head-actions">
          <button onClick={onCreate} title="New notebook">+ New</button>
        </span>
      </div>
      {notebooks.length === 0 && (
        <div className="empty">No notebooks. “+ New” to create one, then add SQL / Python cells.</div>
      )}
      <ul className="dashboard-list">
        {notebooks.map((n) => (
          <li key={n.id}>
            <div className="dashboard-row">
              {editing === n.id ? (
                <input
                  className="load-input dash-rename"
                  value={draft}
                  autoFocus
                  onChange={(e) => setDraft(e.target.value)}
                  onBlur={commitRename}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") commitRename();
                    else if (e.key === "Escape") setEditing(null);
                  }}
                />
              ) : (
                <span
                  className="dashboard-name"
                  title={n.name}
                  onClick={() => onOpen(n.id)}
                  onDoubleClick={() => beginRename(n)}
                >
                  {n.name}
                </span>
              )}
              <span className="row-count" title="cells">{n.cells.length}</span>
              <button
                className="sample"
                onClick={() => onOpen(n.id)}
                title="Open notebook"
              >▸</button>
              <button
                className="sample drop"
                onClick={() => { if (window.confirm(`Delete notebook "${n.name}"?`)) onRemove(n.id); }}
                title="Delete notebook"
              >✕</button>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}