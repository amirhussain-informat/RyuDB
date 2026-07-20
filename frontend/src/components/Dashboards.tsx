import { useState } from "react";
import type { Dashboard } from "../lib/types";

interface Props {
  dashboards: Dashboard[];
  onOpen: (id: string) => void;
  onCreate: () => void;
  onRename: (id: string, name: string) => void;
  onRemove: (id: string) => void;
}

/** Sidebar dashboard browser. Lists the saved dashboards (each with its widget
 *  count); click to open one in the dashboard modal, double-click a name to
 *  rename inline, ✕ to delete. "+ New" creates an empty dashboard. Re-mounts
 *  on sidebar switch (conditionally rendered by App). */
export default function Dashboards({ dashboards, onOpen, onCreate, onRename, onRemove }: Props) {
  const [editing, setEditing] = useState<string | null>(null);
  const [draft, setDraft] = useState("");

  const beginRename = (d: Dashboard) => {
    setEditing(d.id);
    setDraft(d.name);
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
        <span>Dashboards</span>
        <span className="head-actions">
          <button onClick={onCreate} title="New dashboard">+ New</button>
        </span>
      </div>
      {dashboards.length === 0 && (
        <div className="empty">No dashboards. Run a query, open the Chart tab, and “Pin to dashboard”.</div>
      )}
      <ul className="dashboard-list">
        {dashboards.map((d) => (
          <li key={d.id}>
            <div className="dashboard-row">
              {editing === d.id ? (
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
                  title={d.name}
                  onClick={() => onOpen(d.id)}
                  onDoubleClick={() => beginRename(d)}
                >
                  {d.name}
                </span>
              )}
              <span className="row-count" title="widgets">{d.widgets.length}</span>
              <button
                className="sample"
                onClick={() => onOpen(d.id)}
                title="Open dashboard"
              >▸</button>
              <button
                className="sample drop"
                onClick={() => { if (window.confirm(`Delete dashboard "${d.name}"?`)) onRemove(d.id); }}
                title="Delete dashboard"
              >✕</button>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}