import { useEffect, useState } from "react";
import type { ChartSpec, Dashboard } from "../lib/types";

interface Props {
  open: boolean;
  spec: ChartSpec | null;
  sql: string;
  dashboards: Dashboard[];
  onClose: () => void;
  onPin: (dashboardId: string | null, title: string) => void;
}

/** A short title suggestion from a SQL statement: the first non-empty, non-
 *  comment line, trimmed and capped. Falls back to "Chart". */
function suggestTitle(sql: string): string {
  for (const raw of sql.split(/\r?\n/)) {
    const line = raw.trim();
    if (!line || line.startsWith("--")) continue;
    return line.length > 60 ? line.slice(0, 57) + "…" : line;
  }
  return "Chart";
}

/** The "Pin to dashboard" flow: pick a target dashboard (an existing one, or
 *  "New dashboard") and a widget title, then hand the choice back to App which
 *  calls `useDashboards.addWidget`. The chart spec + the SQL that produced the
 *  result are supplied by the caller (App), captured from the Chart tab. */
export default function PinWidgetModal({ open, spec, sql, dashboards, onClose, onPin }: Props) {
  const [title, setTitle] = useState("");
  const [targetId, setTargetId] = useState<string>("");

  useEffect(() => {
    if (open) {
      setTitle(suggestTitle(sql));
      setTargetId(dashboards.length > 0 ? dashboards[0].id : "");
    }
  }, [open, sql, dashboards]);

  if (!open || !spec) return null;

  const submit = () => {
    onPin(targetId === "" ? null : targetId, title);
    onClose();
  };

  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && (e.target as HTMLElement).tagName !== "TEXTAREA") {
      e.preventDefault();
      submit();
    } else if (e.key === "Escape") {
      e.preventDefault();
      onClose();
    }
  };

  return (
    <div className="palette-overlay" onClick={onClose}>
      <div className="load-modal" onClick={(e) => e.stopPropagation()} onKeyDown={onKey}>
        <div className="palette-input shortcuts-head vhist-head">
          <span>Pin chart to dashboard</span>
          <button className="vhist-save" onClick={onClose}>Cancel</button>
        </div>
        <div className="load-body">
          <label className="load-label">Widget title</label>
          <input
            className="load-input"
            type="text"
            value={title}
            autoFocus
            onChange={(e) => setTitle(e.target.value)}
            onKeyDown={onKey}
          />
          <label className="load-label">Dashboard</label>
          <select
            className="load-input"
            value={targetId}
            onChange={(e) => setTargetId(e.target.value)}
          >
            <option value="">— New dashboard —</option>
            {dashboards.map((d) => (
              <option key={d.id} value={d.id}>{d.name}</option>
            ))}
          </select>
          <div className="load-hint">
            {spec.kind} chart · X <code>{spec.xCol || "—"}</code> · Y <code>{spec.yCol || "—"}</code>
          </div>
          <div className="load-actions">
            <button onClick={submit}>Pin</button>
          </div>
        </div>
      </div>
    </div>
  );
}