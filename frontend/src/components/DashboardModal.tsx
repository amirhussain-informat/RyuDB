import { useEffect, useState } from "react";
import type { Dashboard, DashboardWidget, Result, ResultMeta } from "../lib/types";
import ChartView from "./ChartView";

interface Props {
  open: boolean;
  dashboard: Dashboard | null;
  connected: boolean;
  /** Run a widget's SQL and return the result (meta + decoded Arrow table), or
   *  null on a non-result / error response. The caller wraps the server `op`. */
  fetchWidget: (sql: string) => Promise<Result | null>;
  onClose: () => void;
  onRemoveWidget: (widgetId: string) => void;
}

/** A full-overlay dashboard view: a responsive grid of chart widgets. Each
 *  widget re-runs its saved SQL on open and on Refresh, then paints its saved
 *  chart spec against the result via the headless `ChartView` (no controls).
 *  A per-widget ✕ drops it. This is the Snowsight "saved visualization" parity
 *  — a named, persisted board of charts that refreshes against the live GPU
 *  engine. Client-side only (localStorage); no server op beyond the widget
 *  SELECTs. */
export default function DashboardModal({ open, dashboard, connected, fetchWidget, onClose, onRemoveWidget }: Props) {
  const [refreshNonce, setRefreshNonce] = useState(0);

  // Reset the refresh nonce whenever a different dashboard is opened so the
  // cards' effect re-runs cleanly.
  useEffect(() => {
    if (open) setRefreshNonce((n) => n + 1);
  }, [open, dashboard?.id]);

  if (!open || !dashboard) return null;

  return (
    <div className="palette-overlay dash-overlay" onClick={onClose}>
      <div className="dash-modal" onClick={(e) => e.stopPropagation()}>
        <div className="palette-input shortcuts-head vhist-head dash-head">
          <span className="dash-title" title={dashboard.name}>{dashboard.name}</span>
          <span className="vhist-actions">
            <button
              onClick={() => setRefreshNonce((n) => n + 1)}
              disabled={!connected || dashboard.widgets.length === 0}
              title="Re-run every widget"
            >⟳ Refresh</button>
            <button className="vhist-save" onClick={onClose}>Close</button>
          </span>
        </div>
        <div className="dash-body">
          {!connected && <div className="empty">Connect to ryudb-server to run dashboard widgets.</div>}
          {connected && dashboard.widgets.length === 0 && (
            <div className="empty">
              No widgets. Run a query, open the Chart tab, and “Pin to dashboard”.
            </div>
          )}
          {connected && dashboard.widgets.length > 0 && (
            <div className="dash-grid">
              {dashboard.widgets.map((w) => (
                <WidgetCard
                  key={w.id}
                  widget={w}
                  refreshNonce={refreshNonce}
                  connected={connected}
                  fetchWidget={fetchWidget}
                  onRemove={() => onRemoveWidget(w.id)}
                />
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

type WidgetState =
  | { tag: "loading" }
  | { tag: "error"; message: string }
  | { tag: "ok"; meta: ResultMeta; table: Result["table"] };

function WidgetCard({
  widget, refreshNonce, connected, fetchWidget, onRemove,
}: {
  widget: DashboardWidget;
  refreshNonce: number;
  connected: boolean;
  fetchWidget: (sql: string) => Promise<Result | null>;
  onRemove: () => void;
}) {
  const [state, setState] = useState<WidgetState>({ tag: "loading" });

  useEffect(() => {
    if (!connected) return;
    let cancelled = false;
    setState({ tag: "loading" });
    fetchWidget(widget.sql)
      .then((res) => {
        if (cancelled) return;
        if (!res || res.meta.op !== "result") {
          const msg = res && res.meta.op === "error"
            ? (res.meta as { message?: string }).message ?? "error"
            : "no result";
          setState({ tag: "error", message: msg });
          return;
        }
        setState({ tag: "ok", meta: res.meta as ResultMeta, table: res.table });
      })
      .catch((e: unknown) => {
        if (!cancelled) setState({ tag: "error", message: (e as Error)?.message ?? String(e) });
      });
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [widget.sql, refreshNonce, connected]);

  return (
    <div className="dash-widget">
      <div className="dash-widget-head">
        <span className="dash-widget-title" title={widget.title}>{widget.title}</span>
        <span className="dash-widget-meta">
          {state.tag === "ok" && `${state.meta.row_count.toLocaleString()} rows · ${state.meta.duration_ms.toFixed(0)} ms`}
          {state.tag === "loading" && "…"}
        </span>
        <button className="sample drop" onClick={onRemove} title="Remove widget">✕</button>
      </div>
      <div className="dash-widget-body">
        {state.tag === "loading" && <div className="empty">running…</div>}
        {state.tag === "error" && <pre className="msg error dash-widget-err">{state.message}</pre>}
        {state.tag === "ok" && (
          <ChartView meta={state.meta} table={state.table} spec={widget.chart} />
        )}
      </div>
    </div>
  );
}