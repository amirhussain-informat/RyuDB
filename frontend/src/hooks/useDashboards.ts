import { useEffect, useRef, useState } from "react";
import type { ChartSpec, Dashboard, DashboardWidget } from "../lib/types";

/** A saved dashboard (a named collection of chart widgets), persisted to
 *  localStorage so the user's dashboards survive a reload. Mirrors the
 *  `useWorksheets` shape. Unlike worksheets, an EMPTY list is allowed (no
 *  seed dashboard is created). */
const STORAGE_KEY = "ryudb.dashboards.v1";

function uid(): string {
  try {
    if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
      return crypto.randomUUID();
    }
  } catch { /* secure-context unavailable -> fall back */ }
  return Math.random().toString(36).slice(2) + Date.now().toString(36);
}

function validWidget(w: unknown): w is DashboardWidget {
  return !!w
    && typeof (w as DashboardWidget).id === "string"
    && typeof (w as DashboardWidget).title === "string"
    && typeof (w as DashboardWidget).sql === "string"
    && !!(w as DashboardWidget).chart
    && typeof (w as DashboardWidget).chart.kind === "string"
    && typeof (w as DashboardWidget).chart.xCol === "string"
    && typeof (w as DashboardWidget).chart.yCol === "string";
}

function validDashboard(d: unknown): d is Dashboard {
  return !!d
    && typeof (d as Dashboard).id === "string"
    && typeof (d as Dashboard).name === "string"
    && Array.isArray((d as Dashboard).widgets)
    && (d as Dashboard).widgets.every(validWidget);
}

function load(): Dashboard[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw) as { dashboards?: Dashboard[] };
      if (Array.isArray(parsed.dashboards)) {
        return parsed.dashboards.filter(validDashboard);
      }
    }
  } catch {
    /* corrupt / unavailable storage -> start empty */
  }
  return [];
}

/** Manages the dashboard list, persisting to localStorage. Widgets are added
 *  by the "Pin to dashboard" flow on the Chart tab. */
export function useDashboards() {
  const initial = useRef<Dashboard[] | null>(null);
  if (initial.current === null) initial.current = load();

  const [dashboards, setDashboards] = useState<Dashboard[]>(initial.current);

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({ dashboards }));
    } catch {
      /* quota / private mode -> best-effort, non-fatal */
    }
  }, [dashboards]);

  const create = (name: string): string => {
    const id = uid();
    const trimmed = name.trim() || `Dashboard ${dashboards.length + 1}`;
    setDashboards((prev) => [...prev, { id, name: trimmed, widgets: [] }]);
    return id;
  };

  const rename = (id: string, name: string) =>
    setDashboards((prev) => prev.map((d) => (d.id === id ? { ...d, name } : d)));

  const remove = (id: string) =>
    setDashboards((prev) => prev.filter((d) => d.id !== id));

  /** Add a widget to a dashboard. Creates the dashboard first when `dashboardId`
   *  is null (the "new dashboard" pin path). Returns the widget id. */
  const addWidget = (dashboardId: string | null, title: string, sql: string, chart: ChartSpec): {
    dashboardId: string;
    widgetId: string;
  } => {
    const widget: DashboardWidget = { id: uid(), title: title.trim() || "Untitled", sql, chart };
    let did = dashboardId;
    if (!did) {
      did = uid();
      const name = `Dashboard ${dashboards.length + 1}`;
      setDashboards((prev) => [...prev, { id: did!, name, widgets: [widget] }]);
      return { dashboardId: did, widgetId: widget.id };
    }
    setDashboards((prev) => prev.map((d) => (d.id === did ? { ...d, widgets: [...d.widgets, widget] } : d)));
    return { dashboardId: did, widgetId: widget.id };
  };

  const removeWidget = (dashboardId: string, widgetId: string) =>
    setDashboards((prev) => prev.map((d) =>
      d.id === dashboardId ? { ...d, widgets: d.widgets.filter((w) => w.id !== widgetId) } : d));

  const updateWidget = (dashboardId: string, widgetId: string, patch: Partial<Pick<DashboardWidget, "title" | "sql" | "chart">>) =>
    setDashboards((prev) => prev.map((d) =>
      d.id === dashboardId
        ? { ...d, widgets: d.widgets.map((w) => (w.id === widgetId ? { ...w, ...patch } : w)) }
        : d));

  return { dashboards, create, rename, remove, addWidget, removeWidget, updateWidget };
}