// Pure chart geometry + helpers shared by the interactive `Chart` tab and the
// headless `ChartView` used by dashboards. Extracted from the React components
// so the layout math is hermetically unit-testable (esbuild, no DOM) and so a
// saved chart spec can be re-rendered without the picker controls.
//
// Everything here operates on PLAIN arrays (values already pulled from Arrow
// vectors by the caller) — there is no apache-arrow import, so the module is
// dependency-free and testable in node. The SVG painting lives in
// `components/ChartSvg.tsx` (React) and consumes the layouts this produces.

export type ChartKind = "bar" | "line" | "scatter";

export const CHART_W = 820;
export const CHART_H = 360;
export const PAD = { l: 48, r: 16, t: 16, b: 40 };
export const MAX_BARS = 60;

export const INNER_W = CHART_W - PAD.l - PAD.r;
export const INNER_H = CHART_H - PAD.t - PAD.b;

export interface Bar {
  i: number;
  label: string;
  val: number | null;
  top: number;
  zero: number;
  bw: number;
}

export interface Point {
  px: number;
  py: number;
  x: number;
  y: number;
}

/** Coerce an Arrow cell value to a finite number, or null. BigInt values
 *  (Arrow int64) are narrowed via Number(); non-numeric values yield null. */
export function toNum(v: unknown): number | null {
  if (v === null || v === undefined) return null;
  if (typeof v === "bigint") {
    const n = Number(v);
    return Number.isFinite(n) ? n : null;
  }
  if (typeof v === "number") return Number.isFinite(v) ? v : null;
  return null;
}

/** A short human-readable label for a tick / tooltip. NULL -> "NULL"; a Date
 *  (Arrow timestamp/date32 decoded by apache-arrow) -> ISO string; a BigInt ->
 *  its decimal string; else String(v). */
export function labelOf(v: unknown): string {
  if (v === null || v === undefined) return "NULL";
  if (typeof v === "bigint") return v.toString();
  if (v instanceof Date) return v.toISOString();
  return String(v);
}

/** Resolve a column name to its index in `columns`, or -1 when absent. The
 *  match is case-sensitive (SQL column names are). Callers fall back to a
 *  default index when this returns -1 (e.g. a renamed/dropped column on a
 *  saved spec). */
export function resolveByName(columns: { name: string }[], name: string): number {
  for (let i = 0; i < columns.length; i++) {
    if (columns[i].name === name) return i;
  }
  return -1;
}

/** Bar geometry for the first `n` (capped at MAX_BARS) rows. `values` are the
 *  Y-column numbers (null allowed); `labels` are the X-column labels. The zero
 *  line sits at the max value so negative bars grow downward; a null value
 *  draws nothing (top === zero). A flat all-equal series uses span 1. */
export function barLayout(values: (number | null)[], labels: string[], n: number): Bar[] {
  const m = Math.max(0, Math.min(n, MAX_BARS));
  if (m === 0) return [];
  const vals = values.slice(0, m);
  const nums = vals.filter((v): v is number => v !== null);
  const max = Math.max(0, ...nums);
  const min = Math.min(0, ...nums);
  const span = max - min || 1;
  const zero = PAD.t + INNER_H * (max / span);
  const bw = INNER_W / m;
  return vals.map((val, i) => ({
    i,
    label: labels[i] ?? String(i),
    val,
    top: val === null ? zero : PAD.t + INNER_H * ((max - val) / span),
    zero,
    bw,
  }));
}

/** Line/scatter point geometry for numeric X + Y over the first `n` rows. Null
 *  on either axis is skipped. The points are scaled to the inner plot rect; a
 *  single-point / flat series uses span 1 (the point lands at the min edge). */
export function pointLayout(xs: (number | null)[], ys: (number | null)[], n: number): Point[] {
  const pts: { x: number; y: number }[] = [];
  const lim = Math.max(0, Math.min(n, xs.length, ys.length));
  for (let i = 0; i < lim; i++) {
    const x = xs[i];
    const y = ys[i];
    if (x !== null && y !== null) pts.push({ x, y });
  }
  if (pts.length === 0) return [];
  const xsAll = pts.map((p) => p.x);
  const ysAll = pts.map((p) => p.y);
  const xmin = Math.min(...xsAll), xmax = Math.max(...xsAll);
  const ymin = Math.min(...ysAll), ymax = Math.max(...ysAll);
  const sx = xmax - xmin || 1;
  const sy = ymax - ymin || 1;
  return pts.map((p) => ({
    x: p.x,
    y: p.y,
    px: PAD.l + ((p.x - xmin) / sx) * INNER_W,
    py: PAD.t + (1 - (p.y - ymin) / sy) * INNER_H,
  }));
}