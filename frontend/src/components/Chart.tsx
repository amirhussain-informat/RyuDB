import { useEffect, useMemo, useState } from "react";
import { Type } from "apache-arrow";
import type { Table, Vector } from "apache-arrow";
import type { ResultMeta } from "../lib/types";

interface Props {
  meta: ResultMeta;
  table: Table | null;
}

type ChartKind = "bar" | "line" | "scatter";

const W = 820;
const H = 360;
const PAD = { l: 48, r: 16, t: 16, b: 40 };
const MAX_BARS = 60;

/** True for an integer or floating-point Arrow vector (plottable on a numeric
 * axis). Bool/temporal/struct/etc are not treated as numeric here. */
function isNumeric(v: Vector | null): boolean {
  if (!v) return false;
  const t = v.type.typeId;
  return t === Type.Int || t === Type.Float;
}

function toNum(v: unknown): number | null {
  if (v === null || v === undefined) return null;
  if (typeof v === "bigint") {
    const n = Number(v);
    return Number.isFinite(n) ? n : null;
  }
  if (typeof v === "number") return Number.isFinite(v) ? v : null;
  return null;
}

function labelOf(v: unknown): string {
  if (v === null || v === undefined) return "NULL";
  if (typeof v === "bigint") return v.toString();
  if (v instanceof Date) return v.toISOString();
  return String(v);
}

/** A self-contained SVG chart tab over the loaded result rows. v1 is
 * client-side over the displayed page (capped at MAX_BARS for bar) — enough for
 * the common aggregated (GROUP BY) result; charting billions of raw rows is a
 * later GPU-aggregation piece. No charting library — hand-rolled SVG keeps the
 * offline/no-CDN ethos. */
export default function Chart({ meta, table }: Props) {
  const columns = meta.columns;
  const vectors = useMemo(() => {
    if (!table) return [] as (Vector | null)[];
    return columns.map((c) => table.getChild(c.name) ?? null);
  }, [table, columns]);

  const numericIdx = useMemo(
    () => columns.map((_, i) => isNumeric(vectors[i])),
    [columns, vectors],
  );
  const hasNumeric = numericIdx.some(Boolean);

  const [kind, setKind] = useState<ChartKind>("bar");
  const [xIdx, setXIdx] = useState(0);
  const [yIdx, setYIdx] = useState(numericIdx.findIndex(Boolean));

  // When the result changes (new columns), reset the axis picks to sensible
  // defaults: X = first column, Y = first numeric column.
  useEffect(() => {
    setXIdx(0);
    setYIdx(numericIdx.findIndex(Boolean));
  }, [columns, numericIdx]);

  if (!table || meta.returned === 0) {
    return <div className="empty">Run a query to chart its results.</div>;
  }
  if (columns.length < 2) {
    return <div className="empty">Need at least two columns to chart.</div>;
  }

  const n = Math.min(meta.returned, table.numRows);
  const xv = vectors[xIdx];
  const yv = vectors[yIdx];

  const innerW = W - PAD.l - PAD.r;
  const innerH = H - PAD.t - PAD.b;

  // ---- bar: one bar per row, X label from the X column, height from Y ----
  const bars = useMemo(() => {
    if (kind !== "bar" || !yv) return [] as { i: number; label: string; val: number | null; top: number; zero: number; bw: number }[];
    const m = Math.min(n, MAX_BARS);
    const vals: (number | null)[] = [];
    for (let i = 0; i < m; i++) vals.push(toNum(yv.get(i)));
    const nums = vals.filter((v): v is number => v !== null);
    const max = Math.max(0, ...nums);
    const min = Math.min(0, ...nums);
    const span = max - min || 1;
    const zero = PAD.t + innerH * (max / span);
    const bw = innerW / m;
    return vals.map((val, i) => ({
      i,
      label: labelOf(xv ? xv.get(i) : i),
      val,
      top: val === null ? zero : PAD.t + innerH * ((max - val) / span),
      zero,
      bw,
    }));
  }, [kind, yv, xv, n]);

  // ---- line / scatter: numeric X + numeric Y ----
  const points = useMemo(() => {
    if (kind === "bar") return [] as { px: number; py: number; x: number; y: number }[];
    if (!xv || !yv || !isNumeric(xv) || !isNumeric(yv)) return [];
    const pts: { x: number; y: number }[] = [];
    for (let i = 0; i < n; i++) {
      const x = toNum(xv.get(i));
      const y = toNum(yv.get(i));
      if (x !== null && y !== null) pts.push({ x, y });
    }
    if (pts.length === 0) return [];
    const xs = pts.map((p) => p.x);
    const ys = pts.map((p) => p.y);
    const xmin = Math.min(...xs), xmax = Math.max(...xs);
    const ymin = Math.min(...ys), ymax = Math.max(...ys);
    const sx = xmax - xmin || 1;
    const sy = ymax - ymin || 1;
    return pts.map((p) => ({
      x: p.x, y: p.y,
      px: PAD.l + (p.x - xmin) / sx * innerW,
      py: PAD.t + (1 - (p.y - ymin) / sy) * innerH,
    }));
  }, [kind, xv, yv, n]);

  const colOpt = (i: number) => (
    <option key={i} value={i}>
      {columns[i].name} ({columns[i].type})
    </option>
  );

  return (
    <div className="chart">
      <div className="chart-controls">
        <label>type
          <select value={kind} onChange={(e) => setKind(e.target.value as ChartKind)}>
            <option value="bar">bar</option>
            <option value="line">line</option>
            <option value="scatter">scatter</option>
          </select>
        </label>
        <label>{kind === "scatter" ? "X (numeric)" : "X"}
          <select value={xIdx} onChange={(e) => setXIdx(Number(e.target.value))}>
            {columns.map((_, i) => colOpt(i))}
          </select>
        </label>
        <label>{kind === "bar" ? "Y (numeric)" : "Y (numeric)"}
          <select value={yIdx} onChange={(e) => setYIdx(Number(e.target.value))}>
            {columns.map((_, i) => numericIdx[i] ? colOpt(i) : null)}
          </select>
        </label>
        <span className="chart-note">
          {kind === "bar"
            ? `first ${Math.min(n, MAX_BARS)} of ${meta.returned} displayed rows`
            : `${points.length} plottable point${points.length === 1 ? "" : "s"}`}
        </span>
      </div>
      {!hasNumeric && (
        <div className="empty">No numeric columns — bar/line/scatter need a numeric Y.</div>
      )}
      {hasNumeric && kind === "bar" && (
        <BarSvg bars={bars} />
      )}
      {hasNumeric && (kind === "line" || kind === "scatter") && (
        <>
          {points.length === 0
            ? <div className="empty">No plottable rows — pick numeric X and Y columns.</div>
            : <PlotSvg points={points} kind={kind} />}
        </>
      )}
    </div>
  );
}

function BarSvg({ bars }: {
  bars: { i: number; label: string; val: number | null; top: number; zero: number; bw: number }[];
}) {
  return (
    <svg className="chart-svg" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet">
      <line x1={PAD.l} y1={bars[0]?.zero ?? PAD.t} x2={W - PAD.r} y2={bars[0]?.zero ?? PAD.t} stroke="var(--border)" />
      {bars.map((b) => {
        if (b.val === null) return null;
        const top = Math.min(b.top, b.zero);
        const height = Math.abs(b.top - b.zero);
        const x = PAD.l + b.i * b.bw;
        return (
          <g key={b.i}>
            <rect
              x={x + 1} y={top} width={Math.max(1, b.bw - 2)} height={Math.max(1, height)}
              fill="var(--accent)" />
            <title>{`${b.label}: ${b.val}`}</title>
          </g>
        );
      })}
    </svg>
  );
}

function PlotSvg({ points, kind }: { points: { px: number; py: number; x: number; y: number }[]; kind: "line" | "scatter" }) {
  const path = points.map((p, i) => (i === 0 ? "M" : "L") + p.px.toFixed(1) + " " + p.py.toFixed(1)).join(" ");
  return (
    <svg className="chart-svg" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet">
      <rect x={PAD.l} y={PAD.t} width={W - PAD.l - PAD.r} height={H - PAD.t - PAD.b}
        fill="none" stroke="var(--border)" />
      {kind === "line" && <path d={path} fill="none" stroke="var(--accent)" strokeWidth={1.5} />}
      {points.map((p, i) => (
        <circle key={i} cx={p.px} cy={p.py} r={kind === "scatter" ? 2.5 : 1.5} fill="var(--accent)">
          <title>{`(${p.x}, ${p.y})`}</title>
        </circle>
      ))}
    </svg>
  );
}