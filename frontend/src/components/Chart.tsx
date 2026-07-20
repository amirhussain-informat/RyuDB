import { useEffect, useMemo, useState } from "react";
import { Type } from "apache-arrow";
import type { Table, Vector } from "apache-arrow";
import type { ResultMeta, ChartSpec } from "../lib/types";
import {
  barLayout, pointLayout, toNum, labelOf,
  MAX_BARS, type ChartKind,
} from "../lib/chartRender";
import ChartSvg from "./ChartSvg";

interface Props {
  meta: ResultMeta;
  table: Table | null;
  /** Optional callback fired by the "Pin to dashboard" button with the current
   *  chart settings (kind + the chosen X/Y column NAMES). Absent -> no button. */
  onPin?: (spec: ChartSpec) => void;
}

/** True for an integer or floating-point Arrow vector (plottable on a numeric
 *  axis). Bool/temporal/struct/etc are not treated as numeric here. */
function isNumeric(v: Vector | null): boolean {
  if (!v) return false;
  const t = v.type.typeId;
  return t === Type.Int || t === Type.Float;
}

/** A self-contained SVG chart tab over the loaded result rows. v1 is
 *  client-side over the displayed page (capped at MAX_BARS for bar) — enough for
 *  the common aggregated (GROUP BY) result; charting billions of raw rows is a
 *  later GPU-aggregation piece. No charting library — hand-rolled SVG keeps the
 *  offline/no-CDN ethos. The geometry lives in `lib/chartRender.ts` (shared with
 *  the headless dashboard `ChartView`) and the painting in `ChartSvg.tsx`. */
export default function Chart({ meta, table, onPin }: Props) {
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

  // ---- bar: one bar per row, X label from the X column, height from Y ----
  const bars = useMemo(() => {
    if (kind !== "bar" || !yv) return [] as ReturnType<typeof barLayout>;
    const m = Math.min(n, MAX_BARS);
    const values: (number | null)[] = [];
    const labels: string[] = [];
    for (let i = 0; i < m; i++) {
      values.push(toNum(yv.get(i)));
      labels.push(labelOf(xv ? xv.get(i) : i));
    }
    return barLayout(values, labels, m);
  }, [kind, yv, xv, n]);

  // ---- line / scatter: numeric X + numeric Y ----
  const points = useMemo(() => {
    if (kind === "bar") return [] as ReturnType<typeof pointLayout>;
    if (!xv || !yv || !isNumeric(xv) || !isNumeric(yv)) return [];
    const xs: (number | null)[] = [];
    const ys: (number | null)[] = [];
    for (let i = 0; i < n; i++) {
      xs.push(toNum(xv.get(i)));
      ys.push(toNum(yv.get(i)));
    }
    return pointLayout(xs, ys, n);
  }, [kind, xv, yv, n]);

  const colOpt = (i: number) => (
    <option key={i} value={i}>
      {columns[i].name} ({columns[i].type})
    </option>
  );

  const pin = () => {
    if (!onPin) return;
    onPin({
      kind,
      xCol: columns[xIdx]?.name ?? "",
      yCol: columns[yIdx]?.name ?? "",
    });
  };

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
        {onPin && (
          <button className="chart-pin" title="Save this chart to a dashboard" onClick={pin}>
            Pin to dashboard
          </button>
        )}
      </div>
      {!hasNumeric && (
        <div className="empty">No numeric columns — bar/line/scatter need a numeric Y.</div>
      )}
      {hasNumeric && kind === "bar" && (
        <ChartSvg bars={bars} points={[]} kind="bar" />
      )}
      {hasNumeric && (kind === "line" || kind === "scatter") && (
        <>
          {points.length === 0
            ? <div className="empty">No plottable rows — pick numeric X and Y columns.</div>
            : <ChartSvg bars={[]} points={points} kind={kind} />}
        </>
      )}
    </div>
  );
}