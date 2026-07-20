import { useMemo } from "react";
import { Type } from "apache-arrow";
import type { Table, Vector } from "apache-arrow";
import type { ResultMeta, ChartSpec } from "../lib/types";
import {
  barLayout, pointLayout, toNum, labelOf, resolveByName,
  MAX_BARS,
} from "../lib/chartRender";
import ChartSvg from "./ChartSvg";

interface Props {
  meta: ResultMeta;
  table: Table | null;
  spec: ChartSpec;
}

/** True for an integer or floating-point Arrow vector (plottable on a numeric
 *  axis). Mirrors the check in `Chart.tsx`. */
function isNumeric(v: Vector | null): boolean {
  if (!v) return false;
  const t = v.type.typeId;
  return t === Type.Int || t === Type.Float;
}

/** Headless chart renderer for a dashboard widget: paints the SVG for a SAVED
 *  `ChartSpec` (kind + X/Y column NAMES) against a result table, with no picker
 *  controls. Column names are resolved against the current result; a name that
 *  no longer exists (a renamed/dropped column, or a spec saved against a
 *  different result shape) falls back to a sensible default (X = first column,
 *  Y = first numeric column) rather than crashing. This is the re-run path for
 *  a pinned visualization — the interactive `Chart` tab is where specs are
 *  authored. */
export default function ChartView({ meta, table, spec }: Props) {
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

  // Resolve the saved column names to indices, falling back to defaults. This
  // is the only place a stale spec is healed — a missing column is NOT an error
  // so a dashboard still renders after the underlying query changes shape.
  const xIdx = useMemo(() => {
    const byName = resolveByName(columns, spec.xCol);
    if (byName >= 0) return byName;
    return 0;
  }, [columns, spec.xCol]);
  const yIdx = useMemo(() => {
    const byName = resolveByName(columns, spec.yCol);
    if (byName >= 0) return byName;
    const firstNum = numericIdx.findIndex(Boolean);
    return firstNum >= 0 ? firstNum : 0;
  }, [columns, spec.yCol, numericIdx]);

  if (!table || meta.returned === 0) {
    return <div className="empty">No rows.</div>;
  }
  if (columns.length < 2) {
    return <div className="empty">Need at least two columns.</div>;
  }
  if (!hasNumeric) {
    return <div className="empty">No numeric column to plot.</div>;
  }

  const n = Math.min(meta.returned, table.numRows);
  const xv = vectors[xIdx];
  const yv = vectors[yIdx];
  const kind = spec.kind;

  if (kind === "bar") {
    if (!yv) return <div className="empty">No Y column.</div>;
    const m = Math.min(n, MAX_BARS);
    const values: (number | null)[] = [];
    const labels: string[] = [];
    for (let i = 0; i < m; i++) {
      values.push(toNum(yv.get(i)));
      labels.push(labelOf(xv ? xv.get(i) : i));
    }
    const bars = barLayout(values, labels, m);
    return <ChartSvg bars={bars} points={[]} kind="bar" />;
  }

  if (!xv || !yv || !isNumeric(xv) || !isNumeric(yv)) {
    return <div className="empty">Pick numeric X and Y for a {kind} chart.</div>;
  }
  const xs: (number | null)[] = [];
  const ys: (number | null)[] = [];
  for (let i = 0; i < n; i++) {
    xs.push(toNum(xv.get(i)));
    ys.push(toNum(yv.get(i)));
  }
  const points = pointLayout(xs, ys, n);
  if (points.length === 0) {
    return <div className="empty">No plottable rows.</div>;
  }
  return <ChartSvg bars={[]} points={points} kind={kind} />;
}