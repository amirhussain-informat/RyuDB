import type { Bar, Point, ChartKind } from "../lib/chartRender";
import { CHART_W, CHART_H, PAD } from "../lib/chartRender";

/** The SVG painting for a chart, shared by the interactive `Chart` tab and the
 *  headless dashboard `ChartView`. Pure presentational — consumes the geometry
 *  from `lib/chartRender.ts`. No state, no controls. */
export default function ChartSvg({
  bars, points, kind,
}: {
  bars: Bar[];
  points: Point[];
  kind: ChartKind;
}) {
  if (kind === "bar") {
    const zero = bars[0]?.zero ?? PAD.t;
    return (
      <svg className="chart-svg" viewBox={`0 0 ${CHART_W} ${CHART_H}`} preserveAspectRatio="xMidYMid meet">
        <line x1={PAD.l} y1={zero} x2={CHART_W - PAD.r} y2={zero} stroke="var(--border)" />
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
  const path = points.map((p, i) => (i === 0 ? "M" : "L") + p.px.toFixed(1) + " " + p.py.toFixed(1)).join(" ");
  return (
    <svg className="chart-svg" viewBox={`0 0 ${CHART_W} ${CHART_H}`} preserveAspectRatio="xMidYMid meet">
      <rect x={PAD.l} y={PAD.t} width={CHART_W - PAD.l - PAD.r} height={CHART_H - PAD.t - PAD.b}
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