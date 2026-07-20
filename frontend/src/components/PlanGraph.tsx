import { useCallback, useMemo, useRef, useState } from "react";
import type { PlanNode } from "../lib/types";
import {
  type LaidOutNode,
  detailOneLine,
  edgePath,
  layoutPlan,
  opCategory,
} from "../lib/planLayout";

/** Snowsight-style query-profile graph: a left-to-right box-and-arrow tree of
 *  the optimized plan. Each node is a colored box (op label + ~rows badge +
 *  fused badge + compact detail) connected to its children by bezier edges.
 *  Wheel zooms toward the cursor; pointer-drag pans; buttons reset/zoom. */
export default function PlanGraph({ tree }: { tree: PlanNode }) {
  const layout = useMemo(() => layoutPlan(tree), [tree]);
  const [zoom, setZoom] = useState(1);
  const [tx, setTx] = useState(0);
  const [ty, setTy] = useState(0);
  const dragRef = useRef<{ x: number; y: number; tx: number; ty: number } | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  const clampZoom = (z: number) => Math.min(3, Math.max(0.25, z));

  const onWheel = useCallback((e: React.WheelEvent) => {
    // Zoom toward the pointer: keep the point under the cursor stationary.
    e.preventDefault();
    setZoom((z) => {
      const nz = clampZoom(z * (e.deltaY < 0 ? 1.1 : 1 / 1.1));
      const rect = containerRef.current?.getBoundingClientRect();
      if (!rect) return nz;
      const px = e.clientX - rect.left;
      const py = e.clientY - rect.top;
      setTx((cur) => px - (px - cur) * (nz / z));
      setTy((cur) => py - (py - cur) * (nz / z));
      return nz;
    });
  }, []);

  const onPointerDown = useCallback((e: React.PointerEvent) => {
    if (e.button !== 0) return;
    (e.target as Element).setPointerCapture?.(e.pointerId);
    dragRef.current = { x: e.clientX, y: e.clientY, tx, ty };
  }, [tx, ty]);

  const onPointerMove = useCallback((e: React.PointerEvent) => {
    const d = dragRef.current;
    if (!d) return;
    setTx(d.tx + (e.clientX - d.x));
    setTy(d.ty + (e.clientY - d.y));
  }, []);

  const onPointerUp = useCallback(() => { dragRef.current = null; }, []);

  const reset = useCallback(() => { setZoom(1); setTx(0); setTy(0); }, []);
  const zoomBy = useCallback((f: number) => setZoom((z) => clampZoom(z * f)), []);

  const { width, height } = layout;

  return (
    <div className="plan-graph-wrap">
      <div className="plan-graph-tools">
        <button onClick={() => zoomBy(1.2)} title="zoom in">＋</button>
        <button onClick={() => zoomBy(1 / 1.2)} title="zoom out">－</button>
        <button onClick={reset} title="reset view">⟲</button>
        <span className="plan-graph-zoom">{Math.round(zoom * 100)}%</span>
      </div>
      <div
        className="plan-graph"
        ref={containerRef}
        onWheel={onWheel}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerLeave={onPointerUp}
      >
        <svg
          width={width}
          height={height}
          viewBox={`0 0 ${width} ${height}`}
          style={{ transform: `translate(${tx}px, ${ty}px) scale(${zoom})`, transformOrigin: "0 0" }}
        >
          <Edges root={layout.root} />
          <Nodes root={layout.root} />
        </svg>
      </div>
    </div>
  );
}

function Edges({ root }: { root: LaidOutNode }) {
  const paths: string[] = [];
  const walk = (n: LaidOutNode) => {
    for (const c of n.children) {
      paths.push(edgePath(n, c));
      walk(c);
    }
  };
  walk(root);
  return (
    <g className="plan-edges">
      {paths.map((d, i) => (
        <path key={i} d={d} className="plan-edge" fill="none" />
      ))}
    </g>
  );
}

function Nodes({ root }: { root: LaidOutNode }) {
  const all = useMemo(() => {
    const out: LaidOutNode[] = [];
    const walk = (n: LaidOutNode) => { out.push(n); n.children.forEach(walk); };
    walk(root);
    return out;
  }, [root]);

  return (
    <g>
      {all.map((n, i) => <NodeBox key={i} n={n} />)}
    </g>
  );
}

function NodeBox({ n }: { n: LaidOutNode }) {
  const { node, x, y, w, h } = n;
  const cat = opCategory(node.op);
  const detailStr = detailOneLine(node.detail);
  // Full detail for the tooltip (key=value, arrays joined).
  const fullDetail = Object.entries(node.detail)
    .map(([k, v]) => `${k}=${fmtFull(v)}`)
    .join("\n");
  const title = fullDetail ? `${node.op}\n${fullDetail}` : node.op;
  return (
    <g className={`pnode pnode-${cat}`} transform={`translate(${x} ${y})`}>
      <rect width={w} height={h} rx={8} className="pnode-box" />
      <text x={12} y={22} className="pnode-op">{node.op}</text>
      {node.est_rows !== null && (
        <text x={12} y={40} className="pnode-est">~{node.est_rows.toLocaleString()} rows</text>
      )}
      {node.fused && (
        <text x={w - 12} y={22} textAnchor="end" className="pnode-fused">fused</text>
      )}
      {detailStr && (
        <text x={12} y={56} className="pnode-detail">{detailStr}</text>
      )}
      <title>{title}</title>
    </g>
  );
}

function fmtFull(v: unknown): string {
  if (Array.isArray(v)) return (v as unknown[]).join(",");
  if (v && typeof v === "object") return JSON.stringify(v);
  return String(v);
}