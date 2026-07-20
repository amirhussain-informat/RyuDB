import type { PlanNode } from "../lib/types";

function Node({ node }: { node: PlanNode }) {
  const detail = node.detail as Record<string, unknown>;
  const detailStr = Object.entries(detail)
    .map(([k, v]) => `${k}=${fmt(v)}`)
    .join("  ");
  return (
    <li className="plan-node">
      <div className="plan-head">
        <span className="plan-op">{node.op}</span>
        {node.fused && <span className="badge fused" title="Aggregate-over-Join: eligible for the fused C++ star-join+aggregate kernel">fused</span>}
        {node.est_rows !== null && (
          <span className="badge est">~{node.est_rows} rows</span>
        )}
        {detailStr && <span className="plan-detail">{detailStr}</span>}
      </div>
      {node.children.length > 0 && (
        <ul>{node.children.map((c, i) => <Node key={i} node={c} />)}</ul>
      )}
    </li>
  );
}

function fmt(v: unknown): string {
  if (Array.isArray(v)) return (v as unknown[]).join(",");
  if (v && typeof v === "object") return JSON.stringify(v);
  return String(v);
}

export default function Explain({ tree }: { tree: PlanNode | null }) {
  if (!tree) return <div className="empty">No plan.</div>;
  return (
    <div className="explain">
      <ul className="plan-tree">
        <Node node={tree} />
      </ul>
    </div>
  );
}