import { useState } from "react";
import type { PlanNode } from "../lib/types";
import PlanGraph from "./PlanGraph";

type View = "graph" | "tree";

/** Explain-plan panel: a Snowsight-style profile graph (box-and-arrow tree of
 *  the optimized plan) with a fallback to the classic indented text tree.
 *  The graph is the default; the tree view is kept for compact plans and for
 *  reading the full per-node detail inline. */
export default function Explain({ tree }: { tree: PlanNode | null }) {
  const [view, setView] = useState<View>("graph");
  if (!tree) return <div className="empty">No plan. Run Explain to build the optimized plan.</div>;
  return (
    <div className="explain">
      <div className="explain-view-tabs">
        <button className={view === "graph" ? "active" : ""} onClick={() => setView("graph")}>Graph</button>
        <button className={view === "tree" ? "active" : ""} onClick={() => setView("tree")}>Tree</button>
      </div>
      {view === "graph" ? <PlanGraph tree={tree} /> : <PlanTree tree={tree} />}
    </div>
  );
}

function PlanTree({ tree }: { tree: PlanNode }) {
  return (
    <ul className="plan-tree">
      <TreeNode node={tree} />
    </ul>
  );
}

function TreeNode({ node }: { node: PlanNode }) {
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
        <ul>{node.children.map((c, i) => <TreeNode key={i} node={c} />)}</ul>
      )}
    </li>
  );
}

function fmt(v: unknown): string {
  if (Array.isArray(v)) return (v as unknown[]).join(",");
  if (v && typeof v === "object") return JSON.stringify(v);
  return String(v);
}