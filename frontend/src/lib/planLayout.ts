// Pure, dependency-free layout for the query-profile graph (left-to-right tree).
// Extracted from the SVG component so the geometry is unit-testable hermetically
// (esbuild-bundled like csv_check / autocomplete_check — no React/DOM needed).
//
// The tree grows rightward: the root is the leftmost column, children one column
// to the right. Vertical positions are assigned by an in-order leaf sweep —
// each leaf gets the next row slot, and an internal node is vertically centered
// over its first..last child. This is the standard "tidy tree" leaf-counting
// layout: it never overlaps siblings (every leaf owns a distinct row) and
// centers parents over their children's span.

import type { PlanNode } from "./types";

export const BOX_W = 176;
export const BOX_H = 66;
export const H_GAP = 56;
export const V_GAP = 28;

export interface LaidOutNode {
  node: PlanNode;
  /** Top-left of the node box, in SVG user units. */
  x: number;
  y: number;
  w: number;
  h: number;
  children: LaidOutNode[];
}

export interface Layout {
  root: LaidOutNode;
  width: number;
  height: number;
}

/** Categorize a plan node for coloring. Write ops share a category; the catch
 *  all is "other". Keep in sync with the CSS classes `.pnode-<category>`. */
export function opCategory(op: string): string {
  switch (op) {
    case "Scan": return "scan";
    case "Join": return "join";
    case "Aggregate": return "agg";
    case "Filter": return "filter";
    case "Sort":
    case "Limit":
    case "Distinct": return "sort";
    case "Project":
    case "Derived":
    case "SetOp":
    case "Window": return "proj";
    case "Insert":
    case "Update":
    case "Delete":
    case "Merge":
    case "TxnControl": return "write";
    default: return "other";
  }
}

/** Layout a plan tree left-to-right. `depth` starts at 0 for the root. */
export function layoutPlan(root: PlanNode): Layout {
  let leafCursor = 0;
  let maxDepth = 0;

  function place(node: PlanNode, depth: number): LaidOutNode {
    if (depth > maxDepth) maxDepth = depth;
    const x = depth * (BOX_W + H_GAP);
    const kids = node.children.map((c) => place(c, depth + 1));
    let y: number;
    if (kids.length === 0) {
      y = leafCursor * (BOX_H + V_GAP);
      leafCursor++;
    } else {
      // center over the span of the first..last child
      y = (kids[0].y + kids[kids.length - 1].y) / 2;
    }
    return { node, x, y, w: BOX_W, h: BOX_H, children: kids };
  }

  const laid = place(root, 0);
  const width = (maxDepth + 1) * BOX_W + maxDepth * H_GAP;
  const height = Math.max(leafCursor, 1) * BOX_H + (Math.max(leafCursor, 1) - 1) * V_GAP;
  return { root: laid, width, height };
}

/** The point on the parent box where an edge to a child starts (right edge,
 *  vertically centered). */
export function edgeStart(n: LaidOutNode): [number, number] {
  return [n.x + n.w, n.y + n.h / 2];
}

/** The point on the child box where an edge from its parent ends (left edge,
 *  vertically centered). */
export function edgeEnd(n: LaidOutNode): [number, number] {
  return [n.x, n.y + n.h / 2];
}

/** A cubic-bezier edge path from parent's right edge to child's left edge,
 *  curving horizontally (control points pulled out from each end). */
export function edgePath(parent: LaidOutNode, child: LaidOutNode): string {
  const [x1, y1] = edgeStart(parent);
  const [x2, y2] = edgeEnd(child);
  const dx = Math.max(20, (x2 - x1) / 2);
  return `M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`;
}

/** Flatten the laid-out tree into a list (for rendering / debugging). */
export function flatten(n: LaidOutNode, out: LaidOutNode[] = []): LaidOutNode[] {
  out.push(n);
  for (const c of n.children) flatten(c, out);
  return out;
}

/** Compact one-line detail string for a node's `detail` object (key=value
 *  pairs, arrays joined with commas), truncated to `max` chars. */
export function detailOneLine(detail: Record<string, unknown>, max = 26): string {
  const s = Object.entries(detail)
    .map(([k, v]) => `${k}=${fmt(v)}`)
    .join("  ");
  return s.length > max ? s.slice(0, max - 1) + "…" : s;
}

function fmt(v: unknown): string {
  if (Array.isArray(v)) return (v as unknown[]).join(",");
  if (v && typeof v === "object") return JSON.stringify(v);
  return String(v);
}