// Runtime regression test for src/lib/planLayout.ts -> layoutPlan +
// opCategory + edgePath + detailOneLine (the pure geometry core of the
// query-profile graph). planLayout.ts is TypeScript with only a type-only
// import from ./types, so we bundle it to a temp ESM file with esbuild and
// dynamic-import it. Run from the frontend dir: `node test/planLayout_check.mjs`.

import { build } from "esbuild";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const dir = mkdtempSync(join(tmpdir(), "ryudb-pl-"));
const out = join(dir, "planLayout.mjs");
await build({
  entryPoints: ["src/lib/planLayout.ts"],
  bundle: true,
  format: "esm",
  platform: "node",
  outfile: out,
  logLevel: "silent",
});
const {
  layoutPlan, opCategory, edgePath, detailOneLine,
  BOX_W, BOX_H, H_GAP, V_GAP,
} = await import(out);

let fail = 0;
function check(name, cond, extra = "") {
  if (cond) {
    console.log(`  ok: ${name}`);
  } else {
    console.error(`  FAIL: ${name} ${extra}`);
    fail++;
  }
}
function approx(a, b, eps = 1e-6) { return Math.abs(a - b) < eps; }

// --- opCategory: every node type maps to a known category, writes share one ---
check("Scan -> scan", opCategory("Scan") === "scan");
check("Join -> join", opCategory("Join") === "join");
check("Aggregate -> agg", opCategory("Aggregate") === "agg");
check("Filter -> filter", opCategory("Filter") === "filter");
check("Sort/Limit/Distinct -> sort", ["Sort", "Limit", "Distinct"].every((o) => opCategory(o) === "sort"));
check("Project/Derived/SetOp/Window -> proj", ["Project", "Derived", "SetOp", "Window"].every((o) => opCategory(o) === "proj"));
check("Insert/Update/Delete/Merge/TxnControl -> write", ["Insert", "Update", "Delete", "Merge", "TxnControl"].every((o) => opCategory(o) === "write"));
check("unknown op -> other", opCategory("SomethingNew") === "other");

// --- single leaf: one row, one column, width/height = one box ---
const leaf = { op: "Scan", est_rows: 100, fused: false, detail: { table: "t" }, children: [] };
const l1 = layoutPlan(leaf);
check("leaf x=0", l1.root.x === 0, String(l1.root.x));
check("leaf y=0", l1.root.y === 0, String(l1.root.y));
check("leaf width = BOX_W", l1.width === BOX_W, `${l1.width} vs ${BOX_W}`);
check("leaf height = BOX_H", l1.height === BOX_H, `${l1.height} vs ${BOX_H}`);

// --- chain of 3 (Project -> Filter -> Scan): each at its own depth column,
//     all vertically centered (single-leaf-per-node chain) ---
const chain = {
  op: "Project", est_rows: null, fused: false, detail: { items: ["a"] },
  children: [{
    op: "Filter", est_rows: null, fused: false, detail: {},
    children: [leaf],
  }],
};
const l2 = layoutPlan(chain);
check("chain root x=0", l2.root.x === 0);
check("chain depth1 x = BOX_W+H_GAP", l2.root.children[0].x === BOX_W + H_GAP, String(l2.root.children[0].x));
check("chain depth2 x = 2*(BOX_W+H_GAP)", l2.root.children[0].children[0].x === 2 * (BOX_W + H_GAP));
// single-leaf chain: all three share y=0 (the only leaf), parents center over it
check("chain all y=0", [l2.root, l2.root.children[0], l2.root.children[0].children[0]].every((n) => n.y === 0));
check("chain width = 3*BOX_W + 2*H_GAP", l2.width === 3 * BOX_W + 2 * H_GAP, `${l2.width}`);
check("chain height = BOX_H", l2.height === BOX_H, String(l2.height));

// --- bushy: a Join with 3 leaf children -> children on distinct rows, parent
//     centered over the first..last child span ---
const bushy = {
  op: "Join", est_rows: null, fused: false, detail: { how: "inner" },
  children: [
    { op: "Scan", est_rows: 10, fused: false, detail: { table: "a" }, children: [] },
    { op: "Scan", est_rows: 20, fused: false, detail: { table: "b" }, children: [] },
    { op: "Scan", est_rows: 30, fused: false, detail: { table: "c" }, children: [] },
  ],
};
const l3 = layoutPlan(bushy);
const [a, b, c] = l3.root.children;
check("bushy children x = BOX_W+H_GAP", [a, b, c].every((n) => n.x === BOX_W + H_GAP));
check("bushy leaf rows distinct + sequential", a.y === 0 && b.y === BOX_H + V_GAP && c.y === 2 * (BOX_H + V_GAP),
  `${a.y},${b.y},${c.y}`);
// parent centered over first..last child
const midY = (a.y + c.y) / 2;
check("bushy parent centered over child span", approx(l3.root.y + BOX_H / 2, midY + BOX_H / 2) || approx(l3.root.y, midY),
  `root.y=${l3.root.y} midY=${midY}`);
check("bushy parent y == (first+last)/2", approx(l3.root.y, (a.y + c.y) / 2), `root.y=${l3.root.y}`);
check("bushy height = 3*BOX_H + 2*V_GAP", l3.height === 3 * BOX_H + 2 * V_GAP, String(l3.height));

// --- edgePath: cubic bezier from parent right edge to child left edge ---
const ep = edgePath(l3.root, a);
check("edgePath starts at parent right edge", ep.startsWith(`M ${l3.root.x + BOX_W} `), ep);
check("edgePath ends at child left edge", ep.includes(` ${a.x} `) && /C /.test(ep), ep);

// --- detailOneLine: key=value joined, arrays comma-joined, truncated ---
check("detailOneLine simple", detailOneLine({ table: "t", alias: "t" }) === "table=t  alias=t");
check("detailOneLine array joined", detailOneLine({ on_left: ["a", "b"] }) === "on_left=a,b");
check("detailOneLine truncates", detailOneLine({ table: "averylongtablename" }, 10).endsWith("…"));
check("detailOneLine empty -> empty string", detailOneLine({}) === "");

if (fail) { console.error(`planLayout_check: ${fail} FAIL`); process.exit(1); }
console.log("PLANLAYOUT OK");