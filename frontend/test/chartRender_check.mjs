// Runtime regression test for src/lib/chartRender.ts -> barLayout / pointLayout
// / resolveByName / toNum / labelOf (the pure chart geometry shared by the
// interactive Chart tab and the headless dashboard ChartView). The module is
// TypeScript with no imports, so we bundle it with esbuild and dynamic-import
// it. Run from the frontend dir: `node test/chartRender_check.mjs`.

import { build } from "esbuild";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const dir = mkdtempSync(join(tmpdir(), "ryudb-chart-"));
const out = join(dir, "chartRender.mjs");
await build({
  entryPoints: ["src/lib/chartRender.ts"],
  bundle: true,
  format: "esm",
  platform: "node",
  outfile: out,
  logLevel: "silent",
});
const {
  barLayout, pointLayout, resolveByName, toNum, labelOf,
  CHART_W, CHART_H, PAD, MAX_BARS, INNER_W, INNER_H,
} = await import(out);

let fail = 0;
function check(name, cond, extra = "") {
  if (cond) { console.log(`  ok: ${name}`); } else { console.error(`  FAIL: ${name} ${extra}`); fail++; }
}
const near = (a, b, eps = 1e-6) => Math.abs(a - b) <= eps;

// --- toNum ---
check("toNum null", toNum(null) === null);
check("toNum number", toNum(3.5) === 3.5);
check("toNum bigint", toNum(42n) === 42);
check("toNum NaN -> null", toNum(NaN) === null);
check("toNum string -> null", toNum("x") === null);

// --- labelOf ---
check("labelOf null", labelOf(null) === "NULL");
check("labelOf undefined", labelOf(undefined) === "NULL");
check("labelOf bigint", labelOf(7n) === "7");
check("labelOf date", labelOf(new Date("2024-01-02T00:00:00.000Z")) === "2024-01-02T00:00:00.000Z");
check("labelOf string", labelOf("N") === "N");

// --- resolveByName ---
const cols = [{ name: "a" }, { name: "b" }, { name: "c" }];
check("resolve found", resolveByName(cols, "b") === 1);
check("resolve absent -> -1", resolveByName(cols, "z") === -1);
check("resolve case-sensitive", resolveByName(cols, "A") === -1);

// --- barLayout ---
const bars = barLayout([10, -10, null, 20], ["p", "q", "r", "s"], 4);
check("bar count", bars.length === 4);
const max = 20, min = -10, span = 30;
const zero = PAD.t + INNER_H * (max / span);
const bw = INNER_W / 4;
check("bar bw", near(bars[0].bw, bw));
check("bar zero line", near(bars[0].zero, zero), `${bars[0].zero} vs ${zero}`);
check("bar positive top", near(bars[0].top, PAD.t + INNER_H * ((max - 10) / span)));
check("bar negative top above zero", bars[1].top > bars[1].zero, `${bars[1].top} > ${bars[1].zero}`);
check("bar null top == zero", near(bars[2].top, zero));
check("bar labels", bars.map((b) => b.label).join("|") === "p|q|r|s");

// bar cap at MAX_BARS
const bigBars = barLayout(new Array(100).fill(1), new Array(100).fill("x"), 100);
check("bar capped at MAX_BARS", bigBars.length === MAX_BARS);
// n larger than values array -> uses values length
const shortBars = barLayout([1, 2, 3], ["a", "b", "c"], 100);
check("bar n > len -> len", shortBars.length === 3);
// empty / n=0
check("bar n=0 -> []", barLayout([1], ["a"], 0).length === 0);
// flat series span=1 (all equal) -> top at PAD.t (max-0)/1 = max=0 -> zero
const flat = barLayout([5, 5, 5], ["a", "b", "c"], 3);
check("bar flat span=1 no NaN", Number.isFinite(flat[0].top) && Number.isFinite(flat[0].zero));

// --- pointLayout ---
const pts = pointLayout([0, 1, 2, 3], [10, 20, 5, 30], 4);
check("point count", pts.length === 4);
check("point px in [l, W-r]", pts.every((p) => p.px >= PAD.l - 1e-6 && p.px <= CHART_W - PAD.r + 1e-6));
check("point py in [t, H-b]", pts.every((p) => p.py >= PAD.t - 1e-6 && p.py <= CHART_H - PAD.b + 1e-6));
check("point min x at left edge", near(pts[0].px, PAD.l));
check("point max x at right edge", near(pts[3].px, CHART_W - PAD.r));
// nulls skipped
const ptsNull = pointLayout([0, null, 2], [10, 20, 30], 3);
check("point null skipped", ptsNull.length === 2);
// empty
check("point empty -> []", pointLayout([], [], 0).length === 0);
check("point all-null -> []", pointLayout([null], [null], 1).length === 0);
// single point span=1 -> lands at left/bottom edges (xmin..xmin, sx=1)
const one = pointLayout([5], [7], 1);
check("point single count", one.length === 1);
check("point single at min edge", near(one[0].px, PAD.l) && near(one[0].py, PAD.t + INNER_H));

if (fail) { console.error(`chartRender_check: ${fail} FAIL`); process.exit(1); }
console.log("CHARTRENDER OK");