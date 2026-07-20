// Runtime regression test for src/lib/diff.ts -> diffLines / diffStats /
// toSideBySide / splitLines (the line-level LCS diff for the SQL compare view).
// The module is TypeScript with no imports, so we bundle it with esbuild and
// dynamic-import it. Run from the frontend dir: `node test/diff_check.mjs`.

import { build } from "esbuild";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const dir = mkdtempSync(join(tmpdir(), "ryudb-diff-"));
const out = join(dir, "diff.mjs");
await build({
  entryPoints: ["src/lib/diff.ts"],
  bundle: true,
  format: "esm",
  platform: "node",
  outfile: out,
  logLevel: "silent",
});
const { diffLines, diffStats, toSideBySide, splitLines } = await import(out);

let fail = 0;
function check(name, cond, extra = "") {
  if (cond) { console.log(`  ok: ${name}`); } else { console.error(`  FAIL: ${name} ${extra}`); fail++; }
}

// --- splitLines ---
check("split empty", splitLines("").length === 0);
check("split one line no newline", JSON.stringify(splitLines("a")) === JSON.stringify(["a"]));
check("split two lines", JSON.stringify(splitLines("a\nb")) === JSON.stringify(["a", "b"]));
check("split trailing newline dropped", JSON.stringify(splitLines("a\n")) === JSON.stringify(["a"]));
check("split blank middle kept", JSON.stringify(splitLines("a\n\nb")) === JSON.stringify(["a", "", "b"]));

// --- diffLines: identical -> all eq ---
const same = diffLines("SELECT 1\nFROM t", "SELECT 1\nFROM t");
check("identical all eq", same.every((d) => d.op === "eq") && same.length === 2);
check("identical line numbers", same[0].aLine === 1 && same[0].bLine === 1 && same[1].aLine === 2 && same[1].bLine === 2);

// --- pure addition ---
const add = diffLines("SELECT 1", "SELECT 1\nFROM t");
check("add: first eq", add[0].op === "eq" && add[0].text === "SELECT 1");
check("add: second add", add[1].op === "add" && add[1].text === "FROM t" && add[1].bLine === 2 && add[1].aLine === null);

// --- pure deletion ---
const del = diffLines("SELECT 1\nFROM t", "SELECT 1");
check("del: first eq", del[0].op === "eq");
check("del: second del", del[1].op === "del" && del[1].text === "FROM t" && del[1].aLine === 2 && del[1].bLine === null);

// --- modification (a line changed) ---
const mod = diffLines("SELECT 1\nFROM a", "SELECT 1\nFROM b");
check("mod: eq + del + add", mod.length === 3 && mod[0].op === "eq" && mod[1].op === "del" && mod[2].op === "add");
check("mod: del is FROM a", mod[1].text === "FROM a");
check("mod: add is FROM b", mod[2].text === "FROM b");

// --- reorder: LCS length is 1 (a/b/c each appear once but in different order;
//     the tie-break decides WHICH one is kept, so only assert the count) ---
const reorder = diffLines("a\nb\nc", "c\nb\na");
const reorderEq = reorder.filter((d) => d.op === "eq").length;
check("reorder LCS length 1", reorderEq === 1, `got ${reorderEq}`);
check("reorder eq line is one of a/b/c", reorder.some((d) => d.op === "eq" && ["a", "b", "c"].includes(d.text)));

// --- stats ---
const st = diffStats(diffLines("a\nb\nc", "a\nx\nc\nd"));
check("stats added", st.added === 2, JSON.stringify(st));   // x, d
check("stats removed", st.removed === 1, JSON.stringify(st)); // b
check("stats equal", st.equal === 2, JSON.stringify(st));   // a, c

// --- toSideBySide alignment ---
const rows = toSideBySide(diffLines("SELECT 1\nFROM a\nWHERE x", "SELECT 1\nFROM b\nWHERE x"));
// eq SELECT 1 ; del FROM a / add FROM b ; eq WHERE x
check("side: row0 eq both", rows[0].aOp === "eq" && rows[0].bOp === "eq" && rows[0].aText === "SELECT 1");
check("side: row1 del left blank right", rows[1].aOp === "del" && rows[1].aText === "FROM a" && rows[1].bText === null);
check("side: row2 add blank left right", rows[2].aText === null && rows[2].bOp === "add" && rows[2].bText === "FROM b");
check("side: row3 eq both WHERE x", rows[3].aOp === "eq" && rows[3].bText === "WHERE x");

// --- empty vs empty ---
check("empty vs empty", diffLines("", "").length === 0);

// --- empty vs non-empty -> all add ---
const fromEmpty = diffLines("", "a\nb");
check("empty->nonempty all add", fromEmpty.length === 2 && fromEmpty.every((d) => d.op === "add"));

if (fail) { console.error(`diff_check: ${fail} FAIL`); process.exit(1); }
console.log("DIFF OK");