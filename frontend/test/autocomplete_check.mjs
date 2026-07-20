// Runtime regression test for src/lib/autocomplete.ts -> buildSqlSuggestions +
// quoteIdent (the pure core of the Monaco SQL autocompleter). autocomplete.ts
// is TypeScript with only a type-only import from ./types, so we bundle it to a
// temp ESM file with esbuild (a transitive devDep via vite) and dynamic-import it.
// Run from the frontend dir: `node test/autocomplete_check.mjs`.

import { build } from "esbuild";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const dir = mkdtempSync(join(tmpdir(), "ryudb-ac-"));
const out = join(dir, "autocomplete.mjs");
await build({
  entryPoints: ["src/lib/autocomplete.ts"],
  bundle: true,
  format: "esm",
  platform: "node",
  outfile: out,
  logLevel: "silent",
});
const { buildSqlSuggestions, quoteIdent } = await import(out);

// A small two-table schema: lineitem + a table whose name + a column need quoting.
const schema = new Map([
  ["lineitem", [
    { name: "l_orderkey", type: "int64", nullable: false },
    { name: "l_quantity", type: "float64", nullable: true },
    { name: "l_returnflag", type: "string", nullable: false },
  ]],
  ["order hdr", [ // table name has a space -> must be quoted on insert
    { name: "orderkey", type: "int64", nullable: false },
    { name: "status", type: "string", nullable: false },
  ]],
]);

let fail = 0;
function check(name, cond, extra = "") {
  if (cond) console.log("  ok: " + name);
  else { console.error("  FAIL: " + name + " " + JSON.stringify(extra)); fail++; }
}
const labels = (list) => list.map((s) => s.label).sort();
const byLabel = (list, label) => list.find((s) => s.label === label);

// Unqualified context: keywords + tables + every (deduped) column.
const unq = buildSqlSuggestions("SELECT  ", schema);
check("unqualified has SELECT keyword", unq.some((s) => s.label === "SELECT" && s.kind === "keyword"));
check("unqualified has lineitem table", unq.some((s) => s.label === "lineitem" && s.kind === "table"));
check("unqualified table detail is column count", byLabel(unq, "lineitem")?.detail === "3 columns");
check("unqualified has l_orderkey column", unq.some((s) => s.label === "l_orderkey" && s.kind === "column"));
check("unqualified columns deduped across tables", unq.filter((s) => s.label === "orderkey").length === 1);
check("unqualified no keyword when schema empty-ish ranking", byLabel(unq, "lineitem")?.sortText < byLabel(unq, "SELECT")?.sortText);

// Qualified context lineitem. -> ONLY lineitem's columns, no keywords/tables.
const q = buildSqlSuggestions("SELECT lineitem.", schema);
check("qualified returns only columns", q.every((s) => s.kind === "column"));
check("qualified has l_orderkey", q.some((s) => s.label === "l_orderkey"));
check("qualified excludes other-table column", !q.some((s) => s.label === "status"));
check("qualified excludes keywords", !q.some((s) => s.kind === "keyword"));
check("qualified nullable detail flagged", byLabel(q, "l_quantity")?.detail?.includes("nullable"));
check("qualified non-null detail not flagged", !byLabel(q, "l_orderkey")?.detail?.includes("nullable"));

// Qualified against an unknown table -> no suggestions (no throw).
const qUnknown = buildSqlSuggestions("SELECT nope.", schema);
check("qualified unknown table -> empty", qUnknown.length === 0, qUnknown);

// Whitespace tolerated between the dot and cursor.
const qSpace = buildSqlSuggestions("SELECT lineitem. ", schema);
check("qualified with trailing space still column-only", qSpace.every((s) => s.kind === "column"));

// No schema at all: only keywords.
const noSch = buildSqlSuggestions("SELECT ", undefined);
check("no schema -> keywords only", noSch.every((s) => s.kind === "keyword") && noSch.length > 0);

// quoteIdent: plain word stays bare, special chars get double-quoted, embedded
// quotes are doubled.
check("quoteIdent plain", quoteIdent("l_orderkey") === "l_orderkey");
check("quoteIdent space", quoteIdent("order hdr") === '"order hdr"');
check("quoteIdent doubles embedded quote", quoteIdent('a"b') === '"a""b"');

if (fail) { console.error(`autocomplete_check: ${fail} FAIL`); process.exit(1); }
console.log("AUTOCOMPLETE OK");