// Runtime regression test for src/lib/nlSql.ts -> nlToSql + helpers (the
// offline schema-aware NL->SQL recognizer). The module is TypeScript with no
// imports, so we bundle it with esbuild and dynamic-import it. Run from the
// frontend dir: `node test/nlSql_check.mjs`.

import { build } from "esbuild";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const dir = mkdtempSync(join(tmpdir(), "ryudb-nl-"));
const out = join(dir, "nlSql.mjs");
await build({
  entryPoints: ["src/lib/nlSql.ts"],
  bundle: true,
  format: "esm",
  platform: "node",
  outfile: out,
  logLevel: "silent",
});
const {
  nlToSql, normalize, editDistance, findTable, findColumn, quoteIdent, isNumericType,
} = await import(out);

let fail = 0;
function check(name, cond, extra = "") {
  if (cond) { console.log(`  ok: ${name}`); } else { console.error(`  FAIL: ${name} ${extra}`); fail++; }
}

// TPC-H-ish schema: lineitem + orders.
const tables = [
  {
    name: "lineitem",
    columns: [
      { name: "l_orderkey", type: "int64" },
      { name: "l_quantity", type: "float64" },
      { name: "l_extendedprice", type: "float64" },
      { name: "l_returnflag", type: "string" },
    ],
  },
  {
    name: "orders",
    columns: [
      { name: "o_orderkey", type: "int64" },
      { name: "o_totalprice", type: "float64" },
      { name: "o_orderstatus", type: "string" },
    ],
  },
];

// --- helpers ---
check("normalize lower + collapse", normalize("  How Many  Rows? ") === "how many rows");
check("editDistance 0", editDistance("abc", "abc") === 0);
check("editDistance 1", editDistance("abc", "abd") === 1);
check("editDistance insert", editDistance("abc", "abcd") === 1);
check("isNumeric int64", isNumericType("int64") === true);
check("isNumeric float", isNumericType("float64") === true);
check("isNumeric string", isNumericType("string") === false);
check("quoteIdent plain", quoteIdent("lineitem") === "lineitem");
check("quoteIdent weird", quoteIdent("weird name") === '"weird name"');

// --- findTable / findColumn fuzzy ---
check("findTable exact", findTable(tables, "lineitem")?.name === "lineitem");
check("findTable fuzzy typo", findTable(tables, "lineitems")?.name === "lineitem");
check("findTable substring", findTable(tables, "order")?.name === "orders");
check("findTable absent", findTable(tables, "zzz") === null);
check("findColumn exact", findColumn(tables[0], "l_extendedprice")?.name === "l_extendedprice");
check("findColumn tail match", findColumn(tables[0], "extendedprice")?.name === "l_extendedprice");

// --- nlToSql patterns ---
const r1 = nlToSql("how many rows in lineitem", tables);
check("count -> count(*)", r1?.sql === "SELECT count(*) FROM lineitem;", r1?.sql);
check("count template", r1?.template === "row count");

const r2 = nlToSql("top 10 lineitem by extendedprice", tables);
check("top N by col", r2?.sql === "SELECT * FROM lineitem ORDER BY l_extendedprice DESC LIMIT 10;", r2?.sql);
check("top template", r2?.template === "top 10 by l_extendedprice");

const r3 = nlToSql("bottom 5 orders by totalprice", tables);
check("bottom N by col", r3?.sql === "SELECT * FROM orders ORDER BY o_totalprice ASC LIMIT 5;", r3?.sql);

const r4 = nlToSql("average extendedprice in lineitem", tables);
check("avg -> avg(col)", r4?.sql === "SELECT avg(l_extendedprice) FROM lineitem;", r4?.sql);
check("avg template", r4?.template === "avg(l_extendedprice)");

const r5 = nlToSql("sum of l_quantity in lineitem", tables);
check("sum -> sum(col)", r5?.sql === "SELECT sum(l_quantity) FROM lineitem;", r5?.sql);

const r6 = nlToSql("group lineitem by l_returnflag", tables);
check("group by -> count grouped", r6?.sql === "SELECT l_returnflag, count(*) FROM lineitem GROUP BY l_returnflag ORDER BY l_returnflag;", r6?.sql);

const r7 = nlToSql("show me orders", tables);
check("show -> sample", r7?.sql === "SELECT * FROM orders LIMIT 100;", r7?.sql);

// default N fallback when no number
const r8 = nlToSql("top lineitem by l_quantity", tables);
check("top default N=10", r8?.sql === "SELECT * FROM lineitem ORDER BY l_quantity DESC LIMIT 10;", r8?.sql);

// last-resort: a bare table name samples it
const r9 = nlToSql("lineitem", tables);
check("bare table -> sample", r9?.sql === "SELECT * FROM lineitem LIMIT 100;", r9?.sql);

// empty / no schema -> null
check("empty question -> null", nlToSql("", tables) === null);
check("no tables -> null", nlToSql("count lineitem", []) === null);

// completely unrecognized but a table resolves -> last-resort sample (not null)
const r10 = nlToSql("something orders", tables);
check("unrecognized but table resolves -> sample", r10 !== null && r10.table === "orders", JSON.stringify(r10));

if (fail) { console.error(`nlSql_check: ${fail} FAIL`); process.exit(1); }
console.log("NLSQL OK");