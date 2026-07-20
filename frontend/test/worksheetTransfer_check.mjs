// Runtime regression test for src/lib/worksheetTransfer.ts -> serializeBundle /
// serializeOne / parseBundle / parseImportFile / sanitizeFileName / fileStem
// (the pure Git-backed worksheet export/import format). The module is
// TypeScript with no imports, so we bundle it with esbuild and dynamic-import
// it. Run from the frontend dir: `node test/worksheetTransfer_check.mjs`.

import { build } from "esbuild";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const dir = mkdtempSync(join(tmpdir(), "ryudb-wt-"));
const out = join(dir, "worksheetTransfer.mjs");
await build({
  entryPoints: ["src/lib/worksheetTransfer.ts"],
  bundle: true,
  format: "esm",
  platform: "node",
  outfile: out,
  logLevel: "silent",
});
const {
  serializeBundle, serializeOne, parseBundle, parseImportFile,
  sanitizeFileName, worksheetFileName, fileStem,
} = await import(out);

let fail = 0;
function check(name, cond, extra = "") {
  if (cond) { console.log(`  ok: ${name}`); } else { console.error(`  FAIL: ${name} ${extra}`); fail++; }
}

// --- sanitizeFileName / worksheetFileName / fileStem ---
check("sanitize plain", sanitizeFileName("Q1 Revenue") === "Q1 Revenue");
check("sanitize strips path sep", sanitizeFileName("a/b\\c:d") === "a_b_c_d");
check("sanitize empty -> worksheet", sanitizeFileName("") === "worksheet");
check("worksheetFileName appends .sql", worksheetFileName("Q1") === "Q1.sql");
check("fileStem basename no ext", fileStem("lineitem q1.sql") === "lineitem q1");
check("fileStem with path", fileStem("/home/u/x/Q2.SQL".replace(/\.SQL$/, ".sql")) === "Q2");
check("fileStem no ext", fileStem("README") === "README");

// --- serializeOne: raw SQL + trailing newline ---
check("serializeOne empty", serializeOne("") === "");
check("serializeOne adds newline", serializeOne("SELECT 1") === "SELECT 1\n");
check("serializeOne keeps existing newline", serializeOne("SELECT 1\n") === "SELECT 1\n");

// --- bundle round-trip: names + SQL preserved (modulo trailing newline) ---
const docs = [
  { name: "Q1 Revenue", sql: "SELECT l_returnflag, count(*)\nFROM lineitem\nGROUP BY l_returnflag;" },
  { name: "Empty", sql: "" },
  { name: "Multi line", sql: "SELECT *\nFROM orders\nWHERE o_totalprice > 1000;\n" },
];
const bundle = serializeBundle(docs);
check("bundle has header", bundle.startsWith("-- ryudb worksheet bundle"));
check("bundle has count line", bundle.includes("-- 3 worksheet(s)"));
check("bundle has @@worksheet headers", bundle.includes("-- @@worksheet: Q1 Revenue") && bundle.includes("-- @@worksheet: Empty"));
const parsed = parseBundle(bundle);
check("bundle round-trip count", parsed.length === 3, JSON.stringify(parsed.map((p) => p.name)));
check("bundle round-trip names", parsed.map((p) => p.name).join("|") === "Q1 Revenue|Empty|Multi line");
check("bundle round-trip sql[0]", parsed[0].sql === docs[0].sql, JSON.stringify(parsed[0].sql));
check("bundle round-trip sql[1] empty", parsed[1].sql === "", JSON.stringify(parsed[1].sql));
check("bundle round-trip sql[2] trims trailing blank", parsed[2].sql === "SELECT *\nFROM orders\nWHERE o_totalprice > 1000;", JSON.stringify(parsed[2].sql));

// --- a `-- @@worksheet:` line is ALWAYS a separator (documented contract):
//     a worksheet whose SQL contains such a line splits on import. The marker
//     is unusual enough that real SQL does not contain it. ---
const tricky = serializeBundle([{ name: "T", sql: "-- @@worksheet: fake\nSELECT 1;" }]);
const tp = parseBundle(tricky);
check("tricky: header-like body line splits", tp.length === 2, JSON.stringify(tp.map((p) => p.name)));
check("tricky: first section T empty", tp[0].name === "T" && tp[0].sql === "", JSON.stringify(tp[0]));
check("tricky: second section fake", tp[1].name === "fake" && tp[1].sql === "SELECT 1;", JSON.stringify(tp[1]));

// --- parseBundle on a plain .sql file (no headers) -> single empty-named doc ---
const plain = parseBundle("SELECT count(*) FROM lineitem;\n");
check("plain file -> one doc", plain.length === 1, String(plain.length));
check("plain file -> empty name (caller names from file)", plain[0].name === "");
check("plain file -> sql preserved", plain[0].sql === "SELECT count(*) FROM lineitem;");

// --- parseImportFile: plain file named from stem ---
const imp = parseImportFile("q3.sql", "SELECT * FROM orders;");
check("import plain -> name from stem", imp.length === 1 && imp[0].name === "q3", JSON.stringify(imp));
check("import plain -> sql", imp[0].sql === "SELECT * FROM orders;");

// --- parseImportFile: a bundle file -> sections (filename ignored) ---
const impBundle = parseImportFile("dump.sql", bundle);
check("import bundle -> 3 sections", impBundle.length === 3, String(impBundle.length));
check("import bundle -> names from headers", impBundle[0].name === "Q1 Revenue");

// --- empty bundle round-trips ---
const empty = serializeBundle([]);
const pe = parseBundle(empty);
check("empty bundle -> zero docs", pe.length === 0, JSON.stringify(pe));

if (fail) { console.error(`worksheetTransfer_check: ${fail} FAIL`); process.exit(1); }
console.log("WORKSHEETTRANSFER OK");