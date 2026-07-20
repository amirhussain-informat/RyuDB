// Runtime regression test for src/lib/csv.ts -> tableToCSV against a real
// apache-arrow Table built from plain arrays (so nulls become validity-bitmap
// entries, not coerced 0s). csv.ts is TypeScript with a type-only arrow import,
// so we bundle it to a temp ESM file with esbuild (a transitive devDep via
// vite) and dynamic-import that. Run from the frontend dir: `node test/csv_check.mjs`.

import { build } from "esbuild";
import { tableFromArrays } from "apache-arrow";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const dir = mkdtempSync(join(tmpdir(), "ryudb-csv-"));
const out = join(dir, "csv.mjs");
await build({
  entryPoints: ["src/lib/csv.ts"],
  bundle: true,
  format: "esm",
  platform: "node",
  outfile: out,
  logLevel: "silent",
});
const { tableToCSV } = await import(out);

const t = tableFromArrays({
  id: [1, 2, 3, null],
  name: ["alice", "b,ob", 'car"ol', null],
  big: [10n, 20n, 30n, 40n],
});
const csv = tableToCSV(t);
const lines = csv.split("\n");
let fail = 0;
function check(name, cond, extra = "") {
  if (cond) console.log("  ok: " + name);
  else { console.error("  FAIL: " + name + " " + JSON.stringify(extra)); fail++; }
}
check("header", lines[0] === "id,name,big", lines[0]);
check("plain row", lines[1] === "1,alice,10", lines[1]);
check("comma quoted", lines[2] === '2,"b,ob",20', lines[2]);
check("quote doubled", lines[3] === '3,"car""ol",30', lines[3]);
check("null -> empty", lines[4] === ",,40", lines[4]);
check("bigint no n-suffix", !csv.includes("10n"), csv);
console.log(fail === 0 ? "CSV OK" : `CSV FAILED: ${fail}`);
process.exit(fail === 0 ? 0 : 1);