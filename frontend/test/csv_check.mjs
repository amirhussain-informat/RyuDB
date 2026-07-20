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
const { tableToCSV, tableToJSON, tableToTSV, viewToTSV } = await import(out);

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
check("csv header", lines[0] === "id,name,big", lines[0]);
check("csv plain row", lines[1] === "1,alice,10", lines[1]);
check("csv comma quoted", lines[2] === '2,"b,ob",20', lines[2]);
check("csv quote doubled", lines[3] === '3,"car""ol",30', lines[3]);
check("csv null -> empty", lines[4] === ",,40", lines[4]);
check("csv bigint no n-suffix", !csv.includes("10n"), csv);

// JSON: array of row objects; bigint -> number (safe range); null preserved.
const json = JSON.parse(tableToJSON(t));
check("json row count", json.length === 4, json.length);
check("json field names", JSON.stringify(Object.keys(json[0])) === '["id","name","big"]', Object.keys(json[0]));
check("json bigint -> number", json[0].big === 10 && typeof json[0].big === "number", json[0].big);
check("json null preserved", json[3].id === null && json[3].name === null, json[3]);
check("json embedded quote preserved", json[2].name === 'car"ol', json[2].name);

// TSV: tab-separated; embedded tabs/newlines replaced with space; null -> empty.
const tsv = tableToTSV(t);
const tlines = tsv.split("\n");
check("tsv header", tlines[0] === "id\tname\tbig", tlines[0]);
check("tsv row", tlines[1] === "1\talice\t10", tlines[1]);
check("tsv null -> empty", tlines[4] === "\t\t40", tlines[4]);
check("tsv no embedded tab in values", !tsv.slice(tsv.indexOf("\n") + 1).includes("\t\t\t"), "triple-tab");

// viewToTSV: TSV of a filtered+sorted view (a list of source-row indices into
// the column VECTORS), so clipboard copy respects the grid's sort/filter.
// Uses vector.get(r) so a NULL int (validity bitmap) renders as empty, not 0.
const vecId = t.getChild("id");
const vecName = t.getChild("name");
const vecBig = t.getChild("big");
const vecs = [vecId, vecName, vecBig];
// reverse row order (rows 3,2,1,0) — exercises non-contiguous reordering +
// the null-int case: row 3 has id=null, which must serialize to empty (not 0).
const reversed = viewToTSV(["id", "name", "big"], vecs, [3, 2, 1, 0]);
const rlines = reversed.split("\n");
check("view header", rlines[0] === "id\tname\tbig", rlines[0]);
check("view reordered row0 null->empty", rlines[1] === "\t\t40", rlines[1]);
check("view reordered row3", rlines[4] === "1\talice\t10", rlines[4]);
// a filtered view (only row index 1) — header + one row, value from row 1.
const filtered = viewToTSV(["id", "name", "big"], vecs, [1]);
const flines = filtered.split("\n");
check("view filtered one row", flines.length === 2 && flines[1] === "2\tb,ob\t20", flines);

console.log(fail === 0 ? "CSV OK" : `CSV FAILED: ${fail}`);
process.exit(fail === 0 ? 0 : 1);