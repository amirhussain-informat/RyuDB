// Node smoke test for the ryudb-server wire client + Arrow IPC decode path
// (the same path the browser uses, minus the DOM). Requires a running
// ryudb-server; set RYUDB_PORT and RYUDB_LINEITEM_PATH (a directory with a
// lineitem parquet) before invoking. Run via `npm run smoke` after starting
// the server, or let the repo wrapper script start one.
//
// Verifies: admin register, SELECT round-trip + Arrow decode + value match,
// GROUP BY result, EXPLAIN plan tree shape, parse error with position.

import { WebSocket } from "ws";
import { tableFromIPC } from "apache-arrow";

const PORT = process.env.RYUDB_PORT || "5430";
const LIPATH = process.env.RYUDB_LINEITEM_PATH;
const url = `ws://127.0.0.1:${PORT}`;
const N = parseInt(process.env.RYUDB_N || "500", 10);

if (!LIPATH) {
  console.error("RYUDB_LINEITEM_PATH not set");
  process.exit(2);
}

function connect() {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(url);
    ws.binaryType = "arraybuffer";
    ws.once("open", () => resolve(ws));
    ws.once("error", reject);
  });
}

function call(ws, obj) {
  const id = String(obj.id ?? `s${Math.random()}`);
  const out = JSON.stringify({ ...obj, id });
  return new Promise((resolve, reject) => {
    let pendingMeta = null;
    const onMsg = (data, isBinary) => {
      if (isBinary) {
        ws.off("message", onMsg);
        const table = tableFromIPC(new Uint8Array(data));
        resolve({ meta: pendingMeta, table });
        return;
      }
      const frame = JSON.parse(data.toString());
      if (frame.id !== id) return;
      if (frame.op === "result") {
        pendingMeta = frame;
        return;
      }
      ws.off("message", onMsg);
      resolve({ meta: frame, table: null });
    };
    ws.on("message", onMsg);
    ws.send(out);
  });
}

function cells(table, row) {
  const out = {};
  for (const f of table.schema.fields) {
    const arr = table.getChild(f.name)?.toArray();
    out[f.name] = arr ? arr[row] : null;
  }
  return out;
}

const ws = await connect();
let failures = 0;
function check(name, cond, extra = "") {
  if (cond) {
    console.log(`  ok: ${name}`);
  } else {
    console.error(`  FAIL: ${name} ${extra}`);
    failures++;
  }
}

// 1. register
const reg = await call(ws, { id: "reg", op: "admin", action: "register", args: { table: "lineitem", path: LIPATH } });
check("register ok", reg.meta.op === "ok" && reg.meta.detail?.registered === "lineitem", JSON.stringify(reg.meta));

// 2. count(*) round-trip + Arrow decode
const cnt = await call(ws, { id: "c", op: "sql", sql: "SELECT count(*) AS c FROM lineitem" });
check("count result meta", cnt.meta.op === "result" && cnt.meta.columns[0].name === "c", JSON.stringify(cnt.meta));
check("count arrow decoded", cnt.table !== null && cnt.table.numRows === 1);
if (cnt.table) check("count value", Number(cells(cnt.table, 0).c) === N, String(cells(cnt.table, 0).c));

// 3. GROUP BY → 3 rows
const grp = await call(ws, { id: "g", op: "sql", sql: "SELECT l_returnflag, count(*) AS c FROM lineitem GROUP BY l_returnflag ORDER BY l_returnflag" });
check("groupby 3 rows", grp.table !== null && grp.table.numRows === 3, String(grp.table?.numRows));

// 4. EXPLAIN fused badge (Aggregate-over-Join)
const ex = await call(ws, { id: "e", op: "explain", sql: "SELECT l_returnflag, count(*) AS c FROM lineitem JOIN lineitem l2 ON l_orderkey=l2.l_orderkey GROUP BY l_returnflag" });
check("explain plan", ex.meta.op === "plan" && ex.meta.tree.op === "Aggregate");
check("explain fused", ex.meta.op === "plan" && ex.meta.tree.fused === true && ex.meta.tree.children[0].op === "Join");

// 5. parse error with position
const pe = await call(ws, { id: "p", op: "sql", sql: "SELECT * FROM" });
check("parse error kind", pe.meta.op === "error" && pe.meta.kind === "parse", JSON.stringify(pe.meta));
check("parse error position", pe.meta.op === "error" && pe.meta.position && "line" in pe.meta.position, JSON.stringify(pe.meta.position));

// 6. cursor paging: sql cursor:true returns first page + cursor_id; fetch pages
//    the rest; close drops it. Concatenated pages cover the full row_count.
//    LIMIT 500 bounds the result so 5 pages of 100 cover it regardless of the
//    registered lineitem's size.
const PAGE = 100;
const cur = await call(ws, { id: "cur", op: "sql", sql: "SELECT l_orderkey, l_returnflag FROM lineitem LIMIT 500", max_rows: PAGE, cursor: true });
check("cursor first page meta", cur.meta.op === "result" && typeof cur.meta.cursor_id === "string" && cur.meta.offset === 0, JSON.stringify(cur.meta));
check("cursor first page rows", cur.table !== null && cur.table.numRows === PAGE, String(cur.table?.numRows));
check("cursor truncated", cur.meta.truncated === true, JSON.stringify(cur.meta));

let totalRows = cur.table ? cur.table.numRows : 0;
let off = totalRows;
let fetches = 0;
const cid = cur.meta.cursor_id;
while (off < cur.meta.row_count && fetches < 100) {
  const f = await call(ws, { id: `f${fetches}`, op: "fetch", cursor_id: cid, offset: off, limit: PAGE });
  if (f.meta.op !== "result" || !f.table) {
    check(`fetch ${fetches}`, false, JSON.stringify(f.meta));
    break;
  }
  totalRows += f.table.numRows;
  off += f.table.numRows;
  fetches++;
  if (f.table.numRows === 0) break;
}
check("cursor paged to full row_count", totalRows === cur.meta.row_count, `${totalRows} vs ${cur.meta.row_count}`);

// 7. close cursor → ok (idempotent); a subsequent fetch → error
const cl = await call(ws, { id: "cl", op: "close", cursor_id: cid });
check("close ok", cl.meta.op === "ok", JSON.stringify(cl.meta));
const cl2 = await call(ws, { id: "cl2", op: "close", cursor_id: cid });
check("close idempotent", cl2.meta.op === "ok", JSON.stringify(cl2.meta));
const afErr = await call(ws, { id: "afe", op: "fetch", cursor_id: cid, offset: 0, limit: 10 });
check("fetch after close errors", afErr.meta.op === "error" && /unknown cursor/.test(afErr.meta.message ?? ""), JSON.stringify(afErr.meta));

// 8. profile op: GPU per-column statistics (JSON only, no binary)
const prof = await call(ws, { id: "pr", op: "profile", name: "lineitem", top_k: 5 });
check("profile op", prof.meta.op === "profile" && prof.meta.name === "lineitem", JSON.stringify(prof.meta));
check("profile row_count", prof.meta.row_count === N, String(prof.meta.row_count));
const pcols = new Map((prof.meta.columns ?? []).map((c) => [c.name, c]));
check("profile columns", ["l_orderkey", "l_quantity", "l_extendedprice", "l_returnflag"].every((n) => pcols.has(n)), JSON.stringify([...pcols.keys()]));
const qty = pcols.get("l_quantity");
check("profile numeric stats", qty && qty.mean != null && qty.stddev != null && qty.histogram && qty.histogram.length === 10, JSON.stringify(qty));
check("profile histogram sums to rows", qty && qty.histogram.reduce((a, b) => a + b.count, 0) === N, String(qty?.histogram?.reduce((a, b) => a + b.count, 0)));
const flag = pcols.get("l_returnflag");
check("profile categorical top", flag && flag.top && flag.top.length > 0 && flag.top.reduce((a, t) => a + t.count, 0) === N, JSON.stringify(flag?.top));
check("profile unknown table errors", (await call(ws, { id: "pr2", op: "profile", name: "nope" })).meta.op === "error");

ws.close();
if (failures === 0) {
  console.log("SMOKE OK");
  process.exit(0);
} else {
  console.error(`SMOKE FAILED: ${failures} check(s)`);
  process.exit(1);
}