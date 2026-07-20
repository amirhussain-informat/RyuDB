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

ws.close();
if (failures === 0) {
  console.log("SMOKE OK");
  process.exit(0);
} else {
  console.error(`SMOKE FAILED: ${failures} check(s)`);
  process.exit(1);
}