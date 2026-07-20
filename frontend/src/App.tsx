import { useEffect, useRef, useState } from "react";
import { tableToIPC, Table } from "apache-arrow";
import { useServer } from "./hooks/useServer";
import Toolbar from "./components/Toolbar";
import SqlEditor, { type EditorHandle } from "./components/Editor";
import Results from "./components/Results";
import Explain from "./components/Explain";
import Catalog from "./components/Catalog";
import History from "./components/History";
import { tableToCSV, downloadBlob } from "./lib/csv";
import type {
  CatalogResp, CatalogTable, ErrorResp, HistoryEntry, PlanNode,
  ResultMeta, Result, TableResp,
} from "./lib/types";

const DEFAULT_URL = "ws://127.0.0.1:5430";
const RUN_ID = "run";
// Interactive page size (rows fetched per page while browsing). The grid
// virtualizes, so this only bounds the per-fetch wire frame + the slice held
// in memory as the user pages.
const PAGE_SIZE = 1000;
// Page size for cursor-backed downloads (fewer round trips than PAGE_SIZE).
const DL_PAGE = 50_000;

type MainTab = "results" | "explain" | "message";

export default function App() {
  const { status, connect, disconnect, op } = useServer();
  const editorRef = useRef<EditorHandle>(null);
  // The server-side cursor id for the current result (null when the result was
  // not opened as a cursor, or exceeded --max-cursor-rows and fell back). Held
  // in a ref so the unmount/disconnect cleanup can close it without stale state.
  const cursorRef = useRef<string | null>(null);
  const [cursorId, setCursorId] = useState<string | null>(null);
  const [loadingMore, setLoadingMore] = useState(false);

  const [sql, setSql] = useState(
    "SELECT l_returnflag, count(*) AS c, sum(l_extendedprice) AS s\n" +
    "FROM lineitem\nGROUP BY l_returnflag\nORDER BY l_returnflag;",
  );
  const [running, setRunning] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const [result, setResult] = useState<Result | null>(null);
  const [plan, setPlan] = useState<PlanNode | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<ErrorResp | null>(null);
  const [mainTab, setMainTab] = useState<MainTab>("results");
  const [sidebar, setSidebar] = useState<"catalog" | "history">("catalog");

  // Close the current cursor best-effort (unknown id is a server-side no-op).
  const closeCursorNow = (id: string | null) => {
    if (!id) return;
    cursorRef.current = null;
    try {
      void op({ id: "cc", op: "close", cursor_id: id }).catch(() => { /* best-effort */ });
    } catch {
      /* not connected */
    }
  };

  // On unmount, drop any live cursor so the server frees its host-memory table.
  useEffect(() => {
    return () => closeCursorNow(cursorRef.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const run = async () => {
    if (running) return;
    setRunning(true);
    setError(null);
    setPlan(null);
    setMessage(null);
    editorRef.current?.setParseError(null);
    closeCursorNow(cursorRef.current);
    setCursorId(null);
    try {
      // cursor: true asks the server to freeze the full result and return the
      // first page + a cursor_id; the rest is paged via loadMore().
      const res = await op({ id: RUN_ID, op: "sql", sql, max_rows: PAGE_SIZE, cursor: true });
      applyResult(res);
    } catch (e) {
      setMessage(`network: ${(e as Error).message}`);
      setMainTab("message");
    } finally {
      setRunning(false);
    }
  };

  const explain = async () => {
    if (running) return;
    setRunning(true);
    setError(null);
    editorRef.current?.setParseError(null);
    try {
      const res = await op({ id: "ex", op: "explain", sql });
      if (res.meta.op === "plan") {
        setPlan(res.meta.tree);
        setMainTab("explain");
      } else if (res.meta.op === "error") {
        showError(res.meta);
      }
    } catch (e) {
      setMessage(`network: ${(e as Error).message}`);
      setMainTab("message");
    } finally {
      setRunning(false);
    }
  };

  const cancel = async () => {
    try {
      await op({ id: "cx", op: "cancel", targets: [RUN_ID] });
    } catch {
      /* ignore */
    }
  };

  const applyResult = (res: Result) => {
    const m = res.meta;
    switch (m.op) {
      case "result": {
        const rm = m as ResultMeta;
        // A cursor-backed first page carries cursor_id; a too-large result
        // carries cursor:false + reason:too_large (no cursor -> not pageable).
        const cid = rm.cursor_id ?? null;
        cursorRef.current = cid;
        setCursorId(cid);
        setResult(res);
        setMainTab("results");
        break;
      }
      case "write":
        setMessage(`${m.rows_affected} row${m.rows_affected === 1 ? "" : "s"} affected (${m.duration_ms.toFixed(1)} ms)`);
        setMainTab("message");
        break;
      case "ok":
        setMessage(m.detail ? `ok: ${JSON.stringify(m.detail)}` : "ok");
        setMainTab("message");
        break;
      case "cancelled":
        setMessage("cancelled");
        setMainTab("message");
        break;
      case "error":
        showError(m);
        break;
      default:
        setMessage(`unexpected op: ${m.op}`);
        setMainTab("message");
    }
  };

  const loadMore = async () => {
    const cid = cursorRef.current;
    const cur = result;
    if (!cid || !cur || cur.meta.op !== "result" || loadingMore) return;
    const m = cur.meta as ResultMeta;
    if (m.returned >= m.row_count) return;
    setLoadingMore(true);
    try {
      const res = await op({ id: "pg", op: "fetch", cursor_id: cid, offset: m.returned, limit: PAGE_SIZE });
      if (res.meta.op !== "result" || !res.table) {
        setMessage(`fetch failed: ${(res.meta as { message?: string }).message ?? "no rows"}`);
        setMainTab("message");
        return;
      }
      // Concatenate the page onto the displayed table and grow returned/truncated.
      const grown = cur.table ? cur.table.concat(res.table) : res.table;
      const rm = res.meta as ResultMeta;
      const merged: ResultMeta = {
        ...m,
        returned: m.returned + res.table.numRows,
        truncated: m.returned + res.table.numRows < m.row_count,
        // keep the cursor_id on the merged meta so Results keeps showing Load more
        cursor_id: cid,
        offset: rm.offset,
      };
      setResult({ meta: merged, table: grown });
    } catch (e) {
      setMessage(`network: ${(e as Error).message}`);
      setMainTab("message");
    } finally {
      setLoadingMore(false);
    }
  };

  const showError = (m: ErrorResp) => {
    setError(m);
    setMessage(`[${m.kind}] ${m.message}`);
    setMainTab("message");
    if (m.kind === "parse" && m.position) {
      editorRef.current?.setParseError(m.position, m.message);
    }
  };

  const sample = async (name: string) => {
    setRunning(true);
    setError(null);
    setPlan(null);
    setMessage(null);
    editorRef.current?.setParseError(null);
    try {
      const res = await op({ id: "sm", op: "sample", name, n: 100 });
      applyResult(res);
    } catch (e) {
      setMessage(`network: ${(e as Error).message}`);
      setMainTab("message");
    } finally {
      setRunning(false);
    }
  };

  const download = async (format: "csv" | "arrow") => {
    const cur = result;
    if (!cur || cur.meta.op !== "result") return;
    const m = cur.meta as ResultMeta;
    // Guard against an accidental giant download (a cross-join can report
    // billions of rows). The fetch loop below would materialize all of them.
    if (m.row_count > 1_000_000) {
      if (!window.confirm(`Download all ${m.row_count.toLocaleString()} rows?`)) return;
    }
    setDownloading(true);
    try {
      let table = cur.table;
      if (m.truncated && table && table.numRows < m.row_count) {
        const cid = cursorRef.current;
        if (cid) {
          // Cursor-backed: page the rest from the frozen server-side result.
          let acc: Table = table;
          let off = table.numRows;
          while (off < m.row_count) {
            const res = await op({ id: "dl", op: "fetch", cursor_id: cid, offset: off, limit: DL_PAGE });
            if (res.meta.op !== "result" || !res.table) {
              setMessage("download: could not fetch full result");
              setMainTab("message");
              return;
            }
            acc = acc.concat(res.table);
            off += res.table.numRows;
            if (res.table.numRows === 0) break; // safety: no progress
          }
          table = acc;
        } else {
          // No cursor (result exceeded --max-cursor-rows): fall back to a
          // single uncapped re-run, as before cursor paging existed.
          const res = await op({ id: "dl", op: "sql", sql, max_rows: m.row_count });
          if (res.meta.op !== "result" || !res.table) {
            setMessage("download: could not fetch full result");
            setMainTab("message");
            return;
          }
          table = res.table;
        }
      }
      if (!table) return;
      if (format === "arrow") {
        downloadBlob("result.arrow", "application/vnd.apache.arrow.stream", tableToIPC(table));
      } else {
        downloadBlob("result.csv", "text/csv", tableToCSV(table));
      }
    } catch (e) {
      setMessage(`download failed: ${(e as Error).message}`);
      setMainTab("message");
    } finally {
      setDownloading(false);
    }
  };

  const fetchCatalog = async (): Promise<CatalogTable[]> => {
    const res = await op({ id: "cat", op: "catalog" });
    return res.meta.op === "catalog" ? (res.meta as CatalogResp).tables : [];
  };
  const fetchTable = async (name: string): Promise<TableResp> => {
    const res = await op({ id: "tbl", op: "table", name });
    return res.meta.op === "table" ? (res.meta as TableResp) : ({} as TableResp);
  };
  const fetchHistory = async (): Promise<HistoryEntry[]> => {
    const res = await op({ id: "hist", op: "history" });
    return res.meta.op === "history" ? res.meta.entries : [];
  };

  const connected = status === "open";

  return (
    <div className="app">
      <Toolbar
        url={DEFAULT_URL}
        status={status}
        running={running}
        onConnect={connect}
        onDisconnect={disconnect}
        onRun={run}
        onExplain={explain}
        onCancel={cancel}
      />
      <div className="body">
        <aside className="sidebar">
          <div className="sidebar-tabs">
            <button className={sidebar === "catalog" ? "active" : ""} onClick={() => setSidebar("catalog")}>Catalog</button>
            <button className={sidebar === "history" ? "active" : ""} onClick={() => setSidebar("history")}>History</button>
          </div>
          {sidebar === "catalog" ? (
            <Catalog
              fetchCatalog={fetchCatalog}
              fetchTable={fetchTable}
              onInsert={(t) => editorRef.current?.insert(t)}
              onSample={sample}
              status={status}
            />
          ) : (
            <History
              fetchHistory={fetchHistory}
              onPick={(s) => setSql(s)}
              status={status}
            />
          )}
        </aside>
        <main className="main">
          <div className="editor-pane">
            <SqlEditor ref={editorRef} value={sql} onChange={setSql} onRun={run} />
          </div>
          <div className="output">
            <div className="output-tabs">
              <button className={mainTab === "results" ? "active" : ""} onClick={() => setMainTab("results")}>Results</button>
              <button className={mainTab === "explain" ? "active" : ""} onClick={() => setMainTab("explain")}>Explain</button>
              {(message || error) && (
                <button className={mainTab === "message" ? "active" : ""} onClick={() => setMainTab("message")}>
                  {error ? "Error" : "Message"}
                </button>
              )}
            </div>
            <div className="output-body">
              {!connected && <div className="empty">Connect to ryudb-server to run queries.</div>}
              {connected && mainTab === "results" && (
                result && result.meta.op === "result"
                  ? <Results meta={result.meta as ResultMeta} table={result.table}
                             onDownload={download} downloading={downloading}
                             onLoadMore={loadMore}
                             hasMore={!!cursorId && (result.meta as ResultMeta).returned < (result.meta as ResultMeta).row_count}
                             loadingMore={loadingMore} />
                  : <div className="empty">Run a query.</div>
              )}
              {connected && mainTab === "explain" && <Explain tree={plan} />}
              {connected && mainTab === "message" && (
                <pre className={error ? "msg error" : "msg"}>{message}</pre>
              )}
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}