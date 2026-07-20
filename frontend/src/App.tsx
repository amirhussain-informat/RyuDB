import { useEffect, useRef, useState } from "react";
import { tableToIPC, Table } from "apache-arrow";
import { useServer } from "./hooks/useServer";
import { useWorksheets } from "./hooks/useWorksheets";
import { useDashboards } from "./hooks/useDashboards";
import { useTheme } from "./hooks/useTheme";
import { useVersions, type Snapshot } from "./hooks/useVersions";
import Toolbar from "./components/Toolbar";
import SqlEditor, { type EditorHandle, type Schema } from "./components/Editor";
import WorksheetTabs from "./components/WorksheetTabs";
import CommandPalette, { type Command } from "./components/CommandPalette";
import ShortcutsHelp from "./components/ShortcutsHelp";
import SearchModal from "./components/SearchModal";
import VersionHistory from "./components/VersionHistory";
import ProfileModal from "./components/ProfileModal";
import LoadDataModal from "./components/LoadDataModal";
import TableDetailModal from "./components/TableDetailModal";
import Results from "./components/Results";
import Chart from "./components/Chart";
import Explain from "./components/Explain";
import Catalog from "./components/Catalog";
import History from "./components/History";
import Dashboards from "./components/Dashboards";
import DashboardModal from "./components/DashboardModal";
import PinWidgetModal from "./components/PinWidgetModal";
import { tableToCSV, tableToJSON, downloadBlob } from "./lib/csv";
import { serializeBundle, serializeOne, parseImportFile, worksheetFileName } from "./lib/worksheetTransfer";
import type {
  CatalogResp, CatalogTable, ChartSpec, ErrorResp, HistoryEntry, PlanNode,
  ProfileResp, ResultMeta, Result, TableResp,
} from "./lib/types";

const DEFAULT_URL = "ws://127.0.0.1:5430";
const RUN_ID = "run";
// Interactive page size (rows fetched per page while browsing). The grid
// virtualizes, so this only bounds the per-fetch wire frame + the slice held
// in memory as the user pages.
const PAGE_SIZE = 1000;
// Page size for cursor-backed downloads (fewer round trips than PAGE_SIZE).
const DL_PAGE = 50_000;
// Rows fetched per dashboard widget (charts are display-only — bar caps at
// MAX_BARS=60, line/scatter plot up to this many points; no cursor needed).
const WIDGET_ROWS = 1000;

type MainTab = "results" | "chart" | "explain" | "message";

// A past SELECT result kept around so the user can switch back through a
// worksheet's result history (multi-result tabs). Holds the decoded Arrow
// result, its server-side cursor id (for load-more / export), the SQL that
// produced it (so Download re-runs THAT statement, not the editor's current
// text), and a timestamp for the tab label. Bounded by MAX_RESULTS.
interface ResultEntry {
  id: string;
  res: Result;
  cursorId: string | null;
  sql: string;
  ts: number;
}
const MAX_RESULTS = 10;

// The per-worksheet view (results / plan / message / error / active sub-tab /
// cursor id / result history / the active result's own sql+ts). Kept in memory
// keyed by worksheet id so switching tabs restores each tab's last view during
// a session (not persisted across reloads).
interface View {
  result: Result | null;
  plan: PlanNode | null;
  message: string | null;
  error: ErrorResp | null;
  mainTab: MainTab;
  cursorId: string | null;
  resultHistory: ResultEntry[];
  resultSql: string | null;
  resultTs: number | null;
}

const EMPTY_VIEW: View = {
  result: null, plan: null, message: null, error: null, mainTab: "results",
  cursorId: null, resultHistory: [], resultSql: null, resultTs: null,
};

/** Compact relative time for a result-tab label ("just now", "3m", "2h"). */
function relTime(ts: number): string {
  const s = Math.max(0, Math.round((Date.now() - ts) / 1000));
  if (s < 5) return "just now";
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m`;
  if (s < 86400) return `${Math.floor(s / 3600)}h`;
  return `${Math.floor(s / 86400)}d`;
}

/** True if keyboard focus is in a text-entry surface (so global letter-key
 * shortcuts like `?` should not fire). Monaco renders a contenteditable-ish
 * tree under `.monaco-editor`. */
function isTypingTarget(t: EventTarget | null): boolean {
  const el = t as HTMLElement | null;
  if (!el) return false;
  if (el.tagName === "INPUT" || el.tagName === "TEXTAREA") return true;
  if (el.isContentEditable) return true;
  if (el.closest && el.closest(".monaco-editor")) return true;
  return false;
}

export default function App() {
  const { status, connect, disconnect, op, upload } = useServer();
  const editorRef = useRef<EditorHandle>(null);
  // The server-side cursor id for the current result (null when the result was
  // not opened as a cursor, or exceeded --max-cursor-rows and fell back). Held
  // in a ref so the unmount/disconnect cleanup can close it without stale state.
  const cursorRef = useRef<string | null>(null);
  const { worksheets, activeId, active, setActive, create, rename, close, updateSql, importWorksheets } = useWorksheets();
  const { dashboards, create: createDashboard, rename: renameDashboard, remove: removeDashboard, addWidget, removeWidget } = useDashboards();
  const { theme, toggle: toggleTheme } = useTheme();
  const { versions, capture, remove, clear, gc } = useVersions();

  // Per-worksheet view state (in-memory). Switching a tab saves the view of the
  // tab being left and restores (or initializes) the view of the new tab.
  const viewsRef = useRef<Map<string, View>>(new Map());
  const activeIdRef = useRef(activeId);
  activeIdRef.current = activeId;

  const [running, setRunning] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [result, setResult] = useState<Result | null>(null);
  const [plan, setPlan] = useState<PlanNode | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<ErrorResp | null>(null);
  const [mainTab, setMainTab] = useState<MainTab>("results");
  const [cursorId, setCursorId] = useState<string | null>(null);
  // The SQL + timestamp of the statement that produced the CURRENT `result`.
  // Download re-runs `resultSql` (not the editor's text) so exporting a past
  // result tab re-runs the statement that made it, not whatever is in the
  // editor now. `resultHistory` is the worksheet's past SELECT results.
  const [resultSql, setResultSql] = useState<string | null>(null);
  const [resultTs, setResultTs] = useState<number | null>(null);
  const [resultHistory, setResultHistory] = useState<ResultEntry[]>([]);
  const resultIdRef = useRef(0);
  // Catalog schema for SQL autocompletion (table name -> typed columns). Built
  // on connect and re-built when a write/DDL lands (`schemaNonce` bumped in
  // applyResult). Passed down to SqlEditor; a SELECT does not bump the nonce.
  const [schema, setSchema] = useState<Schema>(new Map());
  const [schemaNonce, setSchemaNonce] = useState(0);
  // Data-load wizard open state + a catalog refresh signal bumped after DDL.
  const [loadOpen, setLoadOpen] = useState(false);
  const [catalogNonce, setCatalogNonce] = useState(0);
  const [detailName, setDetailName] = useState<string | null>(null);
  const [sidebar, setSidebar] = useState<"catalog" | "history" | "dashboards">("catalog");
  const [dashboardOpenId, setDashboardOpenId] = useState<string | null>(null);
  const [pinSpec, setPinSpec] = useState<ChartSpec | null>(null);
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [shortcutsOpen, setShortcutsOpen] = useState(false);
  const [searchOpen, setSearchOpen] = useState(false);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [profileName, setProfileName] = useState<string | null>(null);

  // Mint a unique id for a result-history entry (a monotonic counter —
  // Date.now/Math.random are fine in the browser but a ref counter is stable
  // and avoids any same-ms collision across rapid runs).
  const nextResultId = () => `r${++resultIdRef.current}`;

  // Best-effort close of a history entry's server cursor (unknown id is a
  // server-side no-op). Frees the host-memory table behind a dropped result.
  const closeEntryCursor = (e: ResultEntry) => {
    if (e.cursorId) {
      try { void op({ id: "cc", op: "close", cursor_id: e.cursorId }).catch(() => {}); }
      catch { /* not connected */ }
    }
  };

  // Prepend a result-history entry, trimming to MAX_RESULTS and closing the
  // cursors of any entries dropped off the tail (so the server frees them).
  const prependHistory = (e: ResultEntry) => {
    setResultHistory((h) => {
      const next = [e, ...h];
      if (next.length > MAX_RESULTS) {
        for (const d of next.slice(MAX_RESULTS)) closeEntryCursor(d);
        return next.slice(0, MAX_RESULTS);
      }
      return next;
    });
  };

  // Apply a stored View to the live state slots.
  const setView = (v: View) => {
    setResult(v.result);
    setPlan(v.plan);
    setMessage(v.message);
    setError(v.error);
    setMainTab(v.mainTab);
    setCursorId(v.cursorId);
    setResultHistory(v.resultHistory);
    setResultSql(v.resultSql);
    setResultTs(v.resultTs);
    cursorRef.current = v.cursorId;
  };

  // Switch to another worksheet: stash the leaving tab's view, restore the new.
  const switchTab = (id: string) => {
    if (id === activeIdRef.current) return;
    viewsRef.current.set(activeIdRef.current, {
      result, plan, message, error, mainTab, cursorId, resultHistory, resultSql, resultTs,
    });
    editorRef.current?.setParseError(null);
    setActive(id);
    setView(viewsRef.current.get(id) ?? EMPTY_VIEW);
  };

  // Close a worksheet tab: best-effort close its cursor + every history
  // entry's cursor, drop its stored view, then remove it (the hook keeps at
  // least one worksheet).
  const closeTab = (id: string) => {
    const v = viewsRef.current.get(id);
    if (v?.cursorId) {
      try { void op({ id: "cc", op: "close", cursor_id: v.cursorId }).catch(() => {}); } catch { /* disconnected */ }
    }
    if (v) for (const h of v.resultHistory) closeEntryCursor(h);
    viewsRef.current.delete(id);
    if (id === activeIdRef.current) {
      const idx = worksheets.findIndex((w) => w.id === id);
      const next = worksheets.filter((w) => w.id !== id);
      if (next.length) {
        const nb = next[Math.min(idx, next.length - 1)];
        setView(viewsRef.current.get(nb.id) ?? EMPTY_VIEW);
      }
    }
    close(id);
  };

  // --- Git-backed worksheet export/import --------------------------------- //
  // Export worksheets as plain .sql text (single = raw SQL; all = a bundle with
  // `-- @@worksheet: <name>` separators) so they are git-diffable. Import reads
  // .sql files (a bundle splits into sections; a plain file is one worksheet).
  const importInputRef = useRef<HTMLInputElement | null>(null);

  const exportActive = () => {
    if (!active) return;
    downloadBlob(worksheetFileName(active.name), "text/sql;charset=utf-8", serializeOne(active.sql));
  };
  const exportAll = () => {
    if (worksheets.length === 0) return;
    downloadBlob("ryudb-worksheets.sql", "text/sql;charset=utf-8",
      serializeBundle(worksheets.map((w) => ({ name: w.name, sql: w.sql }))));
  };
  const pickImport = () => importInputRef.current?.click();
  const onImportFiles = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    const docs: { name: string; sql: string }[] = [];
    for (const f of Array.from(files)) {
      try {
        const text = await f.text();
        docs.push(...parseImportFile(f.name, text));
      } catch {
        /* skip an unreadable file */
      }
    }
    e.target.value = ""; // allow re-picking the same file
    if (docs.length === 0) {
      setError({ op: "error", kind: "runtime", message: "No worksheets found in the selected file(s)." } as ErrorResp);
      setMainTab("message");
      return;
    }
    importWorksheets(docs);
  };

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

  // Drop version-history rings for worksheets that no longer exist (closed),
  // so localStorage doesn't accrue orphaned snapshot rings.
  useEffect(() => {
    gc(worksheets.map((w) => w.id));
  }, [worksheets, gc]);

  // Global keyboard shortcuts: Cmd/Ctrl+K toggles the command palette;
  // Cmd/Ctrl+Shift+F opens global object search; Cmd/Ctrl+Shift+H opens the
  // active worksheet's version history; `?` (when not typing) opens the
  // shortcuts help; Escape closes any open overlay.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const mod = e.ctrlKey || e.metaKey;
      if (mod && (e.key === "k" || e.key === "K")) {
        e.preventDefault();
        setPaletteOpen((o) => !o);
      } else if (mod && e.shiftKey && (e.key === "f" || e.key === "F")) {
        e.preventDefault();
        setSearchOpen(true);
      } else if (mod && e.shiftKey && (e.key === "h" || e.key === "H")) {
        e.preventDefault();
        setHistoryOpen(true);
      } else if (e.key === "?" && !isTypingTarget(e.target)) {
        e.preventDefault();
        setShortcutsOpen(true);
      } else if (e.key === "Escape") {
        setPaletteOpen(false);
        setShortcutsOpen(false);
        setSearchOpen(false);
        setHistoryOpen(false);
        setProfileName(null);
        setPinSpec(null);
        setDashboardOpenId(null);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const sql = active?.sql ?? "";

  const run = async () => {
    if (running) return;
    // Snapshot the SQL being run into this worksheet's version history (a
    // meaningful, recoverable point — deduped against the latest snapshot).
    if (active) capture(active.id, sql);
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
      applyResult(res, sql);
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

  const applyResult = (res: Result, runSql: string) => {
    const m = res.meta;
    switch (m.op) {
      case "result": {
        const rm = m as ResultMeta;
        // A cursor-backed first page carries cursor_id; a too-large result
        // carries cursor:false + reason:too_large (no cursor -> not pageable).
        const cid = rm.cursor_id ?? null;
        // Archive the previously-shown SELECT result into this worksheet's
        // history before replacing it (multi-result tabs). Non-result ops
        // (write/ok/...) don't get archived.
        if (result && result.meta.op === "result") {
          prependHistory({
            id: nextResultId(), res: result, cursorId: cursorRef.current,
            sql: resultSql ?? "", ts: resultTs ?? Date.now(),
          });
        }
        cursorRef.current = cid;
        setCursorId(cid);
        setResult(res);
        setResultSql(runSql);
        setResultTs(Date.now());
        setMainTab("results");
        break;
      }
      case "write":
        setMessage(`${m.rows_affected} row${m.rows_affected === 1 ? "" : "s"} affected (${m.duration_ms.toFixed(1)} ms)`);
        setMainTab("message");
        setSchemaNonce((n) => n + 1);
        break;
      case "ok":
        setMessage(m.detail ? `ok: ${JSON.stringify(m.detail)}` : "ok");
        setMainTab("message");
        setSchemaNonce((n) => n + 1);
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

  // Switch the displayed result to a past one (multi-result tabs): the chosen
  // history entry becomes the current `result`, and the previously-displayed
  // SELECT result is archived at the front of the history. If the current view
  // isn't a SELECT result (e.g. a write/message), just load the entry without
  // archiving. The entry's own cursor/sql/ts come with it (load-more + download
  // then operate on THAT result's statement).
  const selectResult = (id: string) => {
    const entry = resultHistory.find((h) => h.id === id);
    if (!entry) return;
    const curIsResult = !!result && result.meta.op === "result";
    const cur: ResultEntry | null = curIsResult
      ? { id: nextResultId(), res: result, cursorId: cursorRef.current, sql: resultSql ?? "", ts: resultTs ?? Date.now() }
      : null;
    cursorRef.current = entry.cursorId;
    setCursorId(entry.cursorId);
    setResult(entry.res);
    setResultSql(entry.sql);
    setResultTs(entry.ts);
    setMainTab("results");
    setResultHistory((h) => {
      const next = cur ? [cur, ...h.filter((x) => x.id !== id)] : h.filter((x) => x.id !== id);
      if (next.length > MAX_RESULTS) {
        for (const d of next.slice(MAX_RESULTS)) closeEntryCursor(d);
        return next.slice(0, MAX_RESULTS);
      }
      return next;
    });
  };

  // Drop one past result from the history and close its server cursor.
  const closeResult = (id: string) => {
    const entry = resultHistory.find((h) => h.id === id);
    if (entry) closeEntryCursor(entry);
    setResultHistory((h) => h.filter((x) => x.id !== id));
  };

  // Drop the entire result history for this worksheet (closes every cursor).
  const clearResults = () => {
    for (const h of resultHistory) closeEntryCursor(h);
    setResultHistory([]);
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
      applyResult(res, `SELECT * FROM ${name} LIMIT 100`);
    } catch (e) {
      setMessage(`network: ${(e as Error).message}`);
      setMainTab("message");
    } finally {
      setRunning(false);
    }
  };

  const download = async (format: "csv" | "json" | "arrow" | "parquet") => {
    const cur = result;
    if (!cur || cur.meta.op !== "result") return;
    const m = cur.meta as ResultMeta;
    // Re-run the statement that produced THIS result (resultSql), not the
    // editor's current text — the user may be viewing a past result tab whose
    // SQL differs from what's now in the editor.
    const runSql = resultSql ?? sql;
    // Guard against an accidental giant download (a cross-join can report
    // billions of rows). The fetch loop below would materialize all of them.
    if (m.row_count > 1_000_000) {
      if (!window.confirm(`Download all ${m.row_count.toLocaleString()} rows?`)) return;
    }
    setDownloading(true);
    try {
      // Parquet has no in-browser writer for apache-arrow 17 — the server
      // serializes the full result to Parquet (one export op + binary blob) and
      // we save the raw bytes. No client-side paging needed.
      if (format === "parquet") {
        const res = await op({ id: "dl", op: "export", sql: runSql, format: "parquet" });
        if (res.meta.op !== "export" || !res.bytes) {
          setMessage("download: export failed");
          setMainTab("message");
          return;
        }
        downloadBlob("result.parquet", "application/vnd.apache.parquet", res.bytes);
        return;
      }
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
          const res = await op({ id: "dl", op: "sql", sql: runSql, max_rows: m.row_count });
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
      } else if (format === "json") {
        downloadBlob("result.json", "application/json", tableToJSON(table));
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
  // Build the autocompletion schema: the catalog list, then a per-table `table`
  // op fan-out for typed columns. Re-runs on connect and after any write/DDL
  // (schemaNonce). A dropped-between-catalog-and-fetch table yields no columns.
  useEffect(() => {
    if (status !== "open") {
      setSchema(new Map());
      return;
    }
    let cancelled = false;
    void (async () => {
      const tables = await fetchCatalog();
      if (cancelled) return;
      const entries = await Promise.all(
        tables.map(async (t) => [t.name, (await fetchTable(t.name)).columns ?? []] as const),
      );
      if (cancelled) return;
      setSchema(new Map(entries.filter((e) => e[0])));
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [status, schemaNonce]);
  const fetchHistory = async (): Promise<HistoryEntry[]> => {
    const res = await op({ id: "hist", op: "history" });
    return res.meta.op === "history" ? res.meta.entries : [];
  };
  const fetchProfile = async (name: string): Promise<ProfileResp | null> => {
    const res = await op({ id: "prof", op: "profile", name, top_k: 10 });
    return res.meta.op === "profile" ? (res.meta as ProfileResp) : null;
  };
  // DDL via the `admin` op (register/drop/rename/alter — the engine has no
  // SQL-level DDL parser, so all table management goes through admin). Throws
  // with the server's message on an error frame so callers can surface it.
  const fetchAdmin = async (
    action: string,
    args: Record<string, unknown> = {},
  ): Promise<Record<string, unknown>> => {
    const res = await op({ id: "adm", op: "admin", action, args });
    const m = res.meta;
    if (m.op === "ok") return (m as { detail?: Record<string, unknown> }).detail ?? {};
    if (m.op === "error") throw new Error((m as ErrorResp).message ?? "admin error");
    throw new Error(`unexpected admin response: ${m.op}`);
  };
  // After any DDL, re-fetch the catalog list + the autocomplete schema.
  const refreshAfterDdl = () => {
    setCatalogNonce((n) => n + 1);
    setSchemaNonce((n) => n + 1);
  };
  const handleLoad = async (name: string, path: string) => {
    await fetchAdmin("register", { table: name, path });
    refreshAfterDdl();
  };
  const handleDrop = async (name: string) => {
    if (!window.confirm(`Drop table "${name}"?\n\nThis unregisters it from the catalog. Source files are kept.`)) {
      return;
    }
    try {
      await fetchAdmin("drop", { table: name });
      refreshAfterDdl();
    } catch (e) {
      setMessage(`drop failed: ${(e as Error).message}`);
      setMainTab("message");
    }
  };
  const handleRename = async (oldName: string, newName: string) => {
    await fetchAdmin("rename", { old: oldName, new: newName });
    refreshAfterDdl();
  };
  // Client-side pre-check cap for browser uploads (the server enforces its
  // own --max-upload-bytes; this matches the default and avoids a 1009 close
  // for obviously-too-large files). The upload op writes the parquet to
  // <data>/uploads and registers the table.
  const MAX_UPLOAD_BYTES = 256 * 1024 * 1024;
  const handleUpload = async (name: string, bytes: Uint8Array, _fileName: string) => {
    const res = await upload(name, bytes, MAX_UPLOAD_BYTES);
    const m = res.meta;
    if (m.op === "ok") {
      refreshAfterDdl();
      return;
    }
    if (m.op === "error") throw new Error((m as ErrorResp).message ?? "upload failed");
    throw new Error(`unexpected upload response: ${m.op}`);
  };

  // --- Dashboards -------------------------------------------------------- //
  // A dashboard is a saved grid of chart widgets (a SQL query + a chart spec).
  // Widgets re-run their query on open/refresh. The chart spec is captured on
  // the Chart tab via "Pin to dashboard"; the SQL baked into the widget is the
  // statement that produced the charted result (resultSql), not the editor's
  // current text — so re-running the widget reproduces that result.
  const fetchWidget = async (widgetSql: string): Promise<Result | null> => {
    try {
      return await op({ id: "dw", op: "sql", sql: widgetSql, max_rows: WIDGET_ROWS });
    } catch {
      return null;
    }
  };
  const openDashboard = (id: string) => setDashboardOpenId(id);
  const handleCreateDashboard = () => {
    const id = createDashboard("");
    setDashboardOpenId(id);
  };
  const handleRemoveWidget = (widgetId: string) => {
    if (dashboardOpenId) removeWidget(dashboardOpenId, widgetId);
  };
  const handlePin = (dashboardId: string | null, title: string) => {
    if (!pinSpec) return;
    const bakedSql = resultSql ?? active?.sql ?? "";
    const { dashboardId: did } = addWidget(dashboardId, title, bakedSql, pinSpec);
    // Open the dashboard so the user sees the newly pinned widget render.
    setDashboardOpenId(did);
  };

  const connected = status === "open";
  const activeMeta = result?.meta as ResultMeta | undefined;
  const hasMore =
    !!cursorId &&
    !!result && result.meta.op === "result" &&
    !!activeMeta && activeMeta.returned < activeMeta.row_count;

  // Commands surfaced in the palette. Rebuilt each render so the closures see
  // the latest state (running / connected / active id / worksheet list).
  const commands: Command[] = [
    { id: "run", label: "Run query", hint: "Ctrl/Cmd+Enter", group: "Query", disabled: !connected || running, run },
    { id: "explain", label: "Explain plan", group: "Query", disabled: !connected || running, run: explain },
    { id: "cancel", label: "Cancel running query", group: "Query", disabled: !running, run: cancel },
    { id: "connect", label: "Connect", group: "Session", disabled: connected, run: () => connect(DEFAULT_URL) },
    { id: "disconnect", label: "Disconnect", group: "Session", disabled: !connected, run: disconnect },
    {
      id: "search", label: "Search tables, columns, history", hint: "Ctrl/Cmd+Shift+F",
      group: "Search", disabled: !connected, run: () => setSearchOpen(true),
    },
    { id: "new-ws", label: "New worksheet", hint: "+", group: "Worksheets", run: create },
    {
      id: "close-ws", label: "Close current worksheet", group: "Worksheets",
      disabled: worksheets.length <= 1, run: () => closeTab(activeId),
    },
    {
      id: "save-version", label: "Save version of current SQL", group: "Worksheets",
      disabled: !active, run: () => active && capture(active.id, active.sql),
    },
    {
      id: "history", label: "Show version history", hint: "Ctrl/Cmd+Shift+H", group: "Worksheets",
      disabled: !active, run: () => setHistoryOpen(true),
    },
    {
      id: "clear-results", label: "Clear result history", group: "Worksheets",
      disabled: resultHistory.length === 0, run: clearResults,
    },
    {
      id: "export-ws", label: "Export active worksheet as .sql", group: "Worksheets",
      disabled: !active, run: exportActive,
    },
    {
      id: "export-all-ws", label: "Export all worksheets (.sql bundle)", group: "Worksheets",
      disabled: worksheets.length === 0, run: exportAll,
    },
    {
      id: "import-ws", label: "Import worksheets from .sql", group: "Worksheets",
      run: pickImport,
    },
    {
      id: "sidebar-dashboards", label: "Open Dashboards sidebar", group: "View",
      disabled: sidebar === "dashboards", run: () => setSidebar("dashboards"),
    },
    {
      id: "new-dashboard", label: "New dashboard", group: "Dashboards",
      run: handleCreateDashboard,
    },
    {
      id: "pin-chart", label: "Pin current chart to a dashboard", hint: "Chart tab → Pin", group: "Dashboards",
      disabled: !(connected && !!result && result.meta.op === "result"),
      run: () => setMainTab("chart"),
    },
    ...dashboards.map((d) => ({
      id: "dash-" + d.id, label: `Open dashboard ${d.name}`, group: "Dashboards",
      run: () => openDashboard(d.id),
    })),
    ...worksheets.map((w) => ({
      id: "go-" + w.id, label: `Go to ${w.name}`, group: "Worksheets",
      disabled: w.id === activeId, run: () => switchTab(w.id),
    })),
    { id: "theme", label: "Toggle dark / light theme", group: "View", run: toggleTheme },
    { id: "sidebar-catalog", label: "Open Catalog sidebar", group: "View", disabled: sidebar === "catalog", run: () => setSidebar("catalog") },
    { id: "load-data", label: "Load data from parquet path", group: "Data", disabled: status !== "open", run: () => setLoadOpen(true) },
    { id: "sidebar-history", label: "Open History sidebar", group: "View", disabled: sidebar === "history", run: () => setSidebar("history") },
    { id: "shortcuts", label: "Show keyboard shortcuts", hint: "?", group: "Help", run: () => setShortcutsOpen(true) },
  ];

  return (
    <div className="app">
      <Toolbar
        url={DEFAULT_URL}
        status={status}
        running={running}
        theme={theme}
        onToggleTheme={toggleTheme}
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
            <button className={sidebar === "dashboards" ? "active" : ""} onClick={() => setSidebar("dashboards")}>Dashboards</button>
          </div>
          {sidebar === "catalog" ? (
            <Catalog
              fetchCatalog={fetchCatalog}
              fetchTable={fetchTable}
              onInsert={(t) => editorRef.current?.insert(t)}
              onSample={sample}
              onProfile={(name) => setProfileName(name)}
              onDetail={(name) => setDetailName(name)}
              onLoad={() => setLoadOpen(true)}
              onDrop={handleDrop}
              status={status}
              nonce={catalogNonce}
            />
          ) : sidebar === "history" ? (
            <History
              fetchHistory={fetchHistory}
              onPick={(s) => active && updateSql(active.id, s)}
              status={status}
            />
          ) : (
            <Dashboards
              dashboards={dashboards}
              onOpen={openDashboard}
              onCreate={handleCreateDashboard}
              onRename={renameDashboard}
              onRemove={removeDashboard}
            />
          )}
        </aside>
        <main className="main">
          <div className="editor-pane">
            <WorksheetTabs
              worksheets={worksheets}
              activeId={activeId}
              onSelect={switchTab}
              onCreate={create}
              onClose={closeTab}
              onRename={rename}
              onExportActive={exportActive}
              onExportAll={exportAll}
              onImport={pickImport}
            />
            <div className="editor-host">
              <SqlEditor
                ref={editorRef}
                value={sql}
                onChange={(v) => active && updateSql(active.id, v)}
                onRun={run}
                schema={schema}
                theme={theme === "dark" ? "vs-dark" : "vs"}
              />
            </div>
          </div>
          <div className="output">
            <div className="output-tabs">
              <button className={mainTab === "results" ? "active" : ""} onClick={() => setMainTab("results")}>Results</button>
              {result && result.meta.op === "result" && (
                <button className={mainTab === "chart" ? "active" : ""} onClick={() => setMainTab("chart")}>Chart</button>
              )}
              <button className={mainTab === "explain" ? "active" : ""} onClick={() => setMainTab("explain")}>Explain</button>
              {(message || error) && (
                <button className={mainTab === "message" ? "active" : ""} onClick={() => setMainTab("message")}>
                  {error ? "Error" : "Message"}
                </button>
              )}
            </div>
            {(result?.meta.op === "result" || resultHistory.length > 0) && (
              <div className="result-tabs">
                {result && result.meta.op === "result" && (
                  <button className="rtab active" title={resultSql ?? ""}>
                    <span className="rtab-label">
                      {(result.meta as ResultMeta).row_count.toLocaleString()} rows
                      {resultTs && <span className="rtab-time"> · {relTime(resultTs)}</span>}
                    </span>
                  </button>
                )}
                {resultHistory.map((h) => (
                  <button
                    key={h.id}
                    className="rtab"
                    onClick={() => selectResult(h.id)}
                    title={h.sql}
                  >
                    <span className="rtab-label">
                      {(h.res.meta as ResultMeta).row_count.toLocaleString()} rows
                      <span className="rtab-time"> · {relTime(h.ts)}</span>
                    </span>
                    <span
                      className="rtab-close"
                      role="button"
                      tabIndex={0}
                      title="Close this result tab"
                      onClick={(e) => { e.stopPropagation(); closeResult(h.id); }}
                    >×</span>
                  </button>
                ))}
                {resultHistory.length > 0 && (
                  <button className="rtab-clear" title="Clear result history" onClick={clearResults}>
                    Clear
                  </button>
                )}
              </div>
            )}
            <div className="output-body">
              {!connected && <div className="empty">Connect to ryudb-server to run queries.</div>}
              {connected && mainTab === "results" && (
                result && result.meta.op === "result"
                  ? <Results meta={result.meta as ResultMeta} table={result.table}
                             onDownload={download} downloading={downloading}
                             onLoadMore={loadMore}
                             hasMore={hasMore}
                             loadingMore={loadingMore} />
                  : <div className="empty">Run a query.</div>
              )}
              {connected && mainTab === "chart" && (
                result && result.meta.op === "result"
                  ? <Chart meta={result.meta as ResultMeta} table={result.table} onPin={(spec) => setPinSpec(spec)} />
                  : <div className="empty">Run a query to chart its results.</div>
              )}
              {connected && mainTab === "explain" && <Explain tree={plan} />}
              {connected && mainTab === "message" && (
                <pre className={error ? "msg error" : "msg"}>{message}</pre>
              )}
            </div>
          </div>
        </main>
      </div>
      <CommandPalette open={paletteOpen} commands={commands} onClose={() => setPaletteOpen(false)} />
      <input
        ref={importInputRef}
        type="file"
        accept=".sql,text/sql,application/octet-stream"
        multiple
        style={{ display: "none" }}
        onChange={onImportFiles}
      />
      <ShortcutsHelp open={shortcutsOpen} onClose={() => setShortcutsOpen(false)} />
      <SearchModal
        open={searchOpen}
        connected={connected}
        fetchCatalog={fetchCatalog}
        fetchHistory={fetchHistory}
        onClose={() => setSearchOpen(false)}
        onPickTable={(name) => editorRef.current?.insert(name)}
        onPickColumn={(_table, col) => editorRef.current?.insert(col)}
        onPickHistory={(s) => active && updateSql(active.id, s)}
      />
      <VersionHistory
        open={historyOpen}
        worksheetName={active?.name ?? ""}
        snapshots={active ? versions(active.id) : []}
        onClose={() => setHistoryOpen(false)}
        onSaveNow={() => active && capture(active.id, active.sql)}
        onRestore={(snap: Snapshot) => {
          if (active) updateSql(active.id, snap.sql);
          setHistoryOpen(false);
        }}
        onDelete={(snap) => active && remove(active.id, snap.id)}
        onClear={() => active && clear(active.id)}
      />
      <ProfileModal
        open={profileName !== null}
        name={profileName}
        fetchProfile={fetchProfile}
        onClose={() => setProfileName(null)}
      />
      <LoadDataModal
        open={loadOpen}
        onSubmit={handleLoad}
        onUpload={handleUpload}
        maxUploadBytes={MAX_UPLOAD_BYTES}
        onClose={() => setLoadOpen(false)}
      />
      <TableDetailModal
        open={detailName !== null}
        name={detailName}
        fetchTable={fetchTable}
        onRename={handleRename}
        onClose={() => setDetailName(null)}
      />
      <PinWidgetModal
        open={pinSpec !== null}
        spec={pinSpec}
        sql={resultSql ?? active?.sql ?? ""}
        dashboards={dashboards}
        onClose={() => setPinSpec(null)}
        onPin={handlePin}
      />
      <DashboardModal
        open={dashboardOpenId !== null}
        dashboard={dashboards.find((d) => d.id === dashboardOpenId) ?? null}
        connected={connected}
        fetchWidget={fetchWidget}
        onClose={() => setDashboardOpenId(null)}
        onRemoveWidget={handleRemoveWidget}
      />
    </div>
  );
}