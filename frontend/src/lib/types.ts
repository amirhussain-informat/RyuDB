// Wire-protocol types mirroring ryudb/server/PROTOCOL.md.
//
// Text frames are JSON; a successful `sql`/`sample` result is followed by one
// binary Arrow IPC stream frame. Every response echoes the request `id`
// (except server-initiated protocol errors on an unparseable request, which
// carry no `id`).

import type { ChartKind } from "./chartRender";

export type RequestId = string | number | null;

export interface ColumnMeta {
  name: string;
  type: string;
}

export interface PlanNode {
  op: string;
  est_rows: number | null;
  fused: boolean;
  detail: Record<string, unknown>;
  children: PlanNode[];
}

// ---- requests ----

export interface SqlRequest {
  id: RequestId;
  op: "sql";
  sql: string;
  max_rows?: number;
  cursor?: boolean;
}
export interface FetchRequest {
  id: RequestId;
  op: "fetch";
  cursor_id: string;
  offset?: number;
  limit?: number;
}
export interface CloseRequest {
  id: RequestId;
  op: "close";
  cursor_id: string;
}
export interface ExplainRequest {
  id: RequestId;
  op: "explain";
  sql: string;
}
export interface CatalogRequest {
  id: RequestId;
  op: "catalog";
}
export interface TableRequest {
  id: RequestId;
  op: "table";
  name: string;
}
export interface SampleRequest {
  id: RequestId;
  op: "sample";
  name: string;
  n?: number;
}
export interface AdminRequest {
  id: RequestId;
  op: "admin";
  action: string;
  args?: Record<string, unknown>;
}
export interface CancelRequest {
  id: RequestId;
  op: "cancel";
  targets: string[];
}
export interface HistoryRequest {
  id: RequestId;
  op: "history";
}
export interface ProfileRequest {
  id: RequestId;
  op: "profile";
  name: string;
  top_k?: number;
}
export interface ExportRequest {
  id: RequestId;
  op: "export";
  sql: string;
  format?: "parquet";
}
/** Two-frame request: this text meta is followed by one binary frame with the
 *  parquet bytes (browser file-upload ingest). `format` must be "parquet". */
export interface UploadRequest {
  id: RequestId;
  op: "upload";
  name: string;
  format?: "parquet";
}
/** Run a Python notebook cell on the worker; a `sql()` helper + `pd`/`cudf`
 *  are injected. Text-only response (no binary frame). */
export interface PyRequest {
  id: RequestId;
  op: "py";
  code: string;
}
export type Request =
  | SqlRequest
  | FetchRequest
  | CloseRequest
  | ExplainRequest
  | CatalogRequest
  | TableRequest
  | SampleRequest
  | AdminRequest
  | CancelRequest
  | HistoryRequest
  | ProfileRequest
  | ExportRequest
  | UploadRequest
  | PyRequest;

// ---- responses ----

export interface ResultMeta {
  id: RequestId;
  op: "result";
  columns: ColumnMeta[];
  row_count: number;
  returned: number;
  truncated: boolean;
  duration_ms: number;
  frame_count: number;
  // Present when this result is the first page of a cursor (a `sql` request
  // with `cursor: true`) or a `fetch` page. `truncated` doubles as "has more".
  cursor_id?: string;
  offset?: number;
  // Present when a `cursor: true` request's result exceeded --max-cursor-rows
  // and was served as a plain truncated result (no cursor; the rest is not
  // pageable).
  cursor?: boolean;
  reason?: string;
}
export interface WriteResp {
  id: RequestId;
  op: "write";
  rows_affected: number;
  duration_ms: number;
}
export interface OkResp {
  id: RequestId;
  op: "ok";
  detail?: Record<string, unknown>;
  duration_ms?: number;
}
export interface PlanResp {
  id: RequestId;
  op: "plan";
  tree: PlanNode;
}
export interface CatalogTable {
  name: string;
  row_count: number;
  columns: string[];
}
export interface CatalogResp {
  id: RequestId;
  op: "catalog";
  tables: CatalogTable[];
}
export interface TableColumn {
  name: string;
  type: string;
  nullable: boolean;
}
export interface TableResp {
  id: RequestId;
  op: "table";
  name: string;
  columns: TableColumn[];
  constraints: Record<string, unknown>;
  paths: string[];
  row_count: number;
}
export interface HistoryEntry {
  id: RequestId;
  sql: string;
  duration_ms: number;
  rows: number;
  kind: string;
  /** Wall-clock epoch seconds when the entry was recorded (session timestamp). */
  ts?: number;
}
export interface HistoryResp {
  id: RequestId;
  op: "history";
  entries: HistoryEntry[];
}
export interface ProfileBucket {
  lo: number;
  hi: number;
  count: number;
}
export interface ProfileTopValue {
  value: string | number | boolean | null;
  count: number;
}
export interface ProfileColumn {
  name: string;
  type: string;
  row_count: number;
  null_count: number;
  null_pct: number;
  distinct: number;
  min?: string | number | boolean | null;
  max?: string | number | boolean | null;
  mean?: number | null;
  stddev?: number | null;
  histogram?: ProfileBucket[] | null;
  top?: ProfileTopValue[] | null;
}
export interface ProfileResp {
  id: RequestId;
  op: "profile";
  name: string;
  row_count: number;
  columns: ProfileColumn[];
}
export interface ExportResp {
  id: RequestId;
  op: "export";
  format: string;
  row_count: number;
  byte_count: number;
  duration_ms: number;
}
export interface CancelledResp {
  id: RequestId;
  op: "cancelled";
}
/** A Python notebook cell's result. `result` is the repr() of an expression
 *  cell (null for a statement block); `error` carries the formatted traceback
 *  when the cell raised (still a `py` frame, not an `error` frame). */
export interface PyResp {
  id: RequestId;
  op: "py";
  stdout: string;
  result: string | null;
  error: string | null;
  duration_ms: number;
}
export interface ErrorPosition {
  line: number;
  col: number;
}
export interface ErrorResp {
  id: RequestId;
  op: "error";
  kind: "parse" | "runtime" | "protocol";
  message: string;
  position?: ErrorPosition;
}
export type Response =
  | ResultMeta
  | WriteResp
  | OkResp
  | PlanResp
  | CatalogResp
  | TableResp
  | HistoryResp
  | ProfileResp
  | ExportResp
  | CancelledResp
  | PyResp
  | ErrorResp;

// A parsed result (meta + decoded Arrow table). `table` is null for non-result
// responses and for results whose binary frame failed to decode. `bytes` is set
// for `export` responses (a raw binary blob — Parquet — that is NOT Arrow IPC
// and so is kept as bytes rather than decoded into a Table).
export interface Result {
  meta: Response;
  table: import("apache-arrow").Table | null;
  bytes?: Uint8Array;
}

// ---- dashboards (client-only, persisted to localStorage) ----
// A saved chart visualization. The geometry/axis choices a user made on the
// Chart tab, captured by name so they survive a result changing shape. Reused
// by `useDashboards` and the headless `ChartView`.
export interface ChartSpec {
  kind: ChartKind;
  xCol: string;
  yCol: string;
}
export interface DashboardWidget {
  id: string;
  title: string;
  sql: string;
  chart: ChartSpec;
}
export interface Dashboard {
  id: string;
  name: string;
  widgets: DashboardWidget[];
}

// ---- notebooks (client-only, persisted to localStorage) ----
// A SQL+Python notebook: an ordered list of cells. A `sql` cell runs a
// statement through the `sql` op and shows a mini result table; a `python`
// cell runs through the `py` op (with an injected `sql()` helper) and shows
// captured stdout + a result repr (+ traceback on error). Cell outputs are
// transient (not persisted) — only the cell sources are saved.
export type NotebookCellType = "sql" | "python";
export interface NotebookCell {
  id: string;
  type: NotebookCellType;
  code: string;
}
export interface Notebook {
  id: string;
  name: string;
  cells: NotebookCell[];
}