// Wire-protocol types mirroring ryudb/server/PROTOCOL.md.
//
// Text frames are JSON; a successful `sql`/`sample` result is followed by one
// binary Arrow IPC stream frame. Every response echoes the request `id`
// (except server-initiated protocol errors on an unparseable request, which
// carry no `id`).

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
export type Request =
  | SqlRequest
  | ExplainRequest
  | CatalogRequest
  | TableRequest
  | SampleRequest
  | AdminRequest
  | CancelRequest
  | HistoryRequest;

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
}
export interface HistoryResp {
  id: RequestId;
  op: "history";
  entries: HistoryEntry[];
}
export interface CancelledResp {
  id: RequestId;
  op: "cancelled";
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
  | CancelledResp
  | ErrorResp;

// A parsed result (meta + decoded Arrow table). `table` is null for non-result
// responses and for results whose binary frame failed to decode.
export interface Result {
  meta: Response;
  table: import("apache-arrow").Table | null;
}