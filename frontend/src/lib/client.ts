// ryudb-server wire client: one WebSocket, JSON control frames + one Arrow IPC
// binary frame per successful result.
//
// The server's per-connection send lock (see ryudb/server/app.py) guarantees a
// result's binary frame is the very next frame after its meta frame, so we can
// associate each binary frame with the most recent pending result without any
// framing of our own. Requests are multiplexed by `id`; several may be in
// flight at once.

import { Table, tableFromIPC } from "apache-arrow";
import type { Request, Response, Result, RequestId } from "./types";

export type ConnStatus = "idle" | "connecting" | "open" | "closed" | "error";

interface Pending {
  resolve: (r: Result) => void;
  reject: (e: Error) => void;
}

export class RyuDBClient {
  private ws: WebSocket | null = null;
  private url: string;
  private pending = new Map<string, Pending>();
  private awaitingBin: { id: string; meta: Response } | null = null;
  private idCounter = 0;
  onStatus: ((s: ConnStatus) => void) | null = null;

  constructor(url: string) {
    this.url = url;
  }

  get status(): ConnStatus {
    if (this.ws === null) return "idle";
    switch (this.ws.readyState) {
      case WebSocket.CONNECTING:
        return "connecting";
      case WebSocket.OPEN:
        return "open";
      case WebSocket.CLOSING:
      case WebSocket.CLOSED:
        return "closed";
      default:
        return "idle";
    }
  }

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      this.disconnect();
      this.setStatus("connecting");
      const ws = new WebSocket(this.url);
      ws.binaryType = "arraybuffer";
      this.ws = ws;
      ws.onopen = () => {
        this.setStatus("open");
        resolve();
      };
      ws.onerror = () => {
        this.setStatus("error");
        reject(new Error(`WebSocket error connecting to ${this.url}`));
      };
      ws.onclose = () => {
        this.setStatus("closed");
        this.failAll(new Error("connection closed"));
      };
      ws.onmessage = (ev) => this.onMessage(ev);
    });
  }

  disconnect(): void {
    if (this.ws !== null) {
      this.ws.onclose = null;
      this.ws.onerror = null;
      this.ws.onmessage = null;
      try {
        this.ws.close();
      } catch {
        /* ignore */
      }
      this.ws = null;
    }
    this.failAll(new Error("disconnected"));
    this.awaitingBin = null;
  }

  /** Send a request and resolve with the matching response (+ decoded Arrow
   * table for results). The `id` is assigned here if the request has none. */
  request(req: Request): Promise<Result> {
    const ws = this.ws;
    if (ws === null || ws.readyState !== WebSocket.OPEN) {
      return Promise.reject(new Error("not connected"));
    }
    const id = req.id ?? `r${this.idCounter++}`;
    const key = String(id);
    const out: Request = { ...req, id } as Request;
    return new Promise<Result>((resolve, reject) => {
      this.pending.set(key, { resolve, reject });
      ws.send(JSON.stringify(out));
    });
  }

  /** Convenience: run a SQL statement. */
  sql(sql: string, maxRows?: number): Promise<Result> {
    return this.request({ id: null, op: "sql", sql, max_rows: maxRows });
  }

  /** Convenience: open a cursor-backed SQL query (first page + cursor_id). */
  sqlCursor(sql: string, maxRows?: number): Promise<Result> {
    return this.request({ id: null, op: "sql", sql, max_rows: maxRows, cursor: true });
  }

  /** Convenience: fetch the next page of a cursor. */
  fetch(cursorId: string, offset: number, limit?: number): Promise<Result> {
    return this.request({ id: null, op: "fetch", cursor_id: cursorId, offset, limit });
  }

  /** Convenience: close a cursor (best-effort; never throws on unknown id). */
  closeCursor(cursorId: string): Promise<Result> {
    return this.request({ id: null, op: "close", cursor_id: cursorId });
  }

  private onMessage(ev: MessageEvent): void {
    const data = ev.data;
    if (typeof data === "string") {
      let frame: Response;
      try {
        frame = JSON.parse(data) as Response;
      } catch {
        return; // ignore malformed
      }
      this.handleText(frame);
    } else if (data instanceof ArrayBuffer) {
      this.handleBinary(data);
    }
    // Blob never happens: binaryType is arraybuffer.
  }

  private handleText(frame: Response): void {
    // A `result` (Arrow IPC) or `export` (raw blob, e.g. Parquet) meta is
    // followed by its binary frame — defer resolving until it arrives.
    // (Defensive: if a text frame sneaks in while awaiting a binary, the binary
    // was lost — resolve the pending result with a null payload.)
    if (this.awaitingBin !== null && frame.op !== "result" && frame.op !== "export") {
      const { id, meta } = this.awaitingBin;
      this.awaitingBin = null;
      this.resolve(id, { meta, table: null, bytes: undefined });
    }
    if (frame.op === "result" || frame.op === "export") {
      const id = String(frame.id);
      this.awaitingBin = { id, meta: frame };
      return;
    }
    const id = frame.id;
    if (id === null || id === undefined) return; // server-initiated, no id
    this.resolve(String(id), { meta: frame, table: null });
  }

  private handleBinary(buf: ArrayBuffer): void {
    const awaiting = this.awaitingBin;
    if (awaiting === null) return; // orphan binary, ignore
    this.awaitingBin = null;
    if (awaiting.meta.op === "export") {
      // Raw bytes (Parquet) — NOT Arrow IPC; keep as a blob for download.
      this.resolve(awaiting.id, { meta: awaiting.meta, table: null, bytes: new Uint8Array(buf) });
      return;
    }
    const table = decodeIpc(buf);
    this.resolve(awaiting.id, { meta: awaiting.meta, table });
  }

  private resolve(key: string, result: Result): void {
    const p = this.pending.get(key);
    if (p) {
      this.pending.delete(key);
      p.resolve(result);
    }
  }

  private failAll(err: Error): void {
    for (const p of this.pending.values()) p.reject(err);
    this.pending.clear();
  }

  private setStatus(s: ConnStatus): void {
    if (this.onStatus) this.onStatus(s);
  }
}

/** Decode an Arrow IPC *stream* (what ryudb-server writes) into a Table. */
export function decodeIpc(buf: ArrayBuffer): Table | null {
  try {
    const bytes = new Uint8Array(buf);
    // tableFromIPC auto-detects file vs stream by the footer magic; the server
    // writes a stream (no footer), so this resolves to the stream reader.
    return tableFromIPC(bytes);
  } catch (e) {
    console.error("Arrow IPC decode failed:", e);
    return null;
  }
}

export type { RequestId };