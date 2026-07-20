// Serialize an Arrow Table to RFC-4180-ish CSV for client-side download.
//
// Null-aware: uses `vector.get(i)` (which returns null for null bitmap entries)
// rather than `vector.toArray()` (which renders null ints as 0). Strings are
// quoted only when they contain a comma, quote, or newline; quotes are doubled.
// bigint (Int64) and Date cells are stringified; nested/struct cells fall back
// to JSON.

import type { Table } from "apache-arrow";

function csvEscape(s: string): string {
  if (/[",\n\r]/.test(s)) {
    return '"' + s.replace(/"/g, '""') + '"';
  }
  return s;
}

function cellToCsv(v: unknown): string {
  if (v === null || v === undefined) return ""; // SQL NULL -> empty field
  if (typeof v === "bigint") return v.toString();
  if (v instanceof Date) return v.toISOString();
  if (typeof v === "object") {
    try {
      return csvEscape(JSON.stringify(v));
    } catch {
      return csvEscape(String(v));
    }
  }
  return csvEscape(String(v));
}

export function tableToCSV(table: Table): string {
  const fields = table.schema.fields;
  const cols = fields.map((f) => table.getChild(f.name));
  const n = table.numRows;
  const parts: string[] = [fields.map((f) => csvEscape(f.name)).join(",")];
  for (let i = 0; i < n; i++) {
    const row = cols.map((c) => cellToCsv(c ? c.get(i) : null));
    parts.push(row.join(","));
  }
  return parts.join("\n");
}

/** Trigger a browser download of `data` as `filename`. */
export function downloadBlob(filename: string, mime: string, data: Uint8Array | string): void {
  // Copy bytes into a fresh ArrayBuffer-backed Uint8Array: the TS 5.5 lib.dom
  // BlobPart type rejects a Uint8Array<ArrayBufferLike> (could be
  // SharedArrayBuffer-backed); a copy over a plain ArrayBuffer satisfies it.
  const part: BlobPart = typeof data === "string" ? data : new Uint8Array(data);
  const blob = new Blob([part], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}