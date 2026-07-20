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

/** Convert an Arrow cell to a JSON-ready value: bigint → number when it fits in
 * a safe integer, else its string (JSON.stringify would throw on bigint);
 * Date → ISO string; null stays null; nested objects pass through. */
function cellToJson(v: unknown): unknown {
  if (v === null || v === undefined) return null;
  if (typeof v === "bigint") {
    const n = Number(v);
    return Number.isSafeInteger(n) ? n : v.toString();
  }
  if (v instanceof Date) return v.toISOString();
  return v;
}

/** Serialize an Arrow Table as a JSON array of row objects (one field per
 * column). bigint cells outside the safe-integer range become strings. */
export function tableToJSON(table: Table): string {
  const fields = table.schema.fields;
  const cols = fields.map((f) => table.getChild(f.name));
  const rows: Record<string, unknown>[] = [];
  for (let i = 0; i < table.numRows; i++) {
    const obj: Record<string, unknown> = {};
    for (let f = 0; f < fields.length; f++) {
      const c = cols[f];
      obj[fields[f].name] = cellToJson(c ? c.get(i) : null);
    }
    rows.push(obj);
  }
  return JSON.stringify(rows, null, 2);
}

/** A TSV (tab-separated) serialization for clipboard copy — pastes cleanly
 * into Excel / Google Sheets. Uses the same null-aware cell rendering as CSV
 * but tabs as separators and no quoting (TSV trad. doesn't quote; embedded
 * tabs/newlines are replaced with a space so a cell never spans columns/rows). */
function cellToTsv(v: unknown): string {
  if (v === null || v === undefined) return "";
  let s: string;
  if (typeof v === "bigint") s = v.toString();
  else if (v instanceof Date) s = v.toISOString();
  else if (typeof v === "object") {
    try { s = JSON.stringify(v); } catch { s = String(v); }
  } else s = String(v);
  return s.replace(/[\t\r\n]/g, " ");
}

export function tableToTSV(table: Table): string {
  const fields = table.schema.fields;
  const cols = fields.map((f) => table.getChild(f.name));
  const parts: string[] = [fields.map((f) => f.name.replace(/[\t\r\n]/g, " ")).join("\t")];
  for (let i = 0; i < table.numRows; i++) {
    parts.push(cols.map((c) => cellToTsv(c ? c.get(i) : null)).join("\t"));
  }
  return parts.join("\n");
}

/** TSV serialization of a *view* — an explicit list of source-row indices into
 * the column VECTORS (already filtered + sorted by the caller), so clipboard
 * copy respects the grid's current sort/filter rather than always copying the
 * raw server order. Uses `vector.get(r)` (not `.toArray()`) so SQL NULLs are
 * preserved as empty fields instead of being coerced to 0/NaN. */
export function viewToTSV(
  names: string[],
  vectors: { get(i: number): unknown }[],
  view: number[],
): string {
  const parts: string[] = [names.map((n) => n.replace(/[\t\r\n]/g, " ")).join("\t")];
  for (const r of view) {
    parts.push(vectors.map((v) => cellToTsv(v.get(r))).join("\t"));
  }
  return parts.join("\n");
}

/** Write text to the clipboard, falling back to a hidden textarea +
 * execCommand when the async Clipboard API is unavailable (older browsers /
 * non-secure contexts). Returns whether it succeeded. */
export async function copyText(text: string): Promise<boolean> {
  if (navigator.clipboard && window.isSecureContext) {
    try { await navigator.clipboard.writeText(text); return true; } catch { /* fall through */ }
  }
  try {
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    const ok = document.execCommand("copy");
    ta.remove();
    return ok;
  } catch {
    return false;
  }
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