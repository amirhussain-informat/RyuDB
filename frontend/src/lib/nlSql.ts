// Schema-aware natural-language -> SQL heuristic for the offline NL->SQL assistant.
//
// RyuDB runs fully offline (no CDN, no auth, no hosted LLM), so the NL->SQL
// assistant is a RECOGNIZER over the live catalog schema rather than a model
// call: it matches a small grammar of common analytical questions (count,
// top/bottom N by a column, avg/sum/min/max of a column, group-by, select
// columns, filter) and emits a SELECT, resolving table + column names FUZZILY
// against the schema (case-insensitive, then substring, then a small edit
// distance) so "top 10 lineitem by extendedprice" works without the exact
// `l_`-prefixed name. It returns the SQL plus which template matched (so the UI
// can show "interpreted as …"); an unrecognized question returns null and the
// caller falls back to a hint. This is honest about its limits — it is NOT a
// general NL->SQL model — but it covers the commonasks over a known schema with
// no network dependency, and the matched SQL lands in the editor for the user
// to refine before running.
//
// Pure + dependency-free so it bundles under esbuild for a hermetic test
// (test/nlSql_check.mjs), like csv / autocomplete / planLayout / worksheetTransfer.

export interface NlColumn {
  name: string;
  type: string;
}
export interface NlTable {
  name: string;
  columns: NlColumn[];
}
export interface NlResult {
  sql: string;
  /** A short human label for the matched template (shown by the UI). */
  template: string;
  /** The resolved table name (as it appears in the schema). */
  table: string;
}

const NUMERIC_RE = /(int|float|double|decimal|numeric|real|long)/i;

export function isNumericType(type: string): boolean {
  return NUMERIC_RE.test(type);
}

/** Lowercase, collapse whitespace, strip most punctuation but keep words,
 *  numbers, dots (for qualified names), and comparison operators. */
export function normalize(s: string): string {
  return s
    .toLowerCase()
    .replace(/[?.,;!]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

/** Classic two-row edit distance (Levenshtein). Bounded by the longer string;
 *  only used for short tokens (table/column names), so the cost is tiny. */
export function editDistance(a: string, b: string): number {
  if (a === b) return 0;
  if (a.length === 0) return b.length;
  if (b.length === 0) return a.length;
  let prev = new Array(b.length + 1);
  let curr = new Array(b.length + 1);
  for (let j = 0; j <= b.length; j++) prev[j] = j;
  for (let i = 1; i <= a.length; i++) {
    curr[0] = i;
    for (let j = 1; j <= b.length; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      curr[j] = Math.min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost);
    }
    [prev, curr] = [curr, prev];
  }
  return prev[b.length];
}

/** Find the best-matching table for a token. Exact (case-insensitive) wins;
 *  else a name that starts with the token; else one that contains it; else the
 *  smallest edit distance <= 2. Returns null when nothing is close. */
export function findTable(tables: NlTable[], token: string): NlTable | null {
  const t = token.toLowerCase();
  if (!t) return null;
  let best: NlTable | null = null;
  let bestDist = Infinity;
  for (const tbl of tables) {
    const name = tbl.name.toLowerCase();
    if (name === t) return tbl;
    if (name.startsWith(t) && (!best || tbl.name.length < best.name.length)) best = tbl;
    if (name.includes(t) && (!best || tbl.name.length < best.name.length)) best = tbl;
    const d = editDistance(t, name);
    if (d <= 2 && d < bestDist) { bestDist = d; best = tbl; }
  }
  return best;
}

/** Find the best-matching column within a table by token (same ranking as
 *  findTable, scoped to this table's columns). */
export function findColumn(table: NlTable, token: string): NlColumn | null {
  const t = token.toLowerCase();
  if (!t) return null;
  let best: NlColumn | null = null;
  let bestDist = Infinity;
  for (const col of table.columns) {
    const name = col.name.toLowerCase();
    if (name === t) return col;
    // Match on the last segment of a dotted name too (l_extendedprice <- extendedprice).
    const tail = name.includes(".") ? name.split(".").pop()! : name;
    if ((name.startsWith(t) || tail === t || tail.startsWith(t)) && (!best || col.name.length < best.name.length)) {
      best = col;
    }
    if (name.includes(t) && (!best || col.name.length < best.name.length)) best = col;
    const d = Math.min(editDistance(t, name), editDistance(t, tail));
    if (d <= 2 && d < bestDist) { bestDist = d; best = col; }
  }
  return best;
}

/** Quote an identifier if it needs quoting (contains non-alphanumeric/_ or is
 *  reserved-ish). Conservative: only quotes when necessary. */
export function quoteIdent(name: string): string {
  return /^[A-Za-z_][A-Za-z0-9_]*$/.test(name) ? name : `"${name.replace(/"/g, '""')}"`;
}

function numberFrom(tokens: string[], fallback: number): number {
  for (const tk of tokens) {
    const n = parseInt(tk, 10);
    if (Number.isFinite(n) && n > 0) return n;
  }
  return fallback;
}

/** Tokenize the normalized question into words (operators kept by normalize
 *  only as spaces, so tokens are pure words/numbers/dots). */
function words(q: string): string[] {
  return q.split(/\s+/).filter(Boolean);
}

/** Try to recognize the question against the schema. Returns the SQL + matched
 *  template + resolved table, or null when no pattern matches. Patterns are
 *  tried in priority order; the first match wins. */
export function nlToSql(question: string, tables: NlTable[]): NlResult | null {
  const q = normalize(question);
  if (!q || tables.length === 0) return null;
  const w = words(q);
  const has = (...needles: string[]) => needles.every((n) => q.includes(n));
  const hasAny = (...needles: string[]) => needles.some((n) => q.includes(n));

  // Resolve the table mentioned anywhere in the question (first table that
  // matches some token). Several patterns need this; do it once.
  const resolveTable = (): NlTable | null => {
    for (const tk of w) {
      const tbl = findTable(tables, tk);
      if (tbl) return tbl;
    }
    return null;
  };

  // 1. "how many rows in <table>" / "count <table>" / "count rows"
  if (hasAny("how many rows", "count rows", "number of rows") || /^count\b/.test(q)) {
    const tbl = resolveTable();
    if (tbl) {
      return { sql: `SELECT count(*) FROM ${quoteIdent(tbl.name)};`, template: "row count", table: tbl.name };
    }
  }
  if (has("count", "*") || hasAny("count all")) {
    const tbl = resolveTable();
    if (tbl) return { sql: `SELECT count(*) FROM ${quoteIdent(tbl.name)};`, template: "row count", table: tbl.name };
  }

  // 2. "top N <table> by <col>" / "largest N" / "biggest" -> ORDER BY <col> DESC LIMIT N
  if (hasAny("top", "largest", "biggest", "highest", "max ")) {
    const tbl = resolveTable();
    if (tbl) {
      const byIdx = w.findIndex((x) => x === "by");
      let col: NlColumn | null = null;
      if (byIdx >= 0 && byIdx + 1 < w.length) col = findColumn(tbl, w.slice(byIdx + 1).join(" "));
      if (!col) col = firstNumeric(tbl);
      const n = numberFrom(w, 10);
      if (col) {
        return {
          sql: `SELECT * FROM ${quoteIdent(tbl.name)} ORDER BY ${quoteIdent(col.name)} DESC LIMIT ${n};`,
          template: `top ${n} by ${col.name}`,
          table: tbl.name,
        };
      }
    }
  }

  // 3. "bottom N <table> by <col>" / "smallest" / "lowest" -> ORDER BY ASC
  if (hasAny("bottom", "smallest", "lowest", "cheapest")) {
    const tbl = resolveTable();
    if (tbl) {
      const byIdx = w.findIndex((x) => x === "by");
      let col: NlColumn | null = null;
      if (byIdx >= 0 && byIdx + 1 < w.length) col = findColumn(tbl, w.slice(byIdx + 1).join(" "));
      if (!col) col = firstNumeric(tbl);
      const n = numberFrom(w, 10);
      if (col) {
        return {
          sql: `SELECT * FROM ${quoteIdent(tbl.name)} ORDER BY ${quoteIdent(col.name)} ASC LIMIT ${n};`,
          template: `bottom ${n} by ${col.name}`,
          table: tbl.name,
        };
      }
    }
  }

  // 4. "average|avg|mean of <col> in <table>" / "sum|min|max of <col> in <table>"
  //    The "of" is optional ("avg extendedprice in lineitem").
  const aggIdx = w.findIndex((x) => /^(average|avg|mean|sum|min|max)$/.test(x));
  if (aggIdx >= 0) {
    const word = w[aggIdx];
    const fn = word === "average" || word === "mean" || word === "avg" ? "avg" : word;
    const tbl = resolveTable();
    if (tbl) {
      // The column token is the first non-stopword after the agg word, skipping
      // a leading "of". Fall back to the first numeric column.
      let colIdx = aggIdx + 1;
      if (w[colIdx] === "of") colIdx++;
      let col: NlColumn | null = null;
      if (w[colIdx]) col = findColumn(tbl, w[colIdx]);
      if (!col) col = firstNumeric(tbl);
      if (col) {
        return {
          sql: `SELECT ${fn}(${quoteIdent(col.name)}) FROM ${quoteIdent(tbl.name)};`,
          template: `${fn}(${col.name})`,
          table: tbl.name,
        };
      }
    }
  }

  // 5. "group <table> by <col>" / "<table> by <col>" -> SELECT <col>, count(*)
  if (q.includes("group by") || q.includes("grouped by") || /^group\b/.test(q)) {
    const tbl = resolveTable();
    if (tbl) {
      const byIdx = w.findIndex((x) => x === "by");
      let col: NlColumn | null = null;
      if (byIdx >= 0 && byIdx + 1 < w.length) col = findColumn(tbl, w.slice(byIdx + 1).join(" "));
      if (!col) col = tbl.columns[0] ?? null;
      if (col) {
        return {
          sql: `SELECT ${quoteIdent(col.name)}, count(*) FROM ${quoteIdent(tbl.name)} GROUP BY ${quoteIdent(col.name)} ORDER BY ${quoteIdent(col.name)};`,
          template: `group by ${col.name}`,
          table: tbl.name,
        };
      }
    }
  }

  // 6. "show me <table>" / "select * from <table>" / "list <table>"
  if (hasAny("show", "list", "display", "select *", "all rows", "everything")) {
    const tbl = resolveTable();
    if (tbl) {
      const n = numberFrom(w, 100);
      return {
        sql: `SELECT * FROM ${quoteIdent(tbl.name)} LIMIT ${n};`,
        template: `sample ${n} rows`,
        table: tbl.name,
      };
    }
  }

  // 7. Last resort: if exactly one table is resolvable, sample it.
  const tbl = resolveTable();
  if (tbl) {
    return {
      sql: `SELECT * FROM ${quoteIdent(tbl.name)} LIMIT 100;`,
      template: "sample 100 rows",
      table: tbl.name,
    };
  }
  return null;
}

function firstNumeric(tbl: NlTable): NlColumn | null {
  for (const c of tbl.columns) if (isNumericType(c.type)) return c;
  return null;
}