// Line-level diff for the SQL "compare" view. A classic LCS (longest common
// subsequence) dynamic-programming diff over the two line arrays, producing a
// sequence of equal / added / deleted lines with 1-based line numbers on each
// side so a side-by-side view can render gutter numbers. Pure + dependency-free
// so it bundles under esbuild for a hermetic test (test/diff_check.mjs).
//
// SQL queries are short (hundreds of lines at most), so the O(n*m) DP table is
// fine; a Myers diff would be linear in space but is overkill here. Whitespace
// is significant for lines (a re-indented line counts as a change) — that is
// the right call for SQL where indentation can be meaningful, and the user can
// see exactly what moved.

export type DiffOp = "eq" | "add" | "del";

export interface DiffLine {
  op: DiffOp;
  /** 1-based line number in the left (a) text, for eq + del lines; null for add. */
  aLine: number | null;
  /** 1-based line number in the right (b) text, for eq + add lines; null for del. */
  bLine: number | null;
  text: string;
}

export interface DiffStats {
  added: number;
  removed: number;
  equal: number;
}

/** Split text into lines without a trailing empty element for a final newline. */
export function splitLines(s: string): string[] {
  if (s === "") return [];
  const lines = s.split(/\r?\n/);
  // "a\n" -> ["a", ""] ; drop the trailing "" so a terminal newline doesn't
  // invent an extra blank line.
  if (lines.length > 0 && lines[lines.length - 1] === "" && s.endsWith("\n")) {
    lines.pop();
  }
  return lines;
}

/** The LCS-length DP table for line arrays a (rows) and b (cols). dp[i][j] is
 *  the length of the LCS of a[i..] and b[j..]. The last row + column are 0. */
function lcsTable(a: string[], b: string[]): number[][] {
  const n = a.length, m = b.length;
  const dp: number[][] = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0));
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      dp[i][j] = a[i] === b[j]
        ? dp[i + 1][j + 1] + 1
        : Math.max(dp[i + 1][j], dp[i][j + 1]);
    }
  }
  return dp;
}

/** Produce the line diff of `a` (left) vs `b` (right) as a flat op sequence. */
export function diffLines(a: string, b: string): DiffLine[] {
  const A = splitLines(a);
  const B = splitLines(b);
  const dp = lcsTable(A, B);
  const out: DiffLine[] = [];
  let i = 0, j = 0;
  let aLine = 0, bLine = 0;
  while (i < A.length && j < B.length) {
    if (A[i] === B[j]) {
      aLine++; bLine++;
      out.push({ op: "eq", aLine, bLine, text: A[i] });
      i++; j++;
    } else if (dp[i + 1][j] >= dp[i][j + 1]) {
      aLine++;
      out.push({ op: "del", aLine, bLine: null, text: A[i] });
      i++;
    } else {
      bLine++;
      out.push({ op: "add", aLine: null, bLine, text: B[j] });
      j++;
    }
  }
  while (i < A.length) { aLine++; out.push({ op: "del", aLine, bLine: null, text: A[i++] }); }
  while (j < B.length) { bLine++; out.push({ op: "add", aLine: null, bLine, text: B[j++] }); }
  return out;
}

/** Summary counts of added / removed / equal lines. */
export function diffStats(diff: DiffLine[]): DiffStats {
  let added = 0, removed = 0, equal = 0;
  for (const d of diff) {
    if (d.op === "add") added++;
    else if (d.op === "del") removed++;
    else equal++;
  }
  return { added, removed, equal };
}

/** Split a flat diff into two parallel columns for a side-by-side view: the
 *  left column shows eq + del lines (with their a-line numbers); the right
 *  column shows eq + add lines (with their b-line numbers). Added lines have a
 *  blank on the left; deleted lines have a blank on the right. Returns aligned
 *  rows so the two columns have equal height. */
export interface SideRow {
  aLine: number | null;
  aText: string | null;
  aOp: DiffOp | null;
  bLine: number | null;
  bText: string | null;
  bOp: DiffOp | null;
}
export function toSideBySide(diff: DiffLine[]): SideRow[] {
  const rows: SideRow[] = [];
  for (const d of diff) {
    if (d.op === "eq") {
      rows.push({ aLine: d.aLine, aText: d.text, aOp: "eq", bLine: d.bLine, bText: d.text, bOp: "eq" });
    } else if (d.op === "del") {
      rows.push({ aLine: d.aLine, aText: d.text, aOp: "del", bLine: null, bText: null, bOp: null });
    } else {
      rows.push({ aLine: null, aText: null, aOp: null, bLine: d.bLine, bText: d.text, bOp: "add" });
    }
  }
  return rows;
}