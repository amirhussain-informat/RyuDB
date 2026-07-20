// Pure, dependency-free serialize/parse for Git-backed worksheet export/import.
// Extracted from the UI so the format round-trips under a hermetic esbuild test
// (no React/DOM needed). The exchange format is plain `.sql` text so it is
// git-diffable and mergeable:
//
//   - A SINGLE worksheet exports as its raw SQL (a runnable .sql file). The
//     filename carries the worksheet name.
//   - ALL worksheets export as one bundle: a header comment + one
//     `-- @@worksheet: <name>` separator line per worksheet, followed by that
//     worksheet's SQL. The bundle is still plain text (one file, git-friendly).
//
// Import reads one or many .sql files. A file with `-- @@worksheet:` headers is
// split into one worksheet per section; a plain .sql file becomes one worksheet
// named from the file stem. Round-trip (export all -> import) preserves the
// worksheets' names and SQL byte-for-byte (modulo a trailing newline).

export interface WorksheetDoc {
  name: string;
  sql: string;
}

const BUNDLE_HEADER = "-- ryudb worksheet bundle";
// A header line looks like:  -- @@worksheet: <name>
const HEADER_RE = /^--\s*@@worksheet:\s*(.*)$/;

/** Strip path separators and other shell/git-unfriendly chars from a worksheet
 *  name so it is safe as a filename. Falls back to "worksheet" when empty. */
export function sanitizeFileName(name: string): string {
  const base = name.trim().replace(/[/\\:*?"<>|]+/g, "_").replace(/\s+/g, " ").trim();
  return base.length > 0 ? base.slice(0, 80) : "worksheet";
}

/** The `.sql` filename for a single-worksheet export. */
export function worksheetFileName(name: string): string {
  return `${sanitizeFileName(name)}.sql`;
}

/** Serialize one worksheet as raw SQL (a runnable .sql file). Ensures the
 *  content ends with a single trailing newline. */
export function serializeOne(sql: string): string {
  return ensureTrailingNewline(sql);
}

/** Serialize a bundle of worksheets. Each section begins with
 *  `-- @@worksheet: <name>`; the SQL follows until the next header. */
export function serializeBundle(worksheets: WorksheetDoc[]): string {
  const lines: string[] = [BUNDLE_HEADER, `-- ${worksheets.length} worksheet(s)`];
  for (const w of worksheets) {
    lines.push(`-- @@worksheet: ${w.name}`);
    const body = ensureTrailingNewline(w.sql);
    // A trailing newline before the next header keeps sections visually separate
    // and is stripped on parse (parseBundle trims each section).
    lines.push(body);
  }
  return lines.join("\n") + "\n";
}

/** Parse a bundle (or a plain .sql file) into worksheet docs. A line starting
 *  with `-- @@worksheet:` is ALWAYS a section separator — a worksheet whose SQL
 *  contains such a line will be split on import (a documented limitation; the
 *  marker is unusual enough that real SQL does not contain it). If the text has
 *  NO header lines at all, returns a single doc with an empty name (the caller
 *  names it from the filename). */
export function parseBundle(text: string): WorksheetDoc[] {
  const lines = text.split(/\r?\n/);
  // A bundle is either text we serialized (starts with the bundle marker) or
  // any text containing a `-- @@worksheet:` header. An empty bundle (marker +
  // count, no sections) parses to zero docs; a plain .sql file with no marker
  // and no headers parses to one empty-named doc the caller names from the file.
  const isBundle = text.startsWith(BUNDLE_HEADER) || lines.some((l) => HEADER_RE.test(l));
  if (!isBundle) {
    return [{ name: "", sql: trimTrailingBlank(lines).join("\n") }];
  }
  const docs: WorksheetDoc[] = [];
  let curName: string | null = null;
  let curLines: string[] = [];
  const flush = () => {
    if (curName !== null) {
      docs.push({ name: curName, sql: trimTrailingBlank(curLines).join("\n") });
    }
    curName = null;
    curLines = [];
  };
  for (const line of lines) {
    const m = HEADER_RE.exec(line);
    if (m) {
      flush();
      curName = m[1].trim();
    } else if (curName !== null) {
      curLines.push(line);
    }
    // Lines before the first header (the bundle header comments) are dropped.
  }
  flush();
  return docs;
}

/** Parse an imported file. A bundle (with `-- @@worksheet:` headers) yields its
 *  sections; a plain .sql file yields one doc named from `fileName`'s stem. */
export function parseImportFile(fileName: string, text: string): WorksheetDoc[] {
  const docs = parseBundle(text);
  if (docs.length === 1 && docs[0].name === "") {
    return [{ name: fileStem(fileName), sql: docs[0].sql }];
  }
  return docs;
}

/** Extract the file stem (basename without extension) for naming an imported
 *  worksheet. `lineitem q1.sql` -> `lineitem q1`. */
export function fileStem(fileName: string): string {
  const base = fileName.replace(/^.*[\\/]/, "");
  const dot = base.lastIndexOf(".");
  return dot > 0 ? base.slice(0, dot) : base;
}

function ensureTrailingNewline(s: string): string {
  return s.length === 0 ? "" : (s.endsWith("\n") ? s : s + "\n");
}

function trimTrailingBlank(lines: string[]): string[] {
  let end = lines.length;
  while (end > 0 && lines[end - 1].trim() === "") end--;
  return lines.slice(0, end);
}