import type { TableColumn } from "./types";

/** The catalog schema the autocompleter uses: table name -> its typed columns. */
export type Schema = Map<string, TableColumn[]>;

// SQL keywords offered when no table/column prefix is typed. Not exhaustive —
// enough to surface the common statement/option scaffolding ahead of the schema.
export const SQL_KEYWORDS = [
  "SELECT", "FROM", "WHERE", "JOIN", "INNER", "LEFT", "RIGHT", "FULL", "OUTER",
  "CROSS", "ON", "GROUP", "BY", "ORDER", "HAVING", "LIMIT", "OFFSET", "AS", "AND",
  "OR", "NOT", "NULL", "IS", "IN", "BETWEEN", "LIKE", "DISTINCT", "COUNT", "SUM",
  "AVG", "MIN", "MAX", "ASC", "DESC", "UNION", "ALL", "CASE", "WHEN", "THEN", "ELSE",
  "END", "CREATE", "TABLE", "DROP", "ALTER", "INSERT", "INTO", "VALUES", "UPDATE",
  "SET", "DELETE", "WITH", "CAST", "DATE", "TIMESTAMP", "TRUE", "FALSE",
];

export type SuggestionKind = "keyword" | "table" | "column";

export interface SqlSuggestion {
  label: string;
  kind: SuggestionKind;
  insertText: string;
  detail?: string;
  /** Monaco sortText: "0" tables, "1" columns, "2" keywords — tables first. */
  sortText: string;
}

/** Quote an identifier for insertion if it isn't a plain word (handles spaces,
 *  hyphens, reserved words); otherwise insert it bare. */
export function quoteIdent(name: string): string {
  return /^[A-Za-z_][A-Za-z0-9_]*$/.test(name) ? name : `"${name.replace(/"/g, '""')}"`;
}

/** Pure suggestion builder (no monaco dependency) so it is unit-testable.
 *  `lineUntil` is the source text from column 1 up to the start of the word at
 *  the cursor. After `<ident>.` it offers only that table's columns; otherwise
 *  keywords + table names + every column (deduped), with tables ranked first. */
export function buildSqlSuggestions(lineUntil: string, schema: Schema | undefined): SqlSuggestion[] {
  const suggestions: SqlSuggestion[] = [];
  const qualified = lineUntil.match(/([A-Za-z_][A-Za-z0-9_]*)\.\s*$/);
  if (qualified) {
    const cols = schema?.get(qualified[1]);
    if (cols) {
      for (const c of cols) {
        suggestions.push({
          label: c.name,
          kind: "column",
          insertText: quoteIdent(c.name),
          detail: `${c.type}${c.nullable ? " • nullable" : ""}`,
          sortText: "0" + c.name,
        });
      }
    }
    return suggestions;
  }
  for (const kw of SQL_KEYWORDS) {
    suggestions.push({ label: kw, kind: "keyword", insertText: kw, sortText: "2" + kw });
  }
  if (schema) {
    for (const [tname, cols] of schema) {
      suggestions.push({
        label: tname,
        kind: "table",
        insertText: quoteIdent(tname),
        detail: `${cols.length} columns`,
        sortText: "0" + tname,
      });
    }
    const seen = new Set<string>();
    for (const cols of schema.values()) {
      for (const c of cols) {
        if (seen.has(c.name)) continue;
        seen.add(c.name);
        suggestions.push({
          label: c.name,
          kind: "column",
          insertText: quoteIdent(c.name),
          detail: c.type,
          sortText: "1" + c.name,
        });
      }
    }
  }
  return suggestions;
}