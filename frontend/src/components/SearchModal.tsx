import { useEffect, useMemo, useRef, useState } from "react";
import type { CatalogTable, HistoryEntry } from "../lib/types";

interface Props {
  open: boolean;
  connected: boolean;
  fetchCatalog: () => Promise<CatalogTable[]>;
  fetchHistory: () => Promise<HistoryEntry[]>;
  onClose: () => void;
  /** Insert a table name into the active editor at the cursor. */
  onPickTable: (name: string) => void;
  /** Insert a column name into the active editor at the cursor. */
  onPickColumn: (table: string, column: string) => void;
  /** Load a history entry's SQL into the active worksheet. */
  onPickHistory: (sql: string) => void;
}

type Group = "Tables" | "Columns" | "History";

interface Item {
  key: string;
  label: string;
  sub: string;
  group: Group;
  run: () => void;
}

/** A global object search modal: type to fuzzy-filter across tables, columns,
 * and query history; ↑↓ to move, Enter to act, Esc to close. Reaches across
 * the whole catalog + history in one place (the sidebars show them in
 * isolation). */
export default function SearchModal({
  open, connected, fetchCatalog, fetchHistory, onClose,
  onPickTable, onPickColumn, onPickHistory,
}: Props) {
  const [query, setQuery] = useState("");
  const [sel, setSel] = useState(0);
  const [items, setItems] = useState<Item[]>([]);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Fetch catalog + history once per open (when connected). Cancels a stale
  // fetch if the modal closes mid-load.
  useEffect(() => {
    if (!open || !connected) return;
    let cancelled = false;
    setLoading(true);
    setErr(null);
    setItems([]);
    Promise.all([fetchCatalog(), fetchHistory()])
      .then(([tables, hist]) => {
        if (cancelled) return;
        const tablesItems: Item[] = tables.map((t) => ({
          key: "t:" + t.name,
          label: t.name,
          sub: `${t.row_count.toLocaleString()} rows`,
          group: "Tables",
          run: () => { onClose(); onPickTable(t.name); },
        }));
        const colItems: Item[] = tables.flatMap((t) =>
          (t.columns ?? []).map((c) => ({
            key: "c:" + t.name + "." + c,
            label: c,
            sub: t.name,
            group: "Columns" as Group,
            run: () => { onClose(); onPickColumn(t.name, c); },
          })),
        );
        const histItems: Item[] = hist.map((h, i) => ({
          key: "h:" + i,
          label: h.sql,
          sub: `${h.kind} · ${Math.round(h.duration_ms)} ms · ${h.rows} rows`,
          group: "History",
          run: () => { onClose(); onPickHistory(h.sql); },
        }));
        setItems([...tablesItems, ...colItems, ...histItems]);
      })
      .catch((e: unknown) => {
        if (cancelled) return;
        setErr((e as Error)?.message ?? String(e));
      })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
    // fetchers/pickers are stable enough (App rebuilds them each render, but
    // the effect only re-runs when `open"/"connected" toggles).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, connected]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return items;
    return items.filter((it) =>
      `${it.label} ${it.sub} ${it.group}`.toLowerCase().includes(q),
    );
  }, [query, items]);

  useEffect(() => { setSel(0); }, [query, filtered.length]);

  // Clear + focus on open.
  useEffect(() => {
    if (open) {
      setQuery("");
      setSel(0);
      setErr(null);
      const t = setTimeout(() => inputRef.current?.focus(), 0);
      return () => clearTimeout(t);
    }
  }, [open]);

  if (!open) return null;

  const exec = (it: Item) => { it.run(); };
  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowDown") { e.preventDefault(); setSel((s) => Math.min(s + 1, filtered.length - 1)); }
    else if (e.key === "ArrowUp") { e.preventDefault(); setSel((s) => Math.max(s - 1, 0)); }
    else if (e.key === "Enter") { e.preventDefault(); const it = filtered[sel]; if (it) exec(it); }
    else if (e.key === "Escape") { e.preventDefault(); onClose(); }
  };

  return (
    <div className="palette-overlay" onClick={onClose}>
      <div className="palette" onClick={(e) => e.stopPropagation()}>
        <input
          ref={inputRef}
          className="palette-input"
          placeholder="Search tables, columns, history…  (↑↓ to move, Enter to act, Esc to close)"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={onKey}
        />
        <div className="palette-list">
          {!connected && <div className="palette-empty">Connect to ryudb-server to search.</div>}
          {connected && loading && <div className="palette-empty">Loading catalog + history…</div>}
          {connected && !loading && err && <div className="palette-empty">error: {err}</div>}
          {connected && !loading && !err && filtered.length === 0 && (
            <div className="palette-empty">No matches</div>
          )}
          {filtered.map((it, i) => (
            <button
              key={it.key}
              className={"palette-item" + (i === sel ? " sel" : "")}
              onMouseEnter={() => setSel(i)}
              onClick={() => exec(it)}
            >
              <span className="palette-label" title={it.label}>
                {it.label.length > 90 ? it.label.slice(0, 90) + "…" : it.label}
              </span>
              <span className="palette-meta">
                <span className="palette-group">{it.sub}</span>
                <span className="palette-group">{it.group}</span>
              </span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}