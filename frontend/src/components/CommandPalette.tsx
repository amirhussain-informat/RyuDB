import { useEffect, useMemo, useRef, useState } from "react";

/** A single runnable action surfaced in the command palette. `disabled`
 * items render greyed and are not selectable. */
export interface Command {
  id: string;
  label: string;
  /** Shortcut hint shown on the right (e.g. "Ctrl/Cmd+Enter"). */
  hint?: string;
  group: string;
  disabled?: boolean;
  run: () => void;
}

interface Props {
  open: boolean;
  commands: Command[];
  onClose: () => void;
}

/** A Cmd/Ctrl+K command palette: type to fuzzy-filter, arrow keys + Enter to
 * run, Escape to close. Snowsight-style universal command entry. */
export default function CommandPalette({ open, commands, onClose }: Props) {
  const [query, setQuery] = useState("");
  const [sel, setSel] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return commands;
    return commands.filter((c) =>
      `${c.label} ${c.group} ${c.hint ?? ""}`.toLowerCase().includes(q),
    );
  }, [query, commands]);

  // Reset selection whenever the filter changes; focus + clear on open.
  useEffect(() => { setSel(0); }, [query]);
  useEffect(() => {
    if (open) {
      setQuery("");
      setSel(0);
      const t = setTimeout(() => inputRef.current?.focus(), 0);
      return () => clearTimeout(t);
    }
  }, [open]);

  if (!open) return null;

  const exec = (c: Command) => {
    if (c.disabled) return;
    onClose();
    c.run();
  };
  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowDown") { e.preventDefault(); setSel((s) => Math.min(s + 1, filtered.length - 1)); }
    else if (e.key === "ArrowUp") { e.preventDefault(); setSel((s) => Math.max(s - 1, 0)); }
    else if (e.key === "Enter") { e.preventDefault(); const c = filtered[sel]; if (c) exec(c); }
    else if (e.key === "Escape") { e.preventDefault(); onClose(); }
  };

  return (
    <div className="palette-overlay" onClick={onClose}>
      <div className="palette" onClick={(e) => e.stopPropagation()}>
        <input
          ref={inputRef}
          className="palette-input"
          placeholder="Type a command…  (↑↓ to move, Enter to run, Esc to close)"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={onKey}
        />
        <div className="palette-list">
          {filtered.length === 0 && <div className="palette-empty">No matching commands</div>}
          {filtered.map((c, i) => (
            <button
              key={c.id}
              className={"palette-item" + (i === sel ? " sel" : "") + (c.disabled ? " disabled" : "")}
              onMouseEnter={() => setSel(i)}
              onClick={() => exec(c)}
              disabled={c.disabled}
            >
              <span className="palette-label">{c.label}</span>
              <span className="palette-meta">
                {c.hint && <kbd className="palette-hint">{c.hint}</kbd>}
                <span className="palette-group">{c.group}</span>
              </span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}