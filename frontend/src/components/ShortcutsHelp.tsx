interface Props {
  open: boolean;
  onClose: () => void;
}

const SHORTCUTS: { key: string; desc: string }[] = [
  { key: "Ctrl/Cmd + K", desc: "Open the command palette" },
  { key: "Ctrl/Cmd + Shift + F", desc: "Search tables, columns, and history" },
  { key: "Ctrl/Cmd + Enter", desc: "Run the current query (in the editor)" },
  { key: "?", desc: "Show this keyboard-shortcuts help" },
  { key: "Esc", desc: "Close the palette / this help" },
  { key: "↑ / ↓", desc: "Move the selection in the palette" },
  { key: "Enter", desc: "Run the selected command" },
  { key: "Double-click a tab", desc: "Rename a worksheet" },
];

/** A modal listing the keyboard shortcuts. Opened by `?` (when not typing) or
 * the "Show keyboard shortcuts" command in the palette. */
export default function ShortcutsHelp({ open, onClose }: Props) {
  if (!open) return null;
  return (
    <div className="palette-overlay" onClick={onClose}>
      <div className="palette shortcuts" onClick={(e) => e.stopPropagation()}>
        <div className="palette-input shortcuts-head">Keyboard shortcuts</div>
        <div className="palette-list">
          {SHORTCUTS.map((s) => (
            <div className="palette-item static" key={s.key}>
              <span className="palette-label">{s.desc}</span>
              <span className="palette-meta">
                <kbd className="palette-hint">{s.key}</kbd>
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}