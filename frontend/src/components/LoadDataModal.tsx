import { useEffect, useState } from "react";

interface Props {
  open: boolean;
  onSubmit: (name: string, path: string) => Promise<void>;
  onClose: () => void;
}

/** Data-load wizard: register a parquet table from a server-local path. The
 *  engine is parquet-only and reads from the server's filesystem (no upload
 *  mechanism), so the path is a server-side file / directory / glob resolved
 *  to *.parquet by `Catalog.register` (the same path `CREATE TABLE FROM` and
 *  `admin register` use). On success the parent refreshes the catalog +
 *  autocomplete schema. */
export default function LoadDataModal({ open, onSubmit, onClose }: Props) {
  const [name, setName] = useState("");
  const [path, setPath] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (open) {
      setName("");
      setPath("");
      setError(null);
      setSubmitting(false);
    }
  }, [open]);

  if (!open) return null;

  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === "Escape") {
      e.preventDefault();
      if (!submitting) onClose();
    }
  };

  const submit = async () => {
    const n = name.trim();
    const p = path.trim();
    if (!n || !p) {
      setError("Table name and parquet path are required.");
      return;
    }
    setSubmitting(true);
    setError(null);
    try {
      await onSubmit(n, p);
      onClose();
    } catch (e) {
      setError((e as Error).message ?? "register failed");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="palette-overlay" onClick={() => !submitting && onClose()}>
      <div className="load-modal" onClick={(e) => e.stopPropagation()} onKeyDown={onKey}>
        <div className="sidebar-head">
          <span>Load data</span>
          <button onClick={onClose} disabled={submitting}>Close</button>
        </div>
        <div className="load-body">
          <p className="load-hint">
            Register a parquet table from a path on the server filesystem. A
            directory is expanded to <code>*.parquet</code>; a glob works too.
          </p>
          <label className="load-label">Table name</label>
          <input
            className="load-input"
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="e.g. lineitem"
            autoFocus
            disabled={submitting}
          />
          <label className="load-label">Parquet path</label>
          <input
            className="load-input"
            type="text"
            value={path}
            onChange={(e) => setPath(e.target.value)}
            placeholder="e.g. /home/workbench/RyuDB/data/tpch-sf0.1/lineitem"
            disabled={submitting}
          />
          {error && <div className="load-error">{error}</div>}
          <div className="load-actions">
            <button className="primary" onClick={submit} disabled={submitting}>
              {submitting ? "Loading…" : "Load"}
            </button>
            <button onClick={onClose} disabled={submitting}>Cancel</button>
          </div>
        </div>
      </div>
    </div>
  );
}