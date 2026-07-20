import { useEffect, useState } from "react";

interface Props {
  open: boolean;
  /** Register a parquet table from a server-local path (admin register). */
  onSubmit: (name: string, path: string) => Promise<void>;
  /** Ingest a parquet file uploaded from the browser (upload op). The bytes
   *  are the raw file contents; the server writes them to <data>/uploads. */
  onUpload: (name: string, bytes: Uint8Array, fileName: string) => Promise<void>;
  onClose: () => void;
  /** Client-side upload size cap (the server's --max-upload-bytes). A larger
   *  file is refused before sending. */
  maxUploadBytes?: number;
}

type Mode = "path" | "upload";

/** Data-load wizard with two modes:
 *  - "From path": register a parquet table from a server-local file/dir/glob
 *    (admin register / CREATE TABLE FROM — the engine reads the server's
 *    filesystem directly).
 *  - "Upload file": send a parquet from the browser over the wire (upload op);
 *    the server persists it to <data>/uploads and registers the table.
 *  The engine is parquet-only, so both modes accept parquet only. On success
 *  the parent refreshes the catalog + autocomplete schema. */
export default function LoadDataModal({ open, onSubmit, onUpload, onClose, maxUploadBytes }: Props) {
  const [mode, setMode] = useState<Mode>("upload");
  const [name, setName] = useState("");
  const [path, setPath] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (open) {
      setMode("upload");
      setName("");
      setPath("");
      setFile(null);
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

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0] ?? null;
    setFile(f);
    setError(null);
    if (f && !name) {
      // default the table name to the file stem (e.g. lineitem.parquet -> lineitem)
      const stem = f.name.replace(/\.parquet$/i, "");
      setName(stem.replace(/[^A-Za-z0-9_]/g, "_"));
    }
  };

  const submit = async () => {
    const n = name.trim();
    if (!n) {
      setError("Table name is required.");
      return;
    }
    setSubmitting(true);
    setError(null);
    try {
      if (mode === "path") {
        const p = path.trim();
        if (!p) {
          setError("Parquet path is required.");
          setSubmitting(false);
          return;
        }
        await onSubmit(n, p);
      } else {
        if (!file) {
          setError("Choose a parquet file to upload.");
          setSubmitting(false);
          return;
        }
        if (!/\.parquet$/i.test(file.name)) {
          setError("Only .parquet files are supported.");
          setSubmitting(false);
          return;
        }
        if (maxUploadBytes !== undefined && file.size > maxUploadBytes) {
          setError(`File is ${file.size.toLocaleString()} bytes; upload limit is ` +
                   `${maxUploadBytes.toLocaleString()} bytes.`);
          setSubmitting(false);
          return;
        }
        const buf = new Uint8Array(await file.arrayBuffer());
        await onUpload(n, buf, file.name);
      }
      onClose();
    } catch (e) {
      setError((e as Error).message ?? "load failed");
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
          <div className="load-tabs">
            <button className={mode === "upload" ? "active" : ""} onClick={() => setMode("upload")} disabled={submitting}>
              Upload file
            </button>
            <button className={mode === "path" ? "active" : ""} onClick={() => setMode("path")} disabled={submitting}>
              From path
            </button>
          </div>

          {mode === "upload" ? (
            <>
              <p className="load-hint">
                Send a parquet file from your browser. The server writes it to
                <code> &lt;data&gt;/uploads/</code> and registers the table.
                {maxUploadBytes !== undefined && <> Limit {maxUploadBytes.toLocaleString()} bytes.</>}
              </p>
              <label className="load-label">Parquet file</label>
              <input
                className="load-input"
                type="file"
                accept=".parquet,application/vnd.apache.parquet,application/octet-stream"
                onChange={onFileChange}
                disabled={submitting}
              />
              {file && (
                <div className="load-hint">
                  {file.name} · {file.size.toLocaleString()} bytes
                </div>
              )}
            </>
          ) : (
            <>
              <p className="load-hint">
                Register a parquet table from a path on the server filesystem. A
                directory is expanded to <code>*.parquet</code>; a glob works too.
              </p>
              <label className="load-label">Parquet path</label>
              <input
                className="load-input"
                type="text"
                value={path}
                onChange={(e) => setPath(e.target.value)}
                placeholder="e.g. /home/workbench/RyuDB/data/tpch-sf0.1/lineitem"
                disabled={submitting}
              />
            </>
          )}

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
          {error && <div className="load-error">{error}</div>}
          <div className="load-actions">
            <button className="primary" onClick={submit} disabled={submitting}>
              {submitting ? "Loading…" : mode === "upload" ? "Upload" : "Load"}
            </button>
            <button onClick={onClose} disabled={submitting}>Cancel</button>
          </div>
        </div>
      </div>
    </div>
  );
}