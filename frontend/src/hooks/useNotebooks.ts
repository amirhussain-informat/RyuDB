import { useEffect, useRef, useState } from "react";
import type { Notebook, NotebookCell, NotebookCellType } from "../lib/types";

/** A saved SQL+Python notebook (an ordered list of cells), persisted to
 *  localStorage so the user's notebooks survive a reload. Mirrors the
 *  `useDashboards` shape. An EMPTY list is allowed (no seed notebook). Only
 *  cell SOURCES are persisted — outputs are re-run, never stored. */
const STORAGE_KEY = "ryudb.notebooks.v1";

function uid(): string {
  try {
    if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
      return crypto.randomUUID();
    }
  } catch { /* secure-context unavailable -> fall back */ }
  return Math.random().toString(36).slice(2) + Date.now().toString(36);
}

function validCell(c: unknown): c is NotebookCell {
  return !!c
    && typeof (c as NotebookCell).id === "string"
    && ((c as NotebookCell).type === "sql" || (c as NotebookCell).type === "python")
    && typeof (c as NotebookCell).code === "string";
}

function validNotebook(n: unknown): n is Notebook {
  return !!n
    && typeof (n as Notebook).id === "string"
    && typeof (n as Notebook).name === "string"
    && Array.isArray((n as Notebook).cells)
    && (n as Notebook).cells.every(validCell);
}

function load(): Notebook[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw) as { notebooks?: Notebook[] };
      if (Array.isArray(parsed.notebooks)) {
        return parsed.notebooks.filter(validNotebook);
      }
    }
  } catch {
    /* corrupt / unavailable storage -> start empty */
  }
  return [];
}

/** Manages the notebook list, persisting cell SOURCES to localStorage. */
export function useNotebooks() {
  const initial = useRef<Notebook[] | null>(null);
  if (initial.current === null) initial.current = load();

  const [notebooks, setNotebooks] = useState<Notebook[]>(initial.current);

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({ notebooks }));
    } catch {
      /* quota / private mode -> best-effort, non-fatal */
    }
  }, [notebooks]);

  const create = (name: string): string => {
    const id = uid();
    const trimmed = name.trim() || `Notebook ${notebooks.length + 1}`;
    setNotebooks((prev) => [...prev, { id, name: trimmed, cells: [] }]);
    return id;
  };

  const rename = (id: string, name: string) =>
    setNotebooks((prev) => prev.map((n) => (n.id === id ? { ...n, name } : n)));

  const remove = (id: string) =>
    setNotebooks((prev) => prev.filter((n) => n.id !== id));

  const addCell = (notebookId: string, type: NotebookCellType, code = ""): string => {
    const cellId = uid();
    setNotebooks((prev) => prev.map((n) =>
      n.id === notebookId ? { ...n, cells: [...n.cells, { id: cellId, type, code }] } : n));
    return cellId;
  };

  const updateCell = (notebookId: string, cellId: string, patch: Partial<Pick<NotebookCell, "code" | "type">>) =>
    setNotebooks((prev) => prev.map((n) =>
      n.id === notebookId
        ? { ...n, cells: n.cells.map((c) => (c.id === cellId ? { ...c, ...patch } : c)) }
        : n));

  const removeCell = (notebookId: string, cellId: string) =>
    setNotebooks((prev) => prev.map((n) =>
      n.id === notebookId ? { ...n, cells: n.cells.filter((c) => c.id !== cellId) } : n));

  /** Move a cell within its notebook by delta (+1 / -1). No-op at the edges. */
  const moveCell = (notebookId: string, cellId: string, delta: number) =>
    setNotebooks((prev) => prev.map((n) => {
      if (n.id !== notebookId) return n;
      const i = n.cells.findIndex((c) => c.id === cellId);
      const j = i + delta;
      if (i < 0 || j < 0 || j >= n.cells.length) return n;
      const cells = n.cells.slice();
      [cells[i], cells[j]] = [cells[j], cells[i]];
      return { ...n, cells };
    }));

  return { notebooks, create, rename, remove, addCell, updateCell, removeCell, moveCell };
}