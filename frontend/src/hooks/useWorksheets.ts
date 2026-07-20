import { useEffect, useRef, useState } from "react";

/** A single SQL worksheet (one editor buffer). Persisted to localStorage so
 * the user's worksheets survive a reload; the active id is persisted too. */
export interface Worksheet {
  id: string;
  name: string;
  sql: string;
  /** ms epoch of the last edit (informational; used for sort/display later). */
  updatedAt: number;
}

const STORAGE_KEY = "ryudb.worksheets.v1";

const DEFAULT_SQL =
  "SELECT l_returnflag, count(*) AS c, sum(l_extendedprice) AS s\n" +
  "FROM lineitem\nGROUP BY l_returnflag\nORDER BY l_returnflag;";

function uid(): string {
  try {
    if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
      return crypto.randomUUID();
    }
  } catch { /* secure-context unavailable -> fall back */ }
  return Math.random().toString(36).slice(2) + Date.now().toString(36);
}

function load(): { worksheets: Worksheet[]; activeId: string } {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw) as { worksheets?: Worksheet[]; activeId?: string };
      if (Array.isArray(parsed.worksheets) && parsed.worksheets.length > 0) {
        const ws = parsed.worksheets.filter(
          (w) => w && typeof w.id === "string" && typeof w.name === "string" && typeof w.sql === "string",
        );
        if (ws.length > 0) {
          const activeId =
            typeof parsed.activeId === "string" && ws.some((w) => w.id === parsed.activeId)
              ? parsed.activeId!
              : ws[0].id;
          return { worksheets: ws, activeId };
        }
      }
    }
  } catch {
    /* corrupt / unavailable storage -> seed a fresh worksheet */
  }
  const first: Worksheet = { id: uid(), name: "Worksheet 1", sql: DEFAULT_SQL, updatedAt: 0 };
  return { worksheets: [first], activeId: first.id };
}

/** Manages the worksheet list + the active worksheet, persisting both to
 * localStorage. At least one worksheet is always present (close() is a no-op
 * on the last one). */
export function useWorksheets() {
  const initial = useRef<ReturnType<typeof load> | null>(null);
  if (initial.current === null) initial.current = load();

  const [worksheets, setWorksheets] = useState<Worksheet[]>(initial.current.worksheets);
  const [activeId, setActiveId] = useState<string>(initial.current.activeId);

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({ worksheets, activeId }));
    } catch {
      /* quota / private mode -> best-effort, non-fatal */
    }
  }, [worksheets, activeId]);

  const active = worksheets.find((w) => w.id === activeId) ?? worksheets[0];

  const updateSql = (id: string, sql: string) =>
    setWorksheets((prev) => prev.map((w) => (w.id === id ? { ...w, sql, updatedAt: Date.now() } : w)));

  const create = () => {
    const n = worksheets.length + 1;
    const ws: Worksheet = { id: uid(), name: `Worksheet ${n}`, sql: "", updatedAt: 0 };
    const idx = worksheets.findIndex((w) => w.id === activeId);
    setWorksheets(
      idx >= 0 ? [...worksheets.slice(0, idx + 1), ws, ...worksheets.slice(idx + 1)] : [...worksheets, ws],
    );
    setActiveId(ws.id);
  };

  const rename = (id: string, name: string) =>
    setWorksheets((prev) => prev.map((w) => (w.id === id ? { ...w, name } : w)));

  const close = (id: string) => {
    if (worksheets.length <= 1) return; // always keep one worksheet
    const idx = worksheets.findIndex((w) => w.id === id);
    const next = worksheets.filter((w) => w.id !== id);
    setWorksheets(next);
    if (id === activeId) {
      const newActive = next[Math.min(idx, next.length - 1)];
      setActiveId(newActive.id);
    }
  };

  return { worksheets, activeId, active, setActive: setActiveId, create, rename, close, updateSql };
}