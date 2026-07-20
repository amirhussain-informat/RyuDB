import { useCallback, useEffect, useRef, useState } from "react";

/** A saved snapshot of a worksheet's SQL at a point in time. One worksheet
 * has a ring of these (newest first), capped at MAX. Persisted to
 * localStorage so versions survive a reload — this is the worksheet's undo /
 * history, independent of the server-side query history (which is per-DB, not
 * per-worksheet, and records what ran, not what was typed). */
export interface Snapshot {
  id: string;
  /** ms epoch. */
  ts: number;
  sql: string;
}

const STORAGE_KEY = "ryudb.versions.v1";
const MAX = 30;

function uid(): string {
  try {
    if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
      return crypto.randomUUID();
    }
  } catch { /* secure-context unavailable -> fall back */ }
  return Math.random().toString(36).slice(2) + Date.now().toString(36);
}

function loadAll(): Record<string, Snapshot[]> {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw) as Record<string, Snapshot[]>;
      if (parsed && typeof parsed === "object") return parsed;
    }
  } catch { /* corrupt -> start fresh */ }
  return {};
}

/** Per-worksheet version history. `capture(id, sql)` records a snapshot
 * unless the SQL is empty or identical to the newest one (so rapid keystrokes
 * or repeated runs don't flood the ring). `restore` is handled by the caller
 * (it just loads `snap.sql` into the worksheet); this hook only owns storage. */
export function useVersions() {
  const initial = useRef<Record<string, Snapshot[]> | null>(null);
  if (initial.current === null) initial.current = loadAll();
  const [all, setAll] = useState<Record<string, Snapshot[]>>(initial.current);

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(all));
    } catch { /* quota / private mode -> best-effort, non-fatal */ }
  }, [all]);

  const versions = useCallback((id: string): Snapshot[] => all[id] ?? [], [all]);

  const capture = useCallback((id: string, sql: string) => {
    const trimmed = sql.trim();
    if (!trimmed) return;
    setAll((prev) => {
      const ring = prev[id] ?? [];
      if (ring.length > 0 && ring[0].sql === sql) return prev; // dedupe identical
      const snap: Snapshot = { id: uid(), ts: Date.now(), sql };
      const next = [snap, ...ring].slice(0, MAX);
      return { ...prev, [id]: next };
    });
  }, []);

  const remove = useCallback((id: string, snapId: string) => {
    setAll((prev) => {
      const ring = prev[id];
      if (!ring) return prev;
      return { ...prev, [id]: ring.filter((s) => s.id !== snapId) };
    });
  }, []);

  const clear = useCallback((id: string) => {
    setAll((prev) => {
      if (!prev[id]) return prev;
      const { [id]: _drop, ...rest } = prev;
      return rest;
    });
  }, []);

  // When a worksheet is closed, drop its version ring so storage doesn't grow
  // unbounded with orphaned ids. The caller passes the live set of ids.
  const gc = useCallback((liveIds: string[]) => {
    const live = new Set(liveIds);
    setAll((prev) => {
      let changed = false;
      const next: Record<string, Snapshot[]> = {};
      for (const [k, v] of Object.entries(prev)) {
        if (live.has(k)) next[k] = v;
        else changed = true;
      }
      return changed ? next : prev;
    });
  }, []);

  return { versions, capture, remove, clear, gc };
}