import { useEffect, useState } from "react";
import type { ProfileColumn, ProfileResp } from "../lib/types";

interface Props {
  open: boolean;
  name: string | null;
  fetchProfile: (name: string) => Promise<ProfileResp | null>;
  onClose: () => void;
}

function fmtNum(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") {
    if (Number.isInteger(v)) return v.toLocaleString();
    return v.toFixed(3);
  }
  return String(v);
}

function fmtPct(p: number): string {
  return `${(p * 100).toFixed(p < 0.1 && p > 0 ? 2 : 1)}%`;
}

/** A per-column statistics panel for a table, backed by the server `profile`
 * op (GPU-computed via cuDF). Lists null %, distinct count, min/max, mean and
 * stddev for numeric columns, a 10-bucket histogram, and the top-K most
 * frequent values for low-NDV columns. This is the headline GPU differentiator
 * — the stats come back in milliseconds because they are computed on the GPU,
 * not by scanning in the browser. */
export default function ProfileModal({ open, name, fetchProfile, onClose }: Props) {
  const [profile, setProfile] = useState<ProfileResp | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (!open || !name) return;
    let cancelled = false;
    setLoading(true);
    setErr(null);
    setProfile(null);
    fetchProfile(name)
      .then((p) => { if (!cancelled) setProfile(p); })
      .catch((e: unknown) => { if (!cancelled) setErr((e as Error)?.message ?? String(e)); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [open, name, fetchProfile]);

  if (!open) return null;

  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === "Escape") { e.preventDefault(); onClose(); }
  };

  return (
    <div className="palette-overlay" onClick={onClose}>
      <div className="profile-modal" onClick={(e) => e.stopPropagation()} onKeyDown={onKey}>
        <div className="palette-input shortcuts-head vhist-head">
          <span>Column profile — {name}</span>
          <button className="vhist-save" onClick={onClose}>Close</button>
        </div>
        <div className="palette-list profile-body">
          {loading && <div className="palette-empty">Profiling on the GPU…</div>}
          {err && <div className="palette-empty">error: {err}</div>}
          {profile && (
            <>
              <div className="profile-summary">
                {profile.row_count.toLocaleString()} rows · {profile.columns.length} column{profile.columns.length === 1 ? "" : "s"}
              </div>
              {profile.columns.map((c) => <ColumnCard key={c.name} c={c} />)}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function ColumnCard({ c }: { c: ProfileColumn }) {
  const maxHist = c.histogram ? Math.max(1, ...c.histogram.map((b) => b.count)) : 1;
  const maxTop = c.top ? Math.max(1, ...c.top.map((t) => t.count)) : 1;
  const totalRows = c.row_count || 1;
  return (
    <div className="profile-card">
      <div className="profile-card-head">
        <span className="profile-col-name" title={c.name}>{c.name}</span>
        <span className="profile-col-type">{c.type}</span>
      </div>
      <div className="profile-stats">
        <span title="null percentage">{(c.null_count).toLocaleString()} null · {fmtPct(c.null_pct)}</span>
        <span title="distinct values">{c.distinct.toLocaleString()} distinct</span>
        {c.min !== undefined && <span title="min">min {fmtNum(c.min)}</span>}
        {c.max !== undefined && <span title="max">max {fmtNum(c.max)}</span>}
        {c.mean !== undefined && c.mean !== null && <span title="mean">μ {fmtNum(c.mean)}</span>}
        {c.stddev !== undefined && c.stddev !== null && <span title="stddev">σ {fmtNum(c.stddev)}</span>}
      </div>
      {c.histogram && c.histogram.length > 0 && (
        <div className="profile-hist" title="10-bucket equal-width histogram">
          {c.histogram.map((b, i) => (
            <div className="profile-hist-bar" key={i}
              style={{ height: `${(b.count / maxHist) * 100}%` }}
              title={`[${fmtNum(b.lo)}, ${fmtNum(b.hi)}): ${b.count}`} />
          ))}
        </div>
      )}
      {c.top && c.top.length > 0 && (
        <ul className="profile-top">
          {c.top.map((t, i) => (
            <li key={i}>
              <span className="profile-top-val" title={String(t.value)}>{fmtNum(t.value)}</span>
              <span className="profile-top-bar">
                <span className="profile-top-fill" style={{ width: `${(t.count / maxTop) * 100}%` }} />
              </span>
              <span className="profile-top-count">{t.count.toLocaleString()} · {fmtPct(t.count / totalRows)}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}