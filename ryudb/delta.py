"""In-memory delta-store for the immutable-base HTAP write path.

Base Parquet files are never mutated; writes (INSERTs, from step 3 onward) append
small cuDF "batch" frames here, and the read path concatenates them onto the base
at scan time (see ``Engine._merge_delta``). Each batch is a natural WAL record
boundary (step 6 persists these).

Phase 2 step 5 makes the store MVCC-versioned: every committed batch is tagged
with the monotonic ``commit_ts`` assigned by the Engine at commit time. This lets
a transaction's snapshot read see only batches committed at-or-before its
snapshot timestamp (``batches_at``), and lets a full snapshot restore discard
every batch committed after a target timestamp (``rewind``). In single-session
operation no commit can occur while a txn is active, so the ts-filter is
structurally a no-op today; the per-batch ``commit_ts`` is required for RESTORE
and is forward-looking for concurrent connections.

The store is owned by the ``Engine`` (session-scoped, in-memory, GPU-resident).
It is NOT the catalog: it holds row data, not table definitions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import cudf


class DeltaStore:
    """Per-table list of ``(commit_ts, cuDF batch frame)`` tuples.

    Batches are immutable and appended in monotonically increasing ``commit_ts``
    order (the Engine's counter only increments). This monotonic-tail invariant
    is load-bearing: it makes ``rewind(ts)`` correct (dropped batches are always a
    suffix per table) and lets COMMIT/ROLLBACK/RESTORE cache invalidation be a
    safe over-invalidation (the visible ``base ++ batches_at ++ buffer`` series is
    byte-identical pre/post commit). Do not interleave or reorder batches.
    """

    def __init__(self) -> None:
        self._batches: dict[str, list[tuple[int, "cudf.DataFrame"]]] = {}

    def has_unflushed(self, table: str) -> bool:
        return bool(self._batches.get(table))

    def tables(self) -> list[str]:
        """Tables that currently have >=1 committed batch (for invalidation)."""
        return [t for t, lst in self._batches.items() if lst]

    def batches(self, table: str) -> list["cudf.DataFrame"]:
        """ALL committed batch frames for ``table`` (empty list if none), in
        commit order. Used by the autocommit read path, which sees everything.
        Callers read this fresh each scan so an append becomes visible to the next
        read with no invalidation."""
        return [f for _, f in self._batches.get(table) or []]

    def batches_at(self, table: str, ts: int) -> list["cudf.DataFrame"]:
        """Committed batch frames for ``table`` with ``commit_ts <= ts``, in
        commit order. Used by the transactional read path: a txn whose
        ``snapshot_ts`` is ``ts`` sees exactly the state committed at-or-before
        its snapshot. In single-session operation this equals ``batches(table)``
        (no commit happens mid-txn), but the filter is kept for correctness under
        future concurrent connections and is exercised by snapshot restore."""
        return [f for cts, f in self._batches.get(table) or [] if cts <= ts]

    def append(self, table: str, frame: "cudf.DataFrame", commit_ts: int = 0) -> None:
        """Append a committed INSERT batch tagged with ``commit_ts``. The default
        ``0`` keeps the 2-arg call sites (tests, autocommit pre-step-5) working --
        ``0`` is the oldest timestamp and is visible at every snapshot >= 0."""
        self._batches.setdefault(table, []).append((commit_ts, frame))

    def rewind(self, ts: int) -> set[str]:
        """Full snapshot restore: drop every batch with ``commit_ts > ts`` and
        return the set of tables that lost >=1 batch (for cache invalidation).
        Tables whose list becomes empty are popped. Correct because batches are
        appended in monotonically increasing ``commit_ts`` order, so the dropped
        batches are always a tail suffix per table."""
        touched: set[str] = set()
        for table, lst in self._batches.items():
            kept = [(cts, f) for cts, f in lst if cts <= ts]
            if len(kept) != len(lst):
                touched.add(table)
            if kept:
                self._batches[table] = kept
            else:
                # pop empty entries so has_unflushed/tables stay accurate.
                self._batches[table] = []
        # drop now-empty table keys entirely.
        for t in [t for t, lst in self._batches.items() if not lst]:
            del self._batches[t]
        return touched

    def clear(self, table: str | None = None) -> None:
        if table is None:
            self._batches.clear()
        else:
            self._batches.pop(table, None)