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

    Phase 2 step 9 adds a parallel **tombstone** channel (``_tombstones``): a
    DELETE stores the primary-key values of the rows it removes as a tombstone
    batch, and the read path anti-joins the visible rows against the tombstone
    PKs (see ``Engine._merge_delta``). Tombstones are a separate channel from
    inserts so the insert path is byte-unchanged; they share the same MVCC
    ``commit_ts`` tagging and the same monotonic-tail invariant, so ``rewind`` /
    ``clear`` / ``tables`` / ``has_unflushed`` are tombstone-aware and the
    byte-identical-series invariant survives (the visible series is
    ``(base ++ insert_batches_at) anti-join tombstone_batches_at``, deterministic
    per snapshot).
    """

    def __init__(self) -> None:
        self._batches: dict[str, list[tuple[int, "cudf.DataFrame"]]] = {}
        # DELETE tombstones: per-table list of (commit_ts, PK-values frame).
        self._tombstones: dict[str, list[tuple[int, "cudf.DataFrame"]]] = {}

    def has_unflushed(self, table: str) -> bool:
        # A table is pending if it has unflushed INSERTs OR tombstones.
        return bool(self._batches.get(table)) or bool(self._tombstones.get(table))

    def tables(self) -> list[str]:
        """Tables that currently have >=1 committed batch OR tombstone (for
        invalidation + checkpoint targeting)."""
        out: set[str] = {t for t, lst in self._batches.items() if lst}
        out.update(t for t, lst in self._tombstones.items() if lst)
        return list(out)

    def batches(self, table: str) -> list["cudf.DataFrame"]:
        """ALL committed batch frames for ``table`` (empty list if none), in
        commit order. Used by the autocommit read path, which sees everything.
        Callers read this fresh each scan so an append becomes visible to the next
        read with no invalidation."""
        return [f for _, f in self._batches.get(table) or []]

    def batches_with_ts(self, table: str) -> list[tuple[int, "cudf.DataFrame"]]:
        """Like ``batches`` but keeps each batch's ``commit_ts`` (step 9: the
        read path tags each visible row with its insertion ts so a tombstone only
        removes rows that existed at delete time, letting a newer reinsert of the
        same PK survive an older tombstone)."""
        return list(self._batches.get(table) or [])

    def batches_at(self, table: str, ts: int) -> list["cudf.DataFrame"]:
        """Committed batch frames for ``table`` with ``commit_ts <= ts``, in
        commit order. Used by the transactional read path: a txn whose
        ``snapshot_ts`` is ``ts`` sees exactly the state committed at-or-before
        its snapshot. In single-session operation this equals ``batches(table)``
        (no commit happens mid-txn), but the filter is kept for correctness under
        future concurrent connections and is exercised by snapshot restore."""
        return [f for cts, f in self._batches.get(table) or [] if cts <= ts]

    def batches_at_with_ts(self, table: str, ts: int) -> list[tuple[int, "cudf.DataFrame"]]:
        """Like ``batches_at`` but keeps each batch's ``commit_ts`` (step 9)."""
        return [(cts, f) for cts, f in self._batches.get(table) or [] if cts <= ts]

    def append(self, table: str, frame: "cudf.DataFrame", commit_ts: int = 0) -> None:
        """Append a committed INSERT batch tagged with ``commit_ts``. The default
        ``0`` keeps the 2-arg call sites (tests, autocommit pre-step-5) working --
        ``0`` is the oldest timestamp and is visible at every snapshot >= 0."""
        self._batches.setdefault(table, []).append((commit_ts, frame))

    # ----------------------------------------------------------- tombstones

    def append_tombstone(self, table: str, frame: "cudf.DataFrame", commit_ts: int = 0) -> None:
        """Append a committed DELETE tombstone (the PK-value frame of the removed
        rows) tagged with ``commit_ts``. Same MVCC semantics as ``append``."""
        self._tombstones.setdefault(table, []).append((commit_ts, frame))

    def has_tombstones(self, table: str) -> bool:
        return bool(self._tombstones.get(table))

    def tombstones(self, table: str) -> list["cudf.DataFrame"]:
        """ALL committed tombstone frames for ``table`` (autocommit read path)."""
        return [f for _, f in self._tombstones.get(table) or []]

    def tombstones_with_ts(self, table: str) -> list[tuple[int, "cudf.DataFrame"]]:
        """Like ``tombstones`` but keeps each tombstone's ``commit_ts`` (step 9:
        the anti-join only removes rows whose insertion ts is at-or-before the
        tombstone's ts, so a reinsert of the same PK after the delete survives)."""
        return list(self._tombstones.get(table) or [])

    def tombstones_at(self, table: str, ts: int) -> list["cudf.DataFrame"]:
        """Committed tombstone frames for ``table`` with ``commit_ts <= ts`` (the
        transactional read path -- a snapshot sees deletes committed at-or-before
        its snapshot_ts)."""
        return [f for cts, f in self._tombstones.get(table) or [] if cts <= ts]

    def tombstones_at_with_ts(self, table: str, ts: int) -> list[tuple[int, "cudf.DataFrame"]]:
        """Like ``tombstones_at`` but keeps each tombstone's ``commit_ts`` (step 9)."""
        return [(cts, f) for cts, f in self._tombstones.get(table) or [] if cts <= ts]

    def rewind(self, ts: int) -> set[str]:
        """Full snapshot restore: drop every batch AND tombstone with
        ``commit_ts > ts`` and return the set of tables that lost >=1 entry (for
        cache invalidation). Tables whose list becomes empty are popped. Correct
        because batches/tombstones are appended in monotonically increasing
        ``commit_ts`` order, so the dropped entries are always a tail suffix per
        table per channel."""
        touched: set[str] = set()
        for store in (self._batches, self._tombstones):
            for table, lst in store.items():
                kept = [(cts, f) for cts, f in lst if cts <= ts]
                if len(kept) != len(lst):
                    touched.add(table)
                if kept:
                    store[table] = kept
                else:
                    # pop empty entries so has_unflushed/tables stay accurate.
                    store[table] = []
            # drop now-empty table keys entirely.
            for t in [t for t, lst in store.items() if not lst]:
                del store[t]
        return touched

    def clear(self, table: str | None = None) -> None:
        if table is None:
            self._batches.clear()
            self._tombstones.clear()
        else:
            self._batches.pop(table, None)
            self._tombstones.pop(table, None)