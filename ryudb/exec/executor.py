"""Plan executor: lowers physical plan nodes to cuDF operations on the GPU.

The executor walks the plan bottom-up, producing a cuDF DataFrame at each node.
Index hygiene is deliberate: scans and every reshaping op reset to a clean
RangeIndex so that Series and scalar broadcasts line up in Project/Aggregate.
"""

from __future__ import annotations

import os
import re
import json

import cudf
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..catalog import Catalog
from ..delta import DeltaStore
from ..sql.optimize import optimize
from ..sql.parse import ParseError, parse
from ..sql.plan import (
    Aggregate,
    Col,
    Delete,
    Filter,
    Insert,
    Join,
    Limit,
    PlanNode,
    Project,
    Scan,
    Sort,
    Star,
    TxnControl,
)
from ..storage import scan
from ..transaction import Transaction
from ..wal import WAL, wal_path
from .fused import (
    _PendingFrame,
    _arrow_match_dtype,
    fused_aggregate,
    fused_join_aggregate,
    fused_scan_aggregate,
)
from .ops import _literal, eval_expr

_AGG_METHOD = {"SUM": "sum", "AVG": "mean", "MIN": "min", "MAX": "max", "COUNT": "count"}

# Insertion-timestamp sentinel for in-txn buffered writes (step 9). The
# timestamp-aware DELETE anti-join only removes a row when a matching tombstone
# has ``tomb_ts >= ins_ts``; a buffered insert (no commit_ts yet) is tagged
# ``+inf`` so it survives every committed tombstone (it is newer than anything
# committed), and a buffered tombstone (also ``+inf``) removes every committed
# row and every buffered insert of the same PK (read-your-writes). ``1 << 62``
# is far above any realistic monotonic commit_ts (which starts at 1).
_INS_INF = 1 << 62

# Non-standard snapshot/restore SQL (sqlglot has no node for these). Pre-compiled
# and guarded by a cheap prefix check in Engine.sql/explain so the regex never
# runs on the SELECT hot path. Mirrors cli._CREATE_RE.
_SNAPSHOT_RE = re.compile(r"CREATE\s+SNAPSHOT\s+([A-Za-z_][\w]*)\s*;?", re.IGNORECASE)
_RESTORE_RE = re.compile(r"RESTORE\s+TO\s+SNAPSHOT\s+([A-Za-z_][\w]*)\s*;?", re.IGNORECASE)


class Engine:
    """Front door: parse -> optimize -> execute on GPU, returning a cuDF frame."""

    def __init__(self, catalog: Catalog):
        self.catalog = catalog
        # GPU-resident scan cache: (table, frozenset(columns)) -> coerced cuDF
        # frame. Warm (repeated) queries skip the Parquet read + decimal coercion.
        # The cached frame is returned directly (no copy): the fused kernel path
        # never mutates it, and the cuDF fallback paths copy before mutating, so
        # the cached pristine frame is never corrupted.
        self._scan_cache: dict[tuple[str, frozenset], cudf.DataFrame] = {}
        # Lazily-computed factorize codes for group-key columns, keyed by
        # (table, col). cuDF factorize on 60M string rows is itself a hash-groupby
        # (~460 ms for 2 cols); caching the int codes lets warm repeat queries
        # skip it and run just the ~35 ms fused kernel.
        self._code_cache: dict[tuple[str, str], tuple] = {}
        # Cached "is this column a unique key" result for fused star-join
        # eligibility (dimension join keys must be PKs). Like the code index it
        # is a maintained fact about the data, not a query cache; reuse across
        # runs is valid because uniqueness is deterministic for identical data.
        self._pk_cache: dict[tuple[str, str], bool] = {}
        self.cache_enabled: bool = True
        # In-memory delta-store for the immutable-base write path (Phase 2).
        # Empty by default; reads merge live batches onto the base in _scan.
        # Step 2 leaves this empty (reads unchanged); step 3 appends INSERTs;
        # step 5 tags each committed batch with a monotonic commit_ts (MVCC).
        self.delta: DeltaStore = DeltaStore()
        # Phase 2 step 5 -- MVCC transaction layer. Single-session: at most one
        # txn is active and no commit happens mid-txn (INSERTs buffer, not
        # commit), so snapshot isolation is structural, not lock-based. The
        # commit_ts is required for full snapshot restore and is forward-looking
        # for concurrent connections. _txn/_commit_ts/_snapshots are unprotected
        # mutable state -- the Engine is single-session/one-thread (no locks).
        self._commit_ts: int = 0          # monotonic; bumped on each commit
        self._txn: Transaction | None = None
        self._snapshots: dict[str, int] = {}  # name -> commit_ts captured
        # Phase 2 step 6 -- WAL + recovery. Each commit is appended to
        # <data_dir>/ryudb.wal and fsync'd BEFORE the in-memory delta is mutated,
        # so commit_ts doubles as the WAL LSN. On startup we replay the WAL to
        # reconstruct the delta and reset _commit_ts to the highest replayed LSN.
        # Disabled (no-op) when there is no data dir, mirroring Catalog gating.
        self._wal: WAL = WAL(wal_path(catalog.data_dir))
        max_ts = 0
        for cts, table, kind, frame in self._wal.replay():
            if kind == "tombstone":
                self.delta.append_tombstone(table, frame, cts)
            else:
                self.delta.append(table, frame, cts)
            max_ts = max(max_ts, cts)
        self._commit_ts = max_ts

    def clear_scan_cache(self) -> None:
        """Clear the GPU-resident frame cache (forces a re-read on next scan).

        Note: the per-column factorize *code index* (`_code_cache`) is intentionally
        NOT cleared here. It is a dictionary-encoding of group-key columns — a
        maintained index, not a query-result cache — and reusing it across scans is
        valid because factorize codes are positional and deterministic for identical
        data. This makes a scan-cold run (frame evicted, index resident) skip the
        ~460 ms hash-factorize and run just read+coerce+kernel (~380 ms) instead of
        re-paying it. Use `clear_code_cache()` to invalidate it explicitly (e.g. when
        the underlying table is written to, once the HTAP write path exists).

        Phase 5 async-materialise: a pending entry holds a `pending_id` whose
        background gather scratch (~1.4 GB: `ubig`+`d_idxbig`+small arrays) is
        owned by the C++ registry and freed only by `fused_scan_finalize`.
        Finalize every pending entry before dropping our refs so a cold run
        immediately followed by `clear` (bench ryu_cold / tests) doesn't leak or
        fault. Ready frames need no finalization.
        """
        from .. import kernels as _kernels

        if _kernels.fused_scan_finalize is not None:
            for v in self._scan_cache.values():
                pid = getattr(v, "pending_id", None)
                if pid:
                    try:
                        _kernels.fused_scan_finalize(int(pid))
                    except Exception:
                        pass
        self._scan_cache.clear()

    def clear_code_cache(self) -> None:
        self._code_cache.clear()
        self._pk_cache.clear()

    def _invalidate_table_caches(self, table: str) -> None:
        """Drop this table's _code_cache/_pk_cache entries (autocommit hook).

        INSERTs append rows to the delta, so the base-only factorize codes
        (`_code_cache`) and PK-uniqueness facts (`_pk_cache`) -- both keyed by
        just `(table, col)`, the data identity NOT in the key -- go stale: cached
        codes are row-aligned to the pre-INSERT series length (a longer merged
        series reads them OOB), and a cached `True` survives a duplicate-PK
        INSERT (the fused star-join would then collapse joins). Every code/pk
        series is obtained via `_scan(table)` for the SAME table in the key, so
        dropping only `(table, *)` is necessary and sufficient. The scan cache is
        base-only + live-merged in `_scan`, so it is NOT touched here. Step 5's
        transactional commit() will reuse this hook.
        """
        for k in [k for k in self._code_cache if k[0] == table]:
            del self._code_cache[k]
        for k in [k for k in self._pk_cache if k[0] == table]:
            del self._pk_cache[k]

    def _drop_scan_cache_for(self, table: str) -> None:
        """Drop this table's ``_scan_cache`` entries (used by ``checkpoint``).

        The scan cache holds a **base-only** frame (``_merge_delta`` re-merges the
        live delta on top, never writing the merged frame back). After a
        checkpoint rewrites the base to base++delta and clears the delta, that
        cached base-only frame is stale -- it's the OLD base without the now-
        checkpointed rows -- so the next read would serve the wrong rows. Drop
        just this table's entries (finalizing any in-flight Phase 5 async gather,
        mirroring ``clear_scan_cache``) so the next scan re-reads the new base.
        Other tables' warm frames stay resident (checkpoint is rare)."""
        from .. import kernels as _kernels

        keys = [k for k in self._scan_cache if k[0] == table]
        if _kernels.fused_scan_finalize is not None:
            for k in keys:
                pid = getattr(self._scan_cache[k], "pending_id", None)
                if pid:
                    try:
                        _kernels.fused_scan_finalize(int(pid))
                    except Exception:
                        pass
        for k in keys:
            self._scan_cache.pop(k, None)

    # ------------------------------------------------------------------ #
    # Phase 2 step 5 -- MVCC transaction layer + snapshot restore
    # ------------------------------------------------------------------ #

    def has_pending(self, table: str) -> bool:
        """True if ``table`` has unflushed committed delta rows OR a buffered
        in-txn write -- i.e. a read of ``table`` must merge beyond the base.
        Public so the cold Parquet reader (fused.py) can defer to the
        materialising _scan+merge path for both committed-delta and txn-buffer
        states (the cold reader bypasses _scan and would otherwise miss them)."""
        return self.delta.has_unflushed(table) or (
            self._txn is not None and self._txn.has(table)
        )

    def _next_commit_ts(self) -> int:
        self._commit_ts += 1
        return self._commit_ts

    def _txn_control(self, node: TxnControl) -> None:
        if node.kind == "begin":
            self._begin()
        elif node.kind == "commit":
            self._commit()
        elif node.kind == "rollback":
            self._rollback()
        else:
            raise RuntimeError(f"unknown txn control: {node.kind}")

    def _begin(self) -> None:
        if self._txn is not None:
            raise RuntimeError("BEGIN inside an active transaction (nested txns not supported)")
        self._txn = Transaction(snapshot_ts=self._commit_ts)

    def _commit(self) -> None:
        txn = self._txn
        if txn is None:
            raise RuntimeError("COMMIT without an active transaction")
        # Flush the txn's buffered INSERTs AND DELETE tombstones to the shared
        # delta under one new ts -> atomic commit (all-or-nothing). Buffer append
        # order is preserved, so the post-commit visible series
        # (base ++ insert_batches_at) anti-join tombstone_batches_at is
        # byte-identical to the in-txn read-your-writes series. Step 6 routes the
        # flush through _write_commit so the whole commit is one durable WAL
        # record (atomic on the disk side too), written+fsync'd before the
        # in-memory delta mutates. Step 9: batches are (table, kind, frame) so a
        # mixed INSERT+DELETE commit is one record with per-batch kind.
        batches: list[tuple[str, str, cudf.DataFrame]] = [
            (t, "insert", f) for t in txn.tables() for f in txn.buffer_batches(t)
        ] + [
            (t, "tombstone", f) for t in txn.tombstone_tables() for f in txn.tombstone_batches(t)
        ]
        self._write_commit(batches)
        self._txn = None

    def _write_commit(self, batches: list[tuple[str, str, cudf.DataFrame]]) -> int:
        """The single durable-commit seam used by both the autocommit INSERT/DELETE
        paths and the explicit txn COMMIT path.

        ``batches`` is a list of ``(table, kind, frame)`` where ``kind`` is
        ``"insert"`` or ``"tombstone"`` (step 9). Allocates one new commit_ts,
        writes+fsyncs a single WAL record holding every batch with its kind (one
        record per commit => commit atomicity is either a fully CRC-valid durable
        record or a discarded torn tail), and ONLY THEN appends the batches to the
        in-memory delta in the given order -- inserts to ``delta.append``,
        tombstones to ``delta.append_tombstone``. A crash after the fsync but
        before the in-memory append is recovered by WAL replay; a crash during
        the write leaves a torn tail the next replay discards -- so a commit is
        all-or-nothing on both sides of a restart. The in-memory append order is
        identical to today's, preserving the MVCC byte-identical invariant. Empty
        ``batches`` bumps the counter but writes no WAL record (the counter is
        recovered as max replayed ts). Returns ts.
        """
        ts = self._next_commit_ts()
        self._wal.write_commit(ts, batches)  # durable BEFORE in-memory mutation
        for table, kind, frame in batches:
            if kind == "tombstone":
                self.delta.append_tombstone(table, frame, ts)
            else:
                self.delta.append(table, frame, ts)
        for table in dict.fromkeys(t for t, _, _ in batches):
            self._invalidate_table_caches(table)
        return ts

    def _rollback(self) -> None:
        txn = self._txn
        if txn is None:
            raise RuntimeError("ROLLBACK without an active transaction")
        # Undo only this txn's writes: discard the buffer. The committed delta is
        # untouched (the txn never committed). Invalidate the buffered tables'
        # caches -- a read-your-writes SELECT may have populated them against the
        # base++buffer series; after the buffer is gone those are stale.
        for table in txn.tables():
            self._invalidate_table_caches(table)
        self._txn = None

    # -- snapshot / restore (full DB restore, stronger than per-txn ROLLBACK) --

    def snapshot(self, name: str) -> None:
        """Capture the current committed state as a named snapshot. Allowed during
        a txn: it captures the committed state (frozen mid-txn); the txn's buffer
        is excluded. Overwrites an existing name."""
        self._snapshots[name] = self._commit_ts

    def restore(self, name: str) -> None:
        """Restore the whole DB to the named snapshot: discard every committed
        delta batch after the snapshot's ts (committed work after that point is
        lost), rewind the commit counter, and drop any snapshots that now point at
        discarded state. The current in-flight txn (if any) must be rolled back
        first -- restoring mid-txn is rejected."""
        if name not in self._snapshots:
            raise RuntimeError(f"unknown snapshot: {name}")
        self._restore_to(self._snapshots[name])

    def restore_to(self, ts: int) -> None:
        """Restore the whole DB to a raw commit timestamp (same semantics as
        ``restore(name)`` but keyed by ts instead of a snapshot name)."""
        self._restore_to(ts)

    def _restore_to(self, target: int) -> None:
        if self._txn is not None:
            raise RuntimeError("cannot restore during a transaction (ROLLBACK first)")
        if target > self._commit_ts:
            # Defensive: a snapshot whose state was already discarded by a prior
            # restore-to-earlier. Self-cleaning below makes this unreachable for
            # surviving snapshots, but guard anyway.
            raise RuntimeError(
                f"cannot restore to ts {target} > current commit ts {self._commit_ts} "
                "(that state was already discarded)"
            )
        touched = self.delta.rewind(target)
        self._commit_ts = target
        # Phase 2 step 6: durably drop the discarded tail from the WAL too, so a
        # crash right after a restore doesn't replay the discarded batches back
        # in. commit_ts is monotonic so the discarded records are a tail suffix
        # -> physical truncate to the first record past `target` (mirrors the
        # delta rewind). No-op when the WAL is disabled.
        self._wal.truncate(target)
        # Drop snapshots that now point at discarded state (ts > target). This is
        # the dangling-snapshot fix: without it, restore("b") after restore("a")
        # (where b's ts > a) would silently return state rolled back past b.
        for n in [n for n, ts in self._snapshots.items() if ts > target]:
            del self._snapshots[n]
        for table in touched:
            self._invalidate_table_caches(table)

    # ------------------------------------------------------------------ #
    # Phase 2 step 7 -- delta write-back (checkpoint)
    # ------------------------------------------------------------------ #

    def checkpoint(self) -> dict[str, int]:
        """Flush every committed delta table into a new base Parquet file, clear
        the delta, and truncate the WAL -- the durable write-back that keeps the
        delta + WAL from growing unbounded across a session. Returns ``{table:
        row_count}`` for the tables compacted (empty if nothing to flush).

        Full-store only (every committed table is folded in): the WAL is one
        record per COMMIT and a commit may span tables, so a per-table checkpoint
        could not cleanly drop just one table's records from the WAL. A full
        checkpoint makes every WAL record obsolete, so ``wal.truncate(0)`` is
        safe. Per-table checkpoint (with a WAL rewrite grouped by ``commit_ts``)
        is deferred.

        Type fidelity on disk: the merged frame (decimals coerced to float64,
        dates to datetime64 by the read path) is cast back to the catalog's
        declared Arrow schema, so the new Parquet file stores DECIMAL/DATE/BIGINT
        as their logical types. The Phase 5 cold Parquet reader targets the
        DuckDB physical layout (DECIMAL as INT64); pyarrow writes decimal128 as
        FIXED_LEN_BYTE_ARRAY, so the cold reader *defers* (cleanly, via
        ``_Defer``) on a checkpointed table and the warm cuDF path runs instead.
        ``row_count`` staleness is fixed as a side effect (``register``
        recomputes it from the new file's metadata).

        Ordering / invariants:
          * ``_scan(t, None)`` is the cuDF path (never the cold reader), so the
            merged frame is always materialised regardless of cold-reader deferral.
          * ``_commit_ts`` is NOT reset: within the session the next commit gets
            ``checkpoint_ts+1`` (no collision with the kept ``ts == checkpoint_ts``
            snapshot). Across a restart, snapshots are in-memory-only and dropped,
            so a fresh low-ts batch has no old snapshot to collide with -- this is
            the load-bearing reason snapshots stay in-memory-only.
          * Snapshots with ``ts < checkpoint_ts`` are dropped: their state was
            folded into base and a restore to them could not undo the now-base
            rows. ``ts == checkpoint_ts`` is kept (restore-to-tip == base).
        """
        if self._txn is not None:
            raise RuntimeError("cannot checkpoint during a transaction (COMMIT/ROLLBACK first)")
        if self.catalog.data_dir is None:
            raise RuntimeError("cannot checkpoint without a data dir (ephemeral engine)")
        targets = self.delta.tables()  # tables with >=1 committed batch
        if not targets:
            return {}  # nothing to flush
        checkpoint_ts = self._commit_ts  # the tip; all committed work is <= this
        written: dict[str, int] = {}
        for table in targets:
            if not self.delta.has_unflushed(table):
                continue
            info = self.catalog.get(table)
            merged = self._scan(table, None)  # base ++ ALL committed batches (no txn active)
            # Cast the merged frame to the on-disk dtypes from ``info.schema``
            # (DECIMAL/DATE fidelity) WITHOUT adopting its pandas metadata
            # verbatim: ``info.schema`` was captured from the ORIGINAL base, so
            # its ``index_columns`` encodes that base's RangeIndex ``stop`` (the
            # original row count). After any INSERT/DELETE the row count differs,
            # and cudf would reconstruct a wrong-length index on re-read
            # ("Length mismatch"). Drop the ``index_columns`` hint (cudf then
            # builds a default RangeIndex from the actual row count) while
            # keeping the per-column dtype hints that make the typed re-read
            # faithful. (Step 9: surfaced by delete-then-checkpoint, but the same
            # bug affected insert-checkpoint on object-string tables.)
            src = pa.Table.from_pandas(merged.to_pandas(), preserve_index=False)
            md = dict(info.schema.metadata or {})
            pandas_bytes = md.get(b"pandas")
            if pandas_bytes is not None:
                # Only touch pandas metadata when info.schema already carries it
                # (the typed-engine schema has none -- adding a partial dict
                # missing ``columns`` would break cudf's reader). Zero out
                # ``index_columns`` so cudf builds a default RangeIndex from the
                # actual row count; keep the per-column ``columns`` hints.
                pm = json.loads(pandas_bytes)
                pm["index_columns"] = []
                md[b"pandas"] = json.dumps(pm).encode()
            tbl = src.cast(pa.schema(list(info.schema), metadata=md or None))
            d = os.path.dirname(info.paths[0])
            final = os.path.join(d, "ryudb_base.parquet")
            tmp = final + ".tmp"
            pq.write_table(tbl, tmp, compression="snappy")
            old_paths = list(info.paths)
            os.replace(tmp, final)  # atomic publish
            # register re-derives schema + row_count from the new file and
            # preserves any constraints previously set via :alter (PK/UNIQUE/
            # DEFAULT/NOT NULL), then saves the catalog.
            self.catalog.register(table, final)
            self.delta.clear(table)
            for p in old_paths:
                # Skip the file we just published (final may equal an old path on
                # a re-checkpoint) and any already-gone path.
                if p != final and os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
            # The cached base-only frame is the OLD base (without the now-
            # checkpointed rows); code/pk facts are stale (data changed).
            self._drop_scan_cache_for(table)
            self._invalidate_table_caches(table)
            written[table] = self.catalog.get(table).row_count
        # Every committed batch is now in base -> the whole WAL is obsolete.
        # No-op when the WAL is disabled (ephemeral engine, already rejected
        # above, but harmless).
        self._wal.truncate(0)
        # Drop snapshots whose state was folded into base (ts < checkpoint_ts):
        # a restore to them could not undo the now-base rows. Keep
        # ts == checkpoint_ts (restore-to-tip == base, still valid).
        for name in [n for n, ts in self._snapshots.items() if ts < checkpoint_ts]:
            del self._snapshots[name]
        return written

    def is_unique_key(self, table: str, col: str, series) -> bool:
        """Return cached (table, col) uniqueness -- a dimension join key must be
        a primary key for the fused star-join path (a non-unique key would
        silently collapse joins). Cached so warm repeat queries skip the
        hash-count; cleared with the code index by `clear_code_cache`."""
        key = (table, col)
        if self.cache_enabled and key in self._pk_cache:
            return self._pk_cache[key]
        u = int(series.nunique()) == len(series)
        if self.cache_enabled:
            self._pk_cache[key] = u
        return u

    def get_codes(self, table: str, col: str, series):
        """Return cached (int64 codes, uniques list) for a group-key column,
        computing+caching on first use. Codes are positional (row-aligned) and
        deterministic for identical data, so they stay valid across warm runs."""
        key = (table, col)
        if self.cache_enabled and key in self._code_cache:
            return self._code_cache[key]
        codes, uniques = series.factorize()
        uniques = list(uniques.to_pandas())
        if self.cache_enabled:
            self._code_cache[key] = (codes, uniques)
        return codes, uniques

    def _scan(self, table: str, columns: set[str] | None) -> cudf.DataFrame:
        cols = frozenset(columns) if columns else None
        key = (table, cols)
        if self.cache_enabled and key in self._scan_cache:
            v = self._scan_cache[key]
            # Phase 5 async-materialise: a cold fused scan may have stored a
            # _PendingFrame whose background CUDA gather is still in flight.
            # Resolve it now (syncs the gather, builds the cuDF frame) and
            # replace the cache entry so subsequent warm reads hit the ready
            # frame directly. .get() returns None on failure -> fall through to
            # storage.scan (lose the cache, keep correctness).
            if isinstance(v, _PendingFrame):
                v = v.get()
                if v is None:
                    self._scan_cache.pop(key, None)
                else:
                    self._scan_cache[key] = v
            if v is not None and not isinstance(v, _PendingFrame):
                base = v
            else:
                base = scan(self.catalog.get(table), columns)
                if self.cache_enabled:
                    self._scan_cache[key] = base
        else:
            base = scan(self.catalog.get(table), columns)
            if self.cache_enabled:
                self._scan_cache[key] = base
        # Phase 2 delta merge: concatenate any unflushed batches (committed delta
        # OR an in-txn buffer) onto the base. The cache stays base-only (the
        # merged frame is never written back), so the live delta is re-merged each
        # read and a future append is visible with no invalidation. No pending
        # rows -> return base unchanged (zero cost).
        if self.has_pending(table):
            # Step 9: a DELETE tombstone is anti-joined in _merge_delta on ALL
            # primary-key columns, but a column-projected base may have dropped
            # some PK cols (e.g. ``SELECT count(*)`` projects to one column). When
            # tombstones are live, read the FULL base so every PK col is present,
            # merge, then re-project to the requested columns in storage.scan's
            # sorted order. Insert-only deltas don't reference PK cols, so the
            # projected base is kept for them (column pruning + warm cache
            # preserved). Tombstones are transient (cleared at checkpoint), so the
            # full-base read only applies during a live tombstone.
            has_tomb = self.delta.has_tombstones(table) or (
                self._txn is not None and self._txn.has_tombstone(table)
            )
            if has_tomb:
                full_base = scan(self.catalog.get(table), None)
                merged = self._merge_delta(full_base, table)
                if columns is not None:
                    merged = merged[sorted(columns)]
                return merged
            return self._merge_delta(base, table)
        return base

    def _merge_delta(self, base: cudf.DataFrame, table: str) -> cudf.DataFrame:
        """Return base ∪ delta for ``table`` as a fresh frame (base untouched).

        Each batch is cast column-wise to ``base[col].dtype`` before concat. This
        single rule reconciles the datetime-unit divergence (cold-cache base is
        ``datetime64[s]`` while ``storage.scan`` is ``[ms]``) and any int-width
        difference, so concat never fails on dtype mismatch. Column order follows
        ``base`` (already projected/sorted by ``storage.scan`` or the cold cache).
        Batches are assumed full-schema; a missing projected column raises
        KeyError -- a useful failure for a malformed INSERT once the write path
        exists.

        MVCC (step 5): inside a transaction the visible batches are
        ``batches_at(snapshot_ts)`` (committed at-or-before the snapshot) followed
        by the txn's own buffered frames (read-your-writes). Outside a txn
        (autocommit) ``batches(table)`` returns ALL committed frames. The buffer
        frames are appended in the same order at COMMIT, so the visible series is
        byte-identical pre/post commit -- which is what makes COMMIT/ROLLBACK/
        RESTORE cache invalidation a safe over-invalidation.

        Phase 2 step 9 (timestamp-aware tombstone anti-join): every visible row is
        tagged with an insertion ts (``_ins_ts``: base=0, each committed insert
        batch=its ``commit_ts``, in-txn buffered insert=``_INS_INF``). A tombstone
        (PK values + its ``commit_ts``) removes a row iff a matching tombstone has
        ``tomb_ts >= ins_ts`` -- i.e. only rows that existed at delete time. This
        is what lets a DELETE followed by a same-PK reinsert (at a newer ts) work:
        the reinsert's ``ins_ts`` exceeds the tombstone's ``ts``, so it survives.
        Per PK we reduce the tombstones to ``max(tomb_ts)`` and keep rows where
        that max is below the row's ``ins_ts`` (or no tombstone matches). cuDF merge
        has no ``indicator=`` support, so the anti-join is a left merge on the PK
        cols + a keep-where-null-or-less filter. Requires a declared PK (DELETE
        enforces this); a tombstoned table with no PK is unreachable.
        """
        if self._txn is not None:
            ins_batches = self.delta.batches_at_with_ts(table, self._txn.snapshot_ts)
            buf_inserts = self._txn.buffer_batches(table)
            tomb = self.delta.tombstones_at_with_ts(table, self._txn.snapshot_ts)
            buf_tomb = self._txn.tombstone_batches(table)
        else:
            ins_batches = self.delta.batches_with_ts(table)
            buf_inserts = []
            tomb = self.delta.tombstones_with_ts(table)
            buf_tomb = []
        if not ins_batches and not buf_inserts and not tomb and not buf_tomb:
            return base
        cols = list(base.columns)
        # Tag every source with its insertion ts (base=0, committed batch=ts,
        # buffered insert=+inf). ``base.assign`` returns a fresh frame so the
        # cached base is never mutated; batch slices are copies (see _insert).
        parts: list[cudf.DataFrame] = [base.assign(_ins_ts=0)]
        for ts, b in ins_batches:
            sub = b[cols]
            for c in cols:
                if sub[c].dtype != base[c].dtype:
                    sub[c] = sub[c].astype(base[c].dtype)
            sub["_ins_ts"] = ts
            parts.append(sub)
        for b in buf_inserts:
            sub = b[cols]
            for c in cols:
                if sub[c].dtype != base[c].dtype:
                    sub[c] = sub[c].astype(base[c].dtype)
            sub["_ins_ts"] = _INS_INF
            parts.append(sub)
        merged = cudf.concat(parts, axis=0).reset_index(drop=True)
        # Tombstones: committed carry their ts, buffered carry +inf (removes
        # committed rows + buffered inserts of the same PK -> read-your-writes).
        tomb_entries: list[tuple[int, cudf.DataFrame]] = list(tomb) + [
            (_INS_INF, t) for t in buf_tomb
        ]
        if tomb_entries:
            pk = self.catalog.get(table).constraints.primary_key
            if pk is not None:
                tkeys_parts = []
                for ts, t in tomb_entries:
                    tt = t[list(pk)].copy()
                    tt["_tomb_ts"] = ts
                    tkeys_parts.append(tt)
                tkeys = cudf.concat(tkeys_parts, axis=0)
                # max tombstone ts per PK (the most recent delete wins; older
                # tombstones for the same PK are subsumed).
                tomb_max = (
                    tkeys.groupby(list(pk))["_tomb_ts"].max().reset_index()
                )
                # Align tombstone key dtypes to the merged (base) series before
                # the join -- same nullable-Int64-vs-int64 trap as _enforce_unique.
                for c in pk:
                    if str(tomb_max[c].dtype) != str(merged[c].dtype):
                        tomb_max[c] = tomb_max[c].astype(merged[c].dtype)
                merged = merged.merge(tomb_max, on=list(pk), how="left")
                # Non-matching rows get NaN _tomb_ts; fill with -1 (below every
                # real commit_ts >= 0) so a single clean comparison keeps them:
                # keep iff the newest matching tombstone predates the row's
                # insertion (``tomb_ts < ins_ts``). This avoids cuDF's NA-boolean
                # ``|`` (which yields null, not True, for the isna side).
                merged["_tomb_ts"] = merged["_tomb_ts"].fillna(-1)
                keep = merged["_tomb_ts"] < merged["_ins_ts"]
                merged = merged[keep].drop(columns=["_tomb_ts"])
        return merged.drop(columns=["_ins_ts"]).reset_index(drop=True)

    def _enforce_unique(self, table: str, frame: "cudf.DataFrame") -> None:
        """Reject INSERT rows that violate a declared PRIMARY KEY or UNIQUE
        constraint (Phase 2 step 8).

        Declared-constraints-only: reads ``TableConstraints.primary_key`` and
        ``TableConstraints.unique`` -- NOT ``is_unique_key``/``_pk_cache``, which
        are data-uniqueness facts used as the fused star-join's dimension-PK
        eligibility gate. Gating enforcement on those would reject duplicates on
        any column whose data merely happens to be unique with no declared
        constraint. Called in ``_insert`` BEFORE the WAL write / buffer append so
        a rejected insert leaves no durable or in-transaction state
        (all-or-nothing). PK columns are NOT NULL (set by ``set_primary_key``,
        enforced earlier in ``_insert``); UNIQUE columns are nullable and NULLs
        are exempt (standard SQL: NULL != NULL). Cost is a projection-pruned scan
        of the key column(s) per constraint -- naive but correct;
        index-accelerated enforcement is a future optimization.
        """
        info = self.catalog.get(table)
        pk = info.constraints.primary_key
        uniq = info.constraints.unique
        if pk is None and not uniq:
            return
        constraints = ([pk] if pk is not None else []) + list(uniq)
        for key_cols in constraints:
            key_cols = list(key_cols)
            # NULLs are exempt from UNIQUE (NULL != NULL); PK cols are NOT NULL so
            # the dropna is a no-op for PK, but it keeps the UNIQUE check correct.
            non_null = frame.dropna(subset=key_cols)
            # (a) internal duplicates within this batch.
            if len(non_null) != non_null[key_cols].drop_duplicates().shape[0]:
                raise RuntimeError(
                    f"UNIQUE violation: {table}.{key_cols} duplicate within INSERT batch"
                )
            # (b) collisions with the existing visible series. _scan(table, set)
            # is projection-pruned and read-your-writes in a txn (includes
            # already-buffered rows), so a 2nd in-txn INSERT colliding with a 1st
            # buffered INSERT is caught.
            existing = self._scan(table, set(key_cols))
            existing_nn = existing.dropna(subset=key_cols)
            if existing_nn.shape[0]:
                new_keys = non_null[key_cols].reset_index(drop=True)
                ex_keys = existing_nn[key_cols].reset_index(drop=True)
                # Align the new batch's key dtypes to the existing (base) series
                # before the join: the batch is built with nullable Int64
                # (_arrow_match_dtype) while the base scan is int64, and cuDF
                # merge cannot reconcile numpy Int64Dtype vs pandas Int64Dtype.
                for c in key_cols:
                    if str(new_keys[c].dtype) != str(ex_keys[c].dtype):
                        new_keys[c] = new_keys[c].astype(ex_keys[c].dtype)
                merged = new_keys.merge(ex_keys, on=key_cols, how="inner")
                if merged.shape[0]:
                    raise RuntimeError(
                        f"UNIQUE violation: {table}.{key_cols} "
                        f"{merged.shape[0]} row(s) already exist"
                    )

    def _insert(self, node: Insert) -> int:
        """Append ``INSERT ... VALUES`` rows to the table's delta (Phase 2 step 3).

        Resolves the full schema from the catalog, fills DEFAULTs for omitted
        columns, enforces NOT NULL, builds a typed cuDF batch whose columns cast
        cleanly to the base at merge time, and appends it to the table's write
        target. Outside a transaction (autocommit) the batch is appended to
        ``self.delta`` under a fresh commit_ts and the next SELECT re-merges it
        in ``_scan``. Inside a transaction the batch is buffered on the txn
        (read-your-writes) and flushed to ``self.delta`` at COMMIT. Declared
        PRIMARY KEY / UNIQUE uniqueness is enforced by ``_enforce_unique``
        BEFORE the WAL write / buffer append (all-or-nothing: a rejected
        insert leaves no durable or in-transaction state); only NOT NULL +
        DEFAULT + type coercion otherwise. Returns the row count appended.
        """
        info = self.catalog.get(node.table)
        if info is None:
            raise RuntimeError(f"unknown table: {node.table}")
        all_cols = list(info.columns)
        cols = list(node.columns) if node.columns is not None else list(all_cols)
        unknown = [c for c in cols if c not in all_cols]
        if unknown:
            raise ParseError(f"unknown columns in {node.table}: {unknown}")
        if len(set(cols)) != len(cols):
            raise ParseError(f"INSERT column list has duplicates: {cols}")
        for i, row in enumerate(node.rows):
            if len(row) != len(cols):
                raise ParseError(
                    f"INSERT row {i} has {len(row)} values for {len(cols)} columns"
                )

        not_null = info.constraints.not_null
        defaults = info.constraints.defaults
        types = info.types

        # Per-column python value lists in full-schema order (provided value,
        # else DEFAULT, else NULL), with NOT NULL enforced on the resolved value.
        data: dict[str, list] = {c: [] for c in all_cols}
        provided_idx = {c: i for i, c in enumerate(cols)}
        for row in node.rows:
            pyvals = [_literal(cell) for cell in row]
            for c in all_cols:
                if c in provided_idx:
                    v = pyvals[provided_idx[c]]
                elif c in defaults:
                    v = defaults[c]
                else:
                    v = None
                if v is None and c in not_null:
                    raise RuntimeError(
                        f"NOT NULL violation: {node.table}.{c} (row {len(data[c])})"
                    )
                data[c].append(v)

        # Build a typed pandas frame, then move to cuDF. The dtypes follow the
        # base column families via _arrow_match_dtype (decimal->float64 matching
        # storage._coerce_decimals, int->int64, date->datetime64[s]); since
        # _merge_delta casts each delta column to base[c].dtype anyway, the exact
        # unit/width here only needs to be value-preserving. Decimals are passed
        # as float (base is float64) -- never Decimal -- so astype is trivial.
        pdf = {}
        for c in all_cols:
            arr = data[c]
            dt = _arrow_match_dtype(types[c])
            if "datetime" in str(dt):
                pdf[c] = pd.to_datetime(arr, errors="coerce")
            elif str(dt).startswith("int"):
                # Nullable Int64 so a NULL (on a nullable col) is pd.NA, not a
                # coercion error; _merge_delta casts to base int64 at read time.
                pdf[c] = pd.array(arr, dtype="Int64")
            else:
                pdf[c] = pd.array(arr, dtype=dt)
        frame = cudf.DataFrame(pd.DataFrame(pdf))
        # Phase 2 step 8: enforce declared PK/UNIQUE BEFORE the WAL write / buffer
        # append so a rejected insert is all-or-nothing (no torn WAL record, no
        # partial buffer). Runs on the typed frame; never mutates it.
        self._enforce_unique(node.table, frame)
        # Phase 2 step 5: inside a transaction, buffer the frame (visible only to
        # this txn via read-your-writes; flushed to the shared delta at COMMIT).
        # Outside a txn (autocommit) append it to the delta now under a fresh
        # commit_ts. Either way the visible series for this table changes, so the
        # maintained-fact caches must be dropped -- including on buffer-append,
        # or a warm in-txn read would leave stale codes read OOB on the next
        # in-txn read after the buffer grows.
        # Phase 2 step 6: the autocommit path routes through _write_commit so the
        # batch is also written+fsync'd to the WAL before the in-memory delta
        # mutates (durable). The buffered (txn) path never touches the WAL -- an
        # uncommitted txn is implicitly rolled back on process exit.
        if self._txn is not None:
            self._txn.buffer_append(node.table, frame)
            self._invalidate_table_caches(node.table)
        else:
            self._write_commit([(node.table, "insert", frame)])
        return len(frame)

    def _delete(self, node: Delete) -> int:
        """Delete rows matching ``DELETE FROM t [WHERE pred]`` (Phase 2 step 9).

        Evaluates the predicate (if any) against the currently-visible snapshot
        of ``t`` (base ++ committed inserts, tombstones already applied, in-txn
        read-your-writes), collects the **primary-key values** of the matched
        rows, and stores them as a tombstone batch. The read path anti-joins
        tombstones out in ``_merge_delta``. Only rows existing at delete time are
        tombstoned (correct MVCC: a future INSERT of the same PK is allowed --
        ``_enforce_unique`` sees the tombstoned PK as gone). Autocommit routes
        through ``_write_commit`` (durable, all-or-nothing); in a txn the
        tombstone is buffered (read-your-writes). Returns the count of rows
        deleted (matched the predicate against the visible snapshot).

        Requires a declared PRIMARY KEY (row identity is by PK value, not row
        position -- ``storage.scan`` has no row ids and ``_merge_delta`` resets
        the index). Cost is a full-column scan of the visible snapshot (naive,
        like step 8's PK scan -- correct first; index-accelerated DELETE is a
        future optimization).
        """
        info = self.catalog.get(node.table)
        if info is None:
            raise RuntimeError(f"unknown table: {node.table}")
        pk = info.constraints.primary_key
        if pk is None:
            raise RuntimeError(
                f"DELETE requires a declared PRIMARY KEY on {node.table}"
            )
        # Visible snapshot to delete from. _scan applies committed tombstones
        # already, so a second DELETE with the same predicate sees fewer rows.
        visible = self._scan(node.table, None)
        if node.predicate is not None:
            mask = eval_expr(node.predicate, visible)
            if isinstance(mask, cudf.Series):
                targets = visible[mask]
            else:
                targets = visible if mask else visible.iloc[0:0]
        else:
            targets = visible
        if len(targets) == 0:
            return 0
        tombstone = targets[list(pk)]  # PK values of the rows to delete
        if self._txn is not None:
            self._txn.tombstone_append(node.table, tombstone)
            self._invalidate_table_caches(node.table)
        else:
            self._write_commit([(node.table, "tombstone", tombstone)])
        return len(targets)

    def sql(self, sql: str) -> "cudf.DataFrame | int | None":
        # Non-standard snapshot/restore bypass sqlglot entirely (no AST node).
        # Cheap prefix guard so the regex never runs on a SELECT/INSERT/BEGIN.
        s = sql.lstrip()
        if s[:7].upper() in ("CREATE ", "RESTORE"):
            m = _SNAPSHOT_RE.match(s)
            if m:
                self.snapshot(m.group(1))
                return None
            m = _RESTORE_RE.match(s)
            if m:
                self.restore(m.group(1))
                return None
        plan = parse(sql, self.catalog.schema_dict())
        # INSERT, DELETE, and TxnControl are non-relational leaves with no
        # predicate / projection / join to optimize; bypass the optimizer (the
        # rules are pass-through-safe today, but a future Select-shaped rule
        # could choke on an Insert/Delete/TxnControl root). A DELETE's WHERE is a
        # row-selector evaluated in _delete, not a relational Filter.
        if not isinstance(plan, (Insert, Delete, TxnControl)):
            plan = optimize(
                plan,
                self.catalog.schema_dict(),
                self.catalog.stats_dict(),
            )
        return self.execute(plan)

    def explain(self, sql: str) -> str:
        from ..sql.plan import pretty

        s = sql.lstrip()
        if s[:7].upper() in ("CREATE ", "RESTORE"):
            m = _SNAPSHOT_RE.match(s)
            if m:
                return f"CreateSnapshot({m.group(1)})"
            m = _RESTORE_RE.match(s)
            if m:
                return f"RestoreToSnapshot({m.group(1)})"
        plan = parse(sql, self.catalog.schema_dict())
        if not isinstance(plan, (Insert, Delete, TxnControl)):
            plan = optimize(plan, self.catalog.schema_dict(), self.catalog.stats_dict())
        return pretty(plan)

    def execute(self, plan: PlanNode) -> "cudf.DataFrame | int | None":
        return self._exec(plan)

    def _exec(self, node: PlanNode) -> "cudf.DataFrame | int | None":
        if isinstance(node, Scan):
            return self._scan(node.table, node.columns)
        if isinstance(node, Insert):
            return self._insert(node)
        if isinstance(node, Delete):
            return self._delete(node)
        if isinstance(node, TxnControl):
            self._txn_control(node)
            return None
        if isinstance(node, Filter):
            df = self._exec(node.input)
            mask = eval_expr(node.predicate, df)
            if isinstance(mask, cudf.Series):
                return df[mask]
            return df if mask else df.iloc[0:0]
        if isinstance(node, Join):
            left = self._exec(node.left)
            right = self._exec(node.right)
            return left.merge(
                right,
                left_on=node.on_left,
                right_on=node.on_right,
                how=node.how,
                suffixes=("_x", "_y"),
            )
        if isinstance(node, Aggregate):
            return self._aggregate(node)
        if isinstance(node, Project):
            return self._project(node)
        if isinstance(node, Sort):
            return self._sort(node)
        if isinstance(node, Limit):
            return self._limit(node)
        raise NotImplementedError(f"no executor for {type(node).__name__}")

    # ------------------------------------------------------------------ #
    def _aggregate(self, node: Aggregate) -> cudf.DataFrame:
        group_keys = node.group_keys
        aggs = node.aggs
        by_names = [gn for _, gn in group_keys]

        # No-gather optimization: when a Filter sits directly below the Aggregate
        # and every group key is a non-nullable column, fold the predicate into the
        # groupby by nulling the group keys of failing rows (groupby dropna drops
        # them) instead of materialising a filtered copy. On TPC-H Q1 this avoids
        # copying ~98% of 60M rows and cuts compute roughly in half.
        in_node = node.input
        if isinstance(in_node, Filter):
            # Phase 5 step 3: the cold Parquet reader runs the whole Aggregate ->
            # Filter -> Scan straight off the Parquet pages (nvCOMP Snappy ->
            # decode -> filter -> accumulate) WITHOUT materialising the 60M-row
            # cuDF frame, and on success populates _scan_cache (keyed identically
            # to _scan) so warm repeats hit the GPU-resident frame. It is the
            # DEFAULT cold path now (the RYUDB_SCAN_KERNEL opt-in gate is dropped):
            # try it only on a cache miss -- a hit means a prior cold run already
            # cached the frame, so skip straight to the materialising path below
            # which reads that cached frame. Returns None for unsupported shapes
            # and on any C++/metadata fault (correctness never depends on it) ->
            # cuDF fallback below, which also populates the cache via _scan.
            scan_node = in_node.input
            if isinstance(scan_node, Scan):
                _skey = (scan_node.table,
                         frozenset(scan_node.columns) if scan_node.columns else None)
                if _skey not in self._scan_cache:
                    res = fused_scan_aggregate(node, self)
                    if res is not None:
                        return res
            child = self._exec(in_node.input)
            # Phase 3b/4: try the fused C++/CUDA filter+groupby+aggregate kernel
            # first -- it now handles grouped AND global aggregates, and the
            # SUM/AVG/MIN/MAX/COUNT(*) kinds. Returns None for unsupported shapes
            # (no Filter match, OR predicate, COUNT(col), nullable AVG/MIN/MAX
            # args, multi-col numeric GROUP BY, ...) -> cuDF fallback below.
            res = fused_aggregate(node, child, self)
            if res is not None:
                return res
            mask = eval_expr(in_node.predicate, child)
            if not group_keys:
                # Global aggregate, fused-ineligible -> scalar reductions on the
                # filtered frame.
                df = child[mask] if isinstance(mask, cudf.Series) else (child if mask else child.iloc[0:0])
                return self._scalar_global_agg(df, aggs)
            if isinstance(mask, cudf.Series) and self._keys_nonnull(child, group_keys):
                # no-gather path mutates `work`: copy the cached pristine frame so
                # the scan/code caches are never corrupted.
                return self._fused_agg(child.copy(), child, group_keys, aggs, by_names, dropna=True, mask=mask)
            # fall back: gather then aggregate normally
            df = child[mask] if isinstance(mask, cudf.Series) else (child if mask else child.iloc[0:0])
            return self._fused_agg(df, df, group_keys, aggs, by_names, dropna=False, mask=None)

        # No Filter below the Aggregate.
        if not group_keys:
            return self._scalar_global_agg(self._exec(in_node), aggs)
        # Phase 4 step 2: try the fused star-join + aggregate kernel before
        # materialising the joined frame. It works on the plan (not an executed
        # frame) so the join output is never built; returns None instantly when
        # node.input isn't a Join, leaving the Aggregate -> Scan path unchanged.
        res = fused_join_aggregate(node, self)
        if res is not None:
            return res
        df = self._exec(in_node)
        return self._fused_agg(df, df, group_keys, aggs, by_names, dropna=False, mask=None)

    def _scalar_global_agg(self, df: cudf.DataFrame, aggs) -> cudf.DataFrame:
        """Scalar/global aggregate (no GROUP BY): one output row via cuDF
        reductions. This is the fallback for fused-ineligible global aggregates
        (and for `Aggregate -> Scan` shapes with no Filter)."""
        row: dict[str, list] = {}
        for af, n in aggs:
            if af.func == "COUNT" and isinstance(af.arg, Star):
                row[n] = [int(len(df))]
            else:
                col = eval_expr(af.arg, df)
                row[n] = [_scalar_agg(af.func, col)]
        return cudf.DataFrame(row)

    def _keys_nonnull(self, df: cudf.DataFrame, group_keys) -> bool:
        # Only fold when every group key is a plain column reference with no nulls;
        # otherwise nulling keys to drop filtered rows would also drop genuine
        # NULL-key rows (incorrect), or we can't cheaply prove nullability.
        for ge, _ in group_keys:
            if not isinstance(ge, Col):
                return False
            if ge.name not in df.columns:
                return False
            if df[ge.name].null_count != 0:
                return False
        return True

    def _fused_agg(
        self,
        work: cudf.DataFrame,
        src: cudf.DataFrame,
        group_keys,
        aggs,
        by_names: list[str],
        dropna: bool,
        mask: "cudf.Series | None",
    ) -> cudf.DataFrame:
        # Build the frame we group by. In the gather path `work` is already the
        # filtered frame (group keys present). In the no-gather path we replace
        # the group-key columns with mask-nullified copies so failing rows are
        # dropped by the groupby; agg-arg columns are referenced, not copied.
        if mask is not None:
            for ge, gn in group_keys:
                # ge is guaranteed Col here (checked by _keys_nonnull); null its
                # group-key column where the predicate fails so the groupby drops
                # those rows. Other (agg-arg) columns stay referenced and unmasked:
                # failing rows never reach the aggregates because they are dropped
                # by their null keys.
                work[gn] = df_where(work[gn] if gn in work.columns else src[ge.name], mask)
        else:
            for ge, gn in group_keys:
                if gn not in work.columns:
                    work[gn] = eval_expr(ge, src)

        # Fused single-pass aggregation: one groupby.agg({col: [funcs]}) call
        # instead of one kernel launch per aggregate. COUNT(*) folds in via a
        # constant non-null column counted per group.
        work["__cnt"] = 1
        spec: dict[str, list[str]] = {}
        out_map: list[tuple[str, str, str]] = []
        for af, n in aggs:
            if af.func == "COUNT" and isinstance(af.arg, Star):
                spec.setdefault("__cnt", []).append("count")
                out_map.append(("__cnt", "count", n))
                continue
            tmp = f"__a_{n}"
            work[tmp] = eval_expr(af.arg, src)
            func = "count" if af.func == "COUNT" else _AGG_METHOD[af.func]
            spec.setdefault(tmp, []).append(func)
            out_map.append((tmp, func, n))

        grouped = work.groupby(by_names, dropna=dropna)
        res = grouped.agg(spec)
        pieces = [res[(c, f)].rename(n) for c, f, n in out_map]
        out = cudf.concat(pieces, axis=1).reset_index()
        return out[by_names + [n for _, n in aggs]]

    def _project(self, node: Project) -> cudf.DataFrame:
        df = self._exec(node.input)
        # Build on the input's index so Series columns align and scalar columns
        # broadcast without materializing a Python list of len(df) elements.
        out = cudf.DataFrame(index=df.index)
        for e, name in node.items:
            v = eval_expr(e, df)
            out[name] = v  # Series aligns by index; scalar broadcasts to all rows
        return out

    def _sort(self, node: Sort) -> cudf.DataFrame:
        df = self._exec(node.input)
        if not node.keys:
            return df
        by = [k.name for k, _ in node.keys]
        ascending = [a for _, a in node.keys]
        return df.sort_values(by=by, ascending=ascending)

    def _limit(self, node: Limit) -> cudf.DataFrame:
        df = self._exec(node.input)
        end = node.offset + node.n
        return df.iloc[node.offset:end]


def _scalar_agg(func: str, series) -> object:
    if func == "COUNT":
        return int(series.count())
    method = _AGG_METHOD[func]
    return getattr(series, method)()


def df_where(series, mask):
    """Return a copy of `series` with values nullified where `mask` is False.

    Used by the no-gather aggregate path to drop filtered rows from a groupby
    by nulling their group keys (the groupby then drops them via dropna=True),
    without materialising a filtered row copy.
    """
    return series.where(mask)