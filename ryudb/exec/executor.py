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
    Derived,
    Distinct,
    Expr,
    Filter,
    Insert,
    Join,
    Limit,
    PlanNode,
    Project,
    Scan,
    SetOp,
    Sort,
    Star,
    TxnControl,
    Update,
    Window,
    WindowFunc,
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
                self.delta.append_tombstone(table, frame, cts, exclude_same_ts=False)
            elif kind == "tombstone_update":
                # step 10: an UPDATE tombstone removes rows with ins_ts < tomb_ts
                # (strict), so the re-inserted row (same commit ts) survives.
                self.delta.append_tombstone(table, frame, cts, exclude_same_ts=True)
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
        # mixed INSERT+DELETE commit is one record with per-batch kind. Step 10:
        # a tombstone's kind is ``"tombstone"`` (DELETE) or ``"tombstone_update"``
        # (UPDATE) from its buffered ``exclude_same_ts`` flag.
        batches: list[tuple[str, str, cudf.DataFrame]] = [
            (t, "insert", f) for t in txn.tables() for f in txn.buffer_batches(t)
        ] + [
            (t, "tombstone_update" if fl else "tombstone", f)
            for t in txn.tombstone_tables()
            for f, fl in txn.tombstone_batches_with_flag(t)
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
                self.delta.append_tombstone(table, frame, ts, exclude_same_ts=False)
            elif kind == "tombstone_update":
                # step 10: an UPDATE tombstone removes rows with ins_ts < tomb_ts
                # (strict), so the re-inserted row (same commit ts) survives.
                self.delta.append_tombstone(table, frame, ts, exclude_same_ts=True)
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
        (PK values + its ``commit_ts`` + a ``exclude_same_ts`` flag) removes a row
        iff a matching tombstone has ``tomb_ts >= ins_ts`` -- i.e. only rows that
        existed at delete time. This is what lets a DELETE followed by a same-PK
        reinsert (at a newer ts) work: the reinsert's ``ins_ts`` exceeds the
        tombstone's ``ts``, so it survives. Per PK we reduce the tombstones to
        ``max(tomb_ts)`` and keep rows where that max is below the row's ``ins_ts``
        (or no tombstone matches). cuDF merge has no ``indicator=`` support, so
        the anti-join is a left merge on the PK cols + a keep-where-null-or-less
        filter. Requires a declared PK (DELETE/UPDATE enforce this); a tombstoned
        table with no PK is unreachable.

        Phase 2 step 10 (UPDATE tombstone flag): a tombstone carries an
        ``exclude_same_ts`` flag -- ``False`` for a DELETE (removes
        ``ins_ts <= tomb_ts``) and ``True`` for an UPDATE (removes
        ``ins_ts < tomb_ts``, so the re-inserted row at the *same* commit ts
        survives its own tombstone). Per PK we reduce the two kinds *separately*
        to ``max(tomb_ts)``: ``_tomb_del`` over DELETE tombstones and ``_tomb_upd``
        over UPDATE tombstones, then keep iff ``_tomb_del < ins_ts`` AND
        ``_tomb_upd <= ins_ts``. Reducing the two kinds separately (rather than
        tracking the flag at the overall max ts) is equivalent for every
        reachable history and avoids a flag-at-max rejoin -- for all-DELETE
        histories ``_tomb_upd`` is empty so the second clause is trivially true
        and the path is byte-identical to step 9.
        """
        if self._txn is not None:
            ins_batches = self.delta.batches_at_with_ts(table, self._txn.snapshot_ts)
            buf_inserts = self._txn.buffer_batches(table)
            tomb = self.delta.tombstones_at_with_ts(table, self._txn.snapshot_ts)
            buf_tomb = self._txn.tombstone_batches_with_flag(table)
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
        # Tombstones: committed carry (ts, flag), buffered carry (+inf, flag)
        # (removes committed rows + buffered inserts of the same PK -> read-your-
        # writes). ``flag`` is 0 for a DELETE tombstone (removes ins_ts <= tomb_ts)
        # and 1 for an UPDATE tombstone (removes ins_ts < tomb_ts, so the re-inserted
        # row at the same commit ts survives its own tombstone -- step 10).
        tomb_entries: list[tuple[int, cudf.DataFrame, int]] = list(tomb) + [
            (_INS_INF, t, 1 if fl else 0) for t, fl in buf_tomb
        ]
        if tomb_entries:
            pk = self.catalog.get(table).constraints.primary_key
            if pk is not None:
                tkeys_parts = []
                for ts, t, fl in tomb_entries:
                    tt = t[list(pk)].copy()
                    tt["_tomb_ts"] = ts
                    tt["_flag"] = fl
                    tkeys_parts.append(tt)
                tkeys = cudf.concat(tkeys_parts, axis=0)
                # Newest DELETE tombstone (flag=0) and newest UPDATE tombstone
                # (flag=1) per PK. A DELETE removes a row iff tomb_ts >= ins_ts
                # (keep iff tomb_ts < ins_ts); an UPDATE removes iff tomb_ts > ins_ts
                # (keep iff tomb_ts <= ins_ts). Keep iff BOTH keep. Reducing the two
                # kinds separately (rather than the flag at the max ts) is equivalent
                # for every reachable history and avoids a flag-at-max rejoin.
                del_max = (
                    tkeys[tkeys["_flag"] == 0]
                    .groupby(list(pk), as_index=False)["_tomb_ts"].max()
                    .rename(columns={"_tomb_ts": "_tomb_del"})
                )
                upd_max = (
                    tkeys[tkeys["_flag"] == 1]
                    .groupby(list(pk), as_index=False)["_tomb_ts"].max()
                    .rename(columns={"_tomb_ts": "_tomb_upd"})
                )
                # Align tombstone key dtypes to the merged (base) series before
                # the join -- same nullable-Int64-vs-int64 trap as _enforce_unique.
                for tomb in (del_max, upd_max):
                    for c in pk:
                        if str(tomb[c].dtype) != str(merged[c].dtype):
                            tomb[c] = tomb[c].astype(merged[c].dtype)
                merged = merged.merge(del_max, on=list(pk), how="left")
                merged = merged.merge(upd_max, on=list(pk), how="left")
                # Non-matching rows get NaN; fill with -1 (below every real
                # commit_ts >= 0) so the corresponding clause is always satisfied.
                # Avoids cuDF's NA-boolean ``&`` (which yields null, not True).
                merged["_tomb_del"] = merged["_tomb_del"].fillna(-1)
                merged["_tomb_upd"] = merged["_tomb_upd"].fillna(-1)
                keep = (merged["_tomb_del"] < merged["_ins_ts"]) & (
                    merged["_tomb_upd"] <= merged["_ins_ts"]
                )
                merged = merged[keep].drop(columns=["_tomb_del", "_tomb_upd"])
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
        if node.source is not None:
            # INSERT ... SELECT: the rows come from executing the subplan, not
            # from literal cells. The two forms share the durable tail but differ
            # in how the typed batch is built, so dispatch to a dedicated helper.
            return self._insert_select(node)
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

    def _insert_select(self, node: Insert) -> int:
        """Append an ``INSERT ... SELECT`` result to the table's delta.

        The subplan (already optimized in ``sql``/``explain``) is materialized to
        a cuDF frame, whose output columns map **positionally** onto the target
        column list (standard SQL -- SELECT output names are ignored, exactly
        like ``_setop``'s positional rename). Omitted target columns take DEFAULT
        then NULL. The frame is round-tripped through pandas and each column is
        coerced via the same ``_arrow_match_dtype`` path ``_insert`` uses for
        VALUES, so the resulting batch is byte-compatible with a VALUES batch:
        ``_merge_delta`` casts it to ``base[c].dtype`` cleanly, WAL replay
        reconstructs it identically, and ``_enforce_unique`` reconciles key
        dtypes. NOT NULL is enforced on the frame (frame-based ``isna``, like
        ``_update``) BEFORE ``_enforce_unique`` so a NULL PK can't slip past the
        latter's ``dropna``. ``_enforce_unique`` runs BEFORE the WAL write /
        buffer append (all-or-nothing). The txn / autocommit tail is identical to
        the VALUES path.
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

        src = self._exec(node.source)
        # 0 rows: a valid no-op insert (e.g. WHERE false). Avoid a pointless WAL
        # record and any empty-frame edge case in the durable tail.
        if len(src) == 0:
            return 0
        if len(src.columns) != len(cols):
            raise ParseError(
                f"INSERT ... SELECT column count mismatch: "
                f"{len(src.columns)} source vs {len(cols)} target"
            )

        not_null = info.constraints.not_null
        defaults = info.constraints.defaults
        types = info.types

        # Positional map: SELECT output column i -> target cols[i] (names ignored).
        # reset_index: a filtered source frame carries a non-contiguous index
        # (the surviving row positions); without reset, a provided column becomes
        # a Series with that index while an omitted (DEFAULT/NULL) column becomes
        # a Series with a fresh RangeIndex, and pd.DataFrame(out) aligns the two
        # -> "array length ... does not match index length ...". Reset so every
        # column Series shares RangeIndex(0, n); ExtensionArray columns are
        # positional and never align.
        src_pdf = src.to_pandas().reset_index(drop=True)
        provided = {c: src_pdf.iloc[:, i] for i, c in enumerate(cols)}

        # Full-schema pandas frame in catalog order, with DEFAULT/NULL fill and
        # the same per-column dtype coercion _insert uses for VALUES.
        out: dict = {}
        n = len(src_pdf)
        for c in all_cols:
            if c in provided:
                s = provided[c]
            elif c in defaults:
                s = pd.Series([defaults[c]] * n)
            else:
                s = pd.Series([None] * n)
            dt = _arrow_match_dtype(types[c])
            if "datetime" in str(dt):
                out[c] = pd.to_datetime(s, errors="coerce")
            elif str(dt).startswith("int"):
                # Nullable Int64 so a NULL on a nullable col is pd.NA, not a
                # coercion error; _merge_delta casts to base int64 at read time.
                out[c] = pd.array(s, dtype="Int64")
            else:
                out[c] = pd.array(s, dtype=dt)
            if c in not_null and out[c].isna().any():
                raise RuntimeError(f"NOT NULL violation: {node.table}.{c}")
        frame = cudf.DataFrame(pd.DataFrame(out))
        self._enforce_unique(node.table, frame)
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

    def _typed_series(self, arrow_type, arr: list) -> "cudf.Series":
        """Coerce a python value list to a cuDF Series following the base column
        dtype families -- the same per-column path ``_insert`` uses (decimal/
        float->float64, int->nullable Int64 so a NULL is ``pd.NA`` not a coercion
        error, date->datetime64, else->object), so ``_merge_delta`` casts the
        resulting batch column to base cleanly at read time. Used by ``_update``
        to coerce each SET column's evaluated values. (Kept separate from
        ``_insert``'s inline build to avoid perturbing the tested INSERT path.)"""
        dt = _arrow_match_dtype(arrow_type)
        if "datetime" in str(dt):
            return cudf.Series(pd.to_datetime(arr, errors="coerce"))
        if str(dt).startswith("int"):
            # Nullable Int64: a NULL (on a nullable col) is pd.NA, not a coercion
            # error; _merge_delta casts to base int64 at read time.
            return cudf.Series(pd.array(arr, dtype="Int64"))
        return cudf.Series(pd.array(arr, dtype=dt))

    def _update(self, node: Update) -> int:
        """Apply ``UPDATE t SET col = expr [, ...] [WHERE pred]`` (step 10).

        v1 scope: autocommit only. An UPDATE inside an explicit transaction raises
        ``NotImplementedError`` (correct per-row MVCC versioning would need the
        two-ts split this version deliberately avoids -- see the step-10 plan).
        Requires a declared PRIMARY KEY on ``t`` (row identity is by PK value,
        mirroring DELETE): the matched rows are tombstoned by PK and the post-SET
        rows are re-inserted as a fresh batch, both flushed under ONE commit_ts
        so the re-insert (same ts) survives its own UPDATE tombstone via the
        ``exclude_same_ts`` rule in ``_merge_delta`` (``tomb_upd <= ins_ts``).

        Atomicity + read-your-writes enforcement: the autocommit UPDATE runs
        inside an *implicit* transaction so the old rows' UPDATE tombstone is
        visible to ``_enforce_unique`` (the old PKs are gone from the scan -> no
        false self-collision when SET keeps the PK) and so the tombstone +
        reinsert flush as one durable WAL record under one ts. A PK-changing
        UPDATE that collides with a surviving row raises before any durable
        state is written (the implicit txn is discarded). Returns the count of
        rows matched (and updated).
        """
        info = self.catalog.get(node.table)
        if info is None:
            raise RuntimeError(f"unknown table: {node.table}")
        pk = info.constraints.primary_key
        if pk is None:
            raise RuntimeError(
                f"UPDATE requires a declared PRIMARY KEY on {node.table}"
            )
        if self._txn is not None:
            raise NotImplementedError(
                "UPDATE inside an explicit transaction is not supported in v1 "
                "(needs per-row MVCC versioning); use autocommit UPDATE"
            )
        # Validate SET columns up front (cheap, before any scan).
        all_cols = list(info.columns)
        for col, _ in node.assignments:
            if col not in all_cols:
                raise ParseError(f"unknown column in UPDATE {node.table}: {col}")
        if len({c for c, _ in node.assignments}) != len(node.assignments):
            raise ParseError(
                f"UPDATE {node.table} SET has duplicate columns: "
                f"{[c for c, _ in node.assignments]}"
            )

        # Visible snapshot to update from (autocommit read path: all committed).
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
        # reset_index so a later cuDF Series assignment aligns positionally (a
        # boolean-masked frame keeps its pre-mask index, which would misalign a
        # freshly-built SET series and silently fill NA).
        targets = targets.reset_index(drop=True)

        # Build the post-SET frame: copy the matched rows, override each SET
        # column with the evaluated expression coerced to the column's base
        # dtype family (same path _insert uses, so _merge_delta casts cleanly).
        new_frame = targets.copy()
        n = len(targets)
        for col, expr in node.assignments:
            val = eval_expr(expr, targets)
            if isinstance(val, cudf.Series):
                arr = val.to_pandas().tolist()
            else:
                arr = [val] * n
            new_frame[col] = self._typed_series(info.types[col], arr)

        # NOT NULL: a SET to NULL on a NOT NULL column (incl. PK) is rejected
        # before any durable state is written. PK cols are NOT NULL; UNIQUE is
        # NULL-exempt, so this also stops a NULL PK slipping past _enforce_unique's
        # dropna.
        for c in info.constraints.not_null:
            if new_frame[c].isna().any():
                raise RuntimeError(f"NOT NULL violation: {node.table}.{c}")

        tombstone = targets[list(pk)]  # PK values of the rows being replaced

        # Implicit transaction: buffer the UPDATE tombstone so _enforce_unique's
        # read-your-writes scan sees the old rows as gone (no false self-collision
        # when SET keeps the PK), enforce PK/UNIQUE on the new frame, then buffer
        # the reinsert and COMMIT flushes both under one commit_ts (atomic +
        # durable). On any failure the implicit txn is discarded (no half-applied
        # state) and the error re-raised.
        self._txn = Transaction(snapshot_ts=self._commit_ts)
        try:
            self._txn.tombstone_append(node.table, tombstone, exclude_same_ts=True)
            self._enforce_unique(node.table, new_frame)
            self._txn.buffer_append(node.table, new_frame)
            self._invalidate_table_caches(node.table)
            self._commit()
        except BaseException:
            self._txn = None
            self._invalidate_table_caches(node.table)
            raise
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
        # INSERT ... SELECT carries a relational subplan that DOES need
        # optimizing (predicate pushdown, projection pruning, ...); the Insert
        # shell itself is not a relational node, so optimize the child in place
        # before the leaf-bypass below. INSERT ... VALUES has no source.
        if isinstance(plan, Insert) and plan.source is not None:
            plan.source = optimize(
                plan.source,
                self.catalog.schema_dict(),
                self.catalog.stats_dict(),
            )
        # INSERT, DELETE, UPDATE, and TxnControl are non-relational leaves with no
        # predicate / projection / join to optimize; bypass the optimizer (the
        # rules are pass-through-safe today, but a future Select-shaped rule
        # could choke on an Insert/Delete/Update/TxnControl root). A DELETE's or
        # UPDATE's WHERE is a row-selector evaluated in _delete/_update, not a
        # relational Filter.
        if not isinstance(plan, (Insert, Delete, Update, TxnControl)):
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
        # INSERT ... SELECT: optimize the subplan so EXPLAIN shows the optimized
        # plan (predicate pushdown etc.), mirroring sql().
        if isinstance(plan, Insert) and plan.source is not None:
            plan.source = optimize(
                plan.source, self.catalog.schema_dict(), self.catalog.stats_dict()
            )
        if not isinstance(plan, (Insert, Delete, Update, TxnControl)):
            plan = optimize(plan, self.catalog.schema_dict(), self.catalog.stats_dict())
        return pretty(plan)

    def execute(self, plan: PlanNode) -> "cudf.DataFrame | int | None":
        return self._exec(plan)

    def _exec(self, node: PlanNode) -> "cudf.DataFrame | int | None":
        if isinstance(node, Scan):
            return self._scan(node.table, node.columns)
        if isinstance(node, Derived):
            # A FROM-subquery: the subplan's output frame IS the derived relation
            # (its columns are named by the subplan's top Project/Aggregate). No
            # caching -- a derived table re-executes per call, like the semi/anti
            # join subplans.
            return self._exec(node.input)
        if isinstance(node, Insert):
            return self._insert(node)
        if isinstance(node, Delete):
            return self._delete(node)
        if isinstance(node, Update):
            return self._update(node)
        if isinstance(node, TxnControl):
            self._txn_control(node)
            return None
        if isinstance(node, Filter):
            df = self._exec(node.input)
            mask = eval_expr(node.predicate, df)
            if isinstance(mask, cudf.Series):
                # SQL three-valued logic: an NA predicate (e.g. NULL IN / NULL
                # LIKE) drops the row. Aligns with the join-residual path.
                return df[mask.fillna(False)]
            return df if mask else df.iloc[0:0]
        if isinstance(node, Join):
            left = self._exec(node.left)
            right = self._exec(node.right)
            return self._join(node, left, right)
        if isinstance(node, Aggregate):
            return self._aggregate(node)
        if isinstance(node, Window):
            return self._window(node)
        if isinstance(node, Project):
            return self._project(node)
        if isinstance(node, Sort):
            return self._sort(node)
        if isinstance(node, Limit):
            return self._limit(node)
        if isinstance(node, Distinct):
            return self._distinct(node)
        if isinstance(node, SetOp):
            return self._setop(node)
        raise NotImplementedError(f"no executor for {type(node).__name__}")

    # ------------------------------------------------------------------ #
    # Joins
    # ------------------------------------------------------------------ #
    def _join(self, node: Join, left: cudf.DataFrame, right: cudf.DataFrame) -> cudf.DataFrame:
        """Lower a Join node onto cuDF ``merge`` (all on GPU).

        ``how`` is the cuDF merge kind; cuDF spells FULL OUTER as ``"outer"`` (it
        has no ``"full"``). An ON residual (``node.on_predicate``) is applied
        *inside* the join: for inner/cross it is a plain post-filter (no
        null-padding to protect), but for outer joins it must filter only matched
        rows and leave the null-padded unmatched rows alone -- otherwise the
        residual would silently turn the outer join into an inner join. cuDF merge
        cannot report which rows matched (no ``indicator``) and same-named keys
        collapse to one column, so the outer-with-residual case renames both
        sides' keys to temp columns, computes a matched mask from their nullness,
        applies the residual to matched rows only, then restores the key names.
        """
        how = node.how
        merge_how = "outer" if how == "full" else how
        pred = node.on_predicate

        # Disambiguate cross-input column-name collisions (self-join, or any join
        # whose two inputs share a column name) by alias: rename each side's
        # colliding columns to ``{alias}__{name}`` so qualified refs (``a.v`` /
        # ``b.v``) resolve via ``eval_expr`` and same-named equi-join keys no
        # longer collapse to one column (matching DuckDB, which keeps both). The
        # rename only fires when a collision exists; TPC-H's join keys/columns
        # are all prefix-unique, so this path is inert on the bench. USING /
        # NATURAL joins (``node.using``) are skipped: their keys coalesce into a
        # single column by design, and any non-key collision was rejected at
        # parse time, so the only shared columns are the coalescing keys.
        collision = set(left.columns) & set(right.columns)
        if collision and not node.using and how not in ("semi", "anti"):
            if how in ("left", "right", "full") and pred is not None:
                raise NotImplementedError(
                    "outer join with a residual ON predicate and a cross-input "
                    "column-name collision is not supported; use ON with only "
                    "equi-key predicates, or rename the colliding columns"
                )
            lalias = self._side_alias(node.left)
            ralias = self._side_alias(node.right)
            if lalias is None or ralias is None:
                raise NotImplementedError(
                    "a colliding join side must be a base table (optionally "
                    "filtered); joins/subqueries/derived tables with a "
                    "column-name collision are not supported"
                )
            left = left.rename(columns={c: f"{lalias}__{c}" for c in collision})
            right = right.rename(columns={c: f"{ralias}__{c}" for c in collision})
            on_left = [
                f"{lalias}__{c}" if c in collision else c for c in node.on_left
            ]
            on_right = [
                f"{ralias}__{c}" if c in collision else c for c in node.on_right
            ]
            node = Join(
                left=node.left,
                right=node.right,
                how=how,
                on_left=on_left,
                on_right=on_right,
                on_predicate=pred,
                using=node.using,
            )

        if how in ("semi", "anti"):
            return self._semi_anti_join(node, left, right)

        if how == "cross":
            m = left.merge(right, how="cross", suffixes=("_x", "_y"))
            return self._filter_on_predicate(m, pred)

        if how == "inner" or pred is None:
            # No outer null-padding to protect. Pure-equi outer joins have pred
            # None and rely on cuDF merge alone (FULL -> "outer").
            m = left.merge(
                right,
                left_on=node.on_left,
                right_on=node.on_right,
                how=merge_how,
                suffixes=("_x", "_y"),
            )
            return self._filter_on_predicate(m, pred)

        return self._outer_join_with_predicate(node, left, right)

    def _semi_anti_join(
        self, node: Join, left: cudf.DataFrame, right: cudf.DataFrame
    ) -> cudf.DataFrame:
        """Semi/anti join (IN / NOT IN, correlated EXISTS / NOT EXISTS). Preserve
        the left frame; keep left rows that have (semi) / don't have (anti) a
        matching right row on the equi-keys.

        Single-key, no residual: the fast ``isin`` path (NULL-safe via
        ``dropna`` on the key set -- a NULL left key never matches, and NULLs in
        the set never spuriously match a non-NULL key). NOT IN is only correct
        for non-NULL keys on both sides (a NULL in the set makes DuckDB return
        NULL for every non-matching row, which ``~isin`` does not reproduce) --
        the parser/tests guard that.

        Multi-key (a composite correlation, e.g. ``b.k=a.k AND b.j=a.j``):
        inner-merge on the equi-keys, then keep the left rows whose row-id
        appears (semi) / does not appear (anti) among the matched pairs.
        ``dropna`` on the right keys first makes the match NULL-safe (cuDF merge
        would match NULL==NULL, but SQL ``=`` matches nothing on NULL): a NULL
        inner key can't match, and with no NULL right keys a NULL left key finds
        no match either. The right side of a decorrelated EXISTS projects only
        the equi-keys, so no cross-input column collision arises here. The fused
        star-join kernel defers on semi/anti and on multi-key joins, so this path
        is always the cuDF fallback for composite correlations."""
        how = node.how
        on_left = list(node.on_left)
        on_right = list(node.on_right)
        if len(on_left) == 1 and node.on_predicate is None:
            keys = right[on_right[0]].dropna()
            mask = left[on_left[0]].isin(keys)
            if how == "anti":
                mask = ~mask
            return left[mask]

        # Multi-key (composite correlation). Tag left rows so the matched set can
        # be projected back to the surviving left rows in their original order.
        L = left.reset_index(drop=True)
        Lm = L.copy()
        Lm["__lid"] = cudf.Series(range(len(Lm)), dtype="int64")
        R = right.dropna(subset=on_right)
        merged = Lm.merge(
            R,
            left_on=on_left,
            right_on=on_right,
            how="inner",
            suffixes=("_x", "_y"),
        )
        if node.on_predicate is not None:
            merged = self._filter_on_predicate(merged, node.on_predicate)
        matched = merged["__lid"].drop_duplicates()
        ids = cudf.Series(range(len(L)), dtype="int64")
        hit = ids.isin(matched)
        if how == "anti":
            hit = ~hit
        return L[hit]

    @staticmethod
    def _side_alias(node: PlanNode) -> str | None:
        """The base-table alias for a (possibly filtered) join side, used to name
        colliding columns ``{alias}__{col}``. ``Scan`` carries its alias; a
        ``Filter`` sits between the join and its scan and is transparent; a
        ``Derived`` (CTE / FROM-subquery) exposes its output columns under its
        own alias, so its colliding columns are renamed ``{alias}__{col}`` too.
        A ``Join``/``Project`` side has no single-table provenance (deferred)."""
        if isinstance(node, Scan):
            return node.alias
        if isinstance(node, Filter):
            return Engine._side_alias(node.input)
        if isinstance(node, Derived):
            return node.alias
        return None

    def _filter_on_predicate(self, m: cudf.DataFrame, pred: Expr | None) -> cudf.DataFrame:
        """Apply a predicate as a plain post-filter (inner/cross semantics: a row
        with an NA predicate result is dropped -- SQL three-valued logic)."""
        if pred is None:
            return m
        mask = eval_expr(pred, m)
        if isinstance(mask, cudf.Series):
            return m[mask.fillna(False)]
        return m if mask else m.iloc[0:0]

    def _outer_join_with_predicate(
        self, node: Join, left: cudf.DataFrame, right: cudf.DataFrame
    ) -> cudf.DataFrame:
        """Outer join with a non-equi ON residual.

        Semantics: a preserved-side row survives iff it has at least one match on
        the equi keys whose ON residual is true (it is emitted joined to that
        match); otherwise it is emitted once, null-padded on the other side. This
        differs from a naive ``merge(how=left)`` + filter: a left row whose key
        matches a right row but whose residual is false must still be null-padded,
        not dropped. So we build the *satisfied* joined rows (inner key-join,
        filtered by the residual) and then null-pad every preserved-side row that
        has no satisfied match (by row id, so duplicate keys are handled).

        FULL outer null-pads both sides; RIGHT null-pads the right (the optimizer
        rewrites RIGHT to LEFT on side-swap, but a user-written RIGHT JOIN reaches
        us directly). cuDF has no merge ``indicator``, so matchedness is derived
        from row-id membership, not a merge flag.
        """
        how = node.how
        on_left, on_right = node.on_left, node.on_right
        # Temp key names so both sides' keys survive (cuDF collapses same-named
        # keys to one column). rename(columns=) maps old->new -> {original: temp}.
        l_rename = {lk: f"__ryu_l{i}" for i, lk in enumerate(on_left)}
        r_rename = {rk: f"__ryu_r{i}" for i, rk in enumerate(on_right)}
        lcols = list(l_rename.values())
        rcols = list(r_rename.values())
        lr = left.rename(columns=l_rename).reset_index(drop=True)
        rr = right.rename(columns=r_rename).reset_index(drop=True)
        # Row ids identify preserved-side rows for null-padding (handles duplicate
        # keys correctly -- each row is independent, not deduped by key value).
        lr["__lrid"] = cudf.Series(range(len(lr)), dtype="int64")
        rr["__rrid"] = cudf.Series(range(len(rr)), dtype="int64")

        # Satisfied rows = inner equi-join filtered by the ON residual.
        m = lr.merge(rr, left_on=lcols, right_on=rcols, how="inner", suffixes=("_x", "_y"))
        if len(m):
            pred_mask = eval_expr(node.on_predicate, m)
            if isinstance(pred_mask, cudf.Series):
                pred_mask = pred_mask.fillna(False)
            else:
                pred_mask = cudf.Series([bool(pred_mask)] * len(m), dtype="bool")
            sat = m[pred_mask]
        else:
            sat = m  # no key matches at all -> everything gets null-padded below

        sat_lrids = sat["__lrid"] if "__lrid" in sat.columns else cudf.Series([], dtype="int64")
        sat_rrids = sat["__rrid"] if "__rrid" in sat.columns else cudf.Series([], dtype="int64")
        target_cols = list(sat.columns)
        dtypes = {c: sat[c].dtype for c in target_cols}
        lr_cols = list(lr.columns)
        rr_cols = list(rr.columns)

        pieces: list[cudf.DataFrame] = [sat]
        if how in ("left", "full"):
            mask = ~lr["__lrid"].isin(sat_lrids)
            pieces.append(self._null_pad_rows(lr[mask], keep_cols=lr_cols,
                                              target_cols=target_cols, dtypes=dtypes))
        if how in ("right", "full"):
            mask = ~rr["__rrid"].isin(sat_rrids)
            pieces.append(self._null_pad_rows(rr[mask], keep_cols=rr_cols,
                                              target_cols=target_cols, dtypes=dtypes))

        nonempty = [p for p in pieces if len(p)]
        out = cudf.concat(nonempty, axis=0) if nonempty else sat.iloc[0:0]
        out = out.drop(columns=["__lrid", "__rrid"], errors="ignore")
        return self._restore_join_keys(out, on_left, on_right)

    def _null_pad_rows(
        self, sub: cudf.DataFrame, keep_cols: list[str],
        target_cols: list[str], dtypes: dict,
    ) -> cudf.DataFrame:
        """Build null-padded rows: keep ``keep_cols`` from ``sub``, set the rest of
        ``target_cols`` to all-NA, typed to match ``dtypes`` so ``cudf.concat``
        aligns cleanly. Int/bool columns become float NaN, matching cuDF's own
        outer-merge null-padding. Empty ``sub`` yields an empty frame (filtered
        out of the concat by the caller)."""
        if len(sub) == 0:
            return cudf.DataFrame({c: self._na_series(0, dtypes.get(c)) for c in target_cols})
        keep = set(keep_cols)
        cols: dict[str, cudf.Series] = {}
        for c in target_cols:
            if c in keep:
                cols[c] = sub[c].reset_index(drop=True)
            else:
                cols[c] = self._na_series(len(sub), dtypes.get(c))
        return cudf.DataFrame(cols)

    def _na_series(self, n: int, dtype) -> cudf.Series:
        import numpy as np
        # int/float/bool -> float NaN (nullable); everything else -> object NA.
        if dtype is not None and (np.issubdtype(dtype, np.number) or np.issubdtype(dtype, np.bool_)):
            return cudf.Series([float("nan")] * n, dtype="float64")
        return cudf.Series([None] * n, dtype="object")

    def _restore_join_keys(
        self, m: cudf.DataFrame, on_left: list[str], on_right: list[str]
    ) -> cudf.DataFrame:
        """Collapse the temp key columns back to the original names. Same-named
        keys (USING/NATURAL/``ON a.k=b.k``) coalesce into one column (preferring
        the preserved side's value, falling back to the other for null padding);
        differently-named keys are restored to both names (matches cuDF's native
        distinct-key merge output)."""
        for i, (lk, rk) in enumerate(zip(on_left, on_right)):
            lc, rc = f"__ryu_l{i}", f"__ryu_r{i}"
            if lk == rk:
                m[lc] = m[lc].fillna(m[rc])
                m = m.drop(columns=[rc]).rename(columns={lc: lk})
            else:
                m = m.rename(columns={lc: lk, rc: rk})
        return m

    # ------------------------------------------------------------------ #
    def _aggregate(self, node: Aggregate) -> cudf.DataFrame:
        group_keys = node.group_keys
        aggs = node.aggs
        by_names = [gn for _, gn in group_keys]
        # A per-aggregate FILTER (WHERE ...) cannot be honoured by the C++ fused
        # kernels (they read only af.func/af.arg) -> force the cuDF fallback paths
        # (_fused_agg / _scalar_global_agg), which apply the per-agg mask.
        _has_filter = _agg_has_filter(aggs)
        # A DISTINCT-qualified aggregate (F(DISTINCT x)) dedupes its arg within
        # each group before reducing; the fused kernels aggregate over ALL rows
        # (they read only af.func/af.arg, not af.distinct), so DISTINCT must also
        # force the cuDF fallback paths, which honour af.distinct.
        _has_distinct = _agg_has_distinct(aggs)
        _force_fallback = _has_filter or _has_distinct

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
                    res = fused_scan_aggregate(node, self) if not _force_fallback else None
                    if res is not None:
                        return res
            child = self._exec(in_node.input)
            # Phase 3b/4: try the fused C++/CUDA filter+groupby+aggregate kernel
            # first -- it now handles grouped AND global aggregates, and the
            # SUM/AVG/MIN/MAX/COUNT(*) kinds. Returns None for unsupported shapes
            # (no Filter match, OR predicate, COUNT(col), nullable AVG/MIN/MAX
            # args, multi-col numeric GROUP BY, ...) -> cuDF fallback below.
            res = fused_aggregate(node, child, self) if not _force_fallback else None
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
        res = fused_join_aggregate(node, self) if not _force_fallback else None
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
            fmask = eval_expr(af.filter, df) if af.filter is not None else None
            if af.func == "COUNT" and isinstance(af.arg, Star):
                # COUNT(*) FILTER (WHERE p) = count of rows where p is TRUE.
                # fmask.sum() counts True (NULL/False -> 0); df is already
                # WHERE-filtered so this composes with the outer WHERE.
                row[n] = [int(fmask.sum()) if fmask is not None else int(len(df))]
            else:
                col = eval_expr(af.arg, df)
                if fmask is not None:
                    # Null the arg where the FILTER predicate is False/NA; cuDF
                    # reductions skip nulls, so failing rows drop out of this one
                    # aggregate only (sibling aggs and group membership untouched).
                    col = df_where(col, fmask)
                if af.distinct:
                    # F(DISTINCT x): reduce over the set of distinct arg values.
                    # cuDF drop_duplicates treats NaN as equal (keeps one), and the
                    # reductions below skip nulls, so a NULL arg still does not count
                    # toward COUNT / SUM / AVG -- matching SQL (NULLs are not
                    # distinct values). FILTER composes: it nulls failing rows first,
                    # so only passing rows' arg values are deduped.
                    col = col.drop_duplicates()
                row[n] = [_scalar_agg(af.func, col)]
        if not aggs:
            # No aggregates (a bare global "GROUP BY ()" / no-agg ROLLUP grand
            # total): emit the single grand-total row. An empty dict would make
            # cudf.DataFrame({}) a 0-row frame, dropping the row entirely.
            return cudf.DataFrame({"_g_": [0]})
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
        if not aggs:
            # No aggregates: a bare ``GROUP BY`` (or a no-agg ROLLUP/CUBE/GROUPING
            # SETS branch) with only grouping columns == SELECT DISTINCT of the
            # group keys. groupby().size() yields the distinct key combinations
            # (NULL keys kept via dropna=False on the normal path).
            return (
                work.groupby(by_names, dropna=dropna).size().reset_index()[by_names]
            )
        if any(af.distinct for af, _ in aggs):
            # A DISTINCT-qualified aggregate dedupes its arg within each group
            # before reducing; the fused single-pass groupby.agg below cannot do
            # that, so route to the per-agg path (one groupby per aggregate, then
            # concat along the group-key index -- the same alignment the single
            # pass uses). Non-distinct aggs in the same query aggregate over all
            # rows; FILTER composes (nulls failing rows' arg before dedup/reduce).
            return self._distinct_grouped_agg(
                work, src, group_keys, aggs, by_names, dropna
            )
        work["__cnt"] = 1
        spec: dict[str, list[str]] = {}
        out_map: list[tuple[str, str, str]] = []
        for af, n in aggs:
            fmask = eval_expr(af.filter, src) if af.filter is not None else None
            if af.func == "COUNT" and isinstance(af.arg, Star):
                if fmask is not None:
                    # Per-agg null'd ones: COUNT(*) FILTER counts rows where the
                    # predicate is TRUE. A dedicated column (not the shared
                    # __cnt) so two COUNT(*) with different filters don't collide.
                    # work and src share length+index (the existing agg-arg
                    # assignment relies on the same alignment), so df_where of
                    # work["__cnt"] by the src-length mask is sound.
                    tmp = f"__f_{n}"
                    work[tmp] = df_where(work["__cnt"], fmask)
                    spec.setdefault(tmp, []).append("count")
                    out_map.append((tmp, "count", n))
                else:
                    spec.setdefault("__cnt", []).append("count")
                    out_map.append(("__cnt", "count", n))
                continue
            tmp = f"__a_{n}"
            col = eval_expr(af.arg, src)
            if fmask is not None:
                # Null the arg where the FILTER predicate is False/NA; cuDF
                # reductions skip nulls, so failing rows drop out of this one
                # aggregate only (sibling aggs and group membership untouched).
                col = df_where(col, fmask)
            work[tmp] = col
            func = "count" if af.func == "COUNT" else _AGG_METHOD[af.func]
            spec.setdefault(tmp, []).append(func)
            out_map.append((tmp, func, n))

        grouped = work.groupby(by_names, dropna=dropna)
        res = grouped.agg(spec)
        pieces = [res[(c, f)].rename(n) for c, f, n in out_map]
        out = cudf.concat(pieces, axis=1).reset_index()
        return out[by_names + [n for _, n in aggs]]

    def _distinct_grouped_agg(
        self,
        work: cudf.DataFrame,
        src: cudf.DataFrame,
        group_keys,
        aggs,
        by_names: list[str],
        dropna: bool,
    ) -> cudf.DataFrame:
        """Grouped aggregate where at least one aggregate is DISTINCT-qualified.

        The fused single-pass ``groupby.agg({col: [funcs]})`` cannot dedupe per
        group, so each aggregate is computed by its own groupby and the per-agg
        Series are concatenated along the group-key index (the same alignment
        the single-pass path uses -- every group is present in every per-agg
        ``sub``: non-distinct aggs keep all rows, and DISTINCT dedup keeps >=1
        row per group, so the group-key index is identical across pieces and
        ``concat(axis=1)`` aligns without a NaN-key join). DISTINCT aggs dedupe
        ``(group_keys, arg)`` first; non-distinct aggs (incl. COUNT(*))
        aggregate over all rows. FILTER composes: it nulls failing rows' arg
        before the dedup/reduce. NULL args do not count: drop_duplicates treats
        NaN as equal (keeps one) and the reductions skip nulls."""
        pieces: list[cudf.Series] = []
        for af, n in aggs:
            fmask = eval_expr(af.filter, src) if af.filter is not None else None
            if af.func == "COUNT" and isinstance(af.arg, Star):
                # COUNT(*) [FILTER]; distinct+Star is rejected at parse.
                if fmask is not None:
                    # count rows where the FILTER predicate is TRUE per group
                    # (NULL predicate -> False, i.e. the row is excluded).
                    cnt = fmask.fillna(False).astype("int64")
                    sub = work[by_names].copy()
                    sub[n] = cnt
                    s = sub.groupby(by_names, dropna=dropna)[n].sum()
                else:
                    s = work.groupby(by_names, dropna=dropna).size()
                pieces.append(s.rename(n))
                continue
            col = eval_expr(af.arg, src)
            if fmask is not None:
                col = df_where(col, fmask)
            sub = work[by_names].copy()
            sub[n] = col
            if af.distinct:
                # One row per distinct arg value per group; the reduction then
                # collapses the survivors. work and src share length+index (the
                # single-pass path relies on the same alignment for work[tmp]=col).
                sub = sub.drop_duplicates(subset=by_names + [n])
            func = "count" if af.func == "COUNT" else _AGG_METHOD[af.func]
            s = sub.groupby(by_names, dropna=dropna)[n].agg(func)
            pieces.append(s.rename(n))
        out = cudf.concat(pieces, axis=1).reset_index()
        return out[by_names + [n for _, n in aggs]]

    # ------------------------------------------------------------------ #
    # Window functions (Phase F-1 + G-3 frames). A row-preserving "compute"
    # node: each window function is evaluated over the input frame and attached
    # as a new column; the input columns pass through verbatim so the outer
    # Project can reference both. Ranking/offset funcs sort by (partition,
    # order) and compute position within partition via a global-position-minus-
    # boundary method (NOT groupby.cumcount -- that returns NA for NULL-key
    # partitions in cuDF); the original row order is restored via a position
    # sentinel. Aggregate-over-partition (no ORDER BY) uses
    # groupby(dropna=False).transform, which DOES include NULL-key partitions.
    # Running/cumulative aggregates (an ORDER BY on an aggregate window, with
    # the SQL default frame or an explicit ROWS/RANGE frame) use
    # _window_running: per-partition prefix sums (SUM/COUNT/AVG, O(n), any ROWS
    # bounds incl. FOLLOWING) or cummin/cummax (MIN/MAX, cumulative), with
    # peer-group semantics for RANGE (rows with equal order keys share the
    # cumulative value, matching DuckDB's default frame).
    # ------------------------------------------------------------------ #
    def _window(self, node: Window) -> cudf.DataFrame:
        df = self._exec(node.input)
        out = df.reset_index(drop=True).copy()
        for wf, name in node.funcs:
            out[name] = self._window_column(wf, out)
        return out.reset_index(drop=True)

    def _window_column(self, wf: WindowFunc, df: cudf.DataFrame) -> cudf.Series:
        # Materialize PARTITION BY / ORDER BY keys: bare columns keep their own
        # name; an expression key (G-6) is evaluated into a synthetic sort column
        # (_wpN / _woN) on a copy so it never leaks into the output frame. The
        # downstream broadcast/positioned/running paths sort/group by these names
        # uniformly.
        work, part_names, order_names = self._window_keys(wf, df)
        is_agg = wf.func in _AGG_METHOD
        if is_agg and not wf.order_keys:
            col = self._window_broadcast(wf, work, part_names)
            return col.astype("int64") if wf.func == "COUNT" else col
        if is_agg:
            # Running/cumulative aggregate (ORDER BY present, frame resolved at
            # parse). Pure cuDF -- no fused kernel is involved in the window node.
            col = self._window_running(wf, work, part_names, order_names)
            return col.astype("int64") if wf.func == "COUNT" else col
        # ranking / lag / lead: positioned path.
        col = self._window_positioned(wf, work, part_names, order_names)
        if wf.func in ("ROW_NUMBER", "RANK", "DENSE_RANK"):
            return col.astype("int64")
        return col

    def _window_keys(self, wf, df):
        """Return ``(df_copy, part_names, order_names)`` where non-column key
        expressions are evaluated into synthetic columns ``_wpN`` / ``_woN`` on
        the copy (bare columns use their own name). The copy keeps the caller's
        frame (the Window node's output) free of synthetic sort columns.

        A qualified column key on a self-join / same-named-column join
        (``PARTITION BY a.k``) resolves to the alias-renamed ``a__k`` produced by
        ``_join``; the bare ``k`` is no longer in the frame, so it is addressed
        by its renamed name (no synthetic copy needed when that column is
        already present)."""
        out = df.copy()
        part_names: list[str] = []
        for i, p in enumerate(wf.partition_keys):
            part_names.append(self._window_key_name(p, out, "_wp", i))
        order_names: list[str] = []
        for i, (e, _a) in enumerate(wf.order_keys):
            order_names.append(self._window_key_name(e, out, "_wo", i))
        return out, part_names, order_names

    @staticmethod
    def _window_key_name(p: Expr, out: cudf.DataFrame, prefix: str, i: int) -> str:
        """The sort/groupby column name for a window key. A bare unqualified
        column present in the frame uses its own name; a qualified self-join
        column uses the alias-renamed ``{table}__{name}`` column; anything else
        (an expression, or a column not directly present) is materialized into a
        synthetic ``{prefix}{i}`` column."""
        if isinstance(p, Col):
            if p.table is not None and f"{p.table}__{p.name}" in out.columns:
                return f"{p.table}__{p.name}"
            if p.name in out.columns:
                return p.name
        name = f"{prefix}{i}"
        out[name] = eval_expr(p, out)
        return name

    def _window_broadcast(self, wf, df, part_names):
        """Aggregate over the (whole) partition -- no ORDER BY. Broadcast the
        per-partition aggregate to every row (rows keep their order). NULL
        partition keys form their own group (groupby dropna=False)."""
        if not part_names:
            # Whole-frame aggregate broadcast as a constant.
            if wf.func == "COUNT":
                if isinstance(wf.arg, Star):
                    val = len(df)
                else:
                    val = int(eval_expr(wf.arg, df).notna().sum())
            else:
                val = getattr(eval_expr(wf.arg, df), _AGG_METHOD[wf.func])()
            return cudf.Series([val] * len(df), index=df.index)
        frame = df.copy()
        if wf.func == "COUNT" and isinstance(wf.arg, Star):
            frame["_wc"] = 1
            col = frame.groupby(part_names, dropna=False)["_wc"].transform("sum")
        elif wf.func == "COUNT":
            frame["_wc"] = eval_expr(wf.arg, df).notna().astype("int64")
            col = frame.groupby(part_names, dropna=False)["_wc"].transform("sum")
        else:
            frame["_wc"] = eval_expr(wf.arg, df)
            col = frame.groupby(part_names, dropna=False)["_wc"].transform(
                _AGG_METHOD[wf.func]
            )
        return col

    def _window_positioned(self, wf, df, part_names, order_names):
        """Ranking (ROW_NUMBER/RANK/DENSE_RANK) or offset (LAG/LEAD) funcs that
        need an ORDER BY. Sort by (partition, order), compute via a
        global-position-minus-partition-boundary method (null-safe -- no
        groupby.cumcount, which breaks on NULL-key partitions), then restore the
        original row order via a position sentinel. ``order_names`` are the
        materialized sort-column names (bare or synthetic -- see
        ``_window_keys``)."""
        asc = [a for _, a in wf.order_keys]
        # DuckDB's default is NULLS LAST for BOTH ascending and descending
        # (NULLs sort after every non-null value). cuDF sort_values takes a single
        # na_position; NULLs in a non-leading order key follow the leading key's
        # side, which matches DuckDB when the leading key has no NULLs (F-1 tests
        # avoid NULLs in non-leading order keys for multi-key ORDER BY).
        na_position = "last"
        work = df.copy()
        work["_pos"] = cudf.Series(range(len(work)), index=work.index)
        by, asc_all = self._dedup_sort_keys(
            part_names + order_names, [True] * len(part_names) + asc
        )
        work = work.sort_values(by=by, ascending=asc_all, na_position=na_position) \
            .reset_index(drop=True)
        gpos = cudf.Series(range(len(work)), index=work.index)
        boundary = self._partition_boundary(work, part_names, gpos)
        pstart = gpos.where(boundary).ffill()
        pos_in_part = gpos - pstart  # 0-based within partition (sorted order)

        if wf.func in ("ROW_NUMBER", "RANK", "DENSE_RANK"):
            if wf.func == "ROW_NUMBER":
                result = pos_in_part + 1
            else:
                peer_bdry = boundary
                for c in order_names:
                    peer_bdry = peer_bdry | self._col_changed(work[c], work[c].shift(1))
                if wf.func == "RANK":
                    # Rank at a peer-group start = its 1-based position; forward-fill
                    # within the peer group (resets at the next partition boundary).
                    result = (pos_in_part + 1).where(peer_bdry).ffill()
                else:  # DENSE_RANK
                    dense_global = peer_bdry.astype("int64").cumsum()
                    dense_at_start = dense_global.where(boundary).ffill()
                    result = dense_global - dense_at_start + 1
        else:  # LAG / LEAD
            n_off = 1 if wf.offset is None else int(eval_expr(wf.offset, work))
            arg_s = eval_expr(wf.arg, work)
            shifted = arg_s.shift(n_off if wf.func == "LAG" else -n_off)
            # Valid offset positions: within-partition, not the first/last n rows.
            # Use partition size to bound LEAD; LAG bounds by position alone.
            is_last = boundary.shift(-1).fillna(True)
            pend = gpos.where(is_last).bfill()
            psize = pend - pstart + 1
            if wf.func == "LAG":
                valid = pos_in_part >= n_off
            else:
                valid = pos_in_part <= (psize - 1 - n_off)
            if wf.default is not None:
                dval = eval_expr(wf.default, work)
                result = shifted.where(valid, dval)
            else:
                result = shifted.where(valid)

        work = work.copy()
        work["_wf"] = result
        restored = work.sort_values("_pos").reset_index(drop=True)
        return restored["_wf"]

    def _window_running(self, wf, df, part_names, order_names):
        """Running/cumulative aggregate window function (an aggregate with an
        ORDER BY and a resolved frame). Sort by (partition, order), compute the
        per-row windowed aggregate, then restore the original row order via a
        position sentinel. ``order_names`` are the materialized sort-column
        names (bare or synthetic -- see ``_window_keys``).

        Frame semantics:
          - RANGE default (UNBOUNDED PRECEDING .. CURRENT ROW): peer-group
            cumulative -- rows with equal order keys share the cumulative value
            (matches DuckDB's default frame).
          - ROWS: positional sliding window [lo_i, hi_i] within the partition;
            SUM/COUNT/AVG via per-partition prefix sums (O(n), any bounds incl.
            FOLLOWING); MIN/MAX cumulative via cummin/cummax (start UNBOUNDED
            PRECEDING only -- trailing MIN/MAX is deferred at parse).

        NULLs: NULL order keys sort last (NULLS LAST) and form one peer group.
        NULLs in the agg arg are skipped (SQL): SUM/AVG ignore them (an all-null
        or empty window yields NULL, not 0); MIN/MAX skip them via ffill of the
        cumulative; COUNT(col) counts non-nulls; COUNT(*) counts rows. Pure cuDF
        -- no fused kernel is involved in the window node.
        """
        asc = [a for _, a in wf.order_keys]
        na_position = "last"
        work = df.copy()
        work["_pos"] = cudf.Series(range(len(work)), index=work.index)
        by, asc_all = self._dedup_sort_keys(
            part_names + order_names, [True] * len(part_names) + asc
        )
        work = work.sort_values(by=by, ascending=asc_all, na_position=na_position) \
            .reset_index(drop=True)
        gpos = cudf.Series(range(len(work)), index=work.index)
        boundary = self._partition_boundary(work, part_names, gpos)
        pstart = gpos.where(boundary).ffill()
        pos_in_part = gpos - pstart  # 0-based within partition (sorted order)
        is_last = boundary.shift(-1).fillna(True)
        pend = gpos.where(is_last).bfill()
        psize = pend - pstart + 1
        work["_pid"] = boundary.cumsum().astype("int64")  # per-partition group id

        # Peer-group start/end (0-based within partition) for RANGE frames: a
        # peer group is a run of equal order keys (null-aware, NULL == NULL).
        peer_bdry = boundary
        for c in order_names:
            peer_bdry = peer_bdry | self._col_changed(work[c], work[c].shift(1))
        peer_start = gpos.where(peer_bdry).ffill() - pstart
        peer_end = gpos.where(peer_bdry.shift(-1).fillna(True)).bfill()
        peer_end = peer_end.where(peer_end <= pend, pend) - pstart

        lo, hi = self._frame_bounds(wf.frame, pos_in_part, psize, peer_start, peer_end)
        nonempty = lo <= hi

        is_count_star = wf.func == "COUNT" and isinstance(wf.arg, Star)
        if is_count_star:
            # COUNT(*): window row count = hi - lo + 1 (empty -> 0).
            result = (hi - lo + 1).astype("int64").where(nonempty, 0)
        elif wf.func == "COUNT":  # COUNT(col): non-null count in the window.
            arg_s = eval_expr(wf.arg, work)
            work["_nn"] = arg_s.notna().astype("int64")
            pnn = work.groupby("_pid", dropna=False)["_nn"].cumsum()
            cnt = self._gather_diff(pnn, pstart, lo, hi).where(nonempty, 0)
            result = cnt.astype("int64")
        elif wf.func in ("SUM", "AVG"):
            arg_s = eval_expr(wf.arg, work)
            work["_nn"] = arg_s.notna().astype("int64")
            pnn = work.groupby("_pid", dropna=False)["_nn"].cumsum()
            cnt = self._gather_diff(pnn, pstart, lo, hi).where(nonempty, 0)
            work["_a0"] = arg_s.fillna(0)  # fillna BEFORE groupby (skips nulls)
            pref = work.groupby("_pid", dropna=False)["_a0"].cumsum()
            s = self._gather_diff(pref, pstart, lo, hi)
            if wf.func == "SUM":
                result = s.where(cnt > 0)  # all-null / empty window -> NULL
            else:  # AVG
                result = (s / cnt).where(cnt > 0)
        else:  # MIN / MAX (cumulative only; parse rejects non-cumulative)
            arg_s = eval_expr(wf.arg, work)
            result = self._window_running_minmax(wf, work, arg_s, pstart, hi)

        work = work.copy()
        work["_wf"] = result
        restored = work.sort_values("_pos").reset_index(drop=True)
        return restored["_wf"]

    def _frame_bounds(self, frame, pos_in_part, psize, peer_start, peer_end):
        """Resolve a Frame to per-row inclusive window bounds [lo, hi] (0-based
        within the partition), clamped to [0, psize-1]. peer_start/peer_end are
        the peer-group bounds for RANGE; pos_in_part is the row's 0-based
        position; psize is the partition size. Clamping uses .where (not .clip,
        which rejects a Series upper bound in cuDF 26.06)."""
        zero = pos_in_part - pos_in_part  # int zeros, aligned
        if frame.mode == "RANGE":
            # Only UNBOUNDED_PRECEDING / CURRENT_ROW start and CURRENT_ROW /
            # UNBOUNDED_FOLLOWING end are allowed (parse rejects value offsets).
            lo = peer_start if frame.start.kind == "CURRENT_ROW" else zero
            hi = peer_end if frame.end.kind == "CURRENT_ROW" else (psize - 1)
        else:  # ROWS
            lo = self._rows_offset(frame.start, pos_in_part, psize)
            hi = self._rows_offset(frame.end, pos_in_part, psize)
        lo = lo.where(lo >= 0, 0)
        hi = hi.where(hi <= psize - 1, psize - 1)
        return lo, hi

    def _rows_offset(self, b, pos_in_part, psize):
        """A ROWS frame bound as a per-row 0-based within-partition index."""
        if b.kind == "UNBOUNDED_PRECEDING":
            return pos_in_part - pos_in_part  # zeros
        if b.kind == "UNBOUNDED_FOLLOWING":
            return psize - 1
        if b.kind == "CURRENT_ROW":
            return pos_in_part.astype("int64")
        if b.kind == "PRECEDING":
            return pos_in_part - b.n
        return pos_in_part + b.n  # FOLLOWING

    def _gather_diff(self, prefix, pstart, lo, hi):
        """Windowed prefix-sum difference: prefix[hi] - prefix[lo-1], subtracting
        0 where lo == 0 (partition start). prefix is a per-partition cumulative
        sum aligned to the sorted work frame; pstart is each row's partition-
        start absolute position; lo/hi are 0-based within-partition bounds. cuDF
        .iloc[int_series] returns a Series indexed by the index-array's values,
        so reset_index(drop=True) after each gather keeps the operands aligned."""
        pref_r = prefix.reset_index(drop=True)
        lo_r = lo.reset_index(drop=True)
        pstart_r = pstart.reset_index(drop=True)
        abs_hi = pstart_r + hi.reset_index(drop=True)
        abs_lo_m1 = (pstart_r + lo_r - 1).where(lo_r > 0, 0)
        hi_val = pref_r.iloc[abs_hi].reset_index(drop=True)
        sub = pref_r.iloc[abs_lo_m1].reset_index(drop=True).where(lo_r > 0, 0)
        return hi_val - sub

    def _window_running_minmax(self, wf, work, arg_s, pstart, hi):
        """Cumulative MIN/MAX (start UNBOUNDED PRECEDING): min/max over [0..hi]
        per partition, skipping NULLs. cummin/cummax propagate NULL at null-arg
        rows then recover; ffill over null-arg rows yields the last non-null
        running value (SQL MIN/MAX skip nulls); leading all-null windows stay
        NULL. hi may be the peer-group end (RANGE default) or a FOLLOWING offset;
        cummin[hi] = min over [0..hi] either way."""
        work["_arg"] = arg_s
        grp = work.groupby("_pid", dropna=False)["_arg"]
        cm = grp.cummin() if wf.func == "MIN" else grp.cummax()
        cm = cm.where(arg_s.notna()).ffill()
        cm_r = cm.reset_index(drop=True)
        abs_hi = pstart.reset_index(drop=True) + hi.reset_index(drop=True)
        return cm_r.iloc[abs_hi].reset_index(drop=True)

    @staticmethod
    def _dedup_sort_keys(by, asc):
        """Drop duplicate sort-key names (preserving first occurrence) so cuDF
        sort_values does not reject a key that appears in both PARTITION BY and
        ORDER BY (e.g. ``PARTITION BY k ORDER BY k, w``). A duplicate key is a
        no-op for the sort, so the row order is unchanged."""
        seen: set[str] = set()
        out_by, out_asc = [], []
        for name, a in zip(by, asc):
            if name in seen:
                continue
            seen.add(name)
            out_by.append(name)
            out_asc.append(a)
        return out_by, out_asc

    def _partition_boundary(self, work, part_names, gpos):
        """Boolean Series: True at the first row of each partition in the sorted
        frame (null-aware -- consecutive rows with NULL partition keys are the
        same partition). The first row is always a boundary."""
        boundary = (gpos == 0)
        for c in part_names:
            boundary = boundary | self._col_changed(work[c], work[c].shift(1))
        return boundary

    def _col_changed(self, s, prev):
        """Null-aware "changed" test: True where ``s`` differs from ``prev``
        treating NULL == NULL (so consecutive NULL-key rows are NOT a change)."""
        either_na = s.isna() | prev.isna()
        both_na = s.isna() & prev.isna()
        one_na = either_na & ~both_na
        val_neq = (s != prev).fillna(False)
        return one_na | (~either_na & val_neq)

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

        def bare_safe(k: Expr) -> bool:
            # The bare ``k.name`` is the correct sort column when it is
            # unambiguous: an unqualified column, or a qualified column whose
            # alias-renamed ``{table}__{name}`` is present (the renamed collision
            # column itself), or a qualified column with no renamed column AND no
            # deduped sibling (``v_1``) -- i.e. no collision happened. When a
            # self-join dedups ``SELECT a.v, b.v`` to ``v``/``v_1``, a qualified
            # ``ORDER BY b.v`` must NOT use the bare ``v`` (it is ``a.v``); it is
            # resolved against the pre-projection frame below.
            if not isinstance(k, Col) or k.name not in df.columns:
                return False
            if k.table is None:
                return True
            if f"{k.table}__{k.name}" in df.columns:
                return True
            return f"{k.name}_1" not in df.columns

        if all(bare_safe(k) for k, _ in node.keys):
            by = [k.name for k, _ in node.keys]
            ascending = [a for _, a in node.keys]
            return df.sort_values(by=by, ascending=ascending)

        # A key needs materialization (qualified self-join key, or a non-column
        # expression). Resolve qualified keys against the PRE-projection frame
        # (the Project input) where the alias-renamed ``b__v`` still lives: the
        # Project output dedups it to ``v_1`` and loses the alias. The Project is
        # row-preserving (same index), so a key Series from its input aligns with
        # the output and the sort order applies unchanged. Copy ``df`` first so a
        # temp column never mutates a cached frame.
        src = self._exec(node.input.input) if isinstance(node.input, Project) else df
        work = df.copy()
        by: list[str] = []
        temps: list[str] = []
        for i, (k, _) in enumerate(node.keys):
            if bare_safe(k):
                by.append(k.name)
            else:
                tmp = f"__sk{i}"
                work[tmp] = eval_expr(k, src)
                by.append(tmp)
                temps.append(tmp)
        ascending = [a for _, a in node.keys]
        out = work.sort_values(by=by, ascending=ascending)
        if temps:
            out = out.drop(columns=temps)
        return out

    def _limit(self, node: Limit) -> cudf.DataFrame:
        df = self._exec(node.input)
        end = node.offset + node.n
        return df.iloc[node.offset:end]

    def _distinct(self, node: Distinct) -> cudf.DataFrame:
        # cuDF drop_duplicates treats NaN as equal, so DISTINCT is NULL-correct
        # and matches DuckDB's row-distinct semantics.
        df = self._exec(node.input)
        return df.drop_duplicates().reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Set operators (UNION [ALL] / INTERSECT / EXCEPT)
    # ------------------------------------------------------------------ #
    def _setop(self, node: SetOp) -> cudf.DataFrame:
        """Lower a SetOp onto cuDF concat / drop_duplicates / merge (all on GPU).

        UNION [ALL] is ``cudf.concat`` (ALL) or concat + ``drop_duplicates``
        (DISTINCT). INTERSECT / EXCEPT are DISTINCT-only (ALL variants raise
        ``NotImplementedError``) and use ``merge`` -- cuDF merge matches nulls,
        so a row whose key column is NULL is intersected/excluded correctly with
        no sentinel trick; both sides are deduped first so the result is the SQL
        DISTINCT set. Output column names come from the left child's projection;
        the right child's columns are renamed positionally (SQL names a set op's
        outputs from the left side). ``cudf.concat`` auto-promotes int+float ->
        float, matching DuckDB's UNION type coercion for compatible types.
        """
        left = self._exec(node.left)
        right = self._exec(node.right)
        op = node.op
        if len(left.columns) != len(right.columns):
            raise ParseError(
                f"{op.upper()} column count mismatch: "
                f"{len(left.columns)} vs {len(right.columns)}"
            )
        # Align the right side positionally to the left side's output names.
        right = right.copy()
        right.columns = list(left.columns)
        cols = list(left.columns)

        if op == "union":
            out = cudf.concat([left, right], axis=0)
            if node.distinct:
                out = out.drop_duplicates()
            return out.reset_index(drop=True)

        if not node.distinct:
            raise NotImplementedError(f"{op.upper()} ALL is not supported")

        l_dd = left.drop_duplicates()
        r_dd = right.drop_duplicates()
        if op == "intersect":
            m = l_dd.merge(r_dd, on=cols, how="inner")
            return m.reset_index(drop=True)
        # EXCEPT: left rows with no match in right (incl. null-keyed rows, which
        # merge matches). cuDF merge has no ``indicator``/anti-join, and merging
        # ``on=cols`` (every column is a key) leaves no right-only column to
        # detect unmatched rows. Attach a constant marker to the right side and
        # keep the left rows whose marker came back NULL (no match).
        r_dd = r_dd.assign(_x=cudf.Series([1] * len(r_dd), index=r_dd.index))
        m = l_dd.merge(r_dd, on=cols, how="left")
        m = m[m["_x"].isna()].drop(columns=["_x"])
        return m.reset_index(drop=True)


def _scalar_agg(func: str, series) -> object:
    if func == "COUNT":
        return int(series.count())
    method = _AGG_METHOD[func]
    return getattr(series, method)()


def df_where(series, mask):
    """Return a copy of `series` with values nullified where `mask` is False.

    Used by the no-gather aggregate path to drop filtered rows from a groupby
    by nulling their group keys (the groupby then drops them via dropna=True),
    without materialising a filtered row copy. Also the per-aggregate FILTER
    mechanism: ``series.where(mask)`` nulls on False OR NA, so a NULL (unknown)
    FILTER predicate excludes the row -- exactly SQL FILTER semantics.
    """
    return series.where(mask)


def _agg_has_filter(aggs) -> bool:
    """True if any aggregate carries a ``FILTER (WHERE ...)`` predicate.

    The C++ fused kernels read only ``af.func``/``af.arg`` and would silently
    aggregate over ALL rows (ignoring the filter), so a per-agg filter MUST
    force the cuDF fallback (which honours the filter via df_where).
    """
    return any(af.filter is not None for af, _ in aggs)


def _agg_has_distinct(aggs) -> bool:
    """True if any aggregate is DISTINCT-qualified (``F(DISTINCT x)``).

    The C++ fused kernels read only ``af.func``/``af.arg`` and would aggregate
    over ALL rows (ignoring the dedup), so a DISTINCT aggregate MUST force the
    cuDF fallback (which honours ``af.distinct`` via per-group drop_duplicates).
    """
    return any(af.distinct for af, _ in aggs)