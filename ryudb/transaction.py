"""Per-transaction write buffer for the MVCC transaction layer (Phase 2 step 5).

A ``Transaction`` holds the uncommitted write set of a single in-flight
transaction: a frozen ``snapshot_ts`` (the Engine's commit counter at BEGIN, so
the txn sees exactly the committed state at that point) and a per-table list of
buffered cuDF batch frames produced by ``INSERT``. Buffered frames are visible
only to the txn's own reads (read-your-writes, via ``Engine._merge_delta``) and
never touch the shared ``DeltaStore`` until ``COMMIT`` flushes them under one new
``commit_ts``. ``ROLLBACK`` discards the buffer (undo only this txn's writes; the
committed delta is untouched because the txn never committed).

The buffer is plain ``dict[str, list[frame]]`` with no per-batch timestamp --
the frames are uncommitted, so versioning does not apply until they are flushed
to the delta at commit time.

RyuDB is single-session: at most one ``Transaction`` is active on an Engine at a
time, and no commit can occur while a txn is active (INSERTs buffer, not commit),
so snapshot isolation is enforced by the single-session structural invariant
rather than by locking. The MVCC timestamp is required for full snapshot restore
and is forward-looking for concurrent connections (a later phase).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import cudf


@dataclass
class Transaction:
    """One in-flight transaction's snapshot + uncommitted write buffer."""

    snapshot_ts: int
    _buffer: dict[str, list["cudf.DataFrame"]] = field(default_factory=dict)

    def buffer_append(self, table: str, frame: "cudf.DataFrame") -> None:
        """Buffer an INSERT batch for ``table`` (visible to this txn only)."""
        self._buffer.setdefault(table, []).append(frame)

    def buffer_batches(self, table: str) -> list["cudf.DataFrame"]:
        """This txn's buffered frames for ``table`` in append order (empty if none)."""
        return list(self._buffer.get(table) or [])

    def has(self, table: str) -> bool:
        """True if this txn has buffered >=1 frame for ``table``."""
        return bool(self._buffer.get(table))

    def tables(self) -> list[str]:
        """Tables with >=1 buffered frame (for commit/rollback invalidation)."""
        return [t for t, lst in self._buffer.items() if lst]