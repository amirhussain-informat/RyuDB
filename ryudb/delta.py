"""In-memory delta-store for the immutable-base HTAP write path.

Base Parquet files are never mutated; writes (INSERTs, from step 3 onward) append
small cuDF "batch" frames here, and the read path concatenates them onto the base
at scan time (see ``Engine._merge_delta``). Each batch is a natural WAL record
boundary (step 6 persists these). For step 2 the store is always empty -- the seam
exists so reads are byte-identical to today and step 3 can simply ``append``.

The store is owned by the ``Engine`` (session-scoped, in-memory, GPU-resident).
It is NOT the catalog: it holds row data, not table definitions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import cudf


class DeltaStore:
    """Per-table list of immutable cuDF batch frames awaiting flush/compaction."""

    def __init__(self) -> None:
        self._batches: dict[str, list[cudf.DataFrame]] = {}

    def has_unflushed(self, table: str) -> bool:
        return bool(self._batches.get(table))

    def batches(self, table: str) -> list[cudf.DataFrame]:
        """The live batch list for ``table`` (empty list if none). Callers read
        this fresh each scan so an append becomes visible to the next read with
        no invalidation."""
        return self._batches.get(table) or []

    def append(self, table: str, frame: cudf.DataFrame) -> None:
        """Append an INSERT batch (a cuDF frame with the table's full schema)."""
        self._batches.setdefault(table, []).append(frame)

    def clear(self, table: str | None = None) -> None:
        if table is None:
            self._batches.clear()
        else:
            self._batches.pop(table, None)