"""Write-ahead log for the durable HTAP write path (Phase 2 step 6).

The delta-store (``DeltaStore``) holds committed INSERT batches in GPU memory
tagged with a monotonic ``commit_ts`` (MVCC, step 5). Without this module a
process restart drops every committed write. The WAL makes a commit durable
*before* the in-memory delta is mutated: the ``Engine`` appends one record per
commit, ``fsync``s it, and only then appends the batch to the in-memory delta.
On ``Engine`` startup the WAL is replayed to reconstruct the delta and reset
``_commit_ts`` to the highest replayed LSN -- so ``commit_ts`` doubles as the
WAL LSN.

Scope (step 6):
  * Only **committed** batches are logged. The uncommitted transaction buffer
    (``Transaction._buffer``) is never written -- single-session, and a pending
    txn is implicitly rolled back on process exit.
  * Snapshots stay in-memory only; a restart drops named snapshots. The WAL
    records data, not snapshot markers.
  * One WAL record **per commit** (not per batch): a commit is either a fully
    CRC-valid durable record or it isn't, so commit atomicity is trivial. For
    autocommit INSERT one batch == one record; for a txn ``COMMIT`` flushing N
    buffered batches the single record holds all N in the exact in-memory
    append order, preserving the MVCC byte-identical invariant.

Record framing is struct-packed (no ``pickle`` for the envelope) and CRC32'd so
a torn tail left by a crash mid-write is detected on replay and truncated --
recovery stops at the last good record. The batch payload itself is serialized
as Apache Arrow IPC bytes of the cuDF frame's pandas view, and replayed via the
exact ``cudf.DataFrame(pd.DataFrame(pdf))`` construction ``_insert`` uses, so
replayed frames cast cleanly in ``Engine._merge_delta`` (nullable Int64 /
datetime64[ns] / float64 preserved; honors the cuDF 26.06 "no
``DataFrame.from_pandas``" gotcha).

The WAL is disabled (a no-op) when there is no data dir, mirroring how
``Catalog`` persistence is gated -- an ephemeral ``Engine`` stays in-memory.
"""

from __future__ import annotations

import os
import struct
import zlib
from typing import TYPE_CHECKING, BinaryIO

import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

if TYPE_CHECKING:
    import cudf

_WAL_NAME = "ryudb.wal"

# Header: u32 payload_len, u32 crc32(payload). Payload: u64 commit_ts, u32
# n_batches, then per batch u8 kind (0=insert, 1=tombstone -- step 9) + u32
# name_len + name bytes + u32 arrow_len + arrow bytes. All little-endian.
_HEADER = struct.Struct("<II")
_COMMIT_TS = struct.Struct("<Q")
_U32 = struct.Struct("<I")
_U8 = struct.Struct("<B")

# kind byte <-> tag string. Tombstones (DELETE PK values) take the parallel
# ``_tombstones`` channel in ``DeltaStore``; inserts stay on ``_batches``.
_KIND_INSERT = 0
_KIND_TOMBSTONE = 1
_KIND_BY_TAG = {"insert": _KIND_INSERT, "tombstone": _KIND_TOMBSTONE}
_TAG_BY_KIND = {v: k for k, v in _KIND_BY_TAG.items()}


def _serialize_frame(frame: "cudf.DataFrame") -> bytes:
    """cuDF batch -> Arrow IPC stream bytes of the frame's pandas view.

    ``to_pandas()`` is the stable cuDF->CPU boundary; Arrow IPC round-trips every
    dtype we care about (nullable Int64, datetime64, float64, object strings)."""
    table = pa.Table.from_pandas(frame.to_pandas())
    sink = pa.BufferOutputStream()
    with ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def _deserialize_frame(arrow_bytes: bytes) -> "cudf.DataFrame":
    """Arrow IPC bytes -> cuDF frame via the same construction ``_insert`` uses."""
    import cudf

    table = ipc.open_stream(arrow_bytes).read_all()
    return cudf.DataFrame(pd.DataFrame(table.to_pandas()))


def _build_record(
    commit_ts: int, batches: list[tuple[str, str, "cudf.DataFrame"]]
) -> bytes:
    """Serialize one commit to its full on-disk record bytes (header + payload).

    ``batches`` are ``(table, kind, frame)`` triples where ``kind`` is
    ``"insert"`` or ``"tombstone"`` (step 9 lets one commit mix INSERT and
    DELETE batches; the per-batch kind byte preserves that split on disk)."""
    parts: list[bytes] = [_COMMIT_TS.pack(commit_ts), _U32.pack(len(batches))]
    for table, kind, frame in batches:
        name = table.encode("utf-8")
        arrow = _serialize_frame(frame)
        parts.append(_U8.pack(_KIND_BY_TAG[kind]))
        parts.append(_U32.pack(len(name)))
        parts.append(name)
        parts.append(_U32.pack(len(arrow)))
        parts.append(arrow)
    payload = b"".join(parts)
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    return _HEADER.pack(len(payload), crc) + payload


class WAL:
    """Append-only write-ahead log at ``<data_dir>/ryudb.wal``.

    Disabled (all methods no-op / empty-return) when ``path is None``.
    """

    def __init__(self, path: str | None) -> None:
        self.path = path
        self._fh: BinaryIO | None = None  # lazily opened (see _acquire)

    def _acquire(self, create: bool) -> BinaryIO | None:
        """Return the persistent file handle, opening it lazily on first use.

        ``create=False`` (replay/truncate) opens an *existing* file in ``r+b`` --
        it never creates the file, so an Engine over a data dir that has never
        been written to leaves no stray 0-byte WAL (mirrors how ``Catalog``
        creates its file only on first save). ``create=True`` (write_commit)
        opens ``a+b`` which creates the file on the first commit. ``r+b`` and
        ``a+b`` both allow read+write+seek+truncate; we always seek explicitly,
        so the append-mode flag is irrelevant. No-op (returns None) when the WAL
        is disabled (``path is None``)."""
        if self._fh is None and self.path is not None:
            if os.path.exists(self.path):
                self._fh = open(self.path, "r+b")
            elif create:
                os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
                self._fh = open(self.path, "a+b")
        return self._fh

    # ------------------------------------------------------------------ replay

    def replay(self) -> list[tuple[int, str, str, "cudf.DataFrame"]]:
        """Read every valid record, in order. Stops at the first torn/short/
        CRC-mismatch record and **truncates the file to the last good byte
        offset** (cleans a crashed tail so later appends never strand behind a
        corrupt record), then seeks to EOF for appending. Returns
        ``[(commit_ts, table, kind, frame), ...]`` where ``kind`` is
        ``"insert"`` or ``"tombstone"`` (step 9). No-op (empty) when disabled."""
        fh = self._acquire(create=False)
        if fh is None:
            return []  # disabled, or no WAL file yet -> nothing to replay
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        fh.seek(0)
        records: list[tuple[int, str, str, "cudf.DataFrame"]] = []
        offset = 0
        while offset < size:
            if offset + _HEADER.size > size:
                break  # torn header tail
            fh.seek(offset)
            header = fh.read(_HEADER.size)
            (payload_len, crc) = _HEADER.unpack(header)
            if offset + _HEADER.size + payload_len > size:
                break  # torn payload tail
            payload = fh.read(payload_len)
            if zlib.crc32(payload) & 0xFFFFFFFF != crc:
                break  # corrupt record -- discard this and everything after
            try:
                records.extend(_parse_payload(payload))
            except Exception:
                # A structurally unreadable payload is treated as a corrupt tail
                # too -- drop it and anything after.
                break
            # fh now sits at the byte just past this record's payload.
            offset = fh.tell()
        # Truncate any torn/corrupt tail so the file ends at the last good record.
        if offset != size:
            fh.truncate(offset)
            fh.flush()
            os.fsync(fh.fileno())
        fh.seek(0, os.SEEK_END)
        return records

    # ------------------------------------------------------------ write commit

    def write_commit(
        self, commit_ts: int, batches: list[tuple[str, str, "cudf.DataFrame"]]
    ) -> None:
        """Append one durable commit record. Empty ``batches`` is a no-op (an
        empty commit bumps the in-memory counter but persists no data; the
        counter is recovered as ``max(replayed ts)``). Builds the full record in
        memory, writes it in one call, then ``flush`` + ``fsync`` so the commit
        is durable before the caller mutates the in-memory delta. On ANY error
        the file is truncated back to the pre-write offset (no torn record left
        behind) and the error re-raised. No-op when disabled."""
        if not batches:
            return  # empty commit -> persist nothing
        fh = self._acquire(create=True)
        if fh is None:
            return  # disabled (no data dir) -> in-memory only
        record = _build_record(commit_ts, batches)
        fh.seek(0, os.SEEK_END)
        start = fh.tell()
        try:
            fh.write(record)
            fh.flush()
            os.fsync(fh.fileno())
        except Exception:
            # Restore the file to its last-good length so a later write doesn't
            # append after a half-written (torn) record.
            try:
                fh.truncate(start)
                fh.flush()
            except Exception:
                pass
            raise

    # ----------------------------------------------------------------- truncate

    def truncate(self, max_ts: int) -> None:
        """Drop every record with ``commit_ts > max_ts`` (durable RESTORE). Since
        ``commit_ts`` is monotonic and records are appended in ts order, the
        dropped records are a tail suffix, so this is a physical truncate to the
        byte offset of the first record past ``max_ts`` (+ fsync). Also stops at
        the first corrupt record (mirrors ``DeltaStore.rewind``). No-op when
        disabled."""
        fh = self._acquire(create=False)
        if fh is None:
            return  # disabled, or no WAL file yet -> nothing to truncate
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        fh.seek(0)
        offset = 0
        keep_until = size  # byte offset to truncate to (keep [0, keep_until))
        while offset < size:
            if offset + _HEADER.size > size:
                keep_until = offset  # torn header tail -> drop it
                break
            fh.seek(offset)
            header = fh.read(_HEADER.size)
            (payload_len, crc) = _HEADER.unpack(header)
            rec_end = offset + _HEADER.size + payload_len
            if rec_end > size:
                keep_until = offset  # torn payload tail -> drop it
                break
            payload = fh.read(payload_len)
            if zlib.crc32(payload) & 0xFFFFFFFF != crc:
                keep_until = offset  # corrupt record -> drop it and rest
                break
            try:
                (commit_ts,) = _COMMIT_TS.unpack(payload[: _COMMIT_TS.size])
            except Exception:
                keep_until = offset
                break
            if commit_ts > max_ts:
                keep_until = offset  # first record past max_ts -> drop from here
                break
            offset = rec_end
        if keep_until != size:
            fh.truncate(keep_until)
            fh.flush()
            os.fsync(fh.fileno())
        fh.seek(0, os.SEEK_END)

    # ------------------------------------------------------------------- close

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            finally:
                self._fh = None


def _parse_payload(payload: bytes) -> list[tuple[int, str, str, "cudf.DataFrame"]]:
    """Split a payload into its ``(commit_ts, table, kind, frame)`` records
    (``kind`` is ``"insert"`` or ``"tombstone"``)."""
    pos = 0
    (commit_ts,) = _COMMIT_TS.unpack_from(payload, pos)
    pos += _COMMIT_TS.size
    (n_batches,) = _U32.unpack_from(payload, pos)
    pos += _U32.size
    out: list[tuple[int, str, str, "cudf.DataFrame"]] = []
    for _ in range(n_batches):
        (kind_byte,) = _U8.unpack_from(payload, pos)
        pos += _U8.size
        (name_len,) = _U32.unpack_from(payload, pos)
        pos += _U32.size
        name = payload[pos:pos + name_len].decode("utf-8")
        pos += name_len
        (arrow_len,) = _U32.unpack_from(payload, pos)
        pos += _U32.size
        arrow_bytes = payload[pos:pos + arrow_len]
        pos += arrow_len
        out.append((commit_ts, name, _TAG_BY_KIND[kind_byte], _deserialize_frame(arrow_bytes)))
    return out


def wal_path(data_dir: str | None) -> str | None:
    """Resolve ``<data_dir>/ryudb.wal`` or ``None`` when there is no data dir."""
    if data_dir is None:
        return None
    return os.path.join(data_dir, _WAL_NAME)