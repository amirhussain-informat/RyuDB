"""Wire protocol helpers: JSON text frames + Arrow IPC binary frames.

Transport is WebSocket. Text frames carry JSON control messages (requests,
response meta, errors, events); binary frames carry Arrow IPC stream bytes
(query result data). A SELECT response is exactly: one JSON ``result`` meta
frame followed by one binary Arrow IPC stream frame (Phase 1 sends the whole
result in a single binary frame; the meta ``frame_count`` field reserves the
ability to stream multiple batches later without a breaking change).

cuDF -> Arrow canonicalization (the load-bearing path, verified by the Phase 1
fidelity spike): ``df.reset_index(drop=True).to_arrow().replace_schema_metadata(
None)``. The ``reset_index`` drops a *named* index that some executor paths
(``_sort``/``_limit``/empty-filtered scans) materialize as a column — without it
a spurious ``index: int64`` column leaks into the result. Stripping the pandas
schema metadata keeps the wire schema clean (the JS Arrow reader ignores it
anyway, but it bloats every frame).
"""

from __future__ import annotations

import json
from typing import Any

import pyarrow as pa
import pyarrow.ipc as ipc


class ProtocolError(Exception):
    """A malformed request frame (bad JSON, missing required field, bad op)."""


def loads_request(text: str) -> dict[str, Any]:
    """Parse a JSON text frame into a request dict; raise ProtocolError on bad
    JSON or a non-object payload."""
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"invalid JSON frame: {exc.msg}") from exc
    if not isinstance(obj, dict):
        raise ProtocolError("request frame must be a JSON object")
    return obj


def dumps_json(obj: dict[str, Any]) -> str:
    """Serialize a response/event dict to a JSON text frame."""
    return json.dumps(obj, default=_json_default)


def _json_default(obj: Any) -> Any:
    # pyarrow scalars / numpy ints / dates -> JSON-friendly primitives.
    if isinstance(obj, (pa.Scalar,)):
        return obj.as_py()
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes, dict, list)):
        try:
            return obj.item()
        except Exception:  # noqa: BLE001
            return str(obj)
    return str(obj)


def df_to_arrow(df) -> pa.Table:
    """The canonical cuDF -> pyarrow.Table path: drop any named index and strip
    pandas schema metadata so the wire schema is exactly the result columns."""
    return df.reset_index(drop=True).to_arrow().replace_schema_metadata(None)


def table_to_ipc(table: pa.Table) -> bytes:
    """Serialize a pyarrow Table to an Arrow IPC *stream* bytes buffer."""
    sink = pa.BufferOutputStream()
    writer = ipc.new_stream(sink, table.schema)
    writer.write_table(table)
    writer.close()
    return sink.getvalue().to_pybytes()


def column_meta(table: pa.Table) -> list[dict[str, str]]:
    """Per-column metadata for the result ``meta`` frame: name + a stringified
    Arrow type (the frontend renders types, it does not need the full type
    tree)."""
    return [{"name": f.name, "type": str(f.type)} for f in table.schema]


def error_frame(rid: Any, exc: BaseException) -> dict[str, Any]:
    """Build an ``error`` response frame from an exception, classifying it as
    ``parse`` (sqlglot/RyuDB ParseError, carries line/col) or ``runtime``."""
    # Imported here to avoid a circular import at module load.
    from .errors import classify

    kind, message, position = classify(exc)
    frame: dict[str, Any] = {"id": rid, "op": "error", "kind": kind, "message": message}
    if position is not None:
        frame["position"] = position
    return frame