"""Exception classification for the wire protocol.

A failed request becomes an ``error`` frame with a ``kind`` (``parse`` /
``runtime`` / ``protocol``) and, for parse errors, a ``position`` (line/col) the
frontend can use to squiggle the offending span in the SQL editor.

- ``parse``: sqlglot's ``ParseError`` (a syntax error) carries a list of error
  dicts with ``description``/``line``/``col``; RyuDB's own ``ParseError`` (a
  semantic rejection from ``ryudb/sql/parse.py``, subclass of ``ValueError``)
  carries only a message. Both surface as ``kind: "parse"``.
- ``runtime``: anything raised during execution (``RuntimeError``, ``KeyError``,
  ``NotImplementedError``, ``ValueError`` that isn't a parse error, ...).
- ``protocol``: a malformed frame (set by the connection handler, not here).
"""

from __future__ import annotations


def classify(exc: BaseException) -> tuple[str, str, dict[str, int] | None]:
    """Return (kind, message, position) for an exception.

    ``position`` is ``{"line": int, "col": int}`` when derivable, else ``None``.
    """
    # sqlglot's ParseError: a list of error dicts each with line/col/description.
    errs = getattr(exc, "errors", None)
    if errs:
        first = errs[0] if isinstance(errs, list) and errs else None
        if isinstance(first, dict):
            desc = first.get("description") or str(exc)
            pos = None
            line, col = first.get("line"), first.get("col")
            if line is not None or col is not None:
                pos = {"line": int(line or 0), "col": int(col or 0)}
            return ("parse", desc, pos)
    # RyuDB's own ParseError (ValueError subclass) -> parse, no position.
    name = type(exc).__name__
    if name == "ParseError" and isinstance(exc, ValueError):
        return ("parse", str(exc), None)
    # Everything else is a runtime fault.
    return ("runtime", f"{type(exc).__name__}: {exc}", None)