"""ryudb-server — a network server fronting the RyuDB engine.

A standalone daemon that owns a single ``Engine`` instance and accepts
WebSocket connections speaking a custom JSON + Arrow IPC protocol. All engine
and catalog work is serialized through one background worker thread (the
``Engine`` is single-session / one-thread by design, ``executor.py:152``), so
concurrent connections execute strictly one request at a time.

This is Phase 1 of the Snowsight-like frontend track: the server is the thing
the web console (and a networked CLI) connect to. The engine stays a library;
the server is a thin process around it.

See ``PROTOCOL.md`` for the wire format and ``__main__.py`` for the entrypoint.
"""

from __future__ import annotations

from .app import Server

__all__ = ["Server"]