"""``ryudb-server`` entrypoint — start the RyuDB WebSocket server.

    ryudb-server --data ./data --host 127.0.0.1 --port 5430

Binds a WebSocket server fronting a single ``Engine`` over ``data_dir``. All
flags are overridable by the matching ``RYUDB_*`` env var. Run with ``--help``
for the full surface. See ``ryudb/server/PROTOCOL.md`` for the wire format.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

from .app import Server


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="ryudb-server",
        description="Run the RyuDB engine as a WebSocket server (custom JSON + Arrow IPC protocol).",
    )
    ap.add_argument("--data", default=_env("RYUDB_DATA", "./data"),
                    help="data directory (catalog + parquet); default ./data")
    ap.add_argument("--host", default=_env("RYUDB_HOST", "127.0.0.1"),
                    help="bind host; default 127.0.0.1")
    ap.add_argument("--port", type=int, default=int(_env("RYUDB_PORT", "5430")),
                    help="bind port; default 5430")
    ap.add_argument("--max-rows", type=int,
                    default=int(_env("RYUDB_MAX_ROWS", "200000")),
                    help="max rows returned per SELECT (full result available via "
                         "export); default 200000")
    ap.add_argument("--log-level", default=_env("RYUDB_LOG_LEVEL", "info"),
                    choices=["debug", "info", "warning", "error"],
                    help="log level; default info")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    server = Server(args.data, args.host, args.port, args.max_rows)
    try:
        asyncio.run(server.serve())
    except KeyboardInterrupt:
        log = logging.getLogger("ryudb.server")
        log.info("interrupted, shutting down")
    return 0


if __name__ == "__main__":
    sys.exit(main())