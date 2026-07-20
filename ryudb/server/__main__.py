"""``ryudb-server`` entrypoint — start the RyuDB server.

    ryudb-server --data ./data --host 127.0.0.1 --port 5430 --pg-port 5432

Binds a WebSocket server (custom JSON + Arrow IPC protocol) fronting a single
``Engine`` over ``data_dir``. With ``--pg-port`` set, also binds a Postgres
v3 wire-protocol front on that port (shared engine + per-connection
transactions) so real drivers (``psql``, ``psycopg``, ``pg8000``, ``asyncpg``)
can connect. All flags are overridable by the matching ``RYUDB_*`` env var. Run
with ``--help`` for the full surface. See ``ryudb/server/PROTOCOL.md`` for the
wire formats.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

from .app import Server
from .pgwire import PGServer


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="ryudb-server",
        description="Run the RyuDB engine as a server (WebSocket + optional Postgres wire).",
    )
    ap.add_argument("--data", default=_env("RYUDB_DATA", "./data"),
                    help="data directory (catalog + parquet); default ./data")
    ap.add_argument("--host", default=_env("RYUDB_HOST", "127.0.0.1"),
                    help="bind host; default 127.0.0.1")
    ap.add_argument("--port", type=int, default=int(_env("RYUDB_PORT", "5430")),
                    help="WebSocket port; default 5430")
    ap.add_argument("--pg-port", type=int, default=int(_env("RYUDB_PG_PORT", "0")),
                    help="Postgres wire-protocol port (0 = disabled); default 0")
    ap.add_argument("--pg-max-rows", type=int,
                    default=int(_env("RYUDB_PG_MAX_ROWS", "200000")),
                    help="max rows returned per SELECT over the PG wire; default 200000")
    ap.add_argument("--max-rows", type=int,
                    default=int(_env("RYUDB_MAX_ROWS", "200000")),
                    help="max rows returned per SELECT over WebSocket (full result "
                         "available via export); default 200000")
    ap.add_argument("--workers", type=int,
                    default=int(_env("RYUDB_WORKERS", "1")),
                    help="engine worker pool size. 1 (default) preserves the "
                         "original single-worker semantics; N>1 lets SELECTs run "
                         "concurrently (writes/admin ops still serialize via the "
                         "engine's read/write lock). Each concurrent query holds "
                         "GPU memory, so lower this (toward 1) for very large "
                         "queries to avoid GPU OOM.")
    ap.add_argument("--log-level", default=_env("RYUDB_LOG_LEVEL", "info"),
                    choices=["debug", "info", "warning", "error"],
                    help="log level; default info")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    server = Server(args.data, args.host, args.port, args.max_rows,
                    n_workers=args.workers)
    coros = [server.serve()]
    if args.pg_port:
        pg = PGServer(server, args.host, args.pg_port, args.pg_max_rows)
        coros.append(pg.serve())
    try:
        asyncio.run(asyncio.gather(*coros))
    except KeyboardInterrupt:
        log = logging.getLogger("ryudb.server")
        log.info("interrupted, shutting down")
    return 0


if __name__ == "__main__":
    sys.exit(main())