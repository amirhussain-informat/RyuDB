"""RyuDB CLI / REPL.

Usage:
  ryudb                      # start the REPL (data dir defaults to ./data)
  ryudb -d <data_dir>        # use a different data directory
  ryudb -e "SELECT ..."      # run a single SQL statement and exit
  ryudb -f script.sql        # run a SQL script and exit

REPL dot-commands:
  :tables        list registered tables
  :schema NAME   show columns + row count for a table
  :explain SQL   print the optimized logical plan without running
  :help          show help
  :quit          exit
"""

from __future__ import annotations

import argparse
import re
import sys

import pandas as pd

from .catalog import Catalog
from .exec.executor import Engine

_CREATE_RE = re.compile(
    r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([A-Za-z_][\w]*)\s+FROM\s+'([^']+)'\s*;?",
    re.IGNORECASE,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ryudb", description="RyuDB GPU SQL REPL")
    parser.add_argument("-d", "--data", default="./data", help="data directory")
    parser.add_argument("-e", "--exec", dest="exec_sql", help="run a SQL statement and exit")
    parser.add_argument("-f", "--file", help="run a SQL script file and exit")
    args = parser.parse_args(argv)

    catalog = Catalog(args.data)
    engine = Engine(catalog)

    if args.exec_sql:
        return _run_statement(engine, args.exec_sql, quiet=False)
    if args.file:
        with open(args.file) as fh:
            return _run_script(engine, fh.read())
    return _repl(engine, catalog)


def _run_script(engine: Engine, text: str) -> int:
    for stmt in _split_statements(text):
        stmt = stmt.strip()
        if not stmt:
            continue
        rc = _run_statement(engine, stmt, quiet=False)
        if rc != 0:
            return rc
    return 0


def _run_statement(engine: Engine, stmt: str, quiet: bool) -> int:
    create = _CREATE_RE.match(stmt.strip())
    if create:
        table, path = create.group(1), create.group(2)
        info = engine.catalog.register(table, path)
        print(f"registered {table}: {info.row_count} rows, {len(info.columns)} cols")
        return 0
    if re.match(r"\s*EXPLAIN\s+", stmt, re.IGNORECASE):
        sql = re.sub(r"^\s*EXPLAIN\s+", "", stmt, flags=re.IGNORECASE)
        try:
            print(engine.explain(sql))
        except Exception as exc:  # noqa: BLE001
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0
    try:
        df = engine.sql(stmt)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1
    _print_frame(df)
    return 0


def _repl(engine: Engine, catalog: Catalog) -> int:
    print("RyuDB (GPU, cuDF). Type :help for commands, :quit to exit.")
    buffer: list[str] = []
    while True:
        try:
            prompt = "ryudb> " if not buffer else "   ...> "
            line = input(prompt)
        except (EOFError, KeyboardInterrupt):
            print()
            break
        stripped = line.strip()
        if not buffer and stripped.startswith(":"):
            cmd = stripped[1:].strip()
            if _dot_command(cmd, engine, catalog):
                break
            continue
        buffer.append(line)
        if stripped.endswith(";"):
            stmt = "\n".join(buffer).strip()
            buffer.clear()
            _run_statement(engine, stmt, quiet=False)
    return 0


def _dot_command(cmd: str, engine: Engine, catalog: Catalog) -> bool:
    """Return True to exit the REPL."""
    parts = cmd.split(None, 1)
    name = parts[0].lower() if parts else ""
    arg = parts[1] if len(parts) > 1 else ""
    if name in ("quit", "exit", "q"):
        return True
    if name == "help":
        print(__doc__)
    elif name == "tables":
        print(catalog.describe())
    elif name == "schema":
        try:
            info = catalog.get(arg)
            print(f"{info.name}: {info.row_count} rows, cols={info.columns}")
        except KeyError as exc:
            print(f"error: {exc}")
    elif name == "explain":
        if not arg:
            print("usage: :explain <sql>")
        else:
            try:
                print(engine.explain(arg))
            except Exception as exc:  # noqa: BLE001
                print(f"error: {exc}")
    elif name == "":
        return False
    else:
        print(f"unknown command :{name} (try :help)")
    return False


def _split_statements(text: str) -> list[str]:
    # Naive splitter on ';'. Sufficient for Phase 1 scripts without quoted ';'.
    return [s for s in text.split(";") if s.strip()]


def _print_frame(df) -> None:
    n = len(df)
    if n == 0:
        print("(empty result set)")
        return
    with pd.option_context("display.max_rows", 50, "display.width", 200):
        print(df.to_pandas().to_string())
    if n > 50:
        print(f"... ({n} rows total, showing first 50)")


if __name__ == "__main__":
    raise SystemExit(main())