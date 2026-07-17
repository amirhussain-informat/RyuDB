"""RyuDB CLI / REPL.

Usage:
  ryudb                      # start the REPL (data dir defaults to ./data)
  ryudb -d <data_dir>        # use a different data directory
  ryudb -e "SELECT ..."      # run a single SQL statement and exit
  ryudb -f script.sql        # run a SQL script and exit

REPL dot-commands:
  :tables        list registered tables
  :schema NAME   show columns (typed) + row count + constraints for a table
  :drop NAME     drop a table from the catalog
  :alter NAME ...  alter catalog metadata (no base-file mutation):
                     :alter NAME rename NEW
                     :alter NAME pk c1[,c2,...]
                     :alter NAME notnull COL | :alter NAME notnull- COL
                     :alter NAME unique c1[,c2,...]
                     :alter NAME default COL VALUE
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
        return _run_script(engine, args.exec_sql)
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
        result = engine.sql(stmt)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1
    # INSERT returns an int row count (the write path mutates the delta); SELECT
    # returns a cuDF frame to print.
    if isinstance(result, int):
        print(f"inserted {result} rows")
        return 0
    _print_frame(result)
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
            if info.schema is not None:
                for f in info.schema:
                    print(f"  {f.name}: {f.type}")
            c = info.constraints
            pieces = []
            if c.primary_key:
                pieces.append(f"PRIMARY KEY {list(c.primary_key)}")
            if c.not_null:
                nn = sorted(c.not_null - (set(c.primary_key) if c.primary_key else set()))
                if c.primary_key:
                    pieces.append(f"NOT NULL (incl. PK) {sorted(c.not_null)}")
                else:
                    pieces.append(f"NOT NULL {nn}")
            for u in c.unique:
                pieces.append(f"UNIQUE {list(u)}")
            for col, val in c.defaults.items():
                pieces.append(f"DEFAULT {col}={val!r}")
            if pieces:
                print("  constraints:")
                for p in pieces:
                    print(f"    {p}")
        except KeyError as exc:
            print(f"error: {exc}")
    elif name == "drop":
        try:
            catalog.drop_table(arg)
            print(f"dropped {arg}")
        except KeyError as exc:
            print(f"error: {exc}")
    elif name == "alter":
        _alter_command(catalog, arg)
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


def _coerce_literal(text: str):
    text = text.strip()
    low = text.lower()
    if low == "null":
        return None
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    # strip surrounding quotes
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        return text[1:-1]
    return text


def _alter_command(catalog: Catalog, arg: str) -> None:
    """Handle ``:alter NAME <action> ...`` (catalog metadata only)."""
    parts = arg.split(None, 2)
    if len(parts) < 2:
        print("usage: :alter NAME {rename NEW|pk c1,...|notnull COL|notnull- COL|"
              "unique c1,...|default COL VALUE}")
        return
    table, action = parts[0], parts[1].lower()
    rest = parts[2] if len(parts) > 2 else ""
    try:
        if action == "rename":
            if not rest:
                print("usage: :alter NAME rename NEW")
                return
            catalog.rename_table(table, rest.strip())
            print(f"renamed {table} -> {rest.strip()}")
        elif action == "pk":
            cols = [c.strip() for c in rest.split(",") if c.strip()]
            if not cols:
                print("usage: :alter NAME pk c1[,c2,...]")
                return
            catalog.set_primary_key(table, cols)
            print(f"{table}: primary key set to {cols}")
        elif action in ("notnull", "notnull-"):
            col = rest.strip()
            if not col:
                print("usage: :alter NAME notnull COL")
                return
            catalog.set_not_null(table, col, on=(action == "notnull"))
            print(f"{table}: {col} NOT NULL {'set' if action == 'notnull' else 'dropped'}")
        elif action == "unique":
            cols = [c.strip() for c in rest.split(",") if c.strip()]
            if not cols:
                print("usage: :alter NAME unique c1[,c2,...]")
                return
            catalog.set_unique(table, cols)
            print(f"{table}: UNIQUE({cols}) added")
        elif action == "default":
            sub = rest.split(None, 1)
            if len(sub) < 2:
                print("usage: :alter NAME default COL VALUE")
                return
            catalog.set_default(table, sub[0], _coerce_literal(sub[1]))
            print(f"{table}: {sub[0]} DEFAULT = {_coerce_literal(sub[1])!r}")
        else:
            print(f"unknown alter action: {action!r} (rename|pk|notnull|notnull-|unique|default)")
    except KeyError as exc:
        print(f"error: {exc}")


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