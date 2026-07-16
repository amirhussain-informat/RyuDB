"""Catalog: maps table names to their Parquet files, schema, and row counts.

A table is registered from either a single Parquet file, a glob pattern, or a
directory containing *.parquet files. Schema is read from the first file; row
counts come from Parquet metadata (no data loaded).
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import pyarrow.parquet as pq


@dataclass
class TableInfo:
    name: str
    paths: list[str]
    columns: list[str]
    row_count: int


class Catalog:
    def __init__(self, data_dir: str | None = None):
        self.data_dir = data_dir
        self.tables: dict[str, TableInfo] = {}

    def register(self, table: str, path: str) -> TableInfo:
        paths = _resolve_parquet_paths(path)
        if not paths:
            raise FileNotFoundError(f"no parquet files found for {table!r} at {path!r}")
        schema = pq.read_schema(paths[0])
        columns = list(schema.names)
        row_count = sum(pq.read_metadata(p).num_rows for p in paths)
        info = TableInfo(name=table, paths=paths, columns=columns, row_count=row_count)
        self.tables[table] = info
        return info

    def get(self, table: str) -> TableInfo:
        if table not in self.tables:
            raise KeyError(f"unknown table: {table!r} (registered: {list(self.tables)})")
        return self.tables[table]

    def schema_dict(self) -> dict[str, list[str]]:
        return {t: ti.columns for t, ti in self.tables.items()}

    def stats_dict(self) -> dict[str, int]:
        return {t: ti.row_count for t, ti in self.tables.items()}

    def describe(self) -> str:
        if not self.tables:
            return "(no tables registered)"
        lines = []
        for t, ti in self.tables.items():
            lines.append(f"{t}: {ti.row_count} rows, {len(ti.paths)} file(s), cols={ti.columns}")
        return "\n".join(lines)


def _resolve_parquet_paths(path: str) -> list[str]:
    if os.path.isdir(path):
        return sorted(glob.glob(os.path.join(path, "*.parquet")))
    if os.path.exists(path):
        return [path]
    # treat as glob
    return sorted(glob.glob(path))