"""Catalog: maps table names to their Parquet files, schema, and row counts.

A table is registered from either a single Parquet file, a glob pattern, or a
directory containing *.parquet files. Schema is read from the first file; row
counts come from Parquet metadata (no data loaded).

Phase 2 step 1: the catalog is **typed and persistent**.
  * ``TableInfo`` retains the full ``pyarrow.Schema`` (column types are no longer
    discarded) plus a ``TableConstraints`` block (NOT NULL / PK / UNIQUE /
    DEFAULTs). Constraints are *stored* here, not yet *enforced* — enforcement
    arrives with the write path.
  * The catalog persists itself to ``<data_dir>/ryudb.catalog.json`` and reloads
    on construction, so table definitions survive a process restart. Persistence
    is best-effort: it never raises (a read-only/full disk or a stale entry is
    skipped/logged, never fatal).
  * The read path is untouched: ``schema_dict()`` still returns names only and
    no read site consumes the new typed fields. This is a pure refactor with no
    read-behavior change.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from dataclasses import dataclass, field

import pyarrow as pa
import pyarrow.parquet as pq

_CATALOG_FILE = "ryudb.catalog.json"
_CATALOG_VERSION = 1


@dataclass
class TableConstraints:
    """Declarative per-table constraints.

    Stored on the catalog; **not enforced** in step 1. The write path (step 3+)
    will consult these when accepting INSERTs and maintaining the delta-store.
    """

    not_null: set[str] = field(default_factory=set)
    primary_key: tuple[str, ...] | None = None
    unique: list[tuple[str, ...]] = field(default_factory=list)
    defaults: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "not_null": sorted(self.not_null),
            "primary_key": list(self.primary_key) if self.primary_key else None,
            "unique": [list(u) for u in self.unique],
            "defaults": dict(self.defaults),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TableConstraints":
        return cls(
            not_null=set(d.get("not_null") or []),
            primary_key=tuple(d["primary_key"]) if d.get("primary_key") else None,
            unique=[tuple(u) for u in (d.get("unique") or [])],
            defaults=dict(d.get("defaults") or {}),
        )


@dataclass
class TableInfo:
    name: str
    paths: list[str]
    columns: list[str]
    row_count: int
    schema: pa.Schema = None  # type: ignore[assignment]  # set by register/_load
    constraints: TableConstraints = field(default_factory=TableConstraints)

    @property
    def types(self) -> dict[str, pa.DataType]:
        """Column name -> Arrow type (convenience; not consumed by reads)."""
        if self.schema is None:
            return {}
        return {field.name: field.type for field in self.schema}


class Catalog:
    def __init__(self, data_dir: str | None = None):
        self.data_dir = data_dir
        self.tables: dict[str, TableInfo] = {}
        if data_dir is not None:
            self._load()

    # ------------------------------------------------------------------ persistence

    @property
    def _catalog_path(self) -> str | None:
        if self.data_dir is None:
            return None
        return os.path.join(self.data_dir, _CATALOG_FILE)

    def _load(self) -> None:
        """Best-effort load of the persisted catalog. Never raises.

        Only the table-name -> paths binding and constraints are persisted;
        schema and row_count are re-derived fresh from the on-disk Parquet so a
        loaded entry can never go stale. An entry whose Parquet is missing or
        unreadable is skipped (the table is unusable).
        """
        path = self._catalog_path
        if not path or not os.path.exists(path):
            return
        try:
            with open(path) as fh:
                blob = json.load(fh)
            for entry in blob.get("tables", []):
                name = entry["name"]
                paths = entry["paths"]
                if not paths or not os.path.exists(paths[0]):
                    continue
                try:
                    schema = pq.read_schema(paths[0])
                    row_count = sum(pq.read_metadata(p).num_rows for p in paths)
                except Exception as exc:  # noqa: BLE001
                    print(f"[catalog] skipping {name!r}: unreadable parquet ({exc})", file=sys.stderr)
                    continue
                constraints = TableConstraints.from_dict(entry.get("constraints") or {})
                self.tables[name] = TableInfo(
                    name=name,
                    paths=paths,
                    columns=list(schema.names),
                    row_count=row_count,
                    schema=schema,
                    constraints=constraints,
                )
        except Exception as exc:  # noqa: BLE001
            print(f"[catalog] failed to load {path!r}: {exc}", file=sys.stderr)
            self.tables = {}

    def _save(self) -> None:
        """Best-effort atomic persist. Never raises; no-op without data_dir."""
        path = self._catalog_path
        if path is None:
            return
        try:
            os.makedirs(self.data_dir, exist_ok=True)
            blob = {
                "version": _CATALOG_VERSION,
                "tables": [
                    {
                        "name": name,
                        "paths": ti.paths,
                        "constraints": ti.constraints.to_dict(),
                    }
                    for name, ti in self.tables.items()
                ],
            }
            tmp = path + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(blob, fh, indent=2)
            os.replace(tmp, path)
        except Exception as exc:  # noqa: BLE001
            print(f"[catalog] failed to save {path!r}: {exc}", file=sys.stderr)

    # ------------------------------------------------------------------ mutations

    def register(self, table: str, path: str) -> TableInfo:
        paths = _resolve_parquet_paths(path)
        if not paths:
            raise FileNotFoundError(f"no parquet files found for {table!r} at {path!r}")
        schema = pq.read_schema(paths[0])
        columns = list(schema.names)
        row_count = sum(pq.read_metadata(p).num_rows for p in paths)
        not_null = {f.name for f in schema if not f.nullable}
        info = TableInfo(
            name=table,
            paths=paths,
            columns=columns,
            row_count=row_count,
            schema=schema,
            constraints=TableConstraints(not_null=not_null),
        )
        # Preserve any constraints previously set on this table (e.g. re-register
        # over an auto-loaded entry that had a PK declared).
        prior = self.tables.get(table)
        if prior is not None:
            info.constraints = prior.constraints
            info.constraints.not_null |= not_null
        self.tables[table] = info
        self._save()
        return info

    def drop_table(self, table: str) -> None:
        if table not in self.tables:
            raise KeyError(f"unknown table: {table!r} (registered: {list(self.tables)})")
        del self.tables[table]
        self._save()

    def rename_table(self, old: str, new: str) -> None:
        info = self.get(old)
        if new in self.tables:
            raise KeyError(f"table already exists: {new!r}")
        del self.tables[old]
        info.name = new
        self.tables[new] = info
        self._save()

    def set_primary_key(self, table: str, cols: list[str] | tuple[str, ...]) -> None:
        info = self.get(table)
        cols_t = tuple(cols)
        unknown = [c for c in cols_t if c not in info.columns]
        if unknown:
            raise KeyError(f"unknown columns for {table!r}: {unknown}")
        info.constraints.primary_key = cols_t
        info.constraints.not_null.update(cols_t)
        self._save()

    def set_not_null(self, table: str, col: str, on: bool = True) -> None:
        info = self.get(table)
        if col not in info.columns:
            raise KeyError(f"unknown column for {table!r}: {col!r}")
        if on:
            info.constraints.not_null.add(col)
        else:
            info.constraints.not_null.discard(col)
        self._save()

    def set_unique(self, table: str, cols: list[str] | tuple[str, ...]) -> None:
        info = self.get(table)
        cols_t = tuple(cols)
        unknown = [c for c in cols_t if c not in info.columns]
        if unknown:
            raise KeyError(f"unknown columns for {table!r}: {unknown}")
        if cols_t not in info.constraints.unique:
            info.constraints.unique.append(cols_t)
        self._save()

    def set_default(self, table: str, col: str, value: object) -> None:
        info = self.get(table)
        if col not in info.columns:
            raise KeyError(f"unknown column for {table!r}: {col!r}")
        info.constraints.defaults[col] = value
        self._save()

    def drop_default(self, table: str, col: str) -> None:
        info = self.get(table)
        info.constraints.defaults.pop(col, None)
        self._save()

    # ------------------------------------------------------------------ reads

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
            pk = f" pk={list(ti.constraints.primary_key)}" if ti.constraints.primary_key else ""
            lines.append(f"{t}: {ti.row_count} rows, {len(ti.paths)} file(s), cols={ti.columns}{pk}")
        return "\n".join(lines)


def _resolve_parquet_paths(path: str) -> list[str]:
    if os.path.isdir(path):
        return sorted(glob.glob(os.path.join(path, "*.parquet")))
    if os.path.exists(path):
        return [path]
    # treat as glob
    return sorted(glob.glob(path))