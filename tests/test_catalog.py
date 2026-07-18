"""Phase 2 step 1: typed, persistent catalog.

Covers type retention (Arrow schema kept instead of discarded), the
``schema_dict``/``stats_dict`` shape contract, JSON persistence round-trip
(base64 IPC schema), best-effort load (skip missing files, never raise),
``drop_table``/``alter_table`` persistence, NOT NULL derivation from field
nullability, and the "_save never raises" guarantee.

Every test builds its own Parquet + catalog under a function-scoped ``tmp_path``
so no ``ryudb.catalog.json`` leaks between tests.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ryudb.catalog import Catalog, TableInfo


def _write_typed_parquet(path: str) -> pa.Schema:
    """Write a small typed Parquet file: int64 (NOT NULL), decimal, date, string."""
    schema = pa.schema(
        [
            pa.field("id", pa.int64(), nullable=False),
            pa.field("price", pa.decimal128(15, 2)),
            pa.field("d", pa.date32()),
            pa.field("name", pa.string()),
        ]
    )
    table = pa.table(
        {
            "id": pa.array([1, 2, 3], type=pa.int64()),
            "price": pa.array([Decimal("10.00"), Decimal("20.50"), Decimal("30.00")],
                              type=pa.decimal128(15, 2)),
            "d": pa.array([date(1994, 1, 1), date(1994, 6, 1), date(1995, 1, 1)],
                          type=pa.date32()),
            "name": pa.array(["a", "b", "c"], type=pa.string()),
        },
        schema=schema,
    )
    pq.write_table(table, path)
    return schema


def _new_catalog(tmp_path) -> tuple[Catalog, str, str, pa.Schema]:
    (tmp_path / "t").mkdir()
    fpath = str(tmp_path / "t" / "0.parquet")
    schema = _write_typed_parquet(fpath)
    cat = Catalog(str(tmp_path))
    return cat, fpath, schema


# ---------------------------------------------------------------------- types

def test_register_retains_arrow_types(tmp_path):
    cat, fpath, schema = _new_catalog(tmp_path)
    info = cat.register("t", str(tmp_path / "t"))
    assert isinstance(info, TableInfo)
    assert info.columns == ["id", "price", "d", "name"]
    assert info.types["id"] == pa.int64()
    assert info.types["price"] == pa.decimal128(15, 2)
    assert info.types["d"] == pa.date32()
    assert info.types["name"] == pa.string()
    assert info.schema.equals(schema)
    assert info.row_count == 3


def test_not_null_derived_from_field_nullability(tmp_path):
    cat, _, _ = _new_catalog(tmp_path)
    info = cat.register("t", str(tmp_path / "t"))
    assert "id" in info.constraints.not_null
    for col in ("price", "d", "name"):
        assert col not in info.constraints.not_null
    assert info.constraints.primary_key is None
    assert info.constraints.unique == []
    assert info.constraints.defaults == {}


def test_schema_dict_and_stats_dict_shapes(tmp_path):
    cat, _, _ = _new_catalog(tmp_path)
    cat.register("t", str(tmp_path / "t"))
    sd = cat.schema_dict()
    assert isinstance(sd, dict)
    assert all(isinstance(v, list) and all(isinstance(c, str) for c in v) for v in sd.values())
    assert sd["t"] == ["id", "price", "d", "name"]
    st = cat.stats_dict()
    assert st == {"t": 3}
    assert all(isinstance(v, int) for v in st.values())


# ----------------------------------------------------------------- persistence

def test_persistence_round_trip(tmp_path):
    cat, _, schema = _new_catalog(tmp_path)
    cat.register("t", str(tmp_path / "t"))
    cat.set_primary_key("t", ["id"])  # constraint must survive reload
    cat.set_default("t", "name", "x")

    cat2 = Catalog(str(tmp_path))  # auto-load
    assert "t" in cat2.tables
    info = cat2.get("t")
    assert info.columns == ["id", "price", "d", "name"]
    assert info.types["price"] == pa.decimal128(15, 2)
    assert info.types["d"] == pa.date32()
    assert info.row_count == 3
    assert info.constraints.primary_key == ("id",)
    assert info.constraints.defaults == {"name": "x"}
    # schema re-derived from disk on load matches on-disk schema
    assert info.schema.equals(pq.read_schema(str(tmp_path / "t" / "0.parquet")))


def test_persistence_skips_missing_files(tmp_path):
    cat, fpath, _ = _new_catalog(tmp_path)
    cat.register("t", str(tmp_path / "t"))
    assert "t" in cat.tables
    # Remove the parquet file; a fresh catalog must skip the stale entry, not raise.
    import os
    os.remove(fpath)
    cat2 = Catalog(str(tmp_path))
    assert "t" not in cat2.tables


def test_drop_table_persists(tmp_path):
    cat, _, _ = _new_catalog(tmp_path)
    cat.register("t", str(tmp_path / "t"))
    cat.drop_table("t")
    assert "t" not in cat.tables
    cat2 = Catalog(str(tmp_path))
    assert "t" not in cat2.tables


def test_drop_table_unknown_raises(tmp_path):
    cat, _, _ = _new_catalog(tmp_path)
    with pytest.raises(KeyError):
        cat.drop_table("nope")


def test_alter_persists(tmp_path):
    cat, _, _ = _new_catalog(tmp_path)
    cat.register("t", str(tmp_path / "t"))
    cat.set_primary_key("t", ["id"])
    cat.set_unique("t", ["name"])
    cat.set_not_null("t", "price")
    cat.set_default("t", "price", Decimal("0.00"))
    cat2 = Catalog(str(tmp_path))
    info = cat2.get("t")
    assert info.constraints.primary_key == ("id",)
    assert ("name",) in info.constraints.unique
    assert "price" in info.constraints.not_null
    assert "id" in info.constraints.not_null  # PK implies NOT NULL


def test_alter_unknown_column_raises(tmp_path):
    cat, _, _ = _new_catalog(tmp_path)
    cat.register("t", str(tmp_path / "t"))
    with pytest.raises(KeyError):
        cat.set_primary_key("t", ["bogus"])
    with pytest.raises(KeyError):
        cat.set_not_null("t", "bogus")


def test_rename_table(tmp_path):
    cat, _, _ = _new_catalog(tmp_path)
    cat.register("t", str(tmp_path / "t"))
    cat.rename_table("t", "u")
    assert "t" not in cat.tables
    assert "u" in cat.tables
    assert cat.get("u").name == "u"
    cat2 = Catalog(str(tmp_path))
    assert "u" in cat2.tables and "t" not in cat2.tables


def test_register_preserves_prior_constraints(tmp_path):
    cat, _, _ = _new_catalog(tmp_path)
    cat.register("t", str(tmp_path / "t"))
    cat.set_primary_key("t", ["id"])
    # Re-register (e.g. a re-run CREATE TABLE ... FROM) must keep the declared PK.
    cat.register("t", str(tmp_path / "t"))
    assert cat.get("t").constraints.primary_key == ("id",)


# ----------------------------------------------------- _save never raises

def test_save_failure_does_not_break_register(tmp_path, monkeypatch):
    cat, _, _ = _new_catalog(tmp_path)

    def boom(*a, **kw):
        raise OSError("disk full")

    # Patch an internal call so _save's own try/except is exercised (patching
    # the whole _save method would bypass the guarantee under test).
    monkeypatch.setattr("json.dump", boom)
    info = cat.register("t", str(tmp_path / "t"))
    assert "t" in cat.tables
    assert info.row_count == 3


def test_load_corrupt_json_does_not_raise(tmp_path):
    cat, _, _ = _new_catalog(tmp_path)
    cat.register("t", str(tmp_path / "t"))
    # Corrupt the catalog file; construction must fall back to empty, not raise.
    p = tmp_path / "ryudb.catalog.json"
    p.write_text("{ not valid json")
    cat2 = Catalog(str(tmp_path))
    assert cat2.tables == {}


def test_persistence_file_format(tmp_path):
    cat, _, _ = _new_catalog(tmp_path)
    cat.register("t", str(tmp_path / "t"))
    cat.set_primary_key("t", ["id"])
    import json
    blob = json.loads((tmp_path / "ryudb.catalog.json").read_text())
    assert blob["version"] == 1
    assert len(blob["tables"]) == 1
    entry = blob["tables"][0]
    assert set(entry.keys()) == {"name", "paths", "constraints"}
    assert entry["name"] == "t"
    assert entry["constraints"]["primary_key"] == ["id"]


def test_data_dir_none_skips_persistence(tmp_path):
    cat = Catalog(None)
    cat.tables  # no crash
    # register with no data_dir: _save is a no-op (no file written)
    (tmp_path / "t").mkdir()
    _write_typed_parquet(str(tmp_path / "t" / "0.parquet"))
    info = cat.register("t", str(tmp_path / "t"))
    assert info.row_count == 3
    assert not (tmp_path / "ryudb.catalog.json").exists()