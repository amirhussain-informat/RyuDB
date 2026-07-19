"""Phase 2 step 3: INSERT plan node + parser routing + executor branch.

INSERT builds a typed cuDF batch from literal VALUES, fills DEFAULTs, enforces
NOT NULL, and appends it to the in-memory delta; the next SELECT re-merges the
live delta in ``_scan`` (autocommit, no txn layer yet). PK/UNIQUE uniqueness is
NOT enforced this step. These tests cover parse routing, the write/read
round-trip (float and DECIMAL/DATE schemas), a DuckDB oracle, NOT NULL / DEFAULT
enforcement, multi-row inserts, and the CLI "inserted N rows" output.

Uses the function-scoped ``engine`` / ``typed_engine`` fixtures so the in-memory
delta never pollutes the session ``data_dir``. Constraint tests build a fresh
catalog (own ``tmp_path``) over the same parquet so ``set_not_null`` / ``set_default``
saves never leak into the shared ``data_dir`` catalog file.
"""

from __future__ import annotations

import pytest

from ryudb import Catalog, Engine
from ryudb.sql.parse import parse
from ryudb.sql.plan import Insert

from .conftest import assert_same


# --------------------------------------------------------------------- parse

def test_insert_parse_with_columns():
    plan = parse("INSERT INTO lineitem (l_orderkey, l_quantity) VALUES (1, 2.0), (3, 4.0)")
    assert isinstance(plan, Insert)
    assert plan.table == "lineitem"
    assert plan.columns == ["l_orderkey", "l_quantity"]
    assert len(plan.rows) == 2
    assert [len(r) for r in plan.rows] == [2, 2]


def test_insert_parse_without_columns():
    plan = parse("INSERT INTO lineitem VALUES (1, 2.0, 3.0, date '1998-08-10')")
    assert isinstance(plan, Insert)
    assert plan.columns is None
    assert len(plan.rows) == 1
    assert len(plan.rows[0]) == 4


def test_insert_parse_rejects_value_count_mismatch():
    with pytest.raises(Exception):
        parse("INSERT INTO lineitem (l_orderkey, l_quantity) VALUES (1, 2.0), (3,)")


def test_insert_parse_rejects_on_conflict():
    with pytest.raises(NotImplementedError):
        parse("INSERT INTO lineitem (l_orderkey) VALUES (1) ON CONFLICT DO NOTHING")


def test_insert_parse_select_yields_source():
    # INSERT ... SELECT now builds an Insert carrying a subplan (source) rather
    # than literal rows; the two forms are mutually exclusive.
    plan = parse(
        "INSERT INTO lineitem SELECT * FROM orders",
        {"orders": ["o_orderkey", "o_custkey", "o_totalprice", "o_orderdate"]},
    )
    assert isinstance(plan, Insert)
    assert plan.table == "lineitem"
    assert plan.columns is None
    assert plan.rows == []
    assert plan.source is not None


def test_select_parse_still_routes_to_select():
    # Relaxing the INSERT gate must not perturb SELECT routing.
    plan = parse("SELECT l_orderkey FROM lineitem")
    assert not isinstance(plan, Insert)


# ----------------------------------------------------- round-trip (autocommit)

def test_insert_returns_rowcount_and_visible(engine):
    base_n = int(engine.sql("SELECT count(*) AS n FROM lineitem").to_pandas()["n"].iloc[0])
    rc = engine.sql(
        "INSERT INTO lineitem (l_orderkey, l_quantity, l_extendedprice, l_shipdate) "
        "VALUES (999, 5.0, 50.0, date '1998-08-10')"
    )
    assert isinstance(rc, int)
    assert rc == 1
    # Autocommit: the very next SELECT sees the appended row with no commit call.
    n = int(engine.sql("SELECT count(*) AS n FROM lineitem").to_pandas()["n"].iloc[0])
    assert n == base_n + 1
    row = engine.sql(
        "SELECT l_orderkey, l_quantity, l_extendedprice FROM lineitem WHERE l_orderkey = 999"
    ).to_pandas()
    assert len(row) == 1
    assert int(row["l_orderkey"].iloc[0]) == 999
    assert float(row["l_quantity"].iloc[0]) == 5.0
    assert float(row["l_extendedprice"].iloc[0]) == 50.0


def test_insert_partial_columns_fill_null(engine):
    # No DEFAULT, no NOT NULL on lineitem cols => omitted cols are NULL.
    engine.sql("INSERT INTO lineitem (l_orderkey) VALUES (777)")
    row = engine.sql(
        "SELECT l_quantity, l_extendedprice FROM lineitem WHERE l_orderkey = 777"
    ).to_pandas()
    assert len(row) == 1
    # cuDF null shows as NaN/NaT in pandas.
    assert row["l_quantity"].iloc[0] != row["l_quantity"].iloc[0]  # NaN
    assert row["l_extendedprice"].iloc[0] != row["l_extendedprice"].iloc[0]


def test_insert_multi_row(engine):
    base_n = int(engine.sql("SELECT count(*) AS n FROM lineitem").to_pandas()["n"].iloc[0])
    rc = engine.sql(
        "INSERT INTO lineitem (l_orderkey, l_quantity, l_extendedprice, l_shipdate) "
        "VALUES (1001, 1.0, 10.0, date '1998-01-01'), "
        "(1002, 2.0, 20.0, date '1998-01-02'), "
        "(1003, 3.0, 30.0, date '1998-01-03')"
    )
    assert rc == 3
    n = int(engine.sql("SELECT count(*) AS n FROM lineitem").to_pandas()["n"].iloc[0])
    assert n == base_n + 3
    keys = engine.sql(
        "SELECT l_orderkey FROM lineitem "
        "WHERE l_orderkey >= 1001 AND l_orderkey <= 1003"
    ).to_pandas()["l_orderkey"].tolist()
    assert set(keys) == {1001, 1002, 1003}


# ------------------------------------------------------- DuckDB oracle round-trip

def test_insert_roundtrip_duckdb_oracle(engine, duck):
    # read_parquet views are read-only: materialize a writable temp table.
    duck.execute("CREATE TABLE lineitem_w AS SELECT * FROM lineitem")
    ins = (
        "INSERT INTO lineitem (l_orderkey, l_quantity, l_extendedprice, l_shipdate) "
        "VALUES (4242, 7.5, 88.25, date '1997-03-14')"
    )
    engine.sql(ins)
    duck.execute(ins.replace("lineitem (l_orderkey, l_quantity, l_extendedprice, l_shipdate)",
                             "lineitem_w (l_orderkey, l_quantity, l_extendedprice, l_shipdate)"))
    q = ("SELECT l_orderkey, l_quantity, l_extendedprice FROM lineitem "
         "WHERE l_orderkey >= 4242 ORDER BY l_orderkey")
    ryu = engine.sql(q)
    dft = duck.execute(q.replace("lineitem", "lineitem_w")).fetchdf()
    assert_same(ryu, dft)


def test_insert_typed_decimal_date_roundtrip(typed_engine, typed_duck):
    """DECIMAL(15,2) -> float64 and DATE -> datetime delta casts, with a Q6-shaped
    agg that also exercises the cold-reader delta guard (deferred to _scan+merge)."""
    typed_duck.execute("CREATE TABLE lineitem_w AS SELECT * FROM lineitem")
    ins = (
        "INSERT INTO lineitem (l_orderkey, l_quantity, l_extendedprice, l_discount, "
        "l_tax, l_shipdate) VALUES (500000, 10.0, 1234.56, 0.06, 0.02, date '1994-06-15')"
    )
    rc = typed_engine.sql(ins)
    assert rc == 1
    typed_duck.execute(ins.replace(
        "lineitem (l_orderkey, l_quantity, l_extendedprice, l_discount, l_tax, l_shipdate)",
        "lineitem_w (l_orderkey, l_quantity, l_extendedprice, l_discount, l_tax, l_shipdate)"))
    q = ("SELECT count(*) AS n, sum(l_extendedprice) AS s FROM lineitem "
         "WHERE l_shipdate >= date '1994-01-01' AND l_shipdate < date '1995-01-01' "
         "AND l_discount >= 0.05 AND l_discount <= 0.07")
    typed_engine.clear_scan_cache()  # force the cold path so the delta guard fires
    ryu = typed_engine.sql(q)
    dft = typed_duck.execute(q.replace("lineitem", "lineitem_w")).fetchdf()
    assert_same(ryu, dft)


# --------------------------------------------------------- NOT NULL / DEFAULT

def _fresh_engine(data_dir, tmp_path):
    """A fresh catalog (own dir) over the shared lineitem parquet, so constraint
    saves never leak into the session data_dir catalog file."""
    cat = Catalog(str(tmp_path))
    cat.register("lineitem", str(data_dir / "lineitem"))
    return Engine(cat)


def test_insert_not_null_violation(data_dir, tmp_path):
    eng = _fresh_engine(data_dir, tmp_path)
    eng.catalog.set_not_null("lineitem", "l_orderkey", on=True)
    with pytest.raises(RuntimeError, match="NOT NULL"):
        eng.sql("INSERT INTO lineitem (l_quantity) VALUES (5.0)")


def test_insert_explicit_null_on_not_null(data_dir, tmp_path):
    eng = _fresh_engine(data_dir, tmp_path)
    eng.catalog.set_not_null("lineitem", "l_orderkey", on=True)
    with pytest.raises(RuntimeError, match="NOT NULL"):
        eng.sql("INSERT INTO lineitem (l_orderkey, l_quantity) VALUES (NULL, 5.0)")


def test_insert_default_fill(data_dir, tmp_path):
    eng = _fresh_engine(data_dir, tmp_path)
    eng.catalog.set_default("lineitem", "l_quantity", 42.0)
    eng.sql("INSERT INTO lineitem (l_orderkey, l_extendedprice, l_shipdate) "
            "VALUES (31337, 9.0, date '1998-12-31')")
    row = eng.sql(
        "SELECT l_quantity FROM lineitem WHERE l_orderkey = 31337"
    ).to_pandas()
    assert len(row) == 1
    assert float(row["l_quantity"].iloc[0]) == 42.0


# ----------------------------------------------------------------- CLI smoke

def test_cli_insert_output(engine, capsys):
    from ryudb import cli
    rc = cli._run_statement(
        engine,
        "INSERT INTO lineitem (l_orderkey, l_quantity, l_extendedprice, l_shipdate) "
        "VALUES (555, 1.0, 2.0, date '1998-05-05')",
        quiet=False,
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "inserted 1 rows" in out
    # A following SELECT through the CLI prints the row (not "inserted ...").
    cli._run_statement(engine, "SELECT l_orderkey FROM lineitem WHERE l_orderkey = 555", quiet=False)
    out2 = capsys.readouterr().out
    assert "555" in out2
    assert "inserted" not in out2