"""INSERT ... SELECT: persist a query result into a table's delta.

Extends the step-3 INSERT path from literal VALUES to a frame sourced from a
SELECT subplan. The subplan's output columns map **positionally** onto the
target column list (standard SQL -- SELECT output names are ignored); omitted
target columns take DEFAULT then NULL. The batch is coerced via the same
``_arrow_match_dtype`` path VALUES uses, so the durable tail (delta + WAL +
PK/UNIQUE + MVCC) is shared. DuckDB is the correctness oracle.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ryudb import Catalog, Engine
from ryudb.sql.parse import ParseError, parse
from ryudb.sql.plan import Insert

from .conftest import assert_same


def _fresh_engine(data_dir, tmp_path) -> Engine:
    """A fresh catalog (own dir/WAL) over the shared lineitem parquet, so
    constraint saves and the WAL never leak across tests."""
    cat = Catalog(str(tmp_path))
    cat.register("lineitem", str(data_dir / "lineitem"))
    cat.register("orders", str(data_dir / "orders"))
    return Engine(cat)


# --------------------------------------------------------------------- parse


def test_parse_insert_select_yields_source():
    plan = parse(
        "INSERT INTO lineitem SELECT l_orderkey, l_quantity FROM lineitem",
        {"lineitem": ["l_orderkey", "l_quantity", "l_extendedprice", "l_shipdate"]},
    )
    assert isinstance(plan, Insert)
    assert plan.table == "lineitem"
    assert plan.columns is None
    assert plan.rows == []
    assert plan.source is not None


def test_parse_insert_select_with_columns():
    plan = parse(
        "INSERT INTO lineitem (l_orderkey, l_quantity) SELECT a, b FROM orders",
        {"orders": ["a", "b"]},
    )
    assert isinstance(plan, Insert)
    assert plan.columns == ["l_orderkey", "l_quantity"]
    assert plan.source is not None


def test_parse_insert_values_still_rows():
    plan = parse("INSERT INTO lineitem (l_orderkey) VALUES (1)")
    assert isinstance(plan, Insert)
    assert plan.source is None
    assert len(plan.rows) == 1


# ------------------------------------------------------------- basic round-trip


def test_insert_select_basic(engine, duck):
    duck.execute("CREATE TABLE lineitem_w AS SELECT * FROM lineitem")
    sql = (
        "INSERT INTO lineitem (l_orderkey, l_quantity, l_extendedprice, l_shipdate) "
        "SELECT l_orderkey, l_quantity, l_extendedprice, l_shipdate FROM lineitem "
        "WHERE l_quantity > 4.0"
    )
    rc = engine.sql(sql)
    duck.execute(_duck_swap(sql))
    assert isinstance(rc, int) and rc > 0
    q = ("SELECT l_orderkey, l_quantity, l_extendedprice FROM lineitem "
         "ORDER BY l_orderkey, l_quantity")
    assert_same(engine.sql(q), duck.execute(_duck_swap(q)).fetchdf())


def test_insert_select_star_self_append(engine, duck):
    """INSERT INTO t SELECT * FROM t doubles a non-PK table (SELECT * yields
    catalog order, so the positional map is identity)."""
    duck.execute("CREATE TABLE lineitem_w AS SELECT * FROM lineitem")
    base = int(engine.sql("SELECT count(*) AS n FROM lineitem").to_pandas()["n"].iloc[0])
    sql = ("INSERT INTO lineitem (l_orderkey, l_quantity, l_extendedprice, l_shipdate) "
           "SELECT * FROM lineitem")
    rc = engine.sql(sql)
    duck.execute("INSERT INTO lineitem_w (l_orderkey, l_quantity, l_extendedprice, l_shipdate) "
                 "SELECT * FROM lineitem_w")
    assert rc == base
    assert int(engine.sql("SELECT count(*) AS n FROM lineitem").to_pandas()["n"].iloc[0]) == base * 2
    q = "SELECT l_orderkey, l_quantity FROM lineitem ORDER BY l_orderkey, l_quantity"
    assert_same(engine.sql(q), duck.execute(_duck_swap(q)).fetchdf())


def test_insert_select_positional_reorder(engine, duck):
    """Target list order differs from source order -> positional mapping swaps
    the values (SELECT output NAMES are ignored)."""
    duck.execute("CREATE TABLE lineitem_w AS SELECT * FROM lineitem")
    sql = ("INSERT INTO lineitem (l_orderkey, l_quantity) "
           "SELECT l_quantity, l_orderkey FROM lineitem WHERE l_orderkey = 1")
    engine.sql(sql)
    duck.execute(_duck_swap(sql))
    q = ("SELECT l_orderkey, l_quantity FROM lineitem WHERE l_quantity = 1 "
         "ORDER BY l_orderkey")
    assert_same(engine.sql(q), duck.execute(_duck_swap(q)).fetchdf())


# ----------------------------------------------------- richer SELECT sources


def test_insert_select_grouped_source(engine, duck):
    duck.execute("CREATE TABLE lineitem_w AS SELECT * FROM lineitem")
    sql = ("INSERT INTO lineitem (l_orderkey, l_quantity) "
           "SELECT l_orderkey, sum(l_quantity) AS qty FROM lineitem "
           "GROUP BY l_orderkey ORDER BY l_orderkey")
    engine.sql(sql)
    duck.execute(_duck_swap(sql))
    q = ("SELECT l_orderkey, l_quantity FROM lineitem "
         "WHERE l_extendedprice IS NULL ORDER BY l_orderkey")
    assert_same(engine.sql(q), duck.execute(_duck_swap(q)).fetchdf())


def test_insert_select_cte_source(engine, duck):
    duck.execute("CREATE TABLE lineitem_w AS SELECT * FROM lineitem")
    sql = ("INSERT INTO lineitem (l_orderkey, l_quantity, l_extendedprice, l_shipdate) "
           "WITH big AS (SELECT * FROM lineitem WHERE l_quantity > 5.0) "
           "SELECT * FROM big")
    engine.sql(sql)
    duck.execute(_duck_swap(sql))
    q = ("SELECT l_orderkey, l_quantity FROM lineitem WHERE l_quantity > 5.0 "
         "ORDER BY l_orderkey, l_quantity")
    assert_same(engine.sql(q), duck.execute(_duck_swap(q)).fetchdf())


def test_insert_select_union_source(engine, duck):
    duck.execute("CREATE TABLE lineitem_w AS SELECT * FROM lineitem")
    sql = ("INSERT INTO lineitem (l_orderkey, l_quantity) "
           "SELECT l_orderkey, l_quantity FROM lineitem WHERE l_orderkey = 1 "
           "UNION ALL SELECT l_orderkey, l_quantity FROM lineitem WHERE l_orderkey = 4")
    engine.sql(sql)
    duck.execute(_duck_swap(sql))
    q = ("SELECT l_orderkey, l_quantity FROM lineitem "
           "WHERE l_extendedprice IS NULL ORDER BY l_orderkey, l_quantity")
    assert_same(engine.sql(q), duck.execute(_duck_swap(q)).fetchdf())


def test_insert_select_derived_table_source(engine, duck):
    """Source is a derived table whose column NAMES differ from the target's --
    positional mapping ignores names and routes by position."""
    duck.execute("CREATE TABLE lineitem_w AS SELECT * FROM lineitem")
    sql = ("INSERT INTO lineitem (l_orderkey, l_quantity) "
           "SELECT k, q FROM (SELECT l_orderkey AS k, l_quantity AS q "
           "FROM lineitem WHERE l_orderkey <= 2) d")
    engine.sql(sql)
    duck.execute(_duck_swap(sql))
    q = ("SELECT l_orderkey, l_quantity FROM lineitem "
           "WHERE l_extendedprice IS NULL ORDER BY l_orderkey, l_quantity")
    assert_same(engine.sql(q), duck.execute(_duck_swap(q)).fetchdf())


def test_insert_select_join_source(engine, duck):
    duck.execute("CREATE TABLE lineitem_w AS SELECT * FROM lineitem")
    duck.execute("CREATE TABLE orders_w AS SELECT * FROM orders")
    sql = ("INSERT INTO lineitem (l_orderkey, l_quantity) "
           "SELECT l.l_orderkey, l.l_quantity FROM lineitem l "
           "JOIN orders o ON l.l_orderkey = o.o_orderkey "
           "WHERE o.o_custkey = 10")
    engine.sql(sql)
    duck.execute(_duck_swap_join(sql))
    q = ("SELECT l_orderkey, l_quantity FROM lineitem "
           "WHERE l_extendedprice IS NULL ORDER BY l_orderkey, l_quantity")
    assert_same(engine.sql(q), duck.execute(_duck_swap(q)).fetchdf())


# ------------------------------------------------- typed DECIMAL/DATE coercion


def test_insert_select_typed_decimal_date(typed_engine, typed_duck):
    typed_duck.execute("CREATE TABLE lineitem_w AS SELECT * FROM lineitem")
    sql = ("INSERT INTO lineitem (l_orderkey, l_quantity, l_extendedprice, l_shipdate) "
           "SELECT l_orderkey, l_quantity, l_extendedprice, l_shipdate FROM lineitem "
           "WHERE l_quantity > 40")
    rc = typed_engine.sql(sql)
    typed_duck.execute(
        sql.replace("INSERT INTO lineitem ", "INSERT INTO lineitem_w ").replace(
            "FROM lineitem WHERE", "FROM lineitem_w WHERE"
        )
    )
    assert isinstance(rc, int) and rc > 0
    q = ("SELECT count(*) AS n, sum(l_extendedprice) AS s, max(l_shipdate) AS mx "
         "FROM lineitem")
    typed_engine.clear_scan_cache()  # force the cold path so the delta guard fires
    assert_same(typed_engine.sql(q), typed_duck.execute(q.replace("lineitem", "lineitem_w")).fetchdf())


# --------------------------------------------------------- NOT NULL / PK/UNIQUE


def test_insert_select_not_null_violation(data_dir, tmp_path):
    eng = _fresh_engine(data_dir, tmp_path)
    eng.catalog.set_not_null("lineitem", "l_orderkey", on=True)
    # Omitted target column -> NULL -> NOT NULL violation before any write.
    with pytest.raises(RuntimeError, match="NOT NULL"):
        eng.sql("INSERT INTO lineitem (l_quantity) "
                "SELECT l_quantity FROM lineitem WHERE l_orderkey = 1")


def test_insert_select_pk_collision_existing(data_dir, tmp_path):
    eng = _fresh_engine(data_dir, tmp_path)
    eng.catalog.set_primary_key("lineitem", ["l_orderkey"])
    with pytest.raises(RuntimeError, match="UNIQUE"):
        eng.sql("INSERT INTO lineitem (l_orderkey, l_quantity, l_extendedprice, l_shipdate) "
                "SELECT l_orderkey, l_quantity, l_extendedprice, l_shipdate FROM lineitem "
                "WHERE l_orderkey <= 2")


def test_insert_select_pk_dup_within_batch(data_dir, tmp_path):
    eng = _fresh_engine(data_dir, tmp_path)
    eng.catalog.set_primary_key("lineitem", ["l_orderkey"])
    # New key 100 (not in base) duplicated within the source batch.
    with pytest.raises(RuntimeError, match="UNIQUE"):
        eng.sql("INSERT INTO lineitem (l_orderkey, l_quantity) "
                "SELECT 100, 1.0 FROM lineitem WHERE l_orderkey = 1 "
                "UNION ALL SELECT 100, 2.0 FROM lineitem WHERE l_orderkey = 2")


def test_insert_select_unique_null_exempt(data_dir, tmp_path):
    eng = _fresh_engine(data_dir, tmp_path)
    eng.catalog.set_unique("lineitem", ["l_orderkey"])
    # Two source rows with NULL l_orderkey -> NULL-exempt (no UNIQUE violation).
    rc = eng.sql("INSERT INTO lineitem (l_quantity, l_extendedprice, l_shipdate) "
                 "SELECT l_quantity, l_extendedprice, l_shipdate FROM lineitem "
                 "WHERE l_orderkey IN (1, 2)")
    assert rc == 3  # orderkeys 1,1,2 (l_orderkey omitted -> NULL, NULL-exempt)


def test_insert_select_mixed_provided_omitted_date(tmp_path):
    """Regression: a filtered source carries a non-contiguous index. Providing
    one DATE column (-> Series with that index) while OMITTING another DATE
    column (-> Series with a fresh RangeIndex) made pd.DataFrame(out) misalign
    ("array length ... does not match index length ..."). The source frame is
    reset_index(drop=True) so every column Series shares RangeIndex(0, n)."""
    import duckdb

    d = tmp_path / "t"
    d.mkdir()
    con = duckdb.connect()
    con.execute("CREATE TABLE t (id BIGINT, d1 DATE, d2 DATE, v DOUBLE)")
    con.execute(
        "INSERT INTO t VALUES "
        "(1, '2020-01-01', '2020-01-10', 1.0), "
        "(2, '2020-02-02', '2020-02-20', 2.0), "
        "(3, '2020-03-03', '2020-03-30', 3.0), "
        "(4, '2020-04-04', '2020-04-30', 4.0)"
    )
    con.execute(f"COPY (SELECT * FROM t) TO '{d}/0.parquet' (FORMAT PARQUET)")
    con.close()
    cat = Catalog(str(tmp_path))
    cat.register("t", str(d))
    eng = Engine(cat)
    # Provide id, d1, v (d1 is a DATE -> non-contiguous-index Series); omit d2
    # (DATE -> RangeIndex NULL Series). Before the reset_index fix these two
    # datetime Series had different indices and pd.DataFrame(out) raised.
    rc = eng.sql("INSERT INTO t (id, d1, v) SELECT id, d1, v FROM t WHERE id > 1")
    assert rc == 3
    df = eng.sql("SELECT id, d1, d2, v FROM t WHERE d2 IS NULL ORDER BY id").to_pandas()
    assert len(df) == 3
    assert df["id"].tolist() == [2, 3, 4]
    assert df["d1"].tolist() == [
        pd.Timestamp("2020-02-02"),
        pd.Timestamp("2020-03-03"),
        pd.Timestamp("2020-04-04"),
    ]
    assert df["v"].tolist() == [2.0, 3.0, 4.0]


# ------------------------------------------------------------ transactions


def test_insert_select_in_txn_commit(engine, duck):
    duck.execute("CREATE TABLE lineitem_w AS SELECT * FROM lineitem")
    base = int(engine.sql("SELECT count(*) AS n FROM lineitem").to_pandas()["n"].iloc[0])
    engine.sql("BEGIN")
    engine.sql("INSERT INTO lineitem (l_orderkey, l_quantity) "
               "SELECT l_orderkey, l_quantity FROM lineitem WHERE l_orderkey <= 2")
    mid = int(engine.sql("SELECT count(*) AS n FROM lineitem").to_pandas()["n"].iloc[0])
    assert mid == base + 3  # orderkeys 1,1,2 -> read-your-writes inside the txn
    engine.sql("COMMIT")
    fin = int(engine.sql("SELECT count(*) AS n FROM lineitem").to_pandas()["n"].iloc[0])
    assert fin == base + 3
    duck.execute("INSERT INTO lineitem_w (l_orderkey, l_quantity) "
                 "SELECT l_orderkey, l_quantity FROM lineitem_w WHERE l_orderkey <= 2")
    q = "SELECT l_orderkey, l_quantity FROM lineitem ORDER BY l_orderkey, l_quantity"
    assert_same(engine.sql(q), duck.execute(_duck_swap(q)).fetchdf())


def test_insert_select_in_txn_rollback(engine):
    base = int(engine.sql("SELECT count(*) AS n FROM lineitem").to_pandas()["n"].iloc[0])
    engine.sql("BEGIN")
    engine.sql("INSERT INTO lineitem (l_orderkey, l_quantity) "
               "SELECT l_orderkey, l_quantity FROM lineitem WHERE l_orderkey <= 2")
    engine.sql("ROLLBACK")
    n = int(engine.sql("SELECT count(*) AS n FROM lineitem").to_pandas()["n"].iloc[0])
    assert n == base  # rollback discards the buffered insert


# ----------------------------------------------------------- edge cases


def test_insert_select_zero_rows(engine):
    base = int(engine.sql("SELECT count(*) AS n FROM lineitem").to_pandas()["n"].iloc[0])
    rc = engine.sql("INSERT INTO lineitem (l_orderkey, l_quantity) "
                    "SELECT l_orderkey, l_quantity FROM lineitem WHERE l_quantity > 1000")
    assert rc == 0
    n = int(engine.sql("SELECT count(*) AS n FROM lineitem").to_pandas()["n"].iloc[0])
    assert n == base


def test_insert_select_count_mismatch(engine):
    with pytest.raises(ParseError):
        engine.sql("INSERT INTO lineitem (l_orderkey, l_quantity, l_extendedprice) "
                   "SELECT l_orderkey, l_quantity FROM lineitem WHERE l_orderkey = 1")


def test_insert_select_unknown_target_column(engine):
    with pytest.raises(ParseError):
        engine.sql("INSERT INTO lineitem (no_such_col, l_quantity) "
                   "SELECT l_orderkey, l_quantity FROM lineitem WHERE l_orderkey = 1")


def test_insert_select_durable(data_dir, tmp_path):
    """Autocommit INSERT...SELECT writes a WAL record; a fresh Engine over the
    same dir replays it and sees the rows."""
    eng = _fresh_engine(data_dir, tmp_path)
    eng.sql("INSERT INTO lineitem (l_orderkey, l_quantity) "
            "SELECT l_orderkey, l_quantity FROM lineitem WHERE l_orderkey <= 3")
    n1 = int(eng.sql("SELECT count(*) AS n FROM lineitem").to_pandas()["n"].iloc[0])
    cat2 = Catalog(str(tmp_path))
    cat2.register("lineitem", str(data_dir / "lineitem"))
    eng2 = Engine(cat2)
    n2 = int(eng2.sql("SELECT count(*) AS n FROM lineitem").to_pandas()["n"].iloc[0])
    assert n2 == n1


# ----------------------------------------------------------------- CLI smoke


def test_cli_insert_select_output(engine, capsys):
    from ryudb import cli
    rc = cli._run_statement(
        engine,
        "INSERT INTO lineitem (l_orderkey, l_quantity) "
        "SELECT l_orderkey, l_quantity FROM lineitem WHERE l_orderkey = 1",
        quiet=False,
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "inserted" in out and "rows" in out


# ----------------------------------------------------------------- helpers


def _duck_swap(sql: str) -> str:
    """Rewrite a lineitem-targeting / lineitem-reading statement for the
    DuckDB writable copy ``lineitem_w``. INSERT targets and FROM/JOIN/WHERE
    reads of ``lineitem`` become ``lineitem_w``; ``orders`` stays."""
    s = sql.replace("INSERT INTO lineitem ", "INSERT INTO lineitem_w ")
    s = s.replace("FROM lineitem ", "FROM lineitem_w ")
    s = s.replace("FROM lineitem\n", "FROM lineitem_w\n")
    s = s.replace("JOIN lineitem ", "JOIN lineitem_w ")
    # A bare trailing `FROM lineitem` (no following clause) and the `l` alias
    # form `FROM lineitem l` are covered by the space-suffixed replace above.
    return s


def _duck_swap_join(sql: str) -> str:
    """Variant for the join test: also map the orders read to ``orders_w``."""
    return _duck_swap(sql).replace("FROM orders ", "FROM orders_w ").replace(
        "JOIN orders ", "JOIN orders_w "
    )