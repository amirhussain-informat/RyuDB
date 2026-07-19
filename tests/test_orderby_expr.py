"""ORDER BY scalar expressions.

``ORDER BY`` historically accepted only column references. It now also accepts
arbitrary scalar expressions over source columns (``k + v``, ``-v``, ``v * 2``,
``abs(v)``), which the executor's ``Sort`` materializes via ``eval_expr`` against
the pre-projection frame. Ordering by an aggregate or window function is still
rejected -- the user orders by the function's output alias instead (already
supported as a column reference). DuckDB is the oracle.
"""

from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from ryudb import Catalog, Engine

from .conftest import assert_same


@pytest.fixture
def eengine(tmp_path) -> tuple[Engine, duckdb.DuckDBPyConnection]:
    """``t(k INT, v INT)`` with rows chosen so expression ordering is
    non-trivial (unsorted, with a duplicate ``k`` and a negative ``v``)."""
    d = tmp_path / "t"
    d.mkdir()
    pd.DataFrame(
        {"k": [3, 1, 2, 1, 3], "v": [10, 40, 20, 30, -5]}
    ).to_parquet(str(d / "0.parquet"))
    cat = Catalog(str(tmp_path))
    cat.register("t", str(d))
    eng = Engine(cat)
    duck = duckdb.connect()
    duck.execute(f"CREATE VIEW t AS SELECT * FROM read_parquet('{d}/0.parquet')")
    return eng, duck


def _ryu(eng: Engine, sql: str):
    return eng.sql(sql)


def _duck(duck, sql: str):
    return duck.execute(sql).fetchdf()


# --------------------------------------------------------------------------- #
# Scalar-expression ORDER BY (the new capability)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("desc", [False, True])
def test_order_by_sum(eengine, desc):
    eng, duck = eengine
    d = "DESC" if desc else ""
    sql = f"SELECT k, v FROM t ORDER BY k + v {d}".strip()
    assert_same(_ryu(eng, sql), _duck(duck, sql))


@pytest.mark.parametrize("desc", [False, True])
def test_order_by_negation(eengine, desc):
    eng, duck = eengine
    d = "DESC" if desc else ""
    sql = f"SELECT k, v FROM t ORDER BY -v {d}".strip()
    assert_same(_ryu(eng, sql), _duck(duck, sql))


@pytest.mark.parametrize("desc", [False, True])
def test_order_by_product(eengine, desc):
    eng, duck = eengine
    d = "DESC" if desc else ""
    sql = f"SELECT k, v FROM t ORDER BY v * 2 {d}".strip()
    assert_same(_ryu(eng, sql), _duck(duck, sql))


def test_order_by_function(eengine):
    eng, duck = eengine
    sql = "SELECT k, v FROM t ORDER BY abs(v - 20)"
    assert_same(_ryu(eng, sql), _duck(duck, sql))


def test_order_by_difference(eengine):
    eng, duck = eengine
    sql = "SELECT k, v FROM t ORDER BY k - v"
    assert_same(_ryu(eng, sql), _duck(duck, sql))


def test_order_by_expression_not_in_select(eengine):
    """ORDER BY may reference a source column that is not projected (DuckDB
    allows this; the Sort resolves the expression against the pre-projection
    frame)."""
    eng, duck = eengine
    sql = "SELECT k FROM t ORDER BY k + v"
    assert_same(_ryu(eng, sql), _duck(duck, sql))


def test_order_by_expression_with_alias_selected(eengine):
    """The same expression is also projected under an alias; the ORDER BY term
    is an independent scalar expression (not the alias), so it is recomputed
    against the source -- matching DuckDB."""
    eng, duck = eengine
    sql = "SELECT k, v, k + v AS s FROM t ORDER BY k + v"
    assert_same(_ryu(eng, sql), _duck(duck, sql))


def test_order_by_mixed_expression_and_column(eengine):
    eng, duck = eengine
    sql = "SELECT k, v FROM t ORDER BY k + v, k"
    assert_same(_ryu(eng, sql), _duck(duck, sql))


# --------------------------------------------------------------------------- #
# NULL handling in expression keys
# --------------------------------------------------------------------------- #


def test_order_by_expression_with_nulls(tmp_path):
    d = tmp_path / "tn"
    d.mkdir()
    pd.DataFrame({"k": [1, 2, 3, 4], "v": [10, None, 30, None]}).to_parquet(
        str(d / "0.parquet")
    )
    cat = Catalog(str(tmp_path))
    cat.register("tn", str(d))
    eng = Engine(cat)
    duck = duckdb.connect()
    duck.execute(f"CREATE VIEW tn AS SELECT * FROM read_parquet('{d}/0.parquet')")
    sql = "SELECT k, v FROM tn ORDER BY k + v"
    assert_same(eng.sql(sql), duck.execute(sql).fetchdf())


# --------------------------------------------------------------------------- #
# Aggregates / windows in ORDER BY are rejected (use the output alias)
# --------------------------------------------------------------------------- #


def test_order_by_aggregate_rejected(eengine):
    eng, _duck = eengine
    with pytest.raises(NotImplementedError):
        eng.sql("SELECT k, count(*) AS c FROM t GROUP BY k ORDER BY count(*)")


def test_order_by_aggregate_alias_works(eengine):
    """The supported form: order by the aggregate's output alias."""
    eng, duck = eengine
    sql = "SELECT k, count(*) AS c FROM t GROUP BY k ORDER BY c DESC, k"
    assert_same(_ryu(eng, sql), _duck(duck, sql))


def test_order_by_window_rejected(eengine):
    eng, _duck = eengine
    with pytest.raises(NotImplementedError):
        eng.sql("SELECT k, v FROM t ORDER BY row_number() OVER (ORDER BY v)")


# --------------------------------------------------------------------------- #
# Regression: plain column ORDER BY is unchanged
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("desc", [False, True])
def test_order_by_column_unchanged(eengine, desc):
    eng, duck = eengine
    d = "DESC" if desc else ""
    sql = f"SELECT k, v FROM t ORDER BY v {d}".strip()
    assert_same(_ryu(eng, sql), _duck(duck, sql))


def test_order_by_two_columns_unchanged(eengine):
    eng, duck = eengine
    sql = "SELECT k, v FROM t ORDER BY k, v DESC"
    assert_same(_ryu(eng, sql), _duck(duck, sql))