"""DISTINCT-qualified aggregates: ``count(DISTINCT x)``, ``sum(DISTINCT x)``,
``avg(DISTINCT x)``, ``min(DISTINCT x)``, ``max(DISTINCT x)``.

A DISTINCT-qualified aggregate dedupes its argument within each group before
reducing (so ``count(DISTINCT k)`` counts the distinct non-null ``k`` values,
``sum(DISTINCT v)`` sums the distinct non-null ``v`` values, etc.). NULLs do
not count: ``drop_duplicates`` keeps one NaN and the reduction skips nulls, so
a NULL arg is not a distinct value -- matching DuckDB. DISTINCT composes with
GROUP BY, WHERE, HAVING, and the per-aggregate ``FILTER (WHERE ...)`` clause.

The fused C++/CUDA aggregate kernels read only ``af.func``/``af.arg`` (not
``af.distinct``), so a DISTINCT aggregate forces the cuDF fallback path; TPC-H
uses no DISTINCT aggregates, so this is purely additive. DuckDB is the oracle.
"""

from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from ryudb import Catalog, Engine

from .conftest import assert_same


@pytest.fixture
def dengine(tmp_path) -> tuple[Engine, duckdb.DuckDBPyConnection]:
    """``t(k INT, v INT, g STR)`` with duplicate ``k``/``v`` values within each
    group so DISTINCT vs non-DISTINCT aggregates differ."""
    d = tmp_path / "t"
    d.mkdir()
    pd.DataFrame(
        {
            "k": [1, 1, 2, 2, 3, 3],
            "v": [10, 10, 20, 20, 30, 30],
            "g": ["a", "a", "a", "b", "b", "b"],
        }
    ).to_parquet(str(d / "0.parquet"))
    cat = Catalog(str(tmp_path))
    cat.register("t", str(d))
    eng = Engine(cat)
    duck = duckdb.connect()
    duck.execute(f"CREATE VIEW t AS SELECT * FROM read_parquet('{d}/0.parquet')")
    return eng, duck


@pytest.fixture
def nengine(tmp_path) -> tuple[Engine, duckdb.DuckDBPyConnection]:
    """``tn(k INT, g STR)`` with NULL ``k`` values to pin NULL-distinct
    semantics (a NULL is not a distinct value; it does not count)."""
    d = tmp_path / "tn"
    d.mkdir()
    pd.DataFrame({"k": [1, 1, 2, None, 3], "g": ["a", "a", "a", "a", "b"]}).to_parquet(
        str(d / "0.parquet")
    )
    cat = Catalog(str(tmp_path))
    cat.register("tn", str(d))
    eng = Engine(cat)
    duck = duckdb.connect()
    duck.execute(f"CREATE VIEW tn AS SELECT * FROM read_parquet('{d}/0.parquet')")
    return eng, duck


def _ryu(eng: Engine, sql: str):
    return eng.sql(sql)


def _duck(duck, sql: str):
    return duck.execute(sql).fetchdf()


# --------------------------------------------------------------------------- #
# Global DISTINCT aggregates (no GROUP BY)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT count(DISTINCT k) FROM t",
        "SELECT sum(DISTINCT v) FROM t",
        "SELECT avg(DISTINCT v) FROM t",
        "SELECT min(DISTINCT v) FROM t",
        "SELECT max(DISTINCT v) FROM t",
    ],
)
def test_global_distinct_agg(dengine, sql):
    eng, duck = dengine
    assert_same(_ryu(eng, sql), _duck(duck, sql))


def test_global_distinct_differs_from_nondistinct(dengine):
    """Sanity: ``count(DISTINCT k)`` (3) differs from ``count(k)`` (6) -- the
    DISTINCT path actually dedupes, not just aliases the regular aggregate."""
    eng, duck = dengine
    assert int(_ryu(eng, "SELECT count(DISTINCT k) FROM t").iloc[0, 0]) == 3
    assert int(_ryu(eng, "SELECT count(k) FROM t").iloc[0, 0]) == 6
    assert int(_duck(duck, "SELECT count(DISTINCT k) FROM t").iloc[0, 0]) == 3


def test_mixed_distinct_and_nondistinct(dengine):
    """A DISTINCT and a non-DISTINCT aggregate in the same SELECT: each is
    computed over its own row set (distinct dedupes, non-distinct counts all)."""
    eng, duck = dengine
    sql = "SELECT count(DISTINCT k), sum(v), avg(DISTINCT v) FROM t"
    assert_same(_ryu(eng, sql), _duck(duck, sql))


# --------------------------------------------------------------------------- #
# Grouped DISTINCT aggregates
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT g, count(DISTINCT k) FROM t GROUP BY g",
        "SELECT g, sum(DISTINCT v) FROM t GROUP BY g",
        "SELECT g, avg(DISTINCT v) FROM t GROUP BY g",
        "SELECT g, min(DISTINCT v) FROM t GROUP BY g",
        "SELECT g, max(DISTINCT v) FROM t GROUP BY g",
    ],
)
def test_grouped_distinct_agg(dengine, sql):
    eng, duck = dengine
    assert_same(_ryu(eng, sql), _duck(duck, sql))


def test_grouped_mixed_distinct(dengine):
    """Two DISTINCT aggregates with different args + a non-distinct aggregate,
    all grouped: each DISTINCT agg dedupes its own (group, arg) pairs."""
    eng, duck = dengine
    sql = "SELECT g, count(DISTINCT k), sum(DISTINCT v), count(*) FROM t GROUP BY g"
    assert_same(_ryu(eng, sql), _duck(duck, sql))


def test_grouped_distinct_with_group_key(dengine):
    """The DISTINCT arg is also a group key (``SELECT g, count(DISTINCT g)``)."""
    eng, duck = dengine
    sql = "SELECT g, count(DISTINCT g) AS c FROM t GROUP BY g"
    assert_same(_ryu(eng, sql), _duck(duck, sql))


# --------------------------------------------------------------------------- #
# Composition: WHERE / HAVING / FILTER
# --------------------------------------------------------------------------- #


def test_distinct_with_where(dengine):
    eng, duck = dengine
    sql = "SELECT count(DISTINCT k) FROM t WHERE v > 10"
    assert_same(_ryu(eng, sql), _duck(duck, sql))


def test_distinct_grouped_with_where(dengine):
    eng, duck = dengine
    sql = "SELECT g, count(DISTINCT k) FROM t WHERE v >= 20 GROUP BY g"
    assert_same(_ryu(eng, sql), _duck(duck, sql))


def test_distinct_in_having(dengine):
    """HAVING references the DISTINCT aggregate by its full shape (matched to
    the SELECT alias via sqlglot .sql())."""
    eng, duck = dengine
    sql = (
        "SELECT g, count(DISTINCT k) AS c FROM t GROUP BY g "
        "HAVING count(DISTINCT k) >= 1 ORDER BY g"
    )
    assert_same(_ryu(eng, sql), _duck(duck, sql))


def test_distinct_in_having_not_in_select(dengine):
    """A DISTINCT aggregate in HAVING that is NOT in the SELECT list is added
    as a synthetic aggregate and pruned from the output."""
    eng, duck = dengine
    sql = (
        "SELECT g, count(*) AS c FROM t GROUP BY g "
        "HAVING count(DISTINCT k) >= 1 ORDER BY g"
    )
    assert_same(_ryu(eng, sql), _duck(duck, sql))


def test_distinct_with_filter(dengine):
    """``sum(DISTINCT v) FILTER (WHERE p)``: dedupe the arg of passing rows."""
    eng, duck = dengine
    sql = "SELECT sum(DISTINCT v) FILTER (WHERE k > 1) FROM t"
    assert_same(_ryu(eng, sql), _duck(duck, sql))


def test_distinct_grouped_with_filter(dengine):
    eng, duck = dengine
    sql = (
        "SELECT g, sum(DISTINCT v) FILTER (WHERE k > 1) FROM t GROUP BY g ORDER BY g"
    )
    assert_same(_ryu(eng, sql), _duck(duck, sql))


# --------------------------------------------------------------------------- #
# NULL semantics: a NULL arg is not a distinct value
# --------------------------------------------------------------------------- #


def test_distinct_null_global(nengine):
    eng, duck = nengine
    assert_same(_ryu(eng, "SELECT count(DISTINCT k) FROM tn"), _duck(duck, "SELECT count(DISTINCT k) FROM tn"))


def test_distinct_null_grouped(nengine):
    eng, duck = nengine
    sql = "SELECT g, count(DISTINCT k) FROM tn GROUP BY g"
    assert_same(_ryu(eng, sql), _duck(duck, sql))


def test_distinct_null_sum_avg(nengine):
    eng, duck = nengine
    assert_same(_ryu(eng, "SELECT sum(DISTINCT k) FROM tn"), _duck(duck, "SELECT sum(DISTINCT k) FROM tn"))
    assert_same(_ryu(eng, "SELECT avg(DISTINCT k) FROM tn"), _duck(duck, "SELECT avg(DISTINCT k) FROM tn"))


def test_distinct_all_null_group(nengine):
    """A group whose arg is entirely NULL: count(DISTINCT k) = 0 (no distinct
    non-null values), matching DuckDB."""
    eng, duck = nengine
    sql = "SELECT g, count(DISTINCT k) FROM tn WHERE g = 'a' AND k IS NULL GROUP BY g"
    # DuckDB returns an empty group (no non-null distinct); both should agree.
    assert_same(_ryu(eng, sql), _duck(duck, sql))


# --------------------------------------------------------------------------- #
# Rejections
# --------------------------------------------------------------------------- #


def test_multi_arg_distinct_rejected(dengine):
    eng, _duck = dengine
    with pytest.raises(NotImplementedError):
        eng.sql("SELECT count(DISTINCT k, v) FROM t")


def test_count_distinct_star_rejected(dengine):
    eng, _duck = dengine
    with pytest.raises(NotImplementedError):
        eng.sql("SELECT count(DISTINCT *) FROM t")


# --------------------------------------------------------------------------- #
# Regression: non-DISTINCT aggregates are unchanged
# --------------------------------------------------------------------------- #


def test_nondistinct_unchanged(dengine):
    eng, duck = dengine
    assert_same(_ryu(eng, "SELECT count(*) FROM t"), _duck(duck, "SELECT count(*) FROM t"))
    assert_same(_ryu(eng, "SELECT count(k) FROM t"), _duck(duck, "SELECT count(k) FROM t"))
    assert_same(_ryu(eng, "SELECT sum(v) FROM t"), _duck(duck, "SELECT sum(v) FROM t"))
    assert_same(
        _ryu(eng, "SELECT g, count(*), sum(v) FROM t GROUP BY g"),
        _duck(duck, "SELECT g, count(*), sum(v) FROM t GROUP BY g"),
    )