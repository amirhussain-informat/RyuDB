"""Statistical aggregates: ``stddev(x)`` / ``stddev_samp(x)`` / ``variance(x)`` /
``var_samp(x)`` / ``median(x)``.

These are the **sample** forms (ddof=1 -- the SQL standard default; ``var_samp``
parses to the same ``exp.Variance`` as ``variance``). cuDF's ``.std()`` /
``.var()`` default to ddof=1 and ``.median()`` is the 0.5 quantile, so a single
``_AGG_METHOD`` entry per func serves both the global reduction
(``_scalar_agg`` -> ``getattr(series, method)()``) and the grouped single-pass
(``_fused_agg`` -> ``groupby.agg({col: [method]})``, which supports
``"std"`` / ``"var"`` / ``"median"`` as func strings). The population forms
``stddev_pop`` / ``var_pop`` (ddof=0) need a per-agg ddof path the single-pass
dict-string form can't express, so they are deferred.

NULL semantics match DuckDB: cuDF reductions skip nulls, and a sample
stddev/variance over fewer than 2 non-null values is NULL (ddof=1 divides by
n-1); median over any non-null set is the middle value. The fused C++/CUDA
kernel only handles COUNT/SUM/AVG/MIN/MAX, so a statistical aggregate always
defers to the cuDF path (no fused change). DISTINCT dedupes the arg before the
reduction; FILTER nulls failing rows' arg first; HAVING matches the agg by its
sqlglot ``.sql()``. DuckDB is the oracle.
"""

from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from ryudb import Catalog, Engine

from .conftest import assert_same


@pytest.fixture
def sengine(tmp_path) -> tuple[Engine, duckdb.DuckDBPyConnection]:
    """``t(g STR, v FLOAT, k INT)`` with groups of varying size so sample
    stddev/variance differ from population and single-value groups go NULL."""
    d = tmp_path
    (d / "t").mkdir()
    pd.DataFrame(
        {
            "g": ["a", "a", "a", "a", "b", "b", "b", "c"],
            "v": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 100.0],
            "k": [1, 1, 2, 2, 3, 3, 4, 5],
        }
    ).to_parquet(str(d / "t" / "0.parquet"))
    cat = Catalog(str(d))
    cat.register("t", str(d / "t"))
    eng = Engine(cat)
    duck = duckdb.connect()
    duck.execute(f"CREATE VIEW t AS SELECT * FROM read_parquet('{d}/t/*.parquet')")
    return eng, duck


@pytest.fixture
def nengine(tmp_path) -> tuple[Engine, duckdb.DuckDBPyConnection]:
    """``tn(g STR, v FLOAT)`` with NULL ``v`` values to pin NULL / <2-values
    semantics (a NULL is skipped; sample stddev/variance over <2 non-nulls is
    NULL; median over the non-null set)."""
    d = tmp_path
    (d / "tn").mkdir()
    pd.DataFrame({"g": ["a", "a", "b"], "v": [1.0, None, 10.0]}).to_parquet(
        str(d / "tn" / "0.parquet")
    )
    cat = Catalog(str(d))
    cat.register("tn", str(d / "tn"))
    eng = Engine(cat)
    duck = duckdb.connect()
    duck.execute(f"CREATE VIEW tn AS SELECT * FROM read_parquet('{d}/tn/*.parquet')")
    return eng, duck


def _ryu(eng: Engine, sql: str):
    return eng.sql(sql)


def _duck(duck, sql: str):
    return duck.execute(sql).fetchdf()


# --------------------------------------------------------------------------- #
# Global statistical aggregates
# --------------------------------------------------------------------------- #


def test_global_stddev_variance_median(sengine):
    eng, duck = sengine
    sql = "SELECT stddev(v), variance(v), median(v) FROM t"
    assert_same(_ryu(eng, sql), _duck(duck, sql))


def test_global_stddev_samp_var_samp(sengine):
    eng, duck = sengine
    sql = "SELECT stddev_samp(v), var_samp(v) FROM t"
    assert_same(_ryu(eng, sql), _duck(duck, sql))


def test_global_stddev_equals_stddev_samp(sengine):
    """``stddev`` and ``stddev_samp`` are the same sample standard deviation."""
    eng, duck = sengine
    assert_same(_ryu(eng, "SELECT stddev(v) FROM t"), _duck(duck, "SELECT stddev_samp(v) FROM t"))
    assert_same(_ryu(eng, "SELECT variance(v) FROM t"), _duck(duck, "SELECT var_samp(v) FROM t"))


def test_global_mixed_with_basic_agg(sengine):
    """Statistical aggregates alongside COUNT/SUM/AVG in one global SELECT."""
    eng, duck = sengine
    sql = "SELECT count(*), sum(v), avg(v), stddev(v), median(v) FROM t"
    assert_same(_ryu(eng, sql), _duck(duck, sql))


# --------------------------------------------------------------------------- #
# Grouped statistical aggregates
# --------------------------------------------------------------------------- #


def test_grouped_stddev_variance_median(sengine):
    eng, duck = sengine
    sql = "SELECT g, stddev(v), variance(v), median(v) FROM t GROUP BY g ORDER BY g"
    assert_same(_ryu(eng, sql), _duck(duck, sql))


def test_grouped_single_value_is_null(sengine):
    """Group ``c`` has a single row -> sample stddev/variance NULL (ddof=1),
    median = the value itself."""
    eng, duck = sengine
    sql = "SELECT g, stddev(v), variance(v), median(v) FROM t GROUP BY g ORDER BY g"
    r = _ryu(eng, sql)
    assert_same(r, _duck(duck, sql))


def test_grouped_mixed_with_basic(sengine):
    """Statistical + basic aggregates, grouped, over the same and different
    columns (exercises the single-pass ``groupby.agg`` multi-func spec)."""
    eng, duck = sengine
    sql = "SELECT g, count(*), sum(v), avg(v), stddev(v), median(v), max(k) FROM t GROUP BY g ORDER BY g"
    assert_same(_ryu(eng, sql), _duck(duck, sql))


def test_grouped_stddev_samp(sengine):
    eng, duck = sengine
    sql = "SELECT g, stddev_samp(v), var_samp(v) FROM t GROUP BY g ORDER BY g"
    assert_same(_ryu(eng, sql), _duck(duck, sql))


# --------------------------------------------------------------------------- #
# DISTINCT-qualified statistical aggregates
# --------------------------------------------------------------------------- #


def test_global_distinct_stddev(sengine):
    eng, duck = sengine
    sql = "SELECT stddev(DISTINCT k) FROM t"
    assert_same(_ryu(eng, sql), _duck(duck, sql))


def test_grouped_distinct_median(sengine):
    eng, duck = sengine
    sql = "SELECT g, median(DISTINCT k) FROM t GROUP BY g ORDER BY g"
    assert_same(_ryu(eng, sql), _duck(duck, sql))


def test_grouped_distinct_variance(sengine):
    eng, duck = sengine
    sql = "SELECT g, variance(DISTINCT k) FROM t GROUP BY g ORDER BY g"
    assert_same(_ryu(eng, sql), _duck(duck, sql))


# --------------------------------------------------------------------------- #
# Composition: WHERE / HAVING / FILTER
# --------------------------------------------------------------------------- #


def test_stddev_with_where(sengine):
    eng, duck = sengine
    sql = "SELECT stddev(v), median(v) FROM t WHERE v > 5"
    assert_same(_ryu(eng, sql), _duck(duck, sql))


def test_grouped_stddev_with_where(sengine):
    eng, duck = sengine
    sql = "SELECT g, stddev(v), variance(v) FROM t WHERE v > 2 GROUP BY g ORDER BY g"
    assert_same(_ryu(eng, sql), _duck(duck, sql))


def test_stddev_in_having(sengine):
    """HAVING references the statistical aggregate by its full shape (matched to
    the SELECT alias via sqlglot .sql())."""
    eng, duck = sengine
    sql = "SELECT g, stddev(v) AS s FROM t GROUP BY g HAVING stddev(v) > 5 ORDER BY g"
    assert_same(_ryu(eng, sql), _duck(duck, sql))


def test_variance_in_having_not_in_select(sengine):
    """A statistical aggregate in HAVING that is NOT in the SELECT list is added
    as a synthetic aggregate and pruned from the output."""
    eng, duck = sengine
    sql = "SELECT g, count(*) AS c FROM t GROUP BY g HAVING variance(v) > 5 ORDER BY g"
    assert_same(_ryu(eng, sql), _duck(duck, sql))


def test_stddev_with_filter(sengine):
    """``stddev(v) FILTER (WHERE p)``: reduce the arg of passing rows only."""
    eng, duck = sengine
    sql = "SELECT stddev(v) FILTER (WHERE v > 5) FROM t"
    assert_same(_ryu(eng, sql), _duck(duck, sql))


def test_grouped_variance_with_filter(sengine):
    eng, duck = sengine
    sql = "SELECT g, variance(v) FILTER (WHERE v > 5) FROM t GROUP BY g ORDER BY g"
    assert_same(_ryu(eng, sql), _duck(duck, sql))


# --------------------------------------------------------------------------- #
# NULL / <2-values semantics
# --------------------------------------------------------------------------- #


def test_null_skipped_grouped(nengine):
    """A NULL arg is skipped; group ``a`` has one non-null (1.0) + one NULL ->
    sample stddev/variance NULL, median 1.0."""
    eng, duck = nengine
    sql = "SELECT g, stddev(v), variance(v), median(v) FROM tn GROUP BY g ORDER BY g"
    assert_same(_ryu(eng, sql), _duck(duck, sql))


def test_single_value_global_is_null(nengine):
    """A single non-null value globally -> sample stddev/variance NULL."""
    eng, duck = nengine
    sql = "SELECT stddev(v), variance(v) FROM tn WHERE g = 'b'"
    assert_same(_ryu(eng, sql), _duck(duck, sql))


def test_median_with_null(nengine):
    """median over a set with a NULL ignores the NULL (median of {1.0, NULL} = 1.0)."""
    eng, duck = nengine
    sql = "SELECT median(v) FROM tn WHERE g = 'a'"
    assert_same(_ryu(eng, sql), _duck(duck, sql))


# --------------------------------------------------------------------------- #
# Rejections (deferred forms)
# --------------------------------------------------------------------------- #


def test_stddev_pop_rejected(sengine):
    eng, _duck = sengine
    with pytest.raises(NotImplementedError):
        eng.sql("SELECT stddev_pop(v) FROM t")


def test_var_pop_rejected(sengine):
    eng, _duck = sengine
    with pytest.raises(NotImplementedError):
        eng.sql("SELECT var_pop(v) FROM t")


def test_grouped_stddev_pop_rejected(sengine):
    eng, _duck = sengine
    with pytest.raises(NotImplementedError):
        eng.sql("SELECT g, stddev_pop(v) FROM t GROUP BY g")


def test_window_stddev_rejected(sengine):
    """A statistical aggregate as a window function (running stddev) is not
    supported yet -- the window builder raises on the unsupported func."""
    eng, _duck = sengine
    with pytest.raises(NotImplementedError):
        eng.sql("SELECT g, stddev(v) OVER (PARTITION BY g) FROM t")


# --------------------------------------------------------------------------- #
# Regression: basic aggregates are unchanged
# --------------------------------------------------------------------------- #


def test_basic_aggs_unchanged(sengine):
    eng, duck = sengine
    assert_same(_ryu(eng, "SELECT count(*), sum(v), avg(v), min(v), max(v) FROM t"),
                _duck(duck, "SELECT count(*), sum(v), avg(v), min(v), max(v) FROM t"))
    assert_same(_ryu(eng, "SELECT g, count(*), sum(v) FROM t GROUP BY g ORDER BY g"),
                _duck(duck, "SELECT g, count(*), sum(v) FROM t GROUP BY g ORDER BY g"))