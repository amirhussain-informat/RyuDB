"""NULL group keys under a WHERE-filtered ``GROUP BY``.

DuckDB retains a NULL group key: ``SELECT a, SUM(x) FROM t WHERE x>2 GROUP BY a``
emits a row for the NULL group (``a IS NULL``), summing the passing rows whose
``a`` is NULL. cuDF's ``groupby(dropna=False)`` matches this. The C++/CUDA
``fused_aggregate`` warm kernel, however, factorises group keys (NA -> -1 code),
which *drops* the NULL group -- so it must defer (return ``None``) whenever any
group-key column has nulls, falling back to the cuDF path. The cold Parquet
reader (``fused_scan_aggregate``) already defers on nullable group keys via its
per-column null-statistics guard. This module pins both paths against DuckDB.

The regression this guards: a filtered GROUP BY over a nullable key silently
lost the NULL group, returning fewer rows than DuckDB. The fix is a null check
in ``fused._match`` (mirroring ``_arg_cols_nonnull``).
"""

from __future__ import annotations

import duckdb
import pytest

from ryudb import Catalog, Engine

from .conftest import assert_same


@pytest.fixture
def null_key_engine(tmp_path) -> tuple[Engine, duckdb.DuckDBPyConnection, str]:
    """``t(a DOUBLE, x BIGINT)`` with NULLs in the group key ``a``.

    ``a`` is DOUBLE so pandas/cuDF represent the NULLs as NA (cuDF
    ``null_count`` > 0); ``x`` is the agg arg. Rows are arranged so the WHERE
    filter keeps some NULL-key rows (x=3, x=30 both > 2) and drops a non-NULL
    row (x=1, the second a=1). Expected groups after ``WHERE x>2``:
    ``a=1 -> 10``, ``a=2 -> 25``, ``a=NULL -> 33``.
    """
    d = tmp_path / "t"
    d.mkdir()
    import pandas as pd

    pd.DataFrame(
        {"a": [1, 2, None, 1, 2, None], "x": [10, 5, 3, 1, 20, 30]}
    ).to_parquet(str(d / "0.parquet"))
    cat = Catalog(str(tmp_path))
    cat.register("t", str(d))
    eng = Engine(cat)
    oracle = duckdb.connect()
    oracle.execute(f"CREATE VIEW t AS SELECT * FROM read_parquet('{d}/0.parquet')")
    return eng, oracle, str(d)


def _warm(eng: Engine) -> None:
    """Prime the scan cache so the next query takes the warm path (the
    ``fused_aggregate`` C++/CUDA kernel, not the cold Parquet reader)."""
    eng.sql("SELECT count(*) FROM t")


def _cold(eng: Engine) -> None:
    """Force the cold Parquet-reader path on the next query."""
    eng.clear_scan_cache()


# ----------------------------------------------------------- single-key cases


def test_warm_filtered_groupby_keeps_null_group(null_key_engine):
    eng, oracle, _ = null_key_engine
    _warm(eng)
    sql = "SELECT a, SUM(x) AS s FROM t WHERE x > 2 GROUP BY a ORDER BY a"
    assert_same(eng.sql(sql), oracle.execute(sql).fetchdf())


def test_cold_filtered_groupby_keeps_null_group(null_key_engine):
    eng, oracle, _ = null_key_engine
    _cold(eng)
    sql = "SELECT a, SUM(x) AS s FROM t WHERE x > 2 GROUP BY a ORDER BY a"
    assert_same(eng.sql(sql), oracle.execute(sql).fetchdf())


def test_warm_filtered_multi_agg_keeps_null_group(null_key_engine):
    """COUNT(*), SUM, MIN, MAX all must aggregate the NULL-key passing rows."""
    eng, oracle, _ = null_key_engine
    _warm(eng)
    sql = (
        "SELECT a, count(*) AS c, sum(x) AS s, min(x) AS mn, max(x) AS mx "
        "FROM t WHERE x > 2 GROUP BY a ORDER BY a"
    )
    assert_same(eng.sql(sql), oracle.execute(sql).fetchdf())


def test_warm_filtered_avg_keeps_null_group(null_key_engine):
    """AVG is fused-ineligible when its arg is nullable, but here x is non-null
    so the warm kernel *would* run -- it must still defer because the group
    key is nullable. AVG of the NULL group = (3+30)/2 = 16.5."""
    eng, oracle, _ = null_key_engine
    _warm(eng)
    sql = "SELECT a, avg(x) AS av FROM t WHERE x > 2 GROUP BY a ORDER BY a"
    assert_same(eng.sql(sql), oracle.execute(sql).fetchdf())


def test_cold_filtered_multi_agg_keeps_null_group(null_key_engine):
    eng, oracle, _ = null_key_engine
    _cold(eng)
    sql = (
        "SELECT a, count(*) AS c, sum(x) AS s, min(x) AS mn, max(x) AS mx "
        "FROM t WHERE x > 2 GROUP BY a ORDER BY a"
    )
    assert_same(eng.sql(sql), oracle.execute(sql).fetchdf())


# --------------------------------------------------- no WHERE (regression)


def test_warm_unfiltered_groupby_keeps_null_group(null_key_engine):
    """Without a WHERE there is no Filter below the Aggregate, so the warm
    kernel never runs (``_match`` requires ``Aggregate -> Filter``). This still
    must keep the NULL group via the cuDF ``dropna=False`` path."""
    eng, oracle, _ = null_key_engine
    _warm(eng)
    sql = "SELECT a, SUM(x) AS s FROM t GROUP BY a ORDER BY a"
    assert_same(eng.sql(sql), oracle.execute(sql).fetchdf())


# ----------------------------------------------------- predicate variants


def test_warm_filtered_groupby_null_in_predicate_col(null_key_engine):
    """A predicate over a *different* nullable column: ``x`` is non-null here,
    so filter on a NULL-bearing expression of x is N/A; instead use a predicate
    that keeps all rows (x > 0) so every NULL-key row passes and is grouped."""
    eng, oracle, _ = null_key_engine
    _warm(eng)
    sql = "SELECT a, SUM(x) AS s FROM t WHERE x > 0 GROUP BY a ORDER BY a"
    assert_same(eng.sql(sql), oracle.execute(sql).fetchdf())


def test_warm_filtered_groupby_conjunction(null_key_engine):
    eng, oracle, _ = null_key_engine
    _warm(eng)
    sql = "SELECT a, SUM(x) AS s FROM t WHERE x > 2 AND x < 25 GROUP BY a ORDER BY a"
    assert_same(eng.sql(sql), oracle.execute(sql).fetchdf())


# ----------------------------------------------------------- composite key


@pytest.fixture
def composite_null_engine(tmp_path) -> tuple[Engine, duckdb.DuckDBPyConnection]:
    """``t(a DOUBLE, b DOUBLE, x BIGINT)`` with NULLs in BOTH key columns."""
    d = tmp_path / "t"
    d.mkdir()
    import pandas as pd

    pd.DataFrame(
        {
            "a": [1, 1, None, 2, None, 1],
            "b": [10, None, 10, 10, 10, None],
            "x": [10, 5, 3, 1, 20, 30],
        }
    ).to_parquet(str(d / "0.parquet"))
    cat = Catalog(str(tmp_path))
    cat.register("t", str(d))
    eng = Engine(cat)
    oracle = duckdb.connect()
    oracle.execute(f"CREATE VIEW t AS SELECT * FROM read_parquet('{d}/0.parquet')")
    return eng, oracle


def test_warm_filtered_composite_key_keeps_null_groups(composite_null_engine):
    """A composite key with NULLs in either column must keep every (a,b) group,
    including (NULL,10), (1,NULL), (NULL,NULL)-style combinations."""
    eng, oracle = composite_null_engine
    eng.sql("SELECT count(*) FROM t")  # warm
    sql = "SELECT a, b, SUM(x) AS s FROM t WHERE x > 2 GROUP BY a, b ORDER BY a, b"
    assert_same(eng.sql(sql), oracle.execute(sql).fetchdf())


def test_cold_filtered_composite_key_keeps_null_groups(composite_null_engine):
    eng, oracle = composite_null_engine
    eng.clear_scan_cache()
    sql = "SELECT a, b, SUM(x) AS s FROM t WHERE x > 2 GROUP BY a, b ORDER BY a, b"
    assert_same(eng.sql(sql), oracle.execute(sql).fetchdf())


# ---------------------------------------------- fused kernel defer check


def test_fused_aggregate_defers_on_nullable_group_key(null_key_engine):
    """Directly assert ``fused_aggregate`` returns ``None`` for a nullable
    group key (the load-bearing fix): the warm kernel must NOT run, deferring
    to the cuDF ``dropna=False`` path."""
    from ryudb.exec import fused
    from ryudb.sql.parse import parse
    from ryudb.sql.optimize import optimize
    import ryudb.sql.plan as P

    eng, _oracle, _ = null_key_engine
    sql = "SELECT a, SUM(x) AS s FROM t WHERE x > 2 GROUP BY a"
    plan = optimize(
        parse(sql, eng.catalog.schema_dict()),
        eng.catalog.schema_dict(),
        eng.catalog.stats_dict(),
    )
    agg = next(n for n in P.walk(plan) if isinstance(n, P.Aggregate))
    child = eng._exec(agg.input.input)  # the Scan frame under the Filter
    assert child["a"].null_count != 0  # precondition: the key really is nullable
    assert fused.fused_aggregate(agg, child, eng) is None