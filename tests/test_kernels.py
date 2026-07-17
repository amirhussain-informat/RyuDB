"""Tests for the C++/CUDA fused-kernel extension (Phase 3b).

These exercise the nvcc+pybind11 backend specifically: the hash-table path for a
high-cardinality numeric GROUP BY (the headline capability), the dense path for
low-cardinality string keys, datetime group keys, the multi-column-numeric
deferral, and the safety-net fallback when the extension is disabled. Correctness
is checked against DuckDB. If the C++ extension is not built, the hash/dense
specifics are skipped (the rest still validate the fallback).
"""

from __future__ import annotations

import cudf
import duckdb
import numpy as np
import pandas as pd
import pytest

from ryudb import Catalog, Engine
from ryudb.exec import fused
from ryudb.sql.optimize import optimize
from ryudb.sql.parse import parse
from ryudb.sql.plan import Aggregate, walk

CPP = fused._kernels.is_available


@pytest.fixture(scope="module")
def hc_dir(tmp_path_factory):
    """A lineitem with enough distinct orderkeys to exercise the hash table."""
    d = tmp_path_factory.mktemp("ryudb_kernels")
    (d / "lineitem").mkdir()
    rng = np.random.default_rng(7)
    n = 20000
    rows = {
        # ~5000 distinct orderkeys -> high-cardinality numeric group key.
        "l_orderkey": rng.integers(1, 5001, size=n).astype(np.int64),
        "l_partkey": rng.integers(1, 2001, size=n).astype(np.int64),
        "l_returnflag": rng.choice(["A", "N", "R"], size=n).astype(object),
        "l_linestatus": rng.choice(["F", "O"], size=n).astype(object),
        "l_quantity": rng.uniform(1, 50, size=n),
        "l_extendedprice": rng.uniform(10, 100, size=n),
        "l_discount": rng.uniform(0, 0.5, size=n),
        "l_tax": rng.uniform(0, 0.2, size=n),
        "l_shipdate": pd.to_datetime(
            rng.choice(pd.date_range("1998-01-01", "1998-12-31"), size=n)
        ),
    }
    cudf.DataFrame(rows).to_pandas().to_parquet(d / "lineitem" / "0.parquet")
    return d


@pytest.fixture
def hc_engine(hc_dir) -> Engine:
    cat = Catalog(str(hc_dir))
    cat.register("lineitem", str(hc_dir / "lineitem"))
    return Engine(cat)


@pytest.fixture
def hc_duck(hc_dir) -> "duckdb.DuckDBPyConnection":
    con = duckdb.connect()
    con.execute(f"CREATE VIEW lineitem AS SELECT * FROM read_parquet('{hc_dir}/lineitem/*.parquet')")
    return con


def _pdf(df):
    return df.to_pandas() if hasattr(df, "to_pandas") else df


def _match(a, b):
    pa, pb = _pdf(a), _pdf(b)
    assert list(pa.columns) == list(pb.columns)
    if len(pa) == 0:
        assert len(pb) == 0
        return
    cols = list(pa.columns)
    pa = pa.sort_values(cols).reset_index(drop=True)
    pb = pb.sort_values(cols).reset_index(drop=True)
    for c in cols:
        try:
            assert np.allclose(pa[c].astype(float).values, pb[c].astype(float).values,
                                rtol=1e-6, atol=1e-2, equal_nan=True)
        except (ValueError, TypeError):
            assert list(pa[c]) == list(pb[c])


def _agg_node(sql, engine):
    plan = optimize(parse(sql, engine.catalog.schema_dict()),
                    engine.catalog.schema_dict(), engine.catalog.stats_dict())
    return next(n for n in walk(plan) if isinstance(n, Aggregate))


HC_ORDERKEY = """
    SELECT l_orderkey, sum(l_quantity) AS sum_qty,
           sum(l_extendedprice * (1 - l_discount)) AS sum_disc_price,
           count(*) AS count_order
      FROM lineitem
     WHERE l_shipdate <= date '1998-09-02'
     GROUP BY l_orderkey
     ORDER BY l_orderkey
"""

HC_DENSE = """
    SELECT l_returnflag, l_linestatus, sum(l_quantity) AS sum_qty, count(*) AS n
      FROM lineitem
     WHERE l_shipdate <= date '1998-09-02'
     GROUP BY l_returnflag, l_linestatus
     ORDER BY l_returnflag, l_linestatus
"""

HC_DATETIME_KEY = """
    SELECT l_shipdate, sum(l_quantity) AS sum_qty, count(*) AS n
      FROM lineitem
     WHERE l_quantity > 5
     GROUP BY l_shipdate
     ORDER BY l_shipdate
"""


def test_cpp_extension_available():
    """The build artefact should be importable in the dev env."""
    if not CPP:
        pytest.skip("C++ fused kernel not built (run python -m ryudb.kernels.build)")
    assert fused._kernels.fused_agg is not None


def test_hash_high_card_orderkey(hc_engine, hc_duck):
    """High-cardinality numeric GROUP BY runs the C++ hash-table path and matches
    DuckDB row-for-row."""
    if not CPP:
        pytest.skip("C++ fused kernel not built")
    agg = _agg_node(HC_ORDERKEY, hc_engine)
    child = hc_engine._exec(agg.input.input)
    res = fused.fused_aggregate(agg, child, hc_engine)
    assert res is not None, "high-card numeric GROUP BY should hit the C++ HASH path"
    _match(hc_engine.sql(HC_ORDERKEY), hc_duck.execute(HC_ORDERKEY).fetchdf())


def test_dense_low_card_strings(hc_engine, hc_duck):
    """Low-cardinality string GROUP BY runs the C++ dense path (codes) and matches
    DuckDB."""
    if not CPP:
        pytest.skip("C++ fused kernel not built")
    agg = _agg_node(HC_DENSE, hc_engine)
    child = hc_engine._exec(agg.input.input)
    res = fused.fused_aggregate(agg, child, hc_engine)
    assert res is not None
    _match(hc_engine.sql(HC_DENSE), hc_duck.execute(HC_DENSE).fetchdf())


def test_datetime_group_key_hash(hc_engine, hc_duck):
    """A datetime group key is normalised to int64 seconds and handled by the C++
    hash path (single-column numeric)."""
    if not CPP:
        pytest.skip("C++ fused kernel not built")
    agg = _agg_node(HC_DATETIME_KEY, hc_engine)
    child = hc_engine._exec(agg.input.input)
    res = fused.fused_aggregate(agg, child, hc_engine)
    assert res is not None, "datetime group key should hit the C++ HASH path"
    _match(hc_engine.sql(HC_DATETIME_KEY), hc_duck.execute(HC_DATETIME_KEY).fetchdf())


def test_multi_column_numeric_groupby_deferred(hc_engine):
    """Multi-column numeric GROUP BY is NOT handled by the C++ hash kernel (it
    supports a single int64 key); fused_aggregate returns None so the executor
    falls back to cuDF."""
    if not CPP:
        pytest.skip("C++ fused kernel not built")
    sql = """
        SELECT l_orderkey, l_partkey, sum(l_extendedprice) AS s
          FROM lineitem
         WHERE l_shipdate <= date '1998-09-02'
         GROUP BY l_orderkey, l_partkey
    """
    agg = _agg_node(sql, hc_engine)
    child = hc_engine._exec(agg.input.input)
    assert fused.fused_aggregate(agg, child, hc_engine) is None


def test_fallback_when_extension_disabled(hc_engine, hc_duck):
    """With the C++ backend disabled, the high-card query falls back to cuDF and
    still matches DuckDB (correctness never depends on the extension)."""
    saved = fused._kernels.is_available
    fused._kernels.is_available = False
    try:
        agg = _agg_node(HC_ORDERKEY, hc_engine)
        child = hc_engine._exec(agg.input.input)
        # No dense accumulator for ~5000 groups * 3 aggs either way -> None -> cuDF.
        assert fused.fused_aggregate(agg, child, hc_engine) is None
        _match(hc_engine.sql(HC_ORDERKEY), hc_duck.execute(HC_ORDERKEY).fetchdf())
    finally:
        fused._kernels.is_available = saved


# --- Phase 4: fused global aggregate + MIN/MAX/AVG -------------------------- #

GLOBAL_AGG = """
    SELECT count(*) AS n, sum(l_extendedprice) AS s, avg(l_quantity) AS q,
           min(l_discount) AS md, max(l_tax) AS mt
      FROM lineitem
     WHERE l_quantity > 25
"""

GLOBAL_AGG_EMPTY = """
    SELECT count(*) AS n, sum(l_extendedprice) AS s, avg(l_quantity) AS q,
           min(l_discount) AS md, max(l_tax) AS mt
      FROM lineitem
     WHERE l_quantity > 100000
"""

GROUPED_MINMAXAVG = """
    SELECT l_returnflag, l_linestatus,
           min(l_quantity) AS qmin, max(l_quantity) AS qmax,
           avg(l_extendedprice) AS eavg, sum(l_discount) AS dsum, count(*) AS n
      FROM lineitem
     WHERE l_shipdate <= date '1998-09-02'
     GROUP BY l_returnflag, l_linestatus
     ORDER BY l_returnflag, l_linestatus
"""

HASH_MIN_GUARD = """
    SELECT l_orderkey, min(l_quantity) AS qmin, max(l_extendedprice) AS emax
      FROM lineitem
     WHERE l_shipdate <= date '1998-09-02'
     GROUP BY l_orderkey
"""


def test_global_aggregate_matches_duckdb(hc_engine, hc_duck):
    """A global aggregate (no GROUP BY) with count/sum/avg/min/max runs through
    the fused C++ DENSE path (n_groups=1) and matches DuckDB."""
    if not CPP:
        pytest.skip("C++ fused kernel not built")
    agg = _agg_node(GLOBAL_AGG, hc_engine)
    child = hc_engine._exec(agg.input.input)
    res = fused.fused_aggregate(agg, child, hc_engine)
    assert res is not None, "global agg should hit the C++ DENSE path"
    assert len(res) == 1
    _match(hc_engine.sql(GLOBAL_AGG), hc_duck.execute(GLOBAL_AGG).fetchdf())


def test_global_aggregate_empty_matches_duckdb(hc_engine, hc_duck):
    """A global aggregate over zero matching rows returns ONE row with count=0
    and NULL (NaN) for sum/avg/min/max -- SQL semantics, matching DuckDB."""
    if not CPP:
        pytest.skip("C++ fused kernel not built")
    agg = _agg_node(GLOBAL_AGG_EMPTY, hc_engine)
    child = hc_engine._exec(agg.input.input)
    res = fused.fused_aggregate(agg, child, hc_engine)
    assert res is not None
    assert len(res) == 1
    pdf = res.to_pandas()
    assert int(pdf["n"].iloc[0]) == 0
    assert np.isnan(float(pdf["s"].iloc[0]))
    _match(hc_engine.sql(GLOBAL_AGG_EMPTY), hc_duck.execute(GLOBAL_AGG_EMPTY).fetchdf())


def test_grouped_minmaxavg_matches_duckdb(hc_engine, hc_duck):
    """Low-cardinality grouped aggregate with min/max/avg/sum/count runs through
    the C++ DENSE path and matches DuckDB."""
    if not CPP:
        pytest.skip("C++ fused kernel not built")
    agg = _agg_node(GROUPED_MINMAXAVG, hc_engine)
    child = hc_engine._exec(agg.input.input)
    res = fused.fused_aggregate(agg, child, hc_engine)
    assert res is not None, "grouped min/max/avg should hit the C++ DENSE path"
    _match(hc_engine.sql(GROUPED_MINMAXAVG), hc_duck.execute(GROUPED_MINMAXAVG).fetchdf())


def test_hash_min_max_guard_defers(hc_engine, hc_duck):
    """MIN/MAX over a high-cardinality numeric GROUP BY (HASH path) is NOT
    supported by the C++ hash kernel -> fused_aggregate returns None and the
    query falls back to cuDF, still matching DuckDB."""
    if not CPP:
        pytest.skip("C++ fused kernel not built")
    agg = _agg_node(HASH_MIN_GUARD, hc_engine)
    child = hc_engine._exec(agg.input.input)
    assert fused.fused_aggregate(agg, child, hc_engine) is None
    _match(hc_engine.sql(HASH_MIN_GUARD), hc_duck.execute(HASH_MIN_GUARD).fetchdf())


# --- Phase 4 step 2: fused star-join + aggregate ---------------------------- #
#
# A small snowflake: fact F -> D1 -> D2, GROUP BY D2.label, SUM(F.f_val). The
# data deliberately includes fact rows whose key misses D1 (f_key1 in 20..24,
# D1 has keys 0..19) and D1 rows whose payload misses D2 (d1_next == 5, D2 has
# keys 0..4), so the inner-join drop semantics are exercised end-to-end.

@pytest.fixture(scope="module")
def star_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("ryudb_star")
    rng = np.random.default_rng(11)
    d2 = cudf.DataFrame({
        "d2_key": np.arange(5, dtype=np.int64),
        "label": np.array(["A", "B", "C", "D", "E"], dtype=object),
    })
    # d1_next in 0..5; 5 misses D2 -> second-stage inner-join drop.
    d1 = cudf.DataFrame({
        "d1_key": np.arange(20, dtype=np.int64),
        "d1_next": rng.integers(0, 6, size=20).astype(np.int64),
    })
    # f_key1 in 0..24; 20..24 miss D1 -> first-stage inner-join drop.
    f = cudf.DataFrame({
        "f_key1": rng.integers(0, 25, size=20000).astype(np.int64),
        "f_val": rng.uniform(1, 100, size=20000),
    })
    for name, fr in [("D2", d2), ("D1", d1), ("F", f)]:
        (d / name).mkdir()
        fr.to_pandas().to_parquet(d / name / "0.parquet")
    return d


@pytest.fixture
def star_engine(star_dir) -> Engine:
    cat = Catalog(str(star_dir))
    for t in ("F", "D1", "D2"):
        cat.register(t, str(star_dir / t))
    return Engine(cat)


@pytest.fixture
def star_duck(star_dir) -> "duckdb.DuckDBPyConnection":
    con = duckdb.connect()
    for t in ("F", "D1", "D2"):
        con.execute(f"CREATE VIEW {t} AS SELECT * FROM read_parquet('{star_dir}/{t}/*.parquet')")
    return con


STAR_SNOWFLAKE = """
    SELECT label, sum(f_val) AS revenue
      FROM F
      JOIN D1 ON f_key1 = d1_key
      JOIN D2 ON d1_next = d2_key
     GROUP BY label
     ORDER BY revenue DESC
"""

STAR_MULTIKEY = """
    SELECT label, d1_key, sum(f_val) AS s
      FROM F
      JOIN D1 ON f_key1 = d1_key
      JOIN D2 ON d1_next = d2_key
     GROUP BY label, d1_key
     ORDER BY label, d1_key
"""

STAR_DIM_ARG = """
    SELECT label, sum(d1_next) AS s
      FROM F
      JOIN D1 ON f_key1 = d1_key
      JOIN D2 ON d1_next = d2_key
     GROUP BY label
     ORDER BY label
"""


def test_fused_star_join_matches_duckdb(star_engine, star_duck):
    """A snowflake star-join + grouped SUM runs through the fused C++ kernel
    (stream F, probe D1/D2 HTs, accumulate per label) and matches DuckDB,
    including the inner-join drops on both join legs."""
    if not CPP:
        pytest.skip("C++ fused kernel not built")
    agg = _agg_node(STAR_SNOWFLAKE, star_engine)
    res = fused.fused_join_aggregate(agg, star_engine)
    assert res is not None, "snowflake star-join + SUM should hit the C++ fused path"
    _match(star_engine.sql(STAR_SNOWFLAKE), star_duck.execute(STAR_SNOWFLAKE).fetchdf())


def test_fused_star_join_multikey_defers(star_engine, star_duck):
    """A multi-key GROUP BY over a join is out of scope -> fused_join_aggregate
    returns None and the query falls back to cuDF, still matching DuckDB."""
    if not CPP:
        pytest.skip("C++ fused kernel not built")
    agg = _agg_node(STAR_MULTIKEY, star_engine)
    assert fused.fused_join_aggregate(agg, star_engine) is None
    _match(star_engine.sql(STAR_MULTIKEY), star_duck.execute(STAR_MULTIKEY).fetchdf())


def test_fused_star_join_dim_agg_arg_defers(star_engine, star_duck):
    """An aggregate over a dimension column (not the fact table) is out of scope
    (the kernel reads agg args at the streaming fact row) -> returns None and
    falls back to cuDF, still matching DuckDB."""
    if not CPP:
        pytest.skip("C++ fused kernel not built")
    agg = _agg_node(STAR_DIM_ARG, star_engine)
    assert fused.fused_join_aggregate(agg, star_engine) is None
    _match(star_engine.sql(STAR_DIM_ARG), star_duck.execute(STAR_DIM_ARG).fetchdf())