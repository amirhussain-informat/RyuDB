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
                                rtol=1e-6, atol=1e-2)
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