"""Tests for the fused filter+groupby+aggregate CUDA kernel (Phase 3a).

These build a tiny TPC-H-Q1-shaped lineitem (string group keys, numeric agg
columns, a date filter column) and check the Numba fused kernel path against
DuckDB, plus that an ineligible shape (high-cardinality GROUP BY) falls back to
the cuDF path without error.
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


@pytest.fixture(scope="module")
def q1_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("ryudb_q1")
    (d / "lineitem").mkdir()
    rows = {
        "l_orderkey":    [1, 1, 2, 3, 3, 3, 4, 5, 5, 6],
        "l_returnflag":  ["A", "N", "R", "A", "N", "R", "A", "N", "R", "A"],
        "l_linestatus":  ["F", "O", "F", "O", "F", "O", "F", "O", "F", "O"],
        "l_quantity":    [5.0, 10.0, 2.0, 1.0, 1.0, 1.0, 7.0, 3.0, 4.0, 6.0],
        "l_extendedprice":[50.0, 60.0, 30.0, 10.0, 20.0, 5.0, 90.0, 75.0, 40.0, 80.0],
        "l_discount":     [0.05, 0.10, 0.02, 0.0, 0.50, 0.04, 0.07, 0.03, 0.09, 0.01],
        "l_shipdate": pd.to_datetime(
            ["1998-08-10", "1998-09-20", "1998-08-30", "1998-07-15",
             "1998-07-16", "1998-08-01", "1999-10-05", "1998-09-30",
             "2000-01-01", "1998-08-02"],
        ),
    }
    cudf.DataFrame(rows).to_pandas().to_parquet(d / "lineitem" / "0.parquet")
    return d


@pytest.fixture
def q1_engine(q1_dir) -> Engine:
    cat = Catalog(str(q1_dir))
    cat.register("lineitem", str(q1_dir / "lineitem"))
    return Engine(cat)


@pytest.fixture
def q1_duck(q1_dir) -> "duckdb.DuckDBPyConnection":
    con = duckdb.connect()
    con.execute(f"CREATE VIEW lineitem AS SELECT * FROM read_parquet('{q1_dir}/lineitem/*.parquet')")
    return con


Q1 = """
SELECT l_returnflag, l_linestatus,
       sum(l_quantity) AS sum_qty,
       sum(l_extendedprice) AS sum_base_price,
       sum(l_extendedprice * (1 - l_discount)) AS sum_disc_price,
       count(*) AS count_order
  FROM lineitem
 WHERE l_shipdate <= date '1998-09-02'
 GROUP BY l_returnflag, l_linestatus
 ORDER BY l_returnflag, l_linestatus
"""


def _pdf(df):
    return df.to_pandas() if hasattr(df, "to_pandas") else df


def _match(a, b):
    pa, pb = _pdf(a), _pdf(b)
    assert list(pa.columns) == list(pb.columns)
    if len(pa) == 0:
        assert len(pb) == 0
        return
    pa = pa.sort_values(list(pa.columns)).reset_index(drop=True)
    pb = pb.sort_values(list(pb.columns)).reset_index(drop=True)
    for c in pa.columns:
        try:
            assert np.allclose(pa[c].astype(float).values, pb[c].astype(float).values,
                                rtol=1e-6, atol=1e-2)
        except (ValueError, TypeError):
            assert list(pa[c]) == list(pb[c])


def test_fused_q1_matches_duckdb(q1_engine, q1_duck):
    ryu = q1_engine.sql(Q1)
    duck = q1_duck.execute(Q1).fetchdf()
    _match(ryu, duck)


def test_fused_path_is_taken(q1_engine):
    """The Q1 plan should be eligible for the fused kernel (returns a frame,
    not None)."""
    plan = optimize(parse(Q1, q1_engine.catalog.schema_dict()),
                    q1_engine.catalog.schema_dict(), q1_engine.catalog.stats_dict())
    agg = next(n for n in walk(plan) if isinstance(n, Aggregate))
    child = q1_engine._exec(agg.input.input)
    res = fused.fused_aggregate(agg, child)
    assert res is not None
    assert len(res) > 0


def test_high_card_groupby_matches_duckdb(q1_engine, q1_duck):
    """A high-cardinality numeric GROUP BY (l_orderkey) runs through the C++ HASH
    path (single int64 group key, no factorize) and matches DuckDB."""
    if not fused._kernels.is_available:
        pytest.skip("C++ fused kernel not built")
    sql = """
    SELECT l_orderkey, sum(l_quantity) AS s
      FROM lineitem
     WHERE l_shipdate <= date '1998-09-02'
     GROUP BY l_orderkey
     ORDER BY l_orderkey
    """
    plan = optimize(parse(sql, q1_engine.catalog.schema_dict()),
                    q1_engine.catalog.schema_dict(), q1_engine.catalog.stats_dict())
    agg = next(n for n in walk(plan) if isinstance(n, Aggregate))
    child = q1_engine._exec(agg.input.input)
    res = fused.fused_aggregate(agg, child, q1_engine)
    assert res is not None, "high-card numeric GROUP BY should hit the C++ HASH path"
    ryu = q1_engine.sql(sql)
    duck = q1_duck.execute(sql).fetchdf()
    _match(ryu, duck)


def test_high_card_groupby_falls_back(q1_engine, q1_duck):
    """When the C++ backend is unavailable and the dense accumulator would exceed
    the cell cap, a high-cardinality GROUP BY (l_orderkey) falls back to cuDF and
    still matches DuckDB. This exercises the safety-net fallback, not the hot
    path."""
    sql = """
    SELECT l_orderkey, sum(l_quantity) AS s
      FROM lineitem
     WHERE l_shipdate <= date '1998-09-02'
     GROUP BY l_orderkey
     ORDER BY l_orderkey
    """
    saved_avail = fused._kernels.is_available
    saved_cap = fused.MAX_ACC_CELLS
    fused._kernels.is_available = False
    fused.MAX_ACC_CELLS = 0
    try:
        plan = optimize(parse(sql, q1_engine.catalog.schema_dict()),
                        q1_engine.catalog.schema_dict(), q1_engine.catalog.stats_dict())
        agg = next(n for n in walk(plan) if isinstance(n, Aggregate))
        child = q1_engine._exec(agg.input.input)
        assert fused.fused_aggregate(agg, child, q1_engine) is None  # -> cuDF fallback
        ryu = q1_engine.sql(sql)
        duck = q1_duck.execute(sql).fetchdf()
        _match(ryu, duck)
    finally:
        fused._kernels.is_available = saved_avail
        fused.MAX_ACC_CELLS = saved_cap


def test_fused_no_filter_predicate():
    """A bare Aggregate over a Scan (no Filter) is not the fused shape and must
    fall back without error."""
    from ryudb.sql.plan import AggFunc, Col, Scan, Star

    node = Aggregate(Scan("t"), [(Col("k"), "k")], [(AggFunc("COUNT", Star()), "c")])
    assert fused.fused_aggregate(node, cudf.DataFrame()) is None