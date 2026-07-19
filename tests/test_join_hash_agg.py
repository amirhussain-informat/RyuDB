"""Tests for the fused C++/CUDA HASH accumulator on the join path (high-cardinality
group-from-join).

PR #50 added a HASH strategy to the non-join aggregate path; this lifts the same
strategy to `fused_join_aggregate` so that `GROUP BY` a high-NDV dimension key over a
join (NDV*nagg > MAX_ACC_CELLS) fires a `probe_hash_agg_kernel` instead of deferring
to cuDF. Correctness is checked against DuckDB. If the C++ extension is not built,
the specifics are skipped (the rest still validate the cuDF fallback).
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
import pytest

from ryudb import Catalog, Engine
from ryudb.exec import fused
from ryudb.exec import executor as ex_mod
from ryudb.sql.optimize import optimize
from ryudb.sql.parse import parse
from ryudb.sql.plan import Aggregate, walk

CPP = fused._kernels.is_available


# --------------------------------------------------------------------------- #
# Fixtures: a high-cardinality join (lineitem JOIN orders) with > MAX_ACC_CELLS
# distinct orderkeys, plus a customer dim for the multi-stage chain.
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def hcj_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("ryudb_join_hash")
    rng = np.random.default_rng(7)
    n_li = 60000
    n_ord = 15000   # > 4096 distinct orderkeys -> n_groups * nagg > MAX_ACC_CELLS
    n_cust = 5000   # high-card tail dim for the multi-stage chain

    lineitem = pd.DataFrame({
        "l_orderkey": rng.integers(1, n_ord + 1, size=n_li).astype(np.int64),
        "l_quantity": rng.integers(1, 100, size=n_li).astype(np.int64),
        "l_extendedprice": (rng.integers(1, 1000, size=n_li) * 100).astype(np.int64),
    })
    orders = pd.DataFrame({
        "o_orderkey": np.arange(1, n_ord + 1, dtype=np.int64),
        "o_custkey": rng.integers(1, n_cust + 1, size=n_ord).astype(np.int64),
    })
    customer = pd.DataFrame({
        "c_custkey": np.arange(1, n_cust + 1, dtype=np.int64),
        "c_name": [f"cust_{i}" for i in range(1, n_cust + 1)],
    })
    for name, fr in [("lineitem", lineitem), ("orders", orders), ("customer", customer)]:
        (d / name).mkdir()
        fr.to_parquet(d / name / "0.parquet")
    return d


@pytest.fixture
def hcj_engine(hcj_dir) -> Engine:
    cat = Catalog(str(hcj_dir))
    for t in ("lineitem", "orders", "customer"):
        cat.register(t, str(hcj_dir / t))
    return Engine(cat)


@pytest.fixture
def hcj_duck(hcj_dir) -> "duckdb.DuckDBPyConnection":
    con = duckdb.connect()
    for t in ("lineitem", "orders", "customer"):
        con.execute(f"CREATE VIEW {t} AS SELECT * FROM read_parquet('{hcj_dir}/{t}/*.parquet')")
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
    assert len(pa) == len(pb)
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


# High-card group-from-join: GROUP BY o_orderkey (a dimension key reached by the
# chain) with 3 aggs -> n_groups*nagg = ~15000*3 = 45000 > MAX_ACC_CELLS = 4096.
HIGH_CARD_JOIN = """
    SELECT o_orderkey, sum(l_quantity) AS sum_qty,
           sum(l_extendedprice) AS rev, count(*) AS n
      FROM lineitem
      JOIN orders ON l_orderkey = o_orderkey
     GROUP BY o_orderkey
     ORDER BY o_orderkey
"""

# Two SUMs + COUNT(*) over the same high-card dim key.
HIGH_CARD_JOIN_2SUM = """
    SELECT o_orderkey, sum(l_quantity) AS sum_qty,
           sum(l_extendedprice) AS rev, count(*) AS n
      FROM lineitem
      JOIN orders ON l_orderkey = o_orderkey
     GROUP BY o_orderkey
     ORDER BY o_orderkey
"""

# Multi-stage chain (lineitem -> orders -> customer), GROUP BY the high-card tail
# dim key c_custkey -> HASH at the chain tail.
HIGH_CARD_CHAIN = """
    SELECT c_custkey, sum(l_quantity) AS sum_qty, sum(l_extendedprice) AS rev,
           count(*) AS n
      FROM lineitem
      JOIN orders   ON l_orderkey = o_orderkey
      JOIN customer ON o_custkey = c_custkey
     GROUP BY c_custkey
     ORDER BY c_custkey
"""

# Low-card group-from-join on the same fixture: o_custkey has ~1000 distinct values
# -> n_groups*nagg < MAX_ACC_CELLS -> DENSE (regression check for the unchanged path).
LOW_CARD_JOIN = """
    SELECT o_custkey, sum(l_quantity) AS sum_qty, count(*) AS n
      FROM lineitem
      JOIN orders ON l_orderkey = o_orderkey
     GROUP BY o_custkey
     ORDER BY o_custkey
"""


# --- fires / correctness -------------------------------------------------- #

def test_high_card_join_fires(hcj_engine):
    """High-card group-from-join (NDV*nagg > MAX_ACC_CELLS) hits the C++ HASH path
    instead of deferring to cuDF."""
    if not CPP:
        pytest.skip("C++ fused kernel not built")
    agg = _agg_node(HIGH_CARD_JOIN, hcj_engine)
    assert fused.fused_join_aggregate(agg, hcj_engine) is not None, (
        "high-card group-from-join SUM/COUNT should hit the C++ HASH path")


def test_high_card_join_matches_duckdb(hcj_engine, hcj_duck):
    if not CPP:
        pytest.skip("C++ fused kernel not built")
    assert fused.fused_join_aggregate(_agg_node(HIGH_CARD_JOIN, hcj_engine), hcj_engine) is not None
    _match(hcj_engine.sql(HIGH_CARD_JOIN), hcj_duck.execute(HIGH_CARD_JOIN).fetchdf())


def test_high_card_join_two_sum_count_matches(hcj_engine, hcj_duck):
    if not CPP:
        pytest.skip("C++ fused kernel not built")
    _match(hcj_engine.sql(HIGH_CARD_JOIN_2SUM), hcj_duck.execute(HIGH_CARD_JOIN_2SUM).fetchdf())


def test_high_card_join_chain_matches_duckdb(hcj_engine, hcj_duck):
    """A multi-stage chain with a high-card tail dim group key fires HASH and matches
    DuckDB."""
    if not CPP:
        pytest.skip("C++ fused kernel not built")
    agg = _agg_node(HIGH_CARD_CHAIN, hcj_engine)
    assert fused.fused_join_aggregate(agg, hcj_engine) is not None, (
        "high-card tail-dim group-from-join should hit the C++ HASH path")
    _match(hcj_engine.sql(HIGH_CARD_CHAIN), hcj_duck.execute(HIGH_CARD_CHAIN).fetchdf())


# --- regression: the DENSE join path is unchanged ------------------------- #

def test_low_card_join_dense_fires_and_matches(hcj_engine, hcj_duck):
    """A low-card group-from-join (NDV*nagg <= MAX_ACC_CELLS) still takes the DENSE
    path (byte-unchanged) and matches DuckDB."""
    if not CPP:
        pytest.skip("C++ fused kernel not built")
    agg = _agg_node(LOW_CARD_JOIN, hcj_engine)
    assert fused.fused_join_aggregate(agg, hcj_engine) is not None, (
        "low-card group-from-join should still hit the C++ DENSE path")
    _match(hcj_engine.sql(LOW_CARD_JOIN), hcj_duck.execute(LOW_CARD_JOIN).fetchdf())


# --- group-HT overflow -> defer to cuDF ----------------------------------- #

def test_high_card_join_overflow_defers_and_matches(hcj_engine, hcj_duck):
    """A group HT too small for the NDV overflows in-kernel -> fused_join_aggregate
    returns None and the cuDF fallback still matches DuckDB. Forces a tiny capacity
    via _HASH_ACC_BUDGET so the group HT (32 slots) cannot hold ~15000 groups."""
    if not CPP:
        pytest.skip("C++ fused kernel not built")
    orig_budget = fused._HASH_ACC_BUDGET
    fused._HASH_ACC_BUDGET = 768  # nagg=3 -> 768//24 = 32-slot group HT -> overflow
    try:
        agg = _agg_node(HIGH_CARD_JOIN, hcj_engine)
        assert fused.fused_join_aggregate(agg, hcj_engine) is None, (
            "an overflowing group HT should defer to cuDF")
        _match(hcj_engine.sql(HIGH_CARD_JOIN), hcj_duck.execute(HIGH_CARD_JOIN).fetchdf())
    finally:
        fused._HASH_ACC_BUDGET = orig_budget


# --- disable-extension fallback ------------------------------------------ #

def test_high_card_join_disable_extension_matches(hcj_engine, hcj_duck, monkeypatch):
    """With the C++ extension disabled, the high-card join falls back to cuDF and
    still matches DuckDB."""
    monkeypatch.setattr(fused._kernels, "is_available", False)
    _match(hcj_engine.sql(HIGH_CARD_JOIN), hcj_duck.execute(HIGH_CARD_JOIN).fetchdf())


# --- end-to-end dispatch fires (executor-module-patching gotcha) ---------- #

def test_high_card_join_dispatch_fires(hcj_engine, hcj_duck, monkeypatch):
    """engine.sql actually dispatches to the fused HASH path (not the cuDF fallback):
    patch ryudb.exec.executor.fused_join_aggregate (the name the executor imports,
    NOT fused.fused_join_aggregate) with a counter and assert it is called once."""
    if not CPP:
        pytest.skip("C++ fused kernel not built")
    calls = {"n": 0}
    orig = ex_mod.fused_join_aggregate

    def wrap(node, engine, _o=orig):
        r = _o(node, engine)
        calls["n"] += 1
        return r

    monkeypatch.setattr(ex_mod, "fused_join_aggregate", wrap)
    got = hcj_engine.sql(HIGH_CARD_JOIN)
    assert calls["n"] == 1, "the high-card join should dispatch to the fused path once"
    _match(got, hcj_duck.execute(HIGH_CARD_JOIN).fetchdf())


# --- CLI smoke (the SQL surface, ordered+limited) ------------------------- #

def test_high_card_join_smoke(hcj_engine, hcj_duck):
    """The SQL surface returns the high-card group-from-join shape matching DuckDB
    on the ordered, limited head -- the bench `Q_join_high_card_orderkey` shape."""
    if not CPP:
        pytest.skip("C++ fused kernel not built")
    sql = """
        SELECT o_orderkey, sum(l_quantity) AS sum_qty, sum(l_extendedprice) AS rev,
               count(*) AS n
          FROM lineitem
          JOIN orders ON l_orderkey = o_orderkey
         GROUP BY o_orderkey
         ORDER BY o_orderkey
         LIMIT 5
    """
    ryu = hcj_engine.sql(sql).to_pandas()
    duck = hcj_duck.execute(sql).fetchdf()
    assert list(ryu.columns) == list(duck.columns)
    assert len(ryu) == 5
    for c in ryu.columns:
        x, y = ryu[c].to_numpy(), duck[c].to_numpy()
        if x.dtype.kind in "iu":
            assert np.array_equal(x, y)
        else:
            assert np.allclose(x, y, rtol=1e-6, atol=1e-2)