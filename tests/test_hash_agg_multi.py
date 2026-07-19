"""Tests for the extended HASH aggregate path (multi-col/string group keys +
MIN/MAX/AVG over the high-cardinality C++ hash table).

The fused C++/CUDA kernel has two strategies: DENSE (low-card, shared-mem
accumulator, capped at MAX_ACC_CELLS) and HASH (high-card, global open-addressed
int64->acc hash table). This file covers the shapes the HASH path now accepts:

  * Multi-column numeric GROUP BY (factorize per-col -> stride-combine to one
    int64 perfect-hash code -> single-int64 hash_kernel). Was deferred.
  * High-cardinality string GROUP BY (factorize one string col -> HASH). Was
    DENSE-only and deferred when it blew the dense cap.
  * MIN/MAX/AVG over a single int64 HASH key (hash_kernel's per-slot
    atomic_min/max_d + AVG running-sum/hidden-count dispatch). Was deferred.
  * Any combination of the above (multi-col + string + MIN/MAX/AVG).

DuckDB is the correctness oracle (``assert_same``). Fires/defers assertions call
``fused.fused_aggregate`` / ``fused.fused_scan_aggregate`` directly. The
DISTINCT / per-agg-FILTER deferrals are enforced at the executor dispatch level
(``_force_fallback`` skips the fused call entirely), so they are asserted via
the executor-module-patching counter (patch ``ryudb.exec.executor.fused_aggregate``).
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

from .conftest import assert_same

CPP = fused._kernels.is_available


@pytest.fixture(scope="module")
def hc_dir(tmp_path_factory):
    """A lineitem with high-card int + string keys and a nullable arg/key col."""
    d = tmp_path_factory.mktemp("ryudb_hash_multi")
    (d / "lineitem").mkdir()
    rng = np.random.default_rng(31)
    n = 20000
    # ~5000 distinct orderkeys (high-card int) and ~5000 distinct comments
    # (high-card string); ~2000 distinct partkeys (second int key for multi-col).
    orderkeys = rng.integers(1, 5001, size=n).astype(np.int64)
    partkeys = rng.integers(1, 2001, size=n).astype(np.int64)
    comments = np.array([f"c{rng.integers(0, 5000)}" for _ in range(n)], dtype=object)
    rows = {
        "l_orderkey": orderkeys,
        "l_partkey": partkeys,
        "l_comment": comments,
        "l_returnflag": rng.choice(["A", "N", "R"], size=n).astype(object),
        "l_quantity": rng.uniform(1, 50, size=n),
        "l_extendedprice": rng.uniform(10, 100, size=n),
        "l_discount": rng.uniform(0, 0.5, size=n),
        "l_tax": rng.uniform(0, 0.2, size=n),
        "l_shipdate": pd.to_datetime(
            rng.choice(pd.date_range("1998-01-01", "1998-12-31"), size=n)
        ),
        # Nullable arg col (~10% null): exercises the AVG/MIN/MAX arg-null defer.
        "l_quantity_null": [None if r < 0.1 else float(v)
                            for r, v in zip(rng.random(n), rng.uniform(1, 50, size=n))],
        # Nullable int group key (~5% null): exercises the NULL-group-key defer.
        "l_nullkey": [None if r < 0.05 else int(v)
                      for r, v in zip(rng.random(n), rng.integers(1, 101, size=n))],
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


def _agg_node(sql, engine):
    plan = optimize(parse(sql, engine.catalog.schema_dict()),
                    engine.catalog.schema_dict(), engine.catalog.stats_dict())
    return next(n for n in walk(plan) if isinstance(n, Aggregate))


# --- fires/defers: the new HASH shapes hit the C++ path --------------------- #

def test_hash_multicol_int_fires(hc_engine):
    """Multi-column numeric GROUP BY hits the C++ HASH path (stride-combine)."""
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
    assert fused.fused_aggregate(agg, child, hc_engine) is not None


def test_hash_string_high_card_fires(hc_engine):
    """High-cardinality string GROUP BY hits the C++ HASH path (was DENSE-only,
    deferred when the dense cap blew)."""
    if not CPP:
        pytest.skip("C++ fused kernel not built")
    sql = """
        SELECT l_comment, sum(l_quantity) AS s, count(*) AS n
          FROM lineitem
         WHERE l_shipdate <= date '1998-09-02'
         GROUP BY l_comment
    """
    agg = _agg_node(sql, hc_engine)
    child = hc_engine._exec(agg.input.input)
    assert fused.fused_aggregate(agg, child, hc_engine) is not None


def test_hash_single_int_minmaxavg_fires(hc_engine):
    """MIN/MAX/AVG over a single int64 HASH key hits the C++ path (was deferred)."""
    if not CPP:
        pytest.skip("C++ fused kernel not built")
    sql = """
        SELECT l_orderkey, min(l_quantity) AS qmin, max(l_quantity) AS qmax,
               avg(l_quantity) AS qavg
          FROM lineitem
         WHERE l_shipdate <= date '1998-09-02'
         GROUP BY l_orderkey
    """
    agg = _agg_node(sql, hc_engine)
    child = hc_engine._exec(agg.input.input)
    assert fused.fused_aggregate(agg, child, hc_engine) is not None


def test_hash_multicol_minmaxavg_fires(hc_engine):
    """MIN/MAX/AVG over a multi-col HASH key (stride-combine) hits the C++ path."""
    if not CPP:
        pytest.skip("C++ fused kernel not built")
    sql = """
        SELECT l_orderkey, l_partkey, min(l_quantity) AS qmin, max(l_tax) AS tmax,
               avg(l_extendedprice) AS eavg
          FROM lineitem
         WHERE l_shipdate <= date '1998-09-02'
         GROUP BY l_orderkey, l_partkey
    """
    agg = _agg_node(sql, hc_engine)
    child = hc_engine._exec(agg.input.input)
    assert fused.fused_aggregate(agg, child, hc_engine) is not None


def test_hash_string_minmaxavg_fires(hc_engine):
    """MIN/MAX/AVG over a high-card string HASH key hits the C++ path."""
    if not CPP:
        pytest.skip("C++ fused kernel not built")
    sql = """
        SELECT l_comment, min(l_quantity) AS qmin, max(l_quantity) AS qmax,
               avg(l_extendedprice) AS eavg
          FROM lineitem
         WHERE l_shipdate <= date '1998-09-02'
         GROUP BY l_comment
    """
    agg = _agg_node(sql, hc_engine)
    child = hc_engine._exec(agg.input.input)
    assert fused.fused_aggregate(agg, child, hc_engine) is not None


def test_hash_single_int_sumcount_regression(hc_engine):
    """Regression: a single int64 SUM/COUNT HASH key (the unchanged raw-HASH path)
    still fires -- the MIN/MAX/AVG extension must not regress the headline path."""
    if not CPP:
        pytest.skip("C++ fused kernel not built")
    sql = """
        SELECT l_orderkey, sum(l_quantity) AS s, count(*) AS n
          FROM lineitem
         WHERE l_shipdate <= date '1998-09-02'
         GROUP BY l_orderkey
    """
    agg = _agg_node(sql, hc_engine)
    child = hc_engine._exec(agg.input.input)
    assert fused.fused_aggregate(agg, child, hc_engine) is not None


# --- still-defers: shapes the fused path rejects -> cuDF fallback ----------- #

def test_null_group_key_defers(hc_engine):
    """A nullable group key defers (the kernel factorises NA -> -1 and would drop
    genuine NULL groups; cuDF groupby(dropna=False) keeps them, matching DuckDB)."""
    if not CPP:
        pytest.skip("C++ fused kernel not built")
    sql = """
        SELECT l_nullkey, sum(l_quantity) AS s
          FROM lineitem
         WHERE l_shipdate <= date '1998-09-02'
         GROUP BY l_nullkey
    """
    agg = _agg_node(sql, hc_engine)
    child = hc_engine._exec(agg.input.input)
    assert fused.fused_aggregate(agg, child, hc_engine) is None


def test_null_arg_col_defers(hc_engine):
    """A nullable AVG/MIN/MAX arg defers (the kernel reads raw values and does not
    skip nulls; AVG = sum / passing-row-count is only correct for a non-null arg)."""
    if not CPP:
        pytest.skip("C++ fused kernel not built")
    sql = """
        SELECT l_orderkey, avg(l_quantity_null) AS qavg
          FROM lineitem
         WHERE l_shipdate <= date '1998-09-02'
         GROUP BY l_orderkey
    """
    agg = _agg_node(sql, hc_engine)
    child = hc_engine._exec(agg.input.input)
    assert fused.fused_aggregate(agg, child, hc_engine) is None


def test_cold_multicol_defers(hc_engine):
    """The cold reader can only HASH a single PLAIN int64 key (read raw from the
    Parquet page); a multi-col GROUP BY has no on-disk combined code -> the cold
    `fused_scan_aggregate` defers to the warm path."""
    if not CPP:
        pytest.skip("C++ fused kernel not built")
    sql = """
        SELECT l_orderkey, l_partkey, sum(l_quantity) AS s
          FROM lineitem
         WHERE l_shipdate <= date '1998-09-02'
         GROUP BY l_orderkey, l_partkey
    """
    agg = _agg_node(sql, hc_engine)
    assert fused.fused_scan_aggregate(agg, hc_engine) is None


# --- correctness vs DuckDB -------------------------------------------------- #

def test_multicol_int_with_predicate_matches(hc_engine, hc_duck):
    """Multi-col int HASH + a WHERE predicate (no-gather fold) matches DuckDB."""
    if not CPP:
        pytest.skip("C++ fused kernel not built")
    sql = """
        SELECT l_orderkey, l_partkey, sum(l_extendedprice) AS s, count(*) AS n
          FROM lineitem
         WHERE l_shipdate <= date '1998-09-02' AND l_quantity > 5
         GROUP BY l_orderkey, l_partkey
    """
    assert fused.fused_aggregate(_agg_node(sql, hc_engine),
                                hc_engine._exec(_agg_node(sql, hc_engine).input.input),
                                hc_engine) is not None
    assert_same(hc_engine.sql(sql), hc_duck.execute(sql).fetchdf())


def test_string_high_card_minmaxavg_matches(hc_engine, hc_duck):
    """High-card string HASH + MIN/MAX/AVG matches DuckDB row-for-row."""
    if not CPP:
        pytest.skip("C++ fused kernel not built")
    sql = """
        SELECT l_comment, min(l_quantity) AS qmin, max(l_quantity) AS qmax,
               avg(l_extendedprice) AS eavg
          FROM lineitem
         WHERE l_shipdate <= date '1998-09-02'
         GROUP BY l_comment
    """
    assert_same(hc_engine.sql(sql), hc_duck.execute(sql).fetchdf())


def test_mixed_multicol_string_minmaxavg_count_matches(hc_engine, hc_duck):
    """Mixed multi-col (int + string) + SUM/AVG/MIN/MAX/COUNT(*) in one query
    matches DuckDB (exercises the stride-combine decoder over heterogeneous keys)."""
    if not CPP:
        pytest.skip("C++ fused kernel not built")
    sql = """
        SELECT l_orderkey, l_returnflag,
               sum(l_extendedprice) AS s, avg(l_quantity) AS qavg,
               min(l_discount) AS dmin, max(l_tax) AS tmax, count(*) AS n
          FROM lineitem
         WHERE l_shipdate <= date '1998-09-02'
         GROUP BY l_orderkey, l_returnflag
    """
    assert_same(hc_engine.sql(sql), hc_duck.execute(sql).fetchdf())


def test_null_arg_fallback_matches(hc_engine, hc_duck):
    """A nullable arg col defers to cuDF, which skips nulls correctly and matches
    DuckDB (correctness never depends on the fused extension)."""
    if not CPP:
        pytest.skip("C++ fused kernel not built")
    sql = """
        SELECT l_orderkey, avg(l_quantity_null) AS qavg
          FROM lineitem
         WHERE l_shipdate <= date '1998-09-02'
         GROUP BY l_orderkey
    """
    assert fused.fused_aggregate(_agg_node(sql, hc_engine),
                                hc_engine._exec(_agg_node(sql, hc_engine).input.input),
                                hc_engine) is None
    assert_same(hc_engine.sql(sql), hc_duck.execute(sql).fetchdf())


# --- cold path: single int64 PLAIN key + MIN/MAX/AVG ------------------------ #

COLD_HASH_MINMAXAVG = """
    SELECT l_orderkey, min(l_quantity) AS qmin, max(l_quantity) AS qmax,
           avg(l_quantity) AS qavg
      FROM lineitem
     WHERE l_shipdate <= date '1998-09-02'
     GROUP BY l_orderkey
"""

COLD_HASH_MINMAX = """
    SELECT l_orderkey, min(l_quantity) AS qmin, max(l_quantity) AS qmax
      FROM lineitem
     WHERE l_shipdate <= date '1998-09-02'
     GROUP BY l_orderkey
"""


def _cold_match(a, b):
    pa, pb = (a.to_pandas() if hasattr(a, "to_pandas") else a), (
        b.to_pandas() if hasattr(b, "to_pandas") else b)
    assert list(pa.columns) == list(pb.columns)
    if len(pa) == 0:
        assert len(pb) == 0
        return
    pa = pa.sort_values("l_orderkey").reset_index(drop=True)
    pb = pb.sort_values("l_orderkey").reset_index(drop=True)
    assert len(pa) == len(pb)
    for c in pa.columns:
        x, y = pa[c].to_numpy(), pb[c].to_numpy()
        if x.dtype.kind in "iu":
            assert np.array_equal(x, y), f"col {c} int mismatch"
        else:
            m = np.isfinite(x) & np.isfinite(y)
            assert np.allclose(x[m], y[m], rtol=1e-6, atol=1e-2), f"col {c} float mismatch"
            assert np.sum(np.isnan(x)) == np.sum(np.isnan(y)), f"col {c} nan-count"


def test_cold_hash_minmax_fires_and_matches(typed_engine, typed_duck):
    """Cold HASH MIN/MAX over a single PLAIN int64 key fires (page_hash_kernel's
    per-slot atomic_min/max_d + host acc_init copy) and matches DuckDB."""
    if not CPP:
        pytest.skip("C++ fused kernel not built")
    typed_engine.clear_scan_cache()
    typed_engine.clear_code_cache()
    ryu = typed_engine.sql(COLD_HASH_MINMAX)
    _cold_match(ryu, typed_duck.execute(COLD_HASH_MINMAX).fetchdf())


def test_cold_hash_minmaxavg_matches(typed_engine, typed_duck):
    """Cold HASH MIN/MAX/AVG over a single PLAIN int64 key matches DuckDB (AVG's
    running-sum + hidden-count slots both atomicAdd; divided at read-out)."""
    if not CPP:
        pytest.skip("C++ fused kernel not built")
    typed_engine.clear_scan_cache()
    typed_engine.clear_code_cache()
    ryu = typed_engine.sql(COLD_HASH_MINMAXAVG)
    _cold_match(ryu, typed_duck.execute(COLD_HASH_MINMAXAVG).fetchdf())


def test_cold_multicol_defers_to_warm_matches(typed_engine, typed_duck):
    """A multi-col GROUP BY on the typed lineitem defers cold (no on-disk combined
    code) and is run by the warm `fused_aggregate` path; end-to-end matches DuckDB."""
    if not CPP:
        pytest.skip("C++ fused kernel not built")
    sql = """
        SELECT l_orderkey, l_quantity, min(l_extendedprice) AS emin, max(l_tax) AS tmax
          FROM lineitem
         WHERE l_shipdate <= date '1998-09-02'
         GROUP BY l_orderkey, l_quantity
    """
    assert fused.fused_scan_aggregate(_agg_node(sql, typed_engine), typed_engine) is None
    typed_engine.clear_scan_cache()
    typed_engine.clear_code_cache()
    assert_same(typed_engine.sql(sql), typed_duck.execute(sql).fetchdf())


# --- disable-extension fallback -------------------------------------------- #

def test_fallback_when_extension_disabled(hc_engine, hc_duck, monkeypatch):
    """With the C++ backend disabled, the new HASH shapes fall back to cuDF and
    still match DuckDB (correctness never depends on the extension)."""
    if not CPP:
        pytest.skip("C++ fused kernel not built")
    monkeypatch.setattr(fused._kernels, "is_available", False)
    sql = """
        SELECT l_orderkey, l_partkey, min(l_quantity) AS qmin, max(l_tax) AS tmax,
               avg(l_extendedprice) AS eavg, count(*) AS n
          FROM lineitem
         WHERE l_shipdate <= date '1998-09-02'
         GROUP BY l_orderkey, l_partkey
    """
    assert_same(hc_engine.sql(sql), hc_duck.execute(sql).fetchdf())


# --- end-to-end dispatch fires (executor-module-patching gotcha) ------------- #

def test_dispatch_fires_for_multicol_minmax(hc_engine, hc_duck, monkeypatch):
    """`engine.sql` actually dispatches a high-card multi-col+MIN/MAX query to the
    fused HASH path (not the cuDF fallback). Patches `ryudb.exec.executor.fused_aggregate`
    -- the name the executor imported (NOT `fused.fused_aggregate`) -- with a counter."""
    if not CPP:
        pytest.skip("C++ fused kernel not built")
    import ryudb.exec.executor as ex
    calls = {"n": 0}
    orig = ex.fused_aggregate

    def counting(node, child, engine):
        calls["n"] += 1
        return orig(node, child, engine)

    monkeypatch.setattr(ex, "fused_aggregate", counting)
    sql = """
        SELECT l_orderkey, l_partkey, min(l_quantity) AS qmin, max(l_tax) AS tmax,
               avg(l_extendedprice) AS eavg
          FROM lineitem
         WHERE l_shipdate <= date '1998-09-02'
         GROUP BY l_orderkey, l_partkey
    """
    assert_same(hc_engine.sql(sql), hc_duck.execute(sql).fetchdf())
    assert calls["n"] == 1, f"expected 1 fused_aggregate dispatch, got {calls['n']}"


def test_distinct_defers_dispatch(hc_engine, hc_duck, monkeypatch):
    """A DISTINCT-qualified aggregate forces the cuDF fallback at the executor
    (`_force_fallback` skips the fused call entirely) -> 0 fused dispatches, and
    the cuDF path (honouring af.distinct) matches DuckDB."""
    if not CPP:
        pytest.skip("C++ fused kernel not built")
    import ryudb.exec.executor as ex
    calls = {"n": 0}
    orig = ex.fused_aggregate

    def counting(node, child, engine):
        calls["n"] += 1
        return orig(node, child, engine)

    monkeypatch.setattr(ex, "fused_aggregate", counting)
    sql = """
        SELECT l_orderkey, sum(DISTINCT l_quantity) AS s
          FROM lineitem
         WHERE l_shipdate <= date '1998-09-02'
         GROUP BY l_orderkey
    """
    assert_same(hc_engine.sql(sql), hc_duck.execute(sql).fetchdf())
    assert calls["n"] == 0, "DISTINCT aggregate must force the cuDF fallback (0 fused dispatches)"


def test_per_agg_filter_defers_dispatch(hc_engine, hc_duck, monkeypatch):
    """A per-aggregate FILTER (WHERE ...) forces the cuDF fallback -> 0 fused
    dispatches, and the cuDF path (honouring the filter) matches DuckDB."""
    if not CPP:
        pytest.skip("C++ fused kernel not built")
    import ryudb.exec.executor as ex
    calls = {"n": 0}
    orig = ex.fused_aggregate

    def counting(node, child, engine):
        calls["n"] += 1
        return orig(node, child, engine)

    monkeypatch.setattr(ex, "fused_aggregate", counting)
    sql = """
        SELECT l_orderkey, sum(l_quantity) FILTER (WHERE l_quantity > 5) AS s
          FROM lineitem
         WHERE l_shipdate <= date '1998-09-02'
         GROUP BY l_orderkey
    """
    assert_same(hc_engine.sql(sql), hc_duck.execute(sql).fetchdf())
    assert calls["n"] == 0, "per-agg FILTER must force the cuDF fallback (0 fused dispatches)"


# --- CLI smoke -------------------------------------------------------------- #

def test_cli_high_card_minmax_smoke(typed_engine, typed_duck):
    """The CLI SQL surface returns the new HASH MIN/MAX shape matching DuckDB on
    the typed lineitem (ordered, limited) -- the exact bench `Q_high_card_minmax`
    shape over a single int64 key."""
    if not CPP:
        pytest.skip("C++ fused kernel not built")
    sql = """
        SELECT l_orderkey, min(l_quantity) AS min_qty, max(l_quantity) AS max_qty,
               avg(l_quantity) AS avg_qty
          FROM lineitem
         WHERE l_shipdate <= date '1998-09-02'
         GROUP BY l_orderkey
         ORDER BY l_orderkey
         LIMIT 5
    """
    ryu = typed_engine.sql(sql).to_pandas()
    duck = typed_duck.execute(sql).fetchdf()
    assert list(ryu.columns) == list(duck.columns)
    assert len(ryu) == 5
    for c in ryu.columns:
        x, y = ryu[c].to_numpy(), duck[c].to_numpy()
        if x.dtype.kind in "iu":
            assert np.array_equal(x, y)
        else:
            assert np.allclose(x, y, rtol=1e-6, atol=1e-2)