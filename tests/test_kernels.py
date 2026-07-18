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


# --- Phase 2: fused LEFT-outer aggregate-over-join -------------------------- #
#
# Reuses the star snowflake (F -> D1 -> D2, GROUP BY D2.label) but with LEFT
# joins: a fact row that misses D1 (f_key1 in 20..24) OR whose D1 row misses D2
# (d1_next == 5) is NOT dropped -- it null-pads to a NULL group (label NULL).
# The fused kernel routes a miss at a LEFT stage to group slot 0 (the NULL
# group); a pure-inner stage still drops. Comparison is via conftest.as_sorted
# (NULL -> None) because _match casts to float and cannot hold a NULL string key.

LEFT_SNOWFLAKE = """
    SELECT label, sum(f_val) AS revenue, count(*) AS n
      FROM F
      LEFT JOIN D1 ON f_key1 = d1_key
      LEFT JOIN D2 ON d1_next = d2_key
     GROUP BY label
     ORDER BY revenue DESC
"""

LEFT_SNOWFLAKE_INNER = """
    SELECT label, sum(f_val) AS revenue, count(*) AS n
      FROM F
      JOIN D1 ON f_key1 = d1_key
      JOIN D2 ON d1_next = d2_key
     GROUP BY label
     ORDER BY revenue DESC
"""


def _sorted(df):
    from .conftest import as_sorted
    return as_sorted(df)


def test_fused_left_outer_agg_hits_kernel(star_engine):
    """The LEFT-outer snowflake-agg is now in fused scope (was inner-only): the
    gate accepts how=left/right and returns a frame instead of None."""
    if not CPP:
        pytest.skip("C++ fused kernel not built")
    agg = _agg_node(LEFT_SNOWFLAKE, star_engine)
    assert fused.fused_join_aggregate(agg, star_engine) is not None, (
        "LEFT-outer snowflake + SUM/COUNT should hit the C++ fused path")


def test_fused_left_outer_agg_matches_duckdb(star_engine, star_duck):
    """LEFT-outer snowflake-agg through the fused kernel matches DuckDB, including
    the NULL group (label NULL) for fact rows that miss D1 or D2 -- the rows the
    inner kernel used to drop."""
    if not CPP:
        pytest.skip("C++ fused kernel not built")
    assert fused.fused_join_aggregate(_agg_node(LEFT_SNOWFLAKE, star_engine), star_engine) is not None
    ryu = _sorted(star_engine.sql(LEFT_SNOWFLAKE))
    duk = _sorted(star_duck.execute(LEFT_SNOWFLAKE).fetchdf())
    assert ryu == duk
    # Misses exist in the fixture (f_key1 20..24, d1_next 5) -> the NULL group is
    # present; guards against the kernel silently dropping unmatched fact rows.
    assert any(row[0] is None for row in ryu), "expected a NULL-group row"


def test_fused_left_outer_differs_from_inner(star_engine, star_duck):
    """The NULL group is exactly what distinguishes LEFT from INNER: LEFT has a
    NULL-label row inner lacks. Both match DuckDB. This catches a regression to
    the old inner-only drop semantics."""
    if not CPP:
        pytest.skip("C++ fused kernel not built")
    left = _sorted(star_engine.sql(LEFT_SNOWFLAKE))
    inner = _sorted(star_engine.sql(LEFT_SNOWFLAKE_INNER))
    assert left != inner
    assert any(row[0] is None for row in left)
    assert not any(row[0] is None for row in inner)
    assert left == _sorted(star_duck.execute(LEFT_SNOWFLAKE).fetchdf())
    assert inner == _sorted(star_duck.execute(LEFT_SNOWFLAKE_INNER).fetchdf())


@pytest.fixture(scope="module")
def star_nomiss_dir(tmp_path_factory):
    """All-matching snowflake: every fact key hits D1 and every D1 payload hits
    D2, so a LEFT-outer produces NO NULL group -- the output must be identical to
    the inner-join aggregate (the seen[0]==0 skip guard for slot 0)."""
    d = tmp_path_factory.mktemp("ryudb_star_nomiss")
    d2 = cudf.DataFrame({
        "d2_key": np.arange(5, dtype=np.int64),
        "label": np.array(["A", "B", "C", "D", "E"], dtype=object),
    })
    d1 = cudf.DataFrame({
        "d1_key": np.arange(20, dtype=np.int64),
        "d1_next": np.arange(20, dtype=np.int64) % 5,  # 0..4 -> all hit D2
    })
    f = cudf.DataFrame({
        "f_key1": np.arange(20000, dtype=np.int64) % 20,  # 0..19 -> all hit D1
        "f_val": np.arange(20000, dtype=np.float64),
    })
    for name, fr in [("D2", d2), ("D1", d1), ("F", f)]:
        (d / name).mkdir()
        fr.to_pandas().to_parquet(d / name / "0.parquet")
    return d


@pytest.fixture
def nomiss_engine(star_nomiss_dir) -> Engine:
    cat = Catalog(str(star_nomiss_dir))
    for t in ("F", "D1", "D2"):
        cat.register(t, str(star_nomiss_dir / t))
    return Engine(cat)


@pytest.fixture
def nomiss_duck(star_nomiss_dir):
    con = duckdb.connect()
    for t in ("F", "D1", "D2"):
        con.execute(f"CREATE VIEW {t} AS SELECT * FROM read_parquet('{star_nomiss_dir}/{t}/*.parquet')")
    return con


LEFT_NOMISS = """
    SELECT label, sum(f_val) AS revenue, count(*) AS n
      FROM F
      LEFT JOIN D1 ON f_key1 = d1_key
      LEFT JOIN D2 ON d1_next = d2_key
     GROUP BY label
     ORDER BY label
"""


def test_fused_left_outer_no_miss_identical_to_inner(nomiss_engine, nomiss_duck):
    """Zero misses -> no NULL group -> LEFT-outer fused output equals the inner
    aggregate. The NULL slot (0) is allocated but not emitted (seen[0]==0), so
    the +1 payload offset and the extra slot must not perturb the real groups --
    the inner/star-snowflake regression guard."""
    if not CPP:
        pytest.skip("C++ fused kernel not built")
    assert fused.fused_join_aggregate(_agg_node(LEFT_NOMISS, nomiss_engine), nomiss_engine) is not None
    left = _sorted(nomiss_engine.sql(LEFT_NOMISS))
    inner = _sorted(nomiss_duck.execute(LEFT_NOMISS.replace("LEFT JOIN", "JOIN")).fetchdf())
    assert left == inner
    assert not any(row[0] is None for row in left)


def test_fused_left_outer_fallback_matches(star_engine, star_duck, monkeypatch):
    """With the C++ kernel marked unavailable, the LEFT-outer aggregate falls back
    to the cuDF merge+groupby path and still matches DuckDB -- correctness never
    depends on the extension. (Patching _kernels.is_available makes the gate's
    `if not _kernels.is_available: return None` fire at call time.)"""
    monkeypatch.setattr(fused._kernels, "is_available", False)
    ryu = _sorted(star_engine.sql(LEFT_SNOWFLAKE))
    duk = _sorted(star_duck.execute(LEFT_SNOWFLAKE).fetchdf())
    assert ryu == duk


def test_stale_kernel_guard():
    """The loader refuses a fused.so older than its sources: a stale binary after
    an ABI change would feed wrong descriptors to the kernel (CUDA context
    poison). _stale() reads file mtimes live; is_available is pinned at import."""
    import os
    import time
    from ryudb import kernels
    if not kernels.is_available:
        pytest.skip("C++ fused kernel not built")
    assert not kernels._stale(), "freshly built .so must not read stale"
    so_mtime = kernels._EXT.stat().st_mtime
    try:
        old = so_mtime - 3600
        os.utime(kernels._EXT, (old, old))
        assert kernels._stale(), "a .so older than its .cu must read stale"
    finally:
        now = time.time()
        os.utime(kernels._EXT, (now, now))


# --- Phase 3: fused FULL-outer aggregate-over-join -------------------------- #
#
# FULL = LEFT (a fact miss null-pads to the NULL group) + dim-only groups (a dim
# row no fact row hit -> COUNT=1, SUM=NULL, emitted by the host readout). This
# 2-table fixture exercises both halves: D 40..49 are dim-only (no fact hits
# them); f_key 50..59 are fact-only (NULL group); D 0..39 are matched. Scope is
# single-stage (fact FULL JOIN dim GROUP BY dim.k); a multi-stage FULL chain
# defers to cuDF. Comparison via conftest.as_sorted (NULL -> None) because _match
# casts to float and cannot hold a NULL string group key.


@pytest.fixture(scope="module")
def full_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("ryudb_full")
    # D: d_key 0..49, label L0..L49.
    dfr = cudf.DataFrame({
        "d_key": np.arange(50, dtype=np.int64),
        "label": np.array([f"L{i}" for i in range(50)], dtype=object),
    })
    # F: f_key in {0..39 (matched) , 50..59 (fact-only -> NULL group)}; D 40..49
    # never appear in F -> dim-only.
    fkeys = np.concatenate([np.arange(40), np.arange(50, 60)]).astype(np.int64)
    ffr = cudf.DataFrame({
        "f_key": fkeys,
        "f_val": np.arange(fkeys.size, dtype=np.float64) + 1.0,
    })
    for name, fr in [("D", dfr), ("F", ffr)]:
        (d / name).mkdir()
        fr.to_pandas().to_parquet(d / name / "0.parquet")
    return d


@pytest.fixture
def full_engine(full_dir) -> Engine:
    cat = Catalog(str(full_dir))
    for t in ("F", "D"):
        cat.register(t, str(full_dir / t))
    return Engine(cat)


@pytest.fixture
def full_duck(full_dir):
    con = duckdb.connect()
    for t in ("F", "D"):
        con.execute(f"CREATE VIEW {t} AS SELECT * FROM read_parquet('{full_dir}/{t}/*.parquet')")
    return con


FULL_AGG = """
    SELECT label, sum(f_val) AS revenue, count(*) AS n
      FROM F
      FULL JOIN D ON f_key = d_key
     GROUP BY label
     ORDER BY label
"""

FULL_AGG_LEFT = FULL_AGG.replace("FULL JOIN", "LEFT JOIN")
FULL_AGG_INNER = FULL_AGG.replace("FULL JOIN", "JOIN")


def test_fused_full_outer_agg_hits_kernel(full_engine):
    """The single-stage FULL-outer-agg is in fused scope: the gate accepts
    how=full and returns a frame instead of None."""
    if not CPP:
        pytest.skip("C++ fused kernel not built")
    agg = _agg_node(FULL_AGG, full_engine)
    assert fused.fused_join_aggregate(agg, full_engine) is not None, (
        "single-stage FULL-outer + SUM/COUNT should hit the C++ fused path")


def test_fused_full_outer_agg_matches_duckdb(full_engine, full_duck):
    """FULL-outer-agg through the fused kernel matches DuckDB, including the NULL
    group (fact-only f_key 50..59) and the dim-only groups (D 40..49 with
    COUNT=1, SUM=NULL)."""
    if not CPP:
        pytest.skip("C++ fused kernel not built")
    assert fused.fused_join_aggregate(_agg_node(FULL_AGG, full_engine), full_engine) is not None
    ryu = _sorted(full_engine.sql(FULL_AGG))
    duk = _sorted(full_duck.execute(FULL_AGG).fetchdf())
    assert ryu == duk
    # Both halves of FULL are present: a NULL group (fact-only) and dim-only rows.
    assert any(row[0] is None for row in ryu), "expected a NULL-group row (fact-only)"
    # Dim-only D 40..49: COUNT(*)=1, SUM=NULL. They appear as (Lxx, None, 1).
    dim_only = [r for r in ryu if r[0] in {f"L{i}" for i in range(40, 50)}]
    assert dim_only, "expected dim-only rows (D 40..49 unmatched by any fact)"
    assert all(r[1] is None and r[2] == 1 for r in dim_only), (
        "dim-only rows must have SUM=NULL, COUNT=1")


def test_fused_full_outer_differs_from_left(full_engine, full_duck):
    """FULL differs from LEFT exactly by the dim-only groups (D 40..49): LEFT
    lacks them, FULL has them. Both match DuckDB. Catches a regression that
    forgets the host-side anti-dim emission."""
    if not CPP:
        pytest.skip("C++ fused kernel not built")
    full = _sorted(full_engine.sql(FULL_AGG))
    left = _sorted(full_engine.sql(FULL_AGG_LEFT))
    assert full != left
    # FULL has the dim-only labels LEFT lacks.
    left_labels = {r[0] for r in left}
    assert {f"L{i}" for i in range(40, 50)} <= {r[0] for r in full} - left_labels
    assert full == _sorted(full_duck.execute(FULL_AGG).fetchdf())
    assert left == _sorted(full_duck.execute(FULL_AGG_LEFT).fetchdf())


@pytest.fixture(scope="module")
def full_nomiss_dir(tmp_path_factory):
    """All-matching FULL: every D key is hit by a fact and every fact key hits a
    D row -- no NULL group, no dim-only rows -> FULL == INNER."""
    d = tmp_path_factory.mktemp("ryudb_full_nomiss")
    dfr = cudf.DataFrame({
        "d_key": np.arange(20, dtype=np.int64),
        "label": np.array([f"L{i}" for i in range(20)], dtype=object),
    })
    # f_key covers every D key (0..19); nothing on either side is unmatched.
    ffr = cudf.DataFrame({
        "f_key": np.arange(20000, dtype=np.int64) % 20,
        "f_val": np.arange(20000, dtype=np.float64),
    })
    for name, fr in [("D", dfr), ("F", ffr)]:
        (d / name).mkdir()
        fr.to_pandas().to_parquet(d / name / "0.parquet")
    return d


@pytest.fixture
def full_nomiss_engine(full_nomiss_dir) -> Engine:
    cat = Catalog(str(full_nomiss_dir))
    for t in ("F", "D"):
        cat.register(t, str(full_nomiss_dir / t))
    return Engine(cat)


@pytest.fixture
def full_nomiss_duck(full_nomiss_dir):
    con = duckdb.connect()
    for t in ("F", "D"):
        con.execute(f"CREATE VIEW {t} AS SELECT * FROM read_parquet('{full_nomiss_dir}/{t}/*.parquet')")
    return con


FULL_NOMISS = """
    SELECT label, sum(f_val) AS revenue, count(*) AS n
      FROM F
      FULL JOIN D ON f_key = d_key
     GROUP BY label
     ORDER BY label
"""


def test_fused_full_outer_no_miss_identical_to_inner(full_nomiss_engine, full_nomiss_duck):
    """Zero misses on either side -> no NULL group and no dim-only rows -> the
    FULL-outer fused output equals the inner aggregate. Guards that the +1
    payload offset and the host anti-dim pass do not perturb the matched groups
    when nothing is appended."""
    if not CPP:
        pytest.skip("C++ fused kernel not built")
    assert fused.fused_join_aggregate(_agg_node(FULL_NOMISS, full_nomiss_engine),
                                      full_nomiss_engine) is not None
    full = _sorted(full_nomiss_engine.sql(FULL_NOMISS))
    inner = _sorted(full_nomiss_duck.execute(FULL_NOMISS.replace("FULL JOIN", "JOIN")).fetchdf())
    assert full == inner
    assert not any(row[0] is None for row in full)


def test_fused_full_multi_stage_defers(star_engine, star_duck):
    """A multi-stage FULL-outer chain (F FULL JOIN D1 FULL JOIN D2) defers to
    cuDF: the dim-only semantics across a chain are out of the single-stage scope
    and `fused_join_aggregate` returns None. The cuDF path still matches DuckDB."""
    if not CPP:
        pytest.skip("C++ fused kernel not built")
    sql = """
        SELECT label, sum(f_val) AS revenue, count(*) AS n
          FROM F
          FULL JOIN D1 ON f_key1 = d1_key
          FULL JOIN D2 ON d1_next = d2_key
         GROUP BY label
         ORDER BY label
    """
    agg = _agg_node(sql, star_engine)
    assert fused.fused_join_aggregate(agg, star_engine) is None, (
        "multi-stage FULL-outer-agg must defer to cuDF")
    assert _sorted(star_engine.sql(sql)) == _sorted(star_duck.execute(sql).fetchdf())


def test_fused_full_outer_fallback_matches(full_engine, full_duck, monkeypatch):
    """With the C++ kernel marked unavailable, the FULL-outer aggregate falls
    back to the cuDF merge+groupby path and still matches DuckDB -- correctness
    never depends on the extension."""
    monkeypatch.setattr(fused._kernels, "is_available", False)
    ryu = _sorted(full_engine.sql(FULL_AGG))
    duk = _sorted(full_duck.execute(FULL_AGG).fetchdf())
    assert ryu == duk