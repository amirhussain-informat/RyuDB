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
        "l_suppkey": rng.integers(1, 1001, size=n_li).astype(np.int64),
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


# =========================================================================== #
# PR #52: fold a cross-table WHERE (Filter under the Aggregate) on the join
# path. The optimizer pushes single-table WHERE conjuncts to Filter -> Scan
# under the join tree; fused_join_aggregate collects those, folds fact
# predicates into the kernel's pass_pred and dim predicates into a pre-HT-build
# dim frame filter (the last/group-key dim is factorised directly off the
# filtered frame -- engine.get_codes caches by (table,col) over the full scan).
# OR / cross-table-compound / outer-plan-dim predicates still defer.
# =========================================================================== #

# Fact predicate (l_quantity < 50, on the streamed lineitem) over the high-card
# GROUP BY o_orderkey join -> pass_pred, HASH (NDV*nagg > MAX_ACC_CELLS).
WHERE_FACT = """
    SELECT o_orderkey, sum(l_quantity) AS sq, count(*) AS n
      FROM lineitem
      JOIN orders ON l_orderkey = o_orderkey
     WHERE l_quantity < 50
     GROUP BY o_orderkey
     ORDER BY o_orderkey
"""

# Dim predicate on the LAST (group-key) dim (orders), filtered to ~7327 groups
# -> pre-HT-build dim frame filter + last-dim factorise-off-filtered-frame.
WHERE_DIM_LAST = """
    SELECT o_orderkey, sum(l_quantity) AS sq, count(*) AS n
      FROM lineitem
      JOIN orders ON l_orderkey = o_orderkey
     WHERE o_custkey < 500
     GROUP BY o_orderkey
     ORDER BY o_orderkey
"""

# Fact + last-dim predicate together.
WHERE_FACT_AND_DIM_LAST = """
    SELECT o_orderkey, sum(l_quantity) AS sq, count(*) AS n
      FROM lineitem
      JOIN orders ON l_orderkey = o_orderkey
     WHERE l_quantity < 50 AND o_custkey < 500
     GROUP BY o_orderkey
     ORDER BY o_orderkey
"""

# Dim predicate on a MIDDLE dim (orders), group key on the LAST dim (customer)
# -> the bridging payload is read straight off the filtered frame (no get_codes).
WHERE_DIM_MIDDLE = """
    SELECT c_custkey, sum(l_quantity) AS sq, count(*) AS n
      FROM lineitem
      JOIN orders   ON l_orderkey = o_orderkey
      JOIN customer ON o_custkey = c_custkey
     WHERE o_custkey < 1000
     GROUP BY c_custkey
     ORDER BY c_custkey
"""

# Fact + middle-dim predicate, group key on the last dim, DENSE.
WHERE_FACT_AND_DIM_MIDDLE = """
    SELECT c_custkey, sum(l_extendedprice) AS rev, count(*) AS n
      FROM lineitem
      JOIN orders   ON l_orderkey = o_orderkey
      JOIN customer ON o_custkey = c_custkey
     WHERE l_quantity < 24 AND o_custkey < 800
     GROUP BY c_custkey
     ORDER BY c_custkey
"""

# String dim predicate on the last dim (customer.c_name equality) -> the
# eval_expr string path + last-dim factorise-off-filtered-frame.
WHERE_STRING_DIM = """
    SELECT c_custkey, sum(l_extendedprice) AS rev, count(*) AS n
      FROM lineitem
      JOIN orders   ON l_orderkey = o_orderkey
      JOIN customer ON o_custkey = c_custkey
     WHERE c_name = 'cust_42'
     GROUP BY c_custkey
"""

# OR predicate -> defer (the kernel + this fold are conjunctive only).
WHERE_OR = """
    SELECT o_orderkey, sum(l_quantity) AS sq
      FROM lineitem
      JOIN orders ON l_orderkey = o_orderkey
     WHERE l_quantity < 5 OR l_quantity > 90
     GROUP BY o_orderkey
     ORDER BY o_orderkey
"""

# LEFT join + a dim predicate on the preserved dim -> defer (degrades to inner;
# the kernel null-pads at a LEFT stage, so dim-predicate folding is inner-only).
WHERE_LEFT_DIM = """
    SELECT o_orderkey, sum(l_quantity) AS sq, count(*) AS n
      FROM lineitem
      LEFT JOIN orders ON l_orderkey = o_orderkey
     WHERE o_custkey < 500
     GROUP BY o_orderkey
     ORDER BY o_orderkey
"""


@pytest.mark.parametrize("sql", [WHERE_FACT, WHERE_DIM_LAST, WHERE_FACT_AND_DIM_LAST,
                                 WHERE_DIM_MIDDLE, WHERE_FACT_AND_DIM_MIDDLE,
                                 WHERE_STRING_DIM])
def test_join_where_predicate_fires_and_matches(hcj_engine, hcj_duck, sql):
    """Each foldable WHERE shape fires the fused join path and matches DuckDB."""
    if not CPP:
        pytest.skip("C++ fused kernel not built")
    agg = _agg_node(sql, hcj_engine)
    assert fused.fused_join_aggregate(agg, hcj_engine) is not None, (
        f"a foldable WHERE should hit the fused join path: {sql!r}")
    _match(hcj_engine.sql(sql), hcj_duck.execute(sql).fetchdf())


def test_join_where_or_predicate_defers_and_matches(hcj_engine, hcj_duck):
    """An OR predicate defers (returns None) and the cuDF fallback matches DuckDB."""
    if not CPP:
        pytest.skip("C++ fused kernel not built")
    assert fused.fused_join_aggregate(_agg_node(WHERE_OR, hcj_engine), hcj_engine) is None, (
        "an OR WHERE should defer to cuDF")
    _match(hcj_engine.sql(WHERE_OR), hcj_duck.execute(WHERE_OR).fetchdf())


def test_join_where_dim_predicate_outer_defers(hcj_engine, hcj_duck):
    """A dim predicate on a LEFT (outer) join defers; the cuDF fallback matches."""
    if not CPP:
        pytest.skip("C++ fused kernel not built")
    assert fused.fused_join_aggregate(_agg_node(WHERE_LEFT_DIM, hcj_engine), hcj_engine) is None, (
        "a dim predicate on an outer join should defer to cuDF")
    _match(hcj_engine.sql(WHERE_LEFT_DIM), hcj_duck.execute(WHERE_LEFT_DIM).fetchdf())


def test_join_where_dispatch_fires(hcj_engine, hcj_duck, monkeypatch):
    """engine.sql dispatches a WHERE-bearing join to the fused path (not cuDF):
    patch ryudb.exec.executor.fused_join_aggregate with a counter, assert one call,
    and match DuckDB. Covers the executor's Filter-branch routing (the
    executor-module-patching gotcha: patch the executor's imported name)."""
    if not CPP:
        pytest.skip("C++ fused kernel not built")
    calls = {"n": 0}
    orig = ex_mod.fused_join_aggregate

    def wrap(node, engine, _o=orig):
        r = _o(node, engine)
        calls["n"] += 1
        return r

    monkeypatch.setattr(ex_mod, "fused_join_aggregate", wrap)
    got = hcj_engine.sql(WHERE_FACT_AND_DIM_LAST)
    assert calls["n"] == 1, "a WHERE-bearing join should dispatch to the fused path once"
    _match(got, hcj_duck.execute(WHERE_FACT_AND_DIM_LAST).fetchdf())


# =========================================================================== #
# PR #35: group-key-in-fact on the fused join path. The single GROUP BY key
# lives on the streamed FACT table (not a reached dim). The kernel reads the
# group code from a fact column (group_key_col) instead of the chain tail: the
# fact key is factorised on the host into dense 0..n-1 codes, bound as a NEW fact
# col (NOT the raw key -- the kernel reads cols[group_key_col][i] as the dense
# code, g = fcode*stride + tail), and the last dim's payload is a ZERO array so
# tail=0 -> g = fcode (stride=1). This is CASE 2 of the unified group_key_col /
# group_stride kernel mechanism; the existing chain-tail path (CASE 1,
# group_key_col=-1) is byte-identical. Outer plans with the group key on the
# fact still defer (inner-only v1).
# =========================================================================== #

# Group key on the fact AND it is the join key (l_orderkey), no WHERE. Every
# lineitem row matches orders, so the join does not filter; the group is the
# raw l_orderkey factorised on the fact.
FACT_KEY_NO_WHERE = """
    SELECT l_orderkey, sum(l_quantity) AS sq, sum(l_extendedprice) AS rev,
           count(*) AS n
      FROM lineitem
      JOIN orders ON l_orderkey = o_orderkey
     GROUP BY l_orderkey
     ORDER BY l_orderkey
"""

# Group key on the fact, join kept by a DIM predicate (o_custkey < 500 on
# orders) -- the headline: a fact-key group-by that previously deferred because
# the group key was not a reached dim.
FACT_KEY_WHERE_DIM = """
    SELECT l_orderkey, sum(l_quantity) AS sq, count(*) AS n
      FROM lineitem
      JOIN orders ON l_orderkey = o_orderkey
     WHERE o_custkey < 500
     GROUP BY l_orderkey
     ORDER BY l_orderkey
"""

# Group key on the fact + a FACT predicate (l_quantity < 50 -> pass_pred).
FACT_KEY_WHERE_FACT = """
    SELECT l_orderkey, sum(l_quantity) AS sq, count(*) AS n
      FROM lineitem
      JOIN orders ON l_orderkey = o_orderkey
     WHERE l_quantity < 50
     GROUP BY l_orderkey
     ORDER BY l_orderkey
"""

# Group key on the fact + both a fact and a dim predicate.
FACT_KEY_WHERE_FACT_AND_DIM = """
    SELECT l_orderkey, sum(l_quantity) AS sq, count(*) AS n
      FROM lineitem
      JOIN orders ON l_orderkey = o_orderkey
     WHERE l_quantity < 50 AND o_custkey < 500
     GROUP BY l_orderkey
     ORDER BY l_orderkey
"""

# Group key on the fact that is NOT the join key (l_suppkey) -- exercises the
# separate group_key_col binding (the probe key l_orderkey is distinct from
# the group key l_suppkey). No WHERE.
FACT_KEY_NON_JOIN_KEY = """
    SELECT l_suppkey, sum(l_quantity) AS sq, count(*) AS n
      FROM lineitem
      JOIN orders ON l_orderkey = o_orderkey
     GROUP BY l_suppkey
     ORDER BY l_suppkey
"""

# Group key on the fact (non-join key) + a fact and a dim predicate.
FACT_KEY_NON_JOIN_KEY_WHERE = """
    SELECT l_suppkey, sum(l_quantity) AS sq, count(*) AS n
      FROM lineitem
      JOIN orders ON l_orderkey = o_orderkey
     WHERE l_quantity < 50 AND o_custkey < 500
     GROUP BY l_suppkey
     ORDER BY l_suppkey
"""

# Group key on the fact over a MULTI-STAGE chain (lineitem -> orders ->
# customer); the path is the FULL chain (no group-key dim). A predicate on the
# middle dim (orders) keeps the join.
FACT_KEY_CHAIN_WHERE_MIDDLE = """
    SELECT l_orderkey, sum(l_extendedprice) AS rev, count(*) AS n
      FROM lineitem
      JOIN orders   ON l_orderkey = o_orderkey
      JOIN customer ON o_custkey = c_custkey
     WHERE o_custkey < 1000
     GROUP BY l_orderkey
     ORDER BY l_orderkey
"""

# LEFT join + group key on the fact -> defer (group-key-on-fact outer is
# inner-only v1; a LEFT miss should still group by the fact key, subtler).
FACT_KEY_LEFT_DEFERS = """
    SELECT l_orderkey, sum(l_quantity) AS sq, count(*) AS n
      FROM lineitem
      LEFT JOIN orders ON l_orderkey = o_orderkey
     GROUP BY l_orderkey
     ORDER BY l_orderkey
"""


@pytest.mark.parametrize("sql", [FACT_KEY_NO_WHERE, FACT_KEY_WHERE_DIM,
                                 FACT_KEY_WHERE_FACT, FACT_KEY_WHERE_FACT_AND_DIM,
                                 FACT_KEY_NON_JOIN_KEY, FACT_KEY_NON_JOIN_KEY_WHERE,
                                 FACT_KEY_CHAIN_WHERE_MIDDLE])
def test_join_fact_key_fires_and_matches(hcj_engine, hcj_duck, sql):
    """A GROUP BY a fact-table column fires the fused join path (the group code
    read from a fact column, not the chain tail) and matches DuckDB."""
    if not CPP:
        pytest.skip("C++ fused kernel not built")
    agg = _agg_node(sql, hcj_engine)
    assert fused.fused_join_aggregate(agg, hcj_engine) is not None, (
        f"a fact-key group-from-join should hit the fused join path: {sql!r}")
    _match(hcj_engine.sql(sql), hcj_duck.execute(sql).fetchdf())


def test_join_fact_key_left_defers_and_matches(hcj_engine, hcj_duck):
    """A LEFT (outer) join with the group key on the fact defers (inner-only v1);
    the cuDF fallback matches DuckDB."""
    if not CPP:
        pytest.skip("C++ fused kernel not built")
    assert fused.fused_join_aggregate(
        _agg_node(FACT_KEY_LEFT_DEFERS, hcj_engine), hcj_engine) is None, (
        "a fact-key group-from-join on an outer plan should defer to cuDF")
    _match(hcj_engine.sql(FACT_KEY_LEFT_DEFERS),
           hcj_duck.execute(FACT_KEY_LEFT_DEFERS).fetchdf())


def test_join_fact_key_dispatch_fires(hcj_engine, hcj_duck, monkeypatch):
    """engine.sql dispatches a fact-key group-from-join to the fused path (not
    cuDF): patch ryudb.exec.executor.fused_join_aggregate with a counter, assert
    one call, and match DuckDB (executor-module-patching gotcha)."""
    if not CPP:
        pytest.skip("C++ fused kernel not built")
    calls = {"n": 0}
    orig = ex_mod.fused_join_aggregate

    def wrap(node, engine, _o=orig):
        r = _o(node, engine)
        calls["n"] += 1
        return r

    monkeypatch.setattr(ex_mod, "fused_join_aggregate", wrap)
    got = hcj_engine.sql(FACT_KEY_WHERE_FACT_AND_DIM)
    assert calls["n"] == 1, "a fact-key group-from-join should dispatch once"
    _match(got, hcj_duck.execute(FACT_KEY_WHERE_FACT_AND_DIM).fetchdf())