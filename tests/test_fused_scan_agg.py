"""End-to-end tests for the cold Parquet scan+filter+aggregate path (Phase 5).

`fused_scan_aggregate` runs `Aggregate -> Filter -> Scan` straight off the
Parquet pages (nvCOMP Snappy decompress -> decode -> filter -> accumulate) and
returns None for any shape it can't handle, leaving the existing materialising
path as the correctness safety net. These tests verify, on a typed
DuckDB-written lineitem (decimals as INT64, dates as INT32, Snappy -- the bench
layout):

  * Q6, scan_agg_full (global aggregate, DENSE n_groups=1) and high_card
    (GROUP BY a single plain int64 key, HASH) match DuckDB.
  * The cold path is actually *taken* for the eligible shapes (the
    `fused_scan_agg` kernel is dispatched, not deferred to the warm path).
  * A cold run (scan cache + code index cleared) still matches -- the cold
    reader never touches the scan/code caches, so eviction must not change the
    result.

Deferred shapes (dict-string group keys -> Q1, multi-key GROUP BY, AVG/MIN/MAX
over a HASH key) return None and fall back; those are covered by the existing
`test_fused.py` suite. Decimal money is compared with the same tolerance the
bench uses (sums differ in low-order float bits).
"""

from __future__ import annotations

import numpy as np
import pytest

from ryudb.exec import fused

Q6 = """
    SELECT sum(l_extendedprice * l_discount) AS revenue, count(*) AS n
      FROM lineitem
     WHERE l_shipdate >= date '1994-01-01'
       AND l_shipdate < date '1995-01-01'
       AND l_discount >= 0.05 AND l_discount <= 0.07
       AND l_quantity < 24
"""

SCAN_AGG = """
    SELECT count(*) AS n, sum(l_extendedprice) AS s,
           min(l_discount) AS md, max(l_tax) AS mt
      FROM lineitem WHERE l_quantity > 25
"""

HIGH_CARD = """
    SELECT l_orderkey, sum(l_quantity) AS sum_qty,
           sum(l_extendedprice) AS sum_base_price, count(*) AS count_order
      FROM lineitem
     WHERE l_shipdate <= date '1998-09-02'
     GROUP BY l_orderkey
"""

# A shape the v1 cold reader must defer: two group keys (multi-key GROUP BY).
MULTI_KEY = """
    SELECT l_orderkey, l_quantity, count(*) AS n
      FROM lineitem
     WHERE l_shipdate <= date '1998-09-02'
     GROUP BY l_orderkey, l_quantity
"""


def _pdf(df):
    return df.to_pandas() if hasattr(df, "to_pandas") else df


@pytest.fixture(autouse=True)
def _enable_scan_kernel(monkeypatch):
    """The cold Parquet reader is opt-in (RYUDB_SCAN_KERNEL) because at SF10 it
    is currently slower than the cuDF materialising fallback. These tests
    exercise the reader itself, so enable it for the whole module; the
    multi-key test then proves deferral *under* the enabled flag."""
    monkeypatch.setenv("RYUDB_SCAN_KERNEL", "1")


def _match(a, b):
    pa, pb = _pdf(a), _pdf(b)
    assert list(pa.columns) == list(pb.columns), f"cols {list(pa.columns)} != {list(pb.columns)}"
    if len(pa) == 0:
        assert len(pb) == 0
        return
    # Align by the group key if present (order is not guaranteed across engines).
    if "l_orderkey" in pa.columns:
        pa = pa.sort_values("l_orderkey").reset_index(drop=True)
        pb = pb.sort_values("l_orderkey").reset_index(drop=True)
    assert len(pa) == len(pb), f"len {len(pa)} != {len(pb)}"
    for c in pa.columns:
        x, y = pa[c].to_numpy(), pb[c].to_numpy()
        if x.dtype.kind in "iu":
            assert np.array_equal(x, y), f"col {c} int mismatch"
        else:
            m = np.isfinite(x) & np.isfinite(y)
            assert np.allclose(x[m], y[m], rtol=1e-6, atol=1e-2), f"col {c} float mismatch"
            assert np.sum(np.isnan(x)) == np.sum(np.isnan(y)), f"col {c} nan-count mismatch"


@pytest.fixture
def counting_kernel():
    """Wrap fused_scan_agg to count dispatches; restore on teardown."""
    if not fused._kernels.is_available:
        pytest.skip("C++ fused kernel not built")
    orig = fused._kernels.fused_scan_agg
    calls = {"n": 0}

    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    fused._kernels.fused_scan_agg = counting
    try:
        yield calls
    finally:
        fused._kernels.fused_scan_agg = orig


@pytest.mark.parametrize("sql,name", [(Q6, "Q6"), (SCAN_AGG, "scan_agg"), (HIGH_CARD, "high_card")])
def test_scan_agg_matches_duckdb(typed_engine, typed_duck, sql, name):
    typed_engine.clear_scan_cache()
    typed_engine.clear_code_cache()
    ryu = typed_engine.sql(sql)
    duck = typed_duck.execute(sql).fetchdf()
    _match(ryu, duck)


@pytest.mark.parametrize("sql,name", [(Q6, "Q6"), (HIGH_CARD, "high_card")])
def test_scan_agg_path_is_taken(typed_engine, typed_duck, counting_kernel, sql, name):
    """The eligible shapes must dispatch to fused_scan_agg (cold path), not defer
    to the warm materialising path, and still match DuckDB."""
    typed_engine.clear_scan_cache()
    typed_engine.clear_code_cache()
    ryu = typed_engine.sql(sql)
    assert counting_kernel["n"] == 1, f"{name}: expected 1 fused_scan_agg dispatch, got {counting_kernel['n']}"
    duck = typed_duck.execute(sql).fetchdf()
    _match(ryu, duck)


@pytest.mark.parametrize("sql,name", [(Q6, "Q6"), (SCAN_AGG, "scan_agg"), (HIGH_CARD, "high_card")])
def test_scan_agg_cold_eviction_matches(typed_engine, typed_duck, sql, name):
    """A cold run (both caches cleared right before the query) must match DuckDB:
    the cold reader works off the plan and never touches the scan/code caches, so
    eviction cannot change the result."""
    typed_engine.clear_scan_cache()
    typed_engine.clear_code_cache()
    ryu = typed_engine.sql(sql)
    duck = typed_duck.execute(sql).fetchdf()
    _match(ryu, duck)


def test_multi_key_groupby_defers(typed_engine, typed_duck, counting_kernel):
    """A multi-key GROUP BY is v1-deferred: fused_scan_agg is NOT dispatched and
    the warm cuDF path produces the correct result."""
    typed_engine.clear_scan_cache()
    typed_engine.clear_code_cache()
    ryu = typed_engine.sql(MULTI_KEY)
    assert counting_kernel["n"] == 0, "multi-key GROUP BY should defer, not take the cold path"
    duck = typed_duck.execute(MULTI_KEY).fetchdf()
    _match(ryu, duck)