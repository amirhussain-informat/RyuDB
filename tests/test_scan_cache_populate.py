"""Phase 5 step 3: dropping the RYUDB_SCAN_KERNEL opt-in gate.

The cold Parquet scan path (`fused_scan_aggregate`) now populates
`Engine._scan_cache` from its already-decoded GPU buffers via the
`materialise_kernel`, so warm repeats hit the GPU-resident frame instead of
re-reading Parquet (which was the warm-regression blocker). The scan path is
the DEFAULT cold path now (taken on a `_scan_cache` miss). These tests verify:

  * a cold Q6-shape run dispatches the scan path AND caches the frame under the
    SAME key `_scan` / the executor's cache-miss guard use;
  * the cached frame's columns (sorted projection) and per-column values match
    `storage.scan` -- decimals are float64 in both, the date column is compared
    unit-agnostic (normalised to int64 seconds; the cached frame is
    datetime64[s], storage.scan's cuDF parquet default may be [ms]);
  * a warm re-run hits the cache (the scan path is NOT redispatched) and still
    matches DuckDB.

Correctness never depends on the C++ extension: if the kernel is not built the
tests skip; on any kernel fault `fused_scan_aggregate` returns None and the
materialising path (which also caches via `_scan`) carries correctness.
"""

from __future__ import annotations

import numpy as np
import pytest

from ryudb.exec import fused
from ryudb.storage import scan

Q6 = """
    SELECT sum(l_extendedprice * l_discount) AS revenue, count(*) AS n
      FROM lineitem
     WHERE l_shipdate >= date '1994-01-01'
       AND l_shipdate < date '1995-01-01'
       AND l_discount >= 0.05 AND l_discount <= 0.07
       AND l_quantity < 24
"""

PROJ = {"l_quantity", "l_extendedprice", "l_discount", "l_shipdate"}


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


def _norm_date(s) -> np.ndarray:
    """Normalise a datetime column to int64 seconds for unit-agnostic compare."""
    return s.astype("datetime64[s]").astype("int64").to_numpy()


def test_cold_scan_populates_cache(typed_engine, counting_kernel):
    typed_engine.clear_scan_cache()
    typed_engine.clear_code_cache()

    typed_engine.sql(Q6)  # cold: scan path runs and populates the cache
    assert counting_kernel["n"] == 1, "cold run should dispatch the scan path once"

    key = ("lineitem", frozenset(PROJ))
    assert key in typed_engine._scan_cache, "cold scan did not populate _scan_cache"

    cached = typed_engine._scan_cache[key]
    assert list(cached.columns) == sorted(PROJ), f"cols {list(cached.columns)}"

    ref = scan(typed_engine.catalog.get("lineitem"), PROJ)
    assert list(ref.columns) == sorted(PROJ)
    assert len(cached) == len(ref), f"row count {len(cached)} != {len(ref)}"

    for c in sorted(PROJ):
        if c == "l_shipdate":
            assert np.array_equal(_norm_date(cached[c]), _norm_date(ref[c])), f"date col {c} mismatch"
        else:
            assert str(cached[c].dtype) == "float64", f"{c}: expected float64, got {cached[c].dtype}"
            a, b = cached[c].to_numpy(), ref[c].to_numpy()
            assert np.allclose(a, b, rtol=1e-9, atol=1e-6), f"col {c} value mismatch"


def test_warm_rerun_hits_cache(typed_engine, counting_kernel, typed_duck):
    typed_engine.clear_scan_cache()
    typed_engine.clear_code_cache()

    typed_engine.sql(Q6)  # cold -> dispatches scan path, caches the frame
    assert counting_kernel["n"] == 1
    counting_kernel["n"] = 0

    ryu = typed_engine.sql(Q6)  # warm -> cache hit, scan path NOT redispatched
    assert counting_kernel["n"] == 0, "warm rerun should hit the cache, not redispatch the scan path"

    duck = typed_duck.execute(Q6).fetchdf()
    pa, pb = ryu.to_pandas(), duck
    assert list(pa.columns) == list(pb.columns)
    assert int(pa["n"].iloc[0]) == int(pb["n"].iloc[0]), "row-count mismatch"
    assert np.isclose(float(pa["revenue"].iloc[0]), float(pb["revenue"].iloc[0]),
                      rtol=1e-6, atol=1e-2), "revenue mismatch"