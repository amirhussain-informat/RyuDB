"""Phase 2 step 2: delta-store + scan merge seam.

The delta is empty by default, so reads are byte-identical to today. These tests
append batches directly to ``engine.delta`` (the seam step 3 will wire INSERTs
into) and assert the merge is correct: base ∪ delta, dtypes reconciled to the
base, projection respected, and the cold Parquet reader defers (so the delta is
visible even on the cache-miss cold path).

Uses the function-scoped ``engine`` / ``typed_engine`` fixtures so the in-memory
delta never pollutes the session ``data_dir``.
"""

from __future__ import annotations

import cudf
import pandas as pd
import pytest

from ryudb.delta import DeltaStore


# --------------------------------------------------------------------- empty delta

def test_delta_store_empty():
    d = DeltaStore()
    assert d.has_unflushed("t") is False
    assert d.batches("t") == []


def test_empty_delta_read_unchanged(engine):
    n = int(engine.sql("SELECT count(*) AS n FROM lineitem").to_pandas()["n"].iloc[0])
    assert n > 0
    assert engine.delta.has_unflushed("lineitem") is False


# ------------------------------------------------------------- merge visibility

def _lineitem_delta_frame(orderkey=999, quantity=1.0, extprice=10.0, shipdate="1998-08-10"):
    return cudf.DataFrame(
        {
            "l_orderkey": [orderkey],
            "l_quantity": [quantity],
            "l_extendedprice": [extprice],
            "l_shipdate": pd.Series(pd.to_datetime([shipdate])),
        }
    )


def test_append_count_visible(engine):
    base_n = int(engine.sql("SELECT count(*) AS n FROM lineitem").to_pandas()["n"].iloc[0])
    engine.delta.append("lineitem", _lineitem_delta_frame())
    assert engine.delta.has_unflushed("lineitem") is True
    n = int(engine.sql("SELECT count(*) AS n FROM lineitem").to_pandas()["n"].iloc[0])
    assert n == base_n + 1


def test_clear_delta_back_to_base(engine):
    base_n = int(engine.sql("SELECT count(*) AS n FROM lineitem").to_pandas()["n"].iloc[0])
    engine.delta.append("lineitem", _lineitem_delta_frame())
    assert engine.delta.has_unflushed("lineitem") is True
    engine.delta.clear("lineitem")
    assert engine.delta.has_unflushed("lineitem") is False
    n = int(engine.sql("SELECT count(*) AS n FROM lineitem").to_pandas()["n"].iloc[0])
    assert n == base_n


def test_projection_with_delta(engine):
    base_n = int(engine.sql("SELECT count(*) AS n FROM lineitem").to_pandas()["n"].iloc[0])
    engine.delta.append("lineitem", _lineitem_delta_frame())
    df = engine._scan("lineitem", {"l_orderkey", "l_quantity"})
    # projected + sorted column order, matching storage.scan
    assert list(df.columns) == ["l_orderkey", "l_quantity"]
    assert len(df) == base_n + 1


# --------------------------------------------------- datetime unit reconciliation

def test_merge_reconciles_datetime_units(engine):
    """Base [ms] (storage.scan) + delta [s] (cold-cache style) must concat."""
    base = cudf.DataFrame(
        {"d": pd.Series(pd.to_datetime(["1994-01-01", "1994-06-01"])).astype("datetime64[ms]")}
    )
    delta = cudf.DataFrame(
        {"d": pd.Series(pd.to_datetime(["1995-01-01"])).astype("datetime64[s]")}
    )
    # Only meaningful if the units actually differ; if cuDF normalized them,
    # the concat is trivially fine too.
    engine.delta.append("t", delta)
    merged = engine._merge_delta(base, "t")
    assert len(merged) == 3
    assert merged["d"].dtype == base["d"].dtype
    vals = list(merged["d"].to_pandas())
    assert vals[-1] == pd.Timestamp("1995-01-01")


# ----------------------------- cold-path guard: delta visible on a cold miss

def test_filtered_agg_with_delta_cold_guard(typed_engine):
    """A Q6-shaped agg on the typed lineitem takes the cold Parquet reader on a
    cache miss. With a delta appended, the cold guard must defer to the
    materialising _scan + merge so the delta row is visible. The cold reader
    bypasses _scan, so a *correct* result here proves the guard fired."""
    q = (
        "SELECT sum(l_extendedprice) AS s, count(*) AS n FROM lineitem "
        "WHERE l_shipdate >= date '1994-01-01' AND l_shipdate < date '1995-01-01' "
        "AND l_discount >= 0.05 AND l_discount <= 0.07"
    )
    typed_engine.clear_scan_cache()
    base = typed_engine.sql(q).to_pandas()
    n_base = int(base["n"].iloc[0])
    s_base = float(base["s"].iloc[0])
    assert n_base > 0

    # Append one row that matches the predicate (shipdate in 1994, discount 0.06).
    delta = cudf.DataFrame(
        {
            "l_orderkey": [999999],
            "l_quantity": [1.0],
            "l_extendedprice": [1234.56],
            "l_discount": [0.06],
            "l_tax": [0.01],
            "l_shipdate": pd.Series(pd.to_datetime(["1994-06-15"])),
        }
    )
    typed_engine.delta.append("lineitem", delta)
    # Force a cache miss so _aggregate attempts the cold reader (which must defer).
    typed_engine.clear_scan_cache()
    res = typed_engine.sql(q).to_pandas()
    n_with = int(res["n"].iloc[0])
    s_with = float(res["s"].iloc[0])
    assert n_with == n_base + 1
    assert abs((s_with - s_base) - 1234.56) < 1e-1


def test_delta_not_cached_under_base_key(engine):
    """Appending after a base-cached read must still be visible (live re-merge,
    not a stale cached merged frame)."""
    # Prime the scan cache with the base frame.
    engine.sql("SELECT count(*) AS n FROM lineitem")
    base_n = int(engine.sql("SELECT count(*) AS n FROM lineitem").to_pandas()["n"].iloc[0])
    engine.delta.append("lineitem", _lineitem_delta_frame())
    # The cache key for `SELECT count(*)` is the base-only frame; the merge must
    # still apply on top of the cached base.
    n = int(engine.sql("SELECT count(*) AS n FROM lineitem").to_pandas()["n"].iloc[0])
    assert n == base_n + 1
    # And the base cache entry itself must remain base-only (no double-merge).
    key = ("lineitem", None)
    cached = engine._scan_cache.get(key)
    if cached is not None and not hasattr(cached, "pending_id"):
        assert len(cached) == base_n