"""Phase 2 step 4: cache invalidation on commit (autocommit at INSERT time).

INSERT appends a cuDF batch to the per-table delta (step 3); the next SELECT
re-merges it in ``_scan``. The scan cache is base-only + live-merged, so it needs
no invalidation. But the engine's two maintained-fact caches are keyed only by
``(table, col)`` -- the data identity is NOT in the key -- so an INSERT makes
them silently stale:

  * ``_code_cache[(t, c)] = (codes, uniques)`` -- positional factorize codes,
    row-aligned to the pre-INSERT series length. A longer merged series reads
    them out of bounds; a new group-key value has no code.
  * ``_pk_cache[(t, c)] = bool`` -- nunique()==len(); a cached True survives a
    duplicate-PK INSERT and would let the fused star-join collapse joins.

``_insert`` drops ``(table, *)`` from both caches right after ``delta.append``
(the autocommit point). These tests prove: (1) the hook fires and scopes to the
INSERTed table only, (2) a stale ``_code_cache`` would give wrong results and the
invalidation fixes it (fused DENSE string GROUP BY + a new group key), and (3) a
stale ``_pk_cache`` would collapse a fused star-join and the invalidation fixes
it (duplicate dim PK). Correctness is checked against DuckDB.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import cudf
import duckdb

from ryudb import Catalog, Engine
from ryudb.exec import fused

from .conftest import _clear_wal, assert_same

CPP = fused._kernels.is_available


# --------------------------------------------------------------------- scoping


def test_invalidate_per_table_scopes_to_inserted_table(engine):
    """Populate _code_cache + _pk_cache for lineitem AND orders, INSERT into
    lineitem, and assert only lineitem's entries are evicted (orders stays)."""
    lt = engine._scan("lineitem", None)["l_orderkey"]
    od = engine._scan("orders", None)["o_orderkey"]
    engine.get_codes("lineitem", "l_orderkey", lt)
    engine.get_codes("orders", "o_orderkey", od)
    engine.is_unique_key("lineitem", "l_orderkey", lt)
    engine.is_unique_key("orders", "o_orderkey", od)
    assert ("lineitem", "l_orderkey") in engine._code_cache
    assert ("orders", "o_orderkey") in engine._code_cache
    assert ("lineitem", "l_orderkey") in engine._pk_cache
    assert ("orders", "o_orderkey") in engine._pk_cache

    engine.sql(
        "INSERT INTO lineitem (l_orderkey, l_quantity, l_extendedprice, l_shipdate) "
        "VALUES (999, 5.0, 50.0, date '1998-08-10')"
    )

    # The INSERTed table's maintained-fact caches are stale -> evicted.
    assert ("lineitem", "l_orderkey") not in engine._code_cache
    assert ("lineitem", "l_orderkey") not in engine._pk_cache
    # The untouched table's caches are still valid -> retained.
    assert ("orders", "o_orderkey") in engine._code_cache
    assert ("orders", "o_orderkey") in engine._pk_cache


# -------------------------------------------------- _code_cache end-to-end


HC_DENSE = """
    SELECT l_returnflag, l_linestatus, sum(l_quantity) AS sum_qty, count(*) AS n
      FROM lineitem
     WHERE l_shipdate <= date '1998-09-02'
     GROUP BY l_returnflag, l_linestatus
     ORDER BY l_returnflag, l_linestatus
"""


@pytest.fixture(scope="module")
def hc_dir(tmp_path_factory):
    """A lineitem with low-card string group keys (l_returnflag, l_linestatus)."""
    d = tmp_path_factory.mktemp("ryudb_ci_hc")
    (d / "lineitem").mkdir()
    rng = np.random.default_rng(7)
    n = 20000
    rows = {
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
    _clear_wal(hc_dir)  # module dir reused across tests -> start with empty WAL
    cat = Catalog(str(hc_dir))
    cat.register("lineitem", str(hc_dir / "lineitem"))
    return Engine(cat)


@pytest.fixture
def hc_duck(hc_dir) -> "duckdb.DuckDBPyConnection":
    con = duckdb.connect()
    con.execute(f"CREATE VIEW lineitem AS SELECT * FROM read_parquet('{hc_dir}/lineitem/*.parquet')")
    return con


def test_insert_evicts_code_cache_new_group_key(hc_engine, hc_duck):
    """A warm HC_DENSE populates _code_cache for the string group keys. INSERT a
    row with a NEW l_returnflag ('Z') that passes the WHERE filter; without
    invalidation the stale cached codes (length 20000) would be read against the
    20001-row merged series (OOB / 'Z' has no code) -> wrong grouping. With
    invalidation the codes are re-factorized on the merged series -> correct."""
    hc_engine.sql(HC_DENSE)  # warm: populates _code_cache via the fused DENSE path
    if ("lineitem", "l_returnflag") not in hc_engine._code_cache:
        pytest.skip("fused DENSE path did not populate _code_cache; cannot prove staleness")
    assert ("lineitem", "l_linestatus") in hc_engine._code_cache

    hc_duck.execute("CREATE TABLE lineitem_w AS SELECT * FROM lineitem")
    ins = (
        "INSERT INTO lineitem (l_returnflag, l_linestatus, l_quantity, l_shipdate) "
        "VALUES ('Z', 'O', 10.0, date '1998-08-01')"
    )
    hc_engine.sql(ins)
    hc_duck.execute(ins.replace("lineitem (l_returnflag, l_linestatus, l_quantity, l_shipdate)",
                                "lineitem_w (l_returnflag, l_linestatus, l_quantity, l_shipdate)"))

    # The INSERT evicted this table's _code_cache entries (checked before the
    # re-run, which would re-populate them with the corrected codes).
    assert ("lineitem", "l_returnflag") not in hc_engine._code_cache
    assert ("lineitem", "l_linestatus") not in hc_engine._code_cache

    ryu = hc_engine.sql(HC_DENSE)
    dft = hc_duck.execute(HC_DENSE.replace("lineitem", "lineitem_w")).fetchdf()
    assert_same(ryu, dft)


# --------------------------------------------------- _pk_cache end-to-end


STAR_SNOWFLAKE = """
    SELECT label, sum(f_val) AS revenue
      FROM F
      JOIN D1 ON f_key1 = d1_key
      JOIN D2 ON d1_next = d2_key
     GROUP BY label
     ORDER BY revenue DESC
"""


@pytest.fixture
def star_dir(tmp_path):
    """Function-scoped F->D1->D2 snowflake (mirrors tests/test_kernels.py so the
    fused star-join is eligible): D2(d2_key 0..4, label A..E), D1(d1_key 0..19),
    F(20000 rows). Fresh dir per test so the INSERT delta never leaks."""
    d = tmp_path
    rng = np.random.default_rng(11)
    d2 = cudf.DataFrame({
        "d2_key": np.arange(5, dtype=np.int64),
        "label": np.array(["A", "B", "C", "D", "E"], dtype=object),
    })
    d1 = cudf.DataFrame({
        "d1_key": np.arange(20, dtype=np.int64),
        "d1_next": rng.integers(0, 6, size=20).astype(np.int64),
    })
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


def test_insert_evicts_pk_cache_duplicate_dim_key(star_engine, star_duck):
    """A warm STAR_SNOWFLAKE populates _pk_cache[(D2,d2_key)]=True (the fused
    star-join's dim-PK eligibility gate). INSERT a DUPLICATE d2_key (0 already
    exists); without invalidation the stale True would let the fused star-join
    proceed on a non-unique dim key -> silently collapsed joins. With invalidation
    is_unique_key recomputes False on the merged D2 -> fused defers -> cuDF
    fallback -> correct."""
    if not CPP:
        pytest.skip("fused star-join is C++-only; without it the cuDF fallback is "
                    "always correct and never populates _pk_cache")
    star_engine.sql(STAR_SNOWFLAKE)  # warm: populates _pk_cache via fused_join_aggregate
    if ("D2", "d2_key") not in star_engine._pk_cache:
        pytest.skip("fused star-join did not populate _pk_cache; cannot prove staleness")
    assert star_engine._pk_cache[("D2", "d2_key")] is True

    # Writable oracle copies of every joined table.
    for t in ("F", "D1", "D2"):
        star_duck.execute(f"CREATE TABLE {t}_w AS SELECT * FROM {t}")
    ins = "INSERT INTO D2 (d2_key, label) VALUES (0, 'X')"
    star_engine.sql(ins)
    star_duck.execute("INSERT INTO D2_w (d2_key, label) VALUES (0, 'X')")

    # The INSERT evicted D2's _pk_cache entry (checked before the re-run, which
    # would recompute and cache False).
    assert ("D2", "d2_key") not in star_engine._pk_cache

    q_w = STAR_SNOWFLAKE.replace(" FROM F", " FROM F_w").replace("JOIN D1 ", "JOIN D1_w ") \
        .replace("JOIN D2 ", "JOIN D2_w ")
    ryu = star_engine.sql(STAR_SNOWFLAKE)
    dft = star_duck.execute(q_w).fetchdf()
    assert_same(ryu, dft)
    # Post re-run: is_unique_key recomputed on the merged (now non-unique) D2 key.
    assert star_engine._pk_cache.get(("D2", "d2_key")) is False