"""Shared pytest fixtures: a small Parquet dataset + RyuDB and DuckDB engines.

DuckDB is the correctness oracle: the same SQL is run on both and the sorted
result frames are compared. The dataset mirrors a tiny TPC-H-like schema so the
join/aggregate paths are exercised.
"""

from __future__ import annotations


import os

import cudf
import duckdb
import pandas as pd
import pytest

from ryudb import Catalog, Engine


def _clear_wal(data_dir) -> None:
    """Remove a leftover ``ryudb.wal`` from the data dir so each test starts with
    an empty WAL. Needed only for fixtures whose Engine is function-scoped over
    a session/module-scoped dir: the WAL persists committed INSERTs to that shared
    dir, so without a clear, test A's commits would replay into test B's fresh
    Engine. The catalog file is idempotent so it does NOT need clearing."""
    wal = os.path.join(str(data_dir), "ryudb.wal")
    if os.path.exists(wal):
        os.remove(wal)

# A small but non-trivial dataset. Row counts are intentionally modest so tests
# run in well under a second on the GPU.
_ORDERS = [
    (1, 10, 100.0, "1998-08-01"),
    (2, 20, 200.0, "1998-09-01"),
    (3, 10, 50.0, "1998-07-01"),
    (4, 30, 300.0, "1998-10-01"),
    (5, 20, 75.0, "1998-09-15"),
]
_LINEITEM = [
    (1, 5.0, 50.0, "1998-08-10"),
    (1, 10.0, 60.0, "1998-09-20"),
    (2, 2.0, 30.0, "1998-08-30"),
    (3, 1.0, 10.0, "1998-07-15"),
    (3, 1.0, 20.0, "1998-07-16"),
    (3, 1.0, 5.0, "1998-08-01"),
    (4, 7.0, 90.0, "1998-10-05"),
    (5, 3.0, 75.0, "1998-09-30"),
]
_NATION = [
    (10, "USA"),
    (20, "CANADA"),
    (30, "MEXICO"),
]


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("ryudb_data")
    (d / "orders").mkdir()
    (d / "lineitem").mkdir()
    (d / "nation").mkdir()

    cudf.DataFrame(
        {
            "o_orderkey": [r[0] for r in _ORDERS],
            "o_custkey": [r[1] for r in _ORDERS],
            "o_totalprice": [r[2] for r in _ORDERS],
            "o_orderdate": pd.to_datetime([r[3] for r in _ORDERS]),
        }
    ).to_pandas().to_parquet(d / "orders" / "o.parquet")

    cudf.DataFrame(
        {
            "l_orderkey": [r[0] for r in _LINEITEM],
            "l_quantity": [r[1] for r in _LINEITEM],
            "l_extendedprice": [r[2] for r in _LINEITEM],
            "l_shipdate": pd.to_datetime([r[3] for r in _LINEITEM]),
        }
    ).to_pandas().to_parquet(d / "lineitem" / "l.parquet")

    cudf.DataFrame(
        {
            "n_nationkey": [r[0] for r in _NATION],
            "n_name": [r[1] for r in _NATION],
        }
    ).to_pandas().to_parquet(d / "nation" / "n.parquet")

    return d


@pytest.fixture
def engine(data_dir) -> Engine:
    _clear_wal(data_dir)  # session dir reused across tests -> start each with empty WAL
    cat = Catalog(str(data_dir))
    cat.register("orders", str(data_dir / "orders"))
    cat.register("lineitem", str(data_dir / "lineitem"))
    cat.register("nation", str(data_dir / "nation"))
    return Engine(cat)


# A typed TPC-H lineitem written the same way the bench generates data
# (`bench/run_bench.py`): DuckDB `COPY ... TO parquet`, which stores
# DECIMAL(15,2) as INT64, DATE as INT32, with Snappy compression -- the exact
# on-disk layout the Phase 5 cold Parquet reader (fused_scan_agg) targets. The
# float-typed `lineitem` above is fine for the executor/fused-kernel tests but
# its FLOAT columns make the cold reader defer (it only reads INT physical
# types), so the page-decode + scan-agg tests need this typed copy. ~130k rows
# span DuckDB's default 122,880-row group boundary -> 2 row groups, so the
# per-row-group launch loop and the page-header parser run across >1 group.
# Value ranges are chosen so the Q6 / scan_agg / high-card predicates return
# non-trivial (non-empty, non-all) results.
_TYPED_LINEITEM_ROWS = 130000


@pytest.fixture(scope="session")
def typed_lineitem_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("ryudb_typed")
    (d / "lineitem").mkdir()
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE lineitem (l_orderkey BIGINT, l_quantity DECIMAL(15,2), "
        "l_extendedprice DECIMAL(15,2), l_discount DECIMAL(15,2), "
        "l_tax DECIMAL(15,2), l_shipdate DATE)"
    )
    # Deterministic synthetic rows: orderkey is unique (i+1) so DuckDB stores
    # it PLAIN -- the v1 HASH group-key path reads a raw int64 at values_off and
    # does not gather a dict index. quantity 1..50, discount 0.04..0.07, tax
    # 0.01..0.04, shipdate across 1994-1995+ so the Q6 date window matches a
    # real subset.
    con.execute(
        "INSERT INTO lineitem "
        "SELECT i + 1, ((i % 50) + 1)::DECIMAL(15,2), "
        "((i % 50) + 1) * 10::DECIMAL(15,2), "
        "(0.04 + (i % 4) * 0.01)::DECIMAL(15,2), "
        "(0.01 + (i % 4) * 0.01)::DECIMAL(15,2), "
        "date '1994-01-01' + ((i % 730)::INTEGER) FROM range(130000) t(i)"
    )
    con.execute(
        f"COPY (SELECT * FROM lineitem) TO '{d}/lineitem/0.parquet' "
        "(FORMAT PARQUET, COMPRESSION 'snappy')"
    )
    con.close()
    return d


@pytest.fixture
def typed_engine(typed_lineitem_dir) -> Engine:
    _clear_wal(typed_lineitem_dir)  # session dir reused -> start with empty WAL
    cat = Catalog(str(typed_lineitem_dir))
    cat.register("lineitem", str(typed_lineitem_dir / "lineitem"))
    return Engine(cat)


@pytest.fixture
def typed_duck(typed_lineitem_dir) -> "duckdb.DuckDBPyConnection":
    con = duckdb.connect()
    con.execute(
        f"CREATE VIEW lineitem AS SELECT * FROM read_parquet('{typed_lineitem_dir}/lineitem/*.parquet')"
    )
    return con


@pytest.fixture
def duck(data_dir) -> "duckdb.DuckDBPyConnection":
    con = duckdb.connect()
    con.execute(f"CREATE VIEW orders AS SELECT * FROM read_parquet('{data_dir}/orders/*.parquet')")
    con.execute(f"CREATE VIEW lineitem AS SELECT * FROM read_parquet('{data_dir}/lineitem/*.parquet')")
    con.execute(f"CREATE VIEW nation AS SELECT * FROM read_parquet('{data_dir}/nation/*.parquet')")
    return con


def as_sorted(df) -> list[tuple]:
    """Normalize a frame to a sorted list of tuples for comparison."""
    import pandas as pd

    pdf = df.to_pandas() if hasattr(df, "to_pandas") and not isinstance(df, pd.DataFrame) else df
    if len(pdf) == 0:
        return []
    pdf = pdf.sort_values(list(pdf.columns)).reset_index(drop=True)
    rows = []
    for _, row in pdf.iterrows():
        rows.append(tuple(_clean(v) for v in row))
    return rows


def _clean(v):
    """Round floats for stable comparison across GPU/CPU float differences."""
    if isinstance(v, float):
        return round(v, 6)
    return v


def assert_same(ryu_df, duck_df):
    r = as_sorted(ryu_df)
    d = as_sorted(duck_df)
    assert r == d, f"RyuDB != DuckDB\n ryu={r}\n duck={d}"