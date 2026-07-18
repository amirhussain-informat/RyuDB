"""Phase 2 step 6: WAL + recovery.

Each committed delta batch is persisted to ``<data_dir>/ryudb.wal`` (one record
per commit, fsync'd before the in-memory delta mutates), so ``commit_ts`` doubles
as the WAL LSN. On ``Engine`` startup the WAL is replayed to reconstruct the
in-memory delta and reset ``_commit_ts`` to the highest replayed LSN. These
tests build a fresh ``Engine`` on the SAME data dir after writes to prove the
committed state survives a restart -- the durability contract -- plus the
atomicity (txn COMMIT is one record; ROLLBACK writes nothing), restore-time WAL
truncation, torn-tail CRC recovery, the disabled-WAL path, and dtype fidelity
across the Arrow-IPC serialization round-trip. Correctness is checked against a
writable DuckDB oracle.

The fixtures here intentionally do NOT clear the WAL (unlike the shared ``engine``
/ ``typed_engine`` fixtures, which clear it so a session-scoped dir stays
isolated across tests) -- persistence across two Engine constructions is the
thing under test, so each test gets its own function-scoped ``tmp_path`` dir.
"""

from __future__ import annotations

import os

import cudf
import duckdb
import pandas as pd
import pytest

from ryudb import Catalog, Engine
from ryudb.wal import WAL, wal_path

from .conftest import assert_same

# A tiny lineitem + orders so the base ++ delta merge is exercised (not just an
# empty-base all-delta path). Cols span int / float / datetime / string so the
# Arrow-IPC dtype round-trip is covered.
_LINEITEM = [
    (1, 5.0, 50.0, "1998-08-10", "a"),
    (2, 2.0, 30.0, "1998-08-30", "b"),
    (3, 1.0, 10.0, "1998-07-15", "a"),
]
_ORDERS = [
    (1, 10, 100.0, "1998-08-01"),
    (2, 20, 200.0, "1998-09-01"),
]


def _write_base(d) -> None:
    (d / "lineitem").mkdir()
    (d / "orders").mkdir()
    cudf.DataFrame(
        {
            "l_orderkey": [r[0] for r in _LINEITEM],
            "l_quantity": [r[1] for r in _LINEITEM],
            "l_extendedprice": [r[2] for r in _LINEITEM],
            "l_shipdate": pd.to_datetime([r[3] for r in _LINEITEM]),
            "l_comment": [r[4] for r in _LINEITEM],
        }
    ).to_pandas().to_parquet(d / "lineitem" / "l.parquet")
    cudf.DataFrame(
        {
            "o_orderkey": [r[0] for r in _ORDERS],
            "o_custkey": [r[1] for r in _ORDERS],
            "o_totalprice": [r[2] for r in _ORDERS],
            "o_orderdate": pd.to_datetime([r[3] for r in _ORDERS]),
        }
    ).to_pandas().to_parquet(d / "orders" / "o.parquet")


@pytest.fixture
def wal_dir(tmp_path) -> str:
    """A fresh data dir with base parquet, NOT cleared between engines (so the
    WAL persists across Engine constructions within a test)."""
    _write_base(tmp_path)
    return str(tmp_path)


def _engine(wal_dir: str) -> Engine:
    cat = Catalog(wal_dir)
    cat.register("lineitem", os.path.join(wal_dir, "lineitem"))
    cat.register("orders", os.path.join(wal_dir, "orders"))
    return Engine(cat)


def _writable_duck(wal_dir: str, tables=("lineitem", "orders")) -> "duckdb.DuckDBPyConnection":
    """A writable DuckDB oracle mirroring the parquet tables (for INSERT replay)."""
    con = duckdb.connect()
    for t in tables:
        con.execute(f"CREATE TABLE {t}_w AS SELECT * FROM read_parquet('{wal_dir}/{t}/*.parquet')")
    return con


_LI_COLS = "l_orderkey, l_quantity, l_extendedprice, l_shipdate, l_comment"


# ----------------------------------------------------------- autocommit replay


def test_autocommit_replay(wal_dir):
    eng = _engine(wal_dir)
    eng.sql(
        "INSERT INTO lineitem (l_orderkey, l_quantity, l_extendedprice, l_shipdate, l_comment) "
        "VALUES (999, 5.0, 50.0, date '1998-08-10', 'z')"
    )
    base = int(eng.sql("SELECT count(*) AS n FROM lineitem").to_pandas()["n"].iloc[0])
    assert base == len(_LINEITEM) + 1

    # New engine on the SAME dir -> the committed row survives a restart.
    eng2 = _engine(wal_dir)
    n = int(eng2.sql("SELECT count(*) AS n FROM lineitem").to_pandas()["n"].iloc[0])
    assert n == len(_LINEITEM) + 1
    row = eng2.sql("SELECT * FROM lineitem WHERE l_orderkey = 999").to_pandas()
    assert len(row) == 1
    assert int(row["l_orderkey"].iloc[0]) == 999
    assert float(row["l_quantity"].iloc[0]) == 5.0
    assert row["l_comment"].iloc[0] == "z"


def test_replay_sets_commit_ts(wal_dir):
    eng = _engine(wal_dir)
    # Three autocommit INSERTs -> commit_ts advances 1,2,3.
    for k in (10, 11, 12):
        eng.sql(
            "INSERT INTO lineitem (l_orderkey, l_quantity, l_extendedprice, l_shipdate, l_comment) "
            f"VALUES ({k}, 1.0, 1.0, date '1998-01-01', 'a')"
        )
    assert eng._commit_ts == 3

    eng2 = _engine(wal_dir)
    assert eng2._commit_ts == 3  # recovered to the max replayed LSN

    # A subsequent commit continues monotonically from the recovered LSN (no
    # collision with replayed ts).
    eng2.sql(
        "INSERT INTO lineitem (l_orderkey, l_quantity, l_extendedprice, l_shipdate, l_comment) "
        "VALUES (13, 1.0, 1.0, date '1998-01-02', 'b')"
    )
    assert eng2._commit_ts == 4

    eng3 = _engine(wal_dir)
    n = int(eng3.sql("SELECT count(*) AS n FROM lineitem").to_pandas()["n"].iloc[0])
    assert n == len(_LINEITEM) + 4  # three prior + the post-recovery insert
    assert eng3._commit_ts == 4


# ---------------------------------------------------------- txn commit atomicity


def test_txn_commit_atomic_replay(wal_dir):
    eng = _engine(wal_dir)
    eng.sql("BEGIN")
    eng.sql(
        "INSERT INTO lineitem (l_orderkey, l_quantity, l_extendedprice, l_shipdate, l_comment) "
        "VALUES (100, 1.0, 10.0, date '1998-01-01', 'x')"
    )
    eng.sql(
        "INSERT INTO lineitem (l_orderkey, l_quantity, l_extendedprice, l_shipdate, l_comment) "
        "VALUES (101, 2.0, 20.0, date '1998-01-02', 'y')"
    )
    eng.sql(
        "INSERT INTO orders (o_orderkey, o_custkey, o_totalprice, o_orderdate) "
        "VALUES (50, 5, 500.0, date '1998-01-03')"
    )
    eng.sql("COMMIT")

    # One WAL record for the whole commit -> all three rows survive atomically.
    eng2 = _engine(wal_dir)
    assert int(eng2.sql("SELECT count(*) AS n FROM lineitem").to_pandas()["n"].iloc[0]) == len(_LINEITEM) + 2
    assert int(eng2.sql("SELECT count(*) AS n FROM orders").to_pandas()["n"].iloc[0]) == len(_ORDERS) + 1


def test_rollback_not_persisted(wal_dir):
    eng = _engine(wal_dir)
    # One autocommit first so the WAL file exists with one record; the rollback
    # must NOT grow it (the buffer is never written).
    eng.sql(
        "INSERT INTO lineitem (l_orderkey, l_quantity, l_extendedprice, l_shipdate, l_comment) "
        "VALUES (8, 1.0, 1.0, date '1998-01-01', 'keep')"
    )
    assert os.path.exists(wal_path(wal_dir))
    wal_size_before = os.path.getsize(wal_path(wal_dir))

    eng.sql("BEGIN")
    eng.sql(
        "INSERT INTO lineitem (l_orderkey, l_quantity, l_extendedprice, l_shipdate, l_comment) "
        "VALUES (777, 1.0, 1.0, date '1998-01-01', 'r')"
    )
    eng.sql("ROLLBACK")
    # ROLLBACK writes nothing to the WAL (the buffer never touched the disk).
    assert os.path.getsize(wal_path(wal_dir)) == wal_size_before

    eng2 = _engine(wal_dir)
    # Only the autocommit row survived; the rolled-back row is gone.
    assert int(eng2.sql("SELECT count(*) AS n FROM lineitem").to_pandas()["n"].iloc[0]) == len(_LINEITEM) + 1
    assert eng2.sql("SELECT * FROM lineitem WHERE l_orderkey = 777").to_pandas().empty
    assert eng2.sql("SELECT * FROM lineitem WHERE l_orderkey = 8").to_pandas().shape[0] == 1


# -------------------------------------------------------- restore truncates WAL


def test_restore_truncates_wal(wal_dir):
    eng = _engine(wal_dir)
    eng.sql(
        "INSERT INTO lineitem (l_orderkey, l_quantity, l_extendedprice, l_shipdate, l_comment) "
        "VALUES (1, 1.0, 1.0, date '1998-01-01', 'keep')"  # l_orderkey=1 duplicates base; fine
    )
    eng.snapshot("s")  # captures commit_ts == 1
    eng.sql(
        "INSERT INTO orders (o_orderkey, o_custkey, o_totalprice, o_orderdate) "
        "VALUES (60, 6, 60.0, date '1998-01-04')"
    )
    assert eng._commit_ts == 2
    eng.restore("s")  # discard the orders insert (ts=2); durable WAL truncate

    # On-disk WAL: only the ts=1 record survives.
    recs = WAL(wal_path(wal_dir)).replay()
    assert [r[0] for r in recs] == [1]
    assert all(r[1] == "lineitem" for r in recs)

    # A restart sees the restored state: the orders insert is gone.
    eng2 = _engine(wal_dir)
    assert eng2._commit_ts == 1
    assert int(eng2.sql("SELECT count(*) AS n FROM orders").to_pandas()["n"].iloc[0]) == len(_ORDERS)
    assert int(eng2.sql("SELECT count(*) AS n FROM lineitem").to_pandas()["n"].iloc[0]) == len(_LINEITEM) + 1


def test_torn_tail_truncated(wal_dir):
    eng = _engine(wal_dir)
    eng.sql(
        "INSERT INTO lineitem (l_orderkey, l_quantity, l_extendedprice, l_shipdate, l_comment) "
        "VALUES (5, 5.0, 5.0, date '1998-01-05', 't')"
    )
    good_size = os.path.getsize(wal_path(wal_dir))

    # Simulate a crash mid-write: append garbage that looks like a torn record.
    with open(wal_path(wal_dir), "ab") as fh:
        fh.write(b"\xff" * 37)

    eng2 = _engine(wal_dir)
    # The good record survived; the garbage tail was dropped + file truncated.
    assert int(eng2.sql("SELECT count(*) AS n FROM lineitem").to_pandas()["n"].iloc[0]) == len(_LINEITEM) + 1
    assert os.path.getsize(wal_path(wal_dir)) == good_size  # garbage trimmed

    # Idempotent: a second replay on the now-clean file keeps the same size.
    eng3 = _engine(wal_dir)
    assert os.path.getsize(wal_path(wal_dir)) == good_size
    assert int(eng3.sql("SELECT count(*) AS n FROM lineitem").to_pandas()["n"].iloc[0]) == len(_LINEITEM) + 1


# ----------------------------------------------------------- disabled WAL path


def test_wal_disabled_without_data_dir(wal_dir):
    cat = Catalog(None)  # no data dir -> WAL disabled, catalog not persisted
    cat.register("lineitem", os.path.join(wal_dir, "lineitem"))
    eng = Engine(cat)
    assert eng._wal.path is None
    eng.sql(
        "INSERT INTO lineitem (l_orderkey, l_quantity, l_extendedprice, l_shipdate, l_comment) "
        "VALUES (42, 4.0, 4.0, date '1998-04-04', 'n')"
    )
    assert int(eng.sql("SELECT count(*) AS n FROM lineitem").to_pandas()["n"].iloc[0]) == len(_LINEITEM) + 1

    # No WAL file was ever created (the disabled WAL writes nowhere).
    assert not os.path.exists(os.path.join(os.getcwd(), "ryudb.wal"))

    # A second ephemeral engine re-registers from parquet -> no replay (WAL off).
    cat2 = Catalog(None)
    cat2.register("lineitem", os.path.join(wal_dir, "lineitem"))
    eng2 = Engine(cat2)
    assert int(eng2.sql("SELECT count(*) AS n FROM lineitem").to_pandas()["n"].iloc[0]) == len(_LINEITEM)


# ------------------------------------------------------------ dtype fidelity


@pytest.fixture
def typed_dir(tmp_path) -> str:
    """A small DECIMAL/DATE/BIGINT lineitem written via DuckDB COPY -- the exact
    on-disk layout the cold reader targets -- for the dtype round-trip test."""
    d = tmp_path / "lineitem"
    d.mkdir()
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE lineitem (l_orderkey BIGINT, l_quantity DECIMAL(15,2), "
        "l_extendedprice DECIMAL(15,2), l_shipdate DATE)"
    )
    con.execute(
        "INSERT INTO lineitem SELECT i+1, ((i%50)+1)::DECIMAL(15,2), "
        "(((i%50)+1)*10)::DECIMAL(15,2), date '1994-01-01' + (i%730)::INTEGER "
        "FROM range(50) t(i)"
    )
    con.execute(
        f"COPY (SELECT * FROM lineitem) TO '{d}/0.parquet' (FORMAT PARQUET, COMPRESSION 'snappy')"
    )
    con.close()
    return str(tmp_path)


def test_dtype_fidelity_typed(typed_dir):
    cat = Catalog(typed_dir)
    cat.register("lineitem", os.path.join(typed_dir, "lineitem"))
    eng = Engine(cat)
    duck = duckdb.connect()
    duck.execute(f"CREATE TABLE lineitem_w AS SELECT * FROM read_parquet('{typed_dir}/lineitem/*.parquet')")

    # Insert a row exercising DECIMAL->float64 and DATE round-trip.
    eng.sql(
        "INSERT INTO lineitem (l_orderkey, l_quantity, l_extendedprice, l_shipdate) "
        "VALUES (1001, 37.5, 375.0, date '1995-06-15')"
    )
    duck.execute(
        "INSERT INTO lineitem_w VALUES (1001, 37.5, 375.0, date '1995-06-15')"
    )

    sql = (
        "SELECT l_orderkey, l_quantity, l_extendedprice, l_shipdate FROM lineitem "
        "ORDER BY l_orderkey"
    )
    assert_same(eng.sql(sql), duck.execute(f"{sql.replace('lineitem', 'lineitem_w')}").df())

    # After a restart the replayed row casts cleanly through _merge_delta.
    cat2 = Catalog(typed_dir)
    cat2.register("lineitem", os.path.join(typed_dir, "lineitem"))
    eng2 = Engine(cat2)
    assert_same(eng2.sql(sql), duck.execute(f"{sql.replace('lineitem', 'lineitem_w')}").df())


# ----------------------------------------------------- multi-commit multi-table


def test_multi_commit_multi_table_replay(wal_dir):
    eng = _engine(wal_dir)
    duck = _writable_duck(wal_dir)
    stmts = [
        ("lineitem", f"INSERT INTO lineitem ({_LI_COLS}) VALUES (10, 1.0, 10.0, date '1998-01-10', 'a')"),
        ("orders", "INSERT INTO orders (o_orderkey, o_custkey, o_totalprice, o_orderdate) "
                   "VALUES (10, 1, 10.0, date '1998-01-10')"),
        ("lineitem", f"INSERT INTO lineitem ({_LI_COLS}) VALUES (11, 2.0, 20.0, date '1998-01-11', 'b')"),
        ("orders", "INSERT INTO orders (o_orderkey, o_custkey, o_totalprice, o_orderdate) "
                   "VALUES (11, 2, 20.0, date '1998-01-11')"),
        ("lineitem", f"INSERT INTO lineitem ({_LI_COLS}) VALUES (12, 3.0, 30.0, date '1998-01-12', 'c')"),
    ]
    for _, s in stmts:
        eng.sql(s)
    for t, s in stmts:
        duck.execute(s.replace(f"INTO {t} ", f"INTO {t}_w "))

    eng2 = _engine(wal_dir)
    li_sql = f"SELECT {_LI_COLS} FROM lineitem ORDER BY l_orderkey"
    o_sql = "SELECT o_orderkey, o_custkey, o_totalprice, o_orderdate FROM orders ORDER BY o_orderkey"
    assert_same(eng2.sql(li_sql), duck.execute(li_sql.replace("lineitem", "lineitem_w")).df())
    assert_same(eng2.sql(o_sql), duck.execute(o_sql.replace("orders", "orders_w")).df())
    # Counter recovered to the number of commits (5 autocommits).
    assert eng2._commit_ts == 5