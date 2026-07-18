"""Phase 2 step 7: delta write-back (checkpoint) + DECIMAL/DATE type fidelity.

Committed INSERTs live in the in-memory delta + the WAL until ``checkpoint()``
folds them back into a new base Parquet file (preserving DECIMAL/DATE/BIGINT
logical types on disk), clears the delta, and truncates the WAL. These tests
prove: the delta is flushed and ``row_count`` fixed; the new base is a single
``ryudb_base.parquet`` with typed schema; the state (incl. DECIMAL/DATE values
and NULLs) survives a restart from base alone (empty WAL); the WAL is
truncated; snapshots straddling the checkpoint are invalidated correctly; and
the error/no-op guards hold. Correctness is checked against a writable DuckDB
oracle.

Each test uses a fresh function-scoped ``tmp_path`` (like ``test_wal.py``) so
the base files + WAL persist across the two Engine constructions within a test
(recovery / post-checkpoint restart is the thing under test).
"""

from __future__ import annotations

import glob
import os

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ryudb import Catalog, Engine
from ryudb.wal import wal_path

from .conftest import assert_same

# A small typed lineitem written via DuckDB COPY -- DECIMAL(15,2) as INT64,
# DATE as INT32, Snappy -- the exact on-disk layout the Phase 5 cold reader
# targets and the layout checkpoint must preserve the *logical* types of.
_LI_CREATE = (
    "CREATE TABLE lineitem (l_orderkey BIGINT, l_quantity DECIMAL(15,2), "
    "l_extendedprice DECIMAL(15,2), l_shipdate DATE)"
)
_LI_SEED = (
    "INSERT INTO lineitem SELECT i+1, ((i%50)+1)::DECIMAL(15,2), "
    "(((i%50)+1)*10)::DECIMAL(15,2), date '1994-01-01' + (i%730)::INTEGER "
    "FROM range(40) t(i)"
)
_LI_COLS = "l_orderkey, l_quantity, l_extendedprice, l_shipdate"


def _write_typed_base(d, *, orders: bool = False) -> None:
    """Write a DuckDB-layout typed lineitem base (and optionally a plain orders)."""
    (d / "lineitem").mkdir()
    con = duckdb.connect()
    con.execute(_LI_CREATE)
    con.execute(_LI_SEED)
    con.execute(
        f"COPY (SELECT * FROM lineitem) TO '{d}/lineitem/0.parquet' "
        "(FORMAT PARQUET, COMPRESSION 'snappy')"
    )
    if orders:
        (d / "orders").mkdir()
        con.execute(
            "CREATE TABLE orders (o_orderkey BIGINT, o_totalprice DECIMAL(15,2), "
            "o_orderdate DATE)"
        )
        con.execute(
            "INSERT INTO orders SELECT i+1, ((i+1)*10)::DECIMAL(15,2), "
            "date '1995-01-01' + (i%365)::INTEGER FROM range(20) t(i)"
        )
        con.execute(
            f"COPY (SELECT * FROM orders) TO '{d}/orders/0.parquet' "
            "(FORMAT PARQUET, COMPRESSION 'snappy')"
        )
    con.close()


@pytest.fixture
def ck_dir(tmp_path) -> str:
    _write_typed_base(tmp_path)
    return str(tmp_path)


def _engine(ck_dir: str, *, orders: bool = False) -> Engine:
    cat = Catalog(ck_dir)
    cat.register("lineitem", os.path.join(ck_dir, "lineitem"))
    if orders:
        cat.register("orders", os.path.join(ck_dir, "orders"))
    return Engine(cat)


def _duck(ck_dir: str, *, orders: bool = False) -> "duckdb.DuckDBPyConnection":
    con = duckdb.connect()
    con.execute(f"CREATE TABLE lineitem_w AS SELECT * FROM read_parquet('{ck_dir}/lineitem/*.parquet')")
    if orders:
        con.execute(f"CREATE TABLE orders_w AS SELECT * FROM read_parquet('{ck_dir}/orders/*.parquet')")
    return con


def _li_insert(eng: Engine, k: int, q: str = "1.0", p: str = "10.0", d: str = "1998-01-01") -> None:
    eng.sql(
        f"INSERT INTO lineitem ({_LI_COLS}) "
        f"VALUES ({k}, {q}, {p}, date '{d}')"
    )


# -------------------------------------------------------- flush + type fidelity


def test_checkpoint_flushes_delta_to_base(ck_dir):
    eng = _engine(ck_dir)
    base_rows = int(eng.sql("SELECT count(*) AS n FROM lineitem").to_pandas()["n"].iloc[0])
    # Build the DuckDB oracle from the SEED base BEFORE any insert (it's a
    # materialized table, frozen at 40 rows) so the checkpointed base (40+1)
    # and the oracle (40+1) match -- building it after the checkpoint would
    # read the already-checkpointed base and then double-count the insert.
    duck = _duck(ck_dir)
    _li_insert(eng, 1001, "37.5", "375.0", "1995-06-15")
    duck.execute("INSERT INTO lineitem_w VALUES (1001, 37.5, 375.0, date '1995-06-15')")
    assert eng.delta.has_unflushed("lineitem")

    written = eng.checkpoint()
    assert written == {"lineitem": base_rows + 1}
    # Delta cleared; row_count fixed.
    assert not eng.delta.has_unflushed("lineitem")
    assert eng.catalog.get("lineitem").row_count == base_rows + 1
    # New base is a single typed file; old 0.parquet gone.
    files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(ck_dir, "lineitem", "*.parquet")))
    assert files == ["ryudb_base.parquet"]
    schema = pq.read_schema(os.path.join(ck_dir, "lineitem", "ryudb_base.parquet"))
    assert pa.types.is_decimal(schema.field("l_quantity").type)
    assert pa.types.is_date32(schema.field("l_shipdate").type)
    assert pa.types.is_int64(schema.field("l_orderkey").type)
    # SELECT still correct after the rewrite.
    sql = f"SELECT {_LI_COLS} FROM lineitem ORDER BY l_orderkey"
    assert_same(eng.sql(sql), duck.execute(sql.replace("lineitem", "lineitem_w")).df())


def test_checkpoint_survives_restart(ck_dir):
    eng = _engine(ck_dir)
    base_rows = int(eng.sql("SELECT count(*) AS n FROM lineitem").to_pandas()["n"].iloc[0])
    _li_insert(eng, 1001, "37.5", "375.0", "1995-06-15")
    eng.checkpoint()

    # New engine on the SAME dir: the row is in base, the WAL is empty (no replay).
    eng2 = _engine(ck_dir)
    assert eng2._commit_ts == 0  # empty WAL -> nothing replayed
    assert not os.path.exists(wal_path(ck_dir)) or os.path.getsize(wal_path(ck_dir)) == 0
    n = int(eng2.sql("SELECT count(*) AS n FROM lineitem").to_pandas()["n"].iloc[0])
    assert n == base_rows + 1
    row = eng2.sql("SELECT * FROM lineitem WHERE l_orderkey = 1001").to_pandas()
    assert len(row) == 1 and float(row["l_quantity"].iloc[0]) == 37.5

    # A subsequent insert works from the recovered (empty-WAL) counter with no
    # snapshot to collide against (snapshots are in-memory-only, dropped on restart).
    _li_insert(eng2, 1002)
    assert eng2._commit_ts == 1
    assert int(eng2.sql("SELECT count(*) AS n FROM lineitem").to_pandas()["n"].iloc[0]) == base_rows + 2


def test_checkpoint_truncates_wal(ck_dir):
    eng = _engine(ck_dir)
    _li_insert(eng, 1)
    _li_insert(eng, 2)
    assert os.path.getsize(wal_path(ck_dir)) > 0  # >=2 WAL records
    eng.checkpoint()
    assert os.path.getsize(wal_path(ck_dir)) == 0  # all committed work is in base


def test_checkpoint_clears_delta_and_row_count(ck_dir):
    eng = _engine(ck_dir)
    base_rows = int(eng.sql("SELECT count(*) AS n FROM lineitem").to_pandas()["n"].iloc[0])
    _li_insert(eng, 10)
    _li_insert(eng, 11)
    _li_insert(eng, 12)
    assert eng.delta.has_unflushed("lineitem")
    eng.checkpoint()
    assert not eng.delta.has_unflushed("lineitem")
    assert eng.catalog.get("lineitem").row_count == base_rows + 3
    files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(ck_dir, "lineitem", "*.parquet")))
    assert files == ["ryudb_base.parquet"]


def test_checkpoint_type_fidelity_decimal_date_null(ck_dir):
    eng = _engine(ck_dir)
    duck = _duck(ck_dir)  # frozen seed base; mirror inserts below (see test 1 note)
    # A row with DECIMAL/DATE values AND a NULL in a nullable decimal column.
    eng.sql(
        f"INSERT INTO lineitem ({_LI_COLS}) "
        "VALUES (1001, NULL, 375.0, date '1995-06-15')"
    )
    duck.execute("INSERT INTO lineitem_w VALUES (1001, NULL, 375.0, date '1995-06-15')")
    eng.sql(
        f"INSERT INTO lineitem ({_LI_COLS}) "
        "VALUES (1002, 2.25, 22.5, date '1996-12-31')"
    )
    duck.execute("INSERT INTO lineitem_w VALUES (1002, 2.25, 22.5, date '1996-12-31')")
    eng.checkpoint()

    # Survives a restart from base alone (NULL preserved as NaN, values exact).
    eng2 = _engine(ck_dir)
    sql = f"SELECT {_LI_COLS} FROM lineitem ORDER BY l_orderkey"
    assert_same(eng2.sql(sql), duck.execute(sql.replace("lineitem", "lineitem_w")).df())
    row = eng2.sql("SELECT * FROM lineitem WHERE l_orderkey = 1001").to_pandas()
    assert row["l_quantity"].isna().iloc[0]  # NULL survived the round-trip


# -------------------------------------------------------------- snapshot rules


def test_checkpoint_drops_pre_checkpoint_snapshots(ck_dir):
    eng = _engine(ck_dir)
    _li_insert(eng, 1)
    eng.snapshot("s1")  # ts == 1
    _li_insert(eng, 2)
    eng.snapshot("s2")  # ts == 2
    base_rows = int(eng.sql("SELECT count(*) AS n FROM lineitem").to_pandas()["n"].iloc[0])
    assert base_rows == 40 + 2

    eng.checkpoint()  # checkpoint_ts == 2
    assert "s1" not in eng._snapshots  # ts=1 < 2 -> dropped (state folded into base)
    assert "s2" in eng._snapshots       # ts=2 == checkpoint_ts -> kept

    # restore to the kept tip snapshot == base (through the checkpoint).
    eng.restore("s2")
    assert int(eng.sql("SELECT count(*) AS n FROM lineitem").to_pandas()["n"].iloc[0]) == 42
    # s1 is gone -> restore by name raises.
    with pytest.raises(RuntimeError):
        eng.restore("s1")


def test_checkpoint_mid_txn_rejected(ck_dir):
    eng = _engine(ck_dir)
    eng.sql("BEGIN")
    _li_insert(eng, 1)
    with pytest.raises(RuntimeError):
        eng.checkpoint()
    eng.sql("COMMIT")
    # After COMMIT the checkpoint succeeds.
    written = eng.checkpoint()
    assert written == {"lineitem": 41}


def test_checkpoint_no_data_dir_rejected(ck_dir):
    cat = Catalog(None)  # ephemeral -> no data dir
    cat.register("lineitem", os.path.join(ck_dir, "lineitem"))
    eng = Engine(cat)
    _li_insert(eng, 1)
    with pytest.raises(RuntimeError):
        eng.checkpoint()


def test_checkpoint_noop_when_no_delta(ck_dir):
    eng = _engine(ck_dir)
    # No INSERTs -> nothing to flush -> no base file is written.
    files_before = sorted(glob.glob(os.path.join(ck_dir, "lineitem", "*.parquet")))
    written = eng.checkpoint()
    assert written == {}
    assert sorted(glob.glob(os.path.join(ck_dir, "lineitem", "*.parquet"))) == files_before
    assert "ryudb_base.parquet" not in [os.path.basename(p) for p in files_before]


def test_checkpoint_then_restore_discards_post_checkpoint(ck_dir):
    eng = _engine(ck_dir)
    _li_insert(eng, 1001)
    eng.checkpoint()            # tip == 1, base has 41 rows, delta + WAL empty
    eng.snapshot("s")           # ts == 1 (== checkpoint_ts -> survives checkpoints)
    _li_insert(eng, 1002)      # ts == 2, in delta + WAL
    assert int(eng.sql("SELECT count(*) AS n FROM lineitem").to_pandas()["n"].iloc[0]) == 42

    eng.restore("s")           # rewind to ts=1 -> drop the ts=2 batch + WAL record
    assert int(eng.sql("SELECT count(*) AS n FROM lineitem").to_pandas()["n"].iloc[0]) == 41
    assert eng.sql("SELECT * FROM lineitem WHERE l_orderkey = 1002").to_pandas().empty
    # 1001 was folded into base by the checkpoint and survives the restore.
    assert eng.sql("SELECT * FROM lineitem WHERE l_orderkey = 1001").to_pandas().shape[0] == 1
    # The post-checkpoint WAL record was durably discarded by the restore.
    recs = []
    from ryudb.wal import WAL
    if os.path.exists(wal_path(ck_dir)):
        recs = WAL(wal_path(ck_dir)).replay()
    assert recs == []