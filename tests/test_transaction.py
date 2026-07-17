"""Phase 2 step 5: MVCC transaction layer + full snapshot restore.

The engine now versions every committed delta batch with a monotonic
``commit_ts``. A transaction (``BEGIN``) captures a frozen snapshot of the
committed state, buffers its own ``INSERT`` frames (read-your-writes, visible
only to the txn), and either ``COMMIT``s them atomically to the shared delta
under one new ts or ``ROLLBACK``s them (undo only this txn's writes; the
committed delta is untouched). Full snapshot restore (``CREATE SNAPSHOT`` /
``RESTORE TO SNAPSHOT``, or the ``engine.snapshot``/``engine.restore`` API)
discards every committed batch after a target ts -- discarding committed work,
stronger than per-txn ROLLBACK.

RyuDB is single-session, so snapshot isolation is structural (no commit can
occur mid-txn); the MVCC ts is required for restore and is forward-looking for
concurrent connections. These tests cover: read-your-writes + atomic commit,
ROLLBACK undo, multi-insert atomicity, MVCC snapshot restore (API + SQL
surface), the dangling-snapshot self-cleaning fix, txn x cache invalidation
(read-your-writes through the fused path), cold-scan + txn, error states, and
CLI smoke. Correctness is checked against DuckDB.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import cudf
import duckdb

from ryudb import Catalog, Engine
from ryudb.exec import fused
from ryudb.sql.parse import parse
from ryudb.sql.plan import TxnControl

from .conftest import assert_same

CPP = fused._kernels.is_available

# A writable-DuckDB-oracle helper: materialize writable copies of the tables the
# test will INSERT into, then mirror INSERTs into both RyuDB and the _w tables.
_INS = (
    "INSERT INTO lineitem (l_orderkey, l_quantity, l_extendedprice, l_shipdate) "
    "VALUES ({k}, {q}, {e}, date '{d}')"
)


def _make_writable(duck, tables):
    for t in tables:
        duck.execute(f"CREATE TABLE {t}_w AS SELECT * FROM {t}")


def _ins(k, q, e, d):
    return _INS.format(k=k, q=q, e=e, d=d)


# --------------------------------------------------------------- parse routing


def test_parse_begin_commit_rollback():
    for s, k in [("BEGIN", "begin"), ("BEGIN TRANSACTION", "begin"),
                 ("BEGIN WORK", "begin"), ("COMMIT", "commit"),
                 ("COMMIT WORK", "commit"), ("ROLLBACK", "rollback"),
                 ("ROLLBACK TRANSACTION", "rollback")]:
        p = parse(s)
        assert isinstance(p, TxnControl) and p.kind == k, (s, p)


def test_parse_rejects_commit_chain_and_rollback_savepoint():
    with pytest.raises(NotImplementedError):
        parse("COMMIT AND CHAIN")
    with pytest.raises(NotImplementedError):
        parse("ROLLBACK TO SAVEPOINT s")


def test_txn_control_returns_none(engine):
    assert engine.sql("BEGIN") is None
    assert engine.sql("ROLLBACK") is None  # balance the BEGIN


# --------------------------------------------------- BEGIN/COMMIT read-your-writes


def test_begin_commit_read_your_writes(engine, duck):
    _make_writable(duck, ["lineitem"])
    base = int(engine.sql("SELECT count(*) AS n FROM lineitem").to_pandas()["n"].iloc[0])

    engine.sql("BEGIN")
    engine.sql(_ins(4242, 7.5, 88.25, "1997-03-14"))
    duck.execute("BEGIN")
    duck.execute(_ins(4242, 7.5, 88.25, "1997-03-14").replace(
        "lineitem (", "lineitem_w ("))

    # read-your-writes: the in-txn SELECT sees the buffered row.
    n = int(engine.sql("SELECT count(*) AS n FROM lineitem").to_pandas()["n"].iloc[0])
    assert n == base + 1

    engine.sql("COMMIT")
    duck.execute("COMMIT")

    # post-commit: still visible (now from the shared delta).
    n2 = int(engine.sql("SELECT count(*) AS n FROM lineitem").to_pandas()["n"].iloc[0])
    assert n2 == base + 1
    q = ("SELECT l_orderkey, l_quantity, l_extendedprice FROM lineitem "
         "WHERE l_orderkey = 4242")
    assert_same(engine.sql(q), duck.execute(q.replace("lineitem", "lineitem_w")).fetchdf())


def test_multi_insert_atomic_commit(engine, duck):
    _make_writable(duck, ["lineitem"])
    base = int(engine.sql("SELECT count(*) AS n FROM lineitem").to_pandas()["n"].iloc[0])

    engine.sql("BEGIN")
    duck.execute("BEGIN")
    for k in (1001, 1002):
        engine.sql(_ins(k, float(k), float(k) * 10, "1998-01-02"))
        duck.execute(_ins(k, float(k), float(k) * 10, "1998-01-02").replace(
            "lineitem (", "lineitem_w ("))
    engine.sql("COMMIT")
    duck.execute("COMMIT")

    n = int(engine.sql("SELECT count(*) AS n FROM lineitem").to_pandas()["n"].iloc[0])
    assert n == base + 2
    # Both share one commit_ts (atomic).
    ts = engine._commit_ts
    batches = engine.delta._batches["lineitem"]
    assert all(cts == ts for cts, _ in batches[-2:])
    q = ("SELECT l_orderkey FROM lineitem WHERE l_orderkey >= 1001 "
         "AND l_orderkey <= 1002 ORDER BY l_orderkey")
    assert_same(engine.sql(q), duck.execute(q.replace("lineitem", "lineitem_w")).fetchdf())


# --------------------------------------------------------------- ROLLBACK undo


def test_rollback_undoes_own_writes(engine, duck):
    _make_writable(duck, ["lineitem"])
    base = int(engine.sql("SELECT count(*) AS n FROM lineitem").to_pandas()["n"].iloc[0])

    engine.sql("BEGIN")
    engine.sql(_ins(5555, 1.0, 2.0, "1998-05-05"))
    # read-your-writes sees it...
    assert int(engine.sql("SELECT count(*) AS n FROM lineitem").to_pandas()["n"].iloc[0]) == base + 1
    engine.sql("ROLLBACK")
    # ...but after ROLLBACK it is gone (back to base).
    n = int(engine.sql("SELECT count(*) AS n FROM lineitem").to_pandas()["n"].iloc[0])
    assert n == base
    # The committed delta was never touched: no batch for lineitem gained rows.
    # (Autocommit path is the only other writer; none ran here.)
    q = "SELECT l_orderkey FROM lineitem WHERE l_orderkey = 5555"
    assert len(engine.sql(q).to_pandas()) == 0
    assert_same(engine.sql("SELECT count(*) AS n FROM lineitem"),
                duck.execute("SELECT count(*) AS n FROM lineitem_w").fetchdf())


def test_rollback_does_not_touch_committed_delta(engine, duck):
    """A prior committed INSERT survives a later ROLLBACK."""
    _make_writable(duck, ["lineitem"])
    base = int(engine.sql("SELECT count(*) AS n FROM lineitem").to_pandas()["n"].iloc[0])

    engine.sql(_ins(7000, 1.0, 2.0, "1998-06-06"))  # autocommit -> ts=N
    committed_ts = engine._commit_ts
    engine.sql("BEGIN")
    engine.sql(_ins(7001, 1.0, 2.0, "1998-06-07"))
    engine.sql("ROLLBACK")

    # committed batch still present; counter unchanged by the rolled-back txn.
    assert committed_ts in [cts for cts, _ in engine.delta._batches["lineitem"]]
    assert engine._commit_ts == committed_ts
    n = int(engine.sql("SELECT count(*) AS n FROM lineitem").to_pandas()["n"].iloc[0])
    assert n == base + 1


# ------------------------------------------- MVCC + full snapshot restore (API)


def test_snapshot_restore_api(engine, duck):
    _make_writable(duck, ["lineitem"])
    base = int(engine.sql("SELECT count(*) AS n FROM lineitem").to_pandas()["n"].iloc[0])

    engine.sql(_ins(8001, 1.0, 2.0, "1998-01-01"))  # autocommit ts=1
    duck.execute(_ins(8001, 1.0, 2.0, "1998-01-01").replace("lineitem (", "lineitem_w ("))
    engine.snapshot("s1")
    assert engine._snapshots["s1"] == 1

    engine.sql(_ins(8002, 3.0, 4.0, "1998-01-02"))  # autocommit ts=2
    assert int(engine.sql("SELECT count(*) AS n FROM lineitem").to_pandas()["n"].iloc[0]) == base + 2
    # the ts=2 batch is visible at ts>=2 but not at ts=1.
    assert len(engine.delta.batches_at("lineitem", 1)) == 1
    assert len(engine.delta.batches_at("lineitem", 2)) == 2

    engine.restore("s1")  # discard ts=2 batch, rewind counter to 1
    assert engine._commit_ts == 1
    n = int(engine.sql("SELECT count(*) AS n FROM lineitem").to_pandas()["n"].iloc[0])
    assert n == base + 1
    assert len(engine.delta.batches_at("lineitem", 1)) == 1

    # DuckDB oracle: only the first insert survived.
    assert_same(engine.sql("SELECT l_orderkey FROM lineitem WHERE l_orderkey >= 8001 "
                           "ORDER BY l_orderkey"),
                duck.execute("SELECT l_orderkey FROM lineitem_w WHERE l_orderkey >= 8001 "
                             "ORDER BY l_orderkey").fetchdf())


def test_snapshot_restore_sql_surface(engine, duck):
    """Same as the API test but via CREATE SNAPSHOT / RESTORE TO SNAPSHOT SQL --
    exercises the regex pre-sniff in Engine.sql (bypasses sqlglot)."""
    _make_writable(duck, ["lineitem"])
    base = int(engine.sql("SELECT count(*) AS n FROM lineitem").to_pandas()["n"].iloc[0])

    assert engine.sql("CREATE SNAPSHOT s2") is None
    engine.sql(_ins(9001, 1.0, 2.0, "1998-02-01"))  # ts=1
    duck.execute(_ins(9001, 1.0, 2.0, "1998-02-01").replace("lineitem (", "lineitem_w ("))
    engine.sql("CREATE SNAPSHOT s2b")  # captures ts=1
    engine.sql(_ins(9002, 3.0, 4.0, "1998-02-02"))  # ts=2
    assert int(engine.sql("SELECT count(*) AS n FROM lineitem").to_pandas()["n"].iloc[0]) == base + 2

    assert engine.sql("RESTORE TO SNAPSHOT s2b") is None
    assert engine._commit_ts == 1
    assert int(engine.sql("SELECT count(*) AS n FROM lineitem").to_pandas()["n"].iloc[0]) == base + 1
    assert_same(engine.sql("SELECT l_orderkey FROM lineitem WHERE l_orderkey >= 9001 "
                           "ORDER BY l_orderkey"),
                duck.execute("SELECT l_orderkey FROM lineitem_w WHERE l_orderkey >= 9001 "
                             "ORDER BY l_orderkey").fetchdf())


def test_dangling_snapshot_dropped_on_restore(engine):
    """restore-to-earlier drops snapshots whose ts > target (they point at
    discarded state). After restore('a'), restore('b') must raise."""
    engine.sql(_ins(1001, 1.0, 2.0, "1998-03-01"))  # ts=1
    engine.snapshot("a")                            # a -> ts=1
    engine.sql(_ins(1002, 1.0, 2.0, "1998-03-02"))  # ts=2
    engine.snapshot("b")                            # b -> ts=2
    engine.sql(_ins(1003, 1.0, 2.0, "1998-03-03"))  # ts=3
    assert set(engine._snapshots) == {"a", "b"}

    engine.restore("a")  # rewind to ts=1 -> drops ts>1 batches AND snapshot "b"
    assert "a" in engine._snapshots
    assert "b" not in engine._snapshots
    with pytest.raises(RuntimeError, match="unknown snapshot"):
        engine.restore("b")


def test_restore_to_raw_ts(engine):
    base = int(engine.sql("SELECT count(*) AS n FROM lineitem").to_pandas()["n"].iloc[0])
    engine.sql(_ins(2001, 1.0, 2.0, "1998-04-01"))  # ts=1
    engine.sql(_ins(2002, 1.0, 2.0, "1998-04-02"))  # ts=2
    engine.sql(_ins(2003, 1.0, 2.0, "1998-04-03"))  # ts=3
    assert int(engine.sql("SELECT count(*) AS n FROM lineitem").to_pandas()["n"].iloc[0]) == base + 3

    engine.restore_to(2)
    assert engine._commit_ts == 2
    assert len(engine.delta.batches_at("lineitem", 2)) == 2
    assert int(engine.sql("SELECT count(*) AS n FROM lineitem").to_pandas()["n"].iloc[0]) == base + 2


def test_snapshot_overwrites_name(engine):
    engine.sql(_ins(3001, 1.0, 2.0, "1998-05-01"))  # ts=1
    engine.snapshot("x")                            # x -> 1
    engine.sql(_ins(3002, 1.0, 2.0, "1998-05-02"))  # ts=2
    engine.snapshot("x")                            # x -> 2 (overwrite)
    assert engine._snapshots["x"] == 2
    engine.restore("x")
    assert engine._commit_ts == 2


# ------------------------------------- txn x cache invalidation (fused path)

# A low-card string group-key lineitem (mirrors tests/test_cache_invalidation.py)
# so the fused DENSE path populates _code_cache for the string group keys.
HC_DENSE = """
    SELECT l_returnflag, l_linestatus, sum(l_quantity) AS sum_qty, count(*) AS n
      FROM lineitem
     WHERE l_shipdate <= date '1998-09-02'
     GROUP BY l_returnflag, l_linestatus
     ORDER BY l_returnflag, l_linestatus
"""


@pytest.fixture
def hc_engine(tmp_path) -> Engine:
    d = tmp_path
    (d / "lineitem").mkdir()
    rng = np.random.default_rng(7)
    n = 20000
    rows = {
        "l_orderkey": rng.integers(1, 5001, size=n).astype(np.int64),
        "l_returnflag": rng.choice(["A", "N", "R"], size=n).astype(object),
        "l_linestatus": rng.choice(["F", "O"], size=n).astype(object),
        "l_quantity": rng.uniform(1, 50, size=n),
        "l_extendedprice": rng.uniform(10, 100, size=n),
        "l_shipdate": pd.to_datetime(
            rng.choice(pd.date_range("1998-01-01", "1998-12-31"), size=n)
        ),
    }
    cudf.DataFrame(rows).to_pandas().to_parquet(d / "lineitem" / "0.parquet")
    cat = Catalog(str(d))
    cat.register("lineitem", str(d / "lineitem"))
    return Engine(cat)


@pytest.fixture
def hc_duck(tmp_path) -> "duckdb.DuckDBPyConnection":
    con = duckdb.connect()
    con.execute(f"CREATE VIEW lineitem AS SELECT * FROM read_parquet('{tmp_path}/lineitem/*.parquet')")
    return con


def test_txn_buffer_append_invalidates_code_cache(hc_engine, hc_duck):
    """A warm HC_DENSE populates _code_cache. BEGIN; INSERT a NEW l_returnflag
    'Z' that passes the WHERE; re-run HC_DENSE -- the buffer-append must have
    invalidated the cache so the codes re-factorize over base+snapshot+buffer
    (else stale length-20000 codes read OOB / 'Z' has no code -> wrong grouping).
    COMMIT; re-run -- committed state must still be correct."""
    hc_duck.execute("CREATE TABLE lineitem_w AS SELECT * FROM lineitem")
    hc_engine.sql(HC_DENSE)  # warm
    if ("lineitem", "l_returnflag") not in hc_engine._code_cache:
        pytest.skip("fused DENSE path did not populate _code_cache; cannot prove staleness")

    hc_engine.sql("BEGIN")
    ins = ("INSERT INTO lineitem (l_returnflag, l_linestatus, l_quantity, l_shipdate) "
           "VALUES ('Z', 'O', 10.0, date '1998-08-01')")
    hc_engine.sql(ins)
    # buffer-append evicted this table's _code_cache entries.
    assert ("lineitem", "l_returnflag") not in hc_engine._code_cache
    assert ("lineitem", "l_linestatus") not in hc_engine._code_cache

    hc_duck.execute("BEGIN")
    hc_duck.execute(ins.replace("lineitem (l_returnflag, l_linestatus, l_quantity, l_shipdate)",
                                "lineitem_w (l_returnflag, l_linestatus, l_quantity, l_shipdate)"))

    ryu = hc_engine.sql(HC_DENSE)
    dft = hc_duck.execute(HC_DENSE.replace("lineitem", "lineitem_w")).fetchdf()
    assert_same(ryu, dft)

    hc_engine.sql("COMMIT")
    hc_duck.execute("COMMIT")
    ryu2 = hc_engine.sql(HC_DENSE)
    dft2 = hc_duck.execute(HC_DENSE.replace("lineitem", "lineitem_w")).fetchdf()
    assert_same(ryu2, dft2)


def test_rollback_invalidates_after_read_your_writes(hc_engine):
    """A read-your-writes SELECT populates _code_cache against base+buffer; after
    ROLLBACK the buffer is gone, so those caches must be evicted (else a stale
    length-(N+1) code set would be read against the length-N base)."""
    hc_engine.sql(HC_DENSE)  # warm
    if ("lineitem", "l_returnflag") not in hc_engine._code_cache:
        pytest.skip("fused DENSE path did not populate _code_cache")
    hc_engine.sql("BEGIN")
    hc_engine.sql("INSERT INTO lineitem (l_returnflag, l_linestatus, l_quantity, l_shipdate) "
                  "VALUES ('Z', 'O', 10.0, date '1998-08-01')")
    hc_engine.sql(HC_DENSE)  # read-your-writes -> re-populates _code_cache over base+buffer
    assert ("lineitem", "l_returnflag") in hc_engine._code_cache
    hc_engine.sql("ROLLBACK")
    assert ("lineitem", "l_returnflag") not in hc_engine._code_cache
    assert ("lineitem", "l_linestatus") not in hc_engine._code_cache


# ----------------------------------------------------- cold-scan + txn (RISK 10)


def test_cold_scan_then_txn(typed_engine, typed_duck):
    """A cold SELECT populates a _PendingFrame in _scan_cache; a subsequent
    BEGIN/INSERT/SELECT must resolve the pending frame and merge the buffer
    correctly (assert_same vs DuckDB)."""
    typed_duck.execute("CREATE TABLE lineitem_w AS SELECT * FROM lineitem")
    q = ("SELECT count(*) AS n, sum(l_extendedprice) AS s FROM lineitem "
         "WHERE l_shipdate >= date '1994-01-01' AND l_shipdate < date '1995-01-01' "
         "AND l_discount >= 0.05 AND l_discount <= 0.07")
    typed_engine.clear_scan_cache()
    typed_engine.sql(q)  # cold -> may store a _PendingFrame
    # at least one cache entry now exists (pending or ready)
    assert typed_engine._scan_cache

    typed_engine.sql("BEGIN")
    typed_engine.sql("INSERT INTO lineitem (l_orderkey, l_quantity, l_extendedprice, "
                     "l_discount, l_tax, l_shipdate) VALUES (500000, 10.0, 1234.56, 0.06, "
                     "0.02, date '1994-06-15')")
    typed_duck.execute("BEGIN")
    typed_duck.execute("INSERT INTO lineitem_w (l_orderkey, l_quantity, l_extendedprice, "
                       "l_discount, l_tax, l_shipdate) VALUES (500000, 10.0, 1234.56, 0.06, "
                       "0.02, date '1994-06-15')")
    typed_engine.sql("COMMIT")
    typed_duck.execute("COMMIT")

    typed_engine.clear_scan_cache()  # force a fresh cold read with the delta
    ryu = typed_engine.sql(q)
    dft = typed_duck.execute(q.replace("lineitem", "lineitem_w")).fetchdf()
    assert_same(ryu, dft)


# ------------------------------------------------------------- error states


def test_error_begin_when_active(engine):
    engine.sql("BEGIN")
    with pytest.raises(RuntimeError, match="active transaction"):
        engine.sql("BEGIN")
    engine.sql("ROLLBACK")


def test_error_commit_when_none(engine):
    with pytest.raises(RuntimeError, match="without an active transaction"):
        engine.sql("COMMIT")


def test_error_rollback_when_none(engine):
    with pytest.raises(RuntimeError, match="without an active transaction"):
        engine.sql("ROLLBACK")


def test_error_restore_unknown_snapshot(engine):
    with pytest.raises(RuntimeError, match="unknown snapshot"):
        engine.restore("nope")


def test_error_restore_during_txn(engine):
    engine.sql(_ins(1, 1.0, 2.0, "1998-01-01"))
    engine.snapshot("s")
    engine.sql("BEGIN")
    with pytest.raises(RuntimeError, match="during a transaction"):
        engine.restore("s")
    engine.sql("ROLLBACK")


# --------------------------------------------------------------- CLI smoke


def test_cli_txn_output(engine, capsys):
    from ryudb import cli
    assert cli._run_statement(engine, "BEGIN", quiet=False) == 0
    assert "begin ok" in capsys.readouterr().out
    assert cli._run_statement(
        engine,
        "INSERT INTO lineitem (l_orderkey, l_quantity, l_extendedprice, l_shipdate) "
        "VALUES (123, 1.0, 2.0, date '1998-05-05')",
        quiet=False,
    ) == 0
    out = capsys.readouterr().out
    assert "inserted 1 rows" in out
    assert cli._run_statement(engine, "COMMIT", quiet=False) == 0
    assert "commit ok" in capsys.readouterr().out


def test_cli_snapshot_restore_dot_commands(engine, capsys):
    from ryudb import cli
    engine.sql(_ins(1, 1.0, 2.0, "1998-01-01"))
    assert cli._dot_command("snapshot snap1", engine, engine.catalog) is False
    assert "snapshot snap1 captured" in capsys.readouterr().out
    engine.sql(_ins(2, 1.0, 2.0, "1998-01-02"))
    assert cli._dot_command("restore snap1", engine, engine.catalog) is False
    assert "restored to snapshot snap1" in capsys.readouterr().out
    n = int(engine.sql("SELECT count(*) AS n FROM lineitem").to_pandas()["n"].iloc[0])
    assert n == 9  # base 8 + 1; the second insert was discarded by restore


def test_cli_rollback_output(engine, capsys):
    from ryudb import cli
    cli._run_statement(engine, "BEGIN", quiet=False)
    cli._run_statement(
        engine,
        "INSERT INTO lineitem (l_orderkey, l_quantity, l_extendedprice, l_shipdate) "
        "VALUES (321, 1.0, 2.0, date '1998-05-05')",
        quiet=False,
    )
    capsys.readouterr()  # drain
    assert cli._run_statement(engine, "ROLLBACK", quiet=False) == 0
    assert "rollback ok" in capsys.readouterr().out