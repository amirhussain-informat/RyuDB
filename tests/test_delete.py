"""Phase 2 step 9: DELETE DML via PK-tombstoned delta.

``DELETE FROM t [WHERE pred]`` evaluates the predicate against the currently-
visible snapshot, collects the **primary-key values** of the matched rows, and
stores them as a tombstone batch. The read path (``Engine._merge_delta``)
anti-joins the visible rows against the tombstone PKs **timestamp-aware**: every
visible row is tagged with its insertion ``commit_ts`` (base=0, each committed
insert batch=its ts, in-txn buffered insert=+inf) and a tombstone removes a row
only when ``tomb_ts >= ins_ts`` -- so a row inserted *after* a delete of the same
PK survives the older tombstone (delete-then-reinsert-same-PK works). Requires
a declared PRIMARY KEY (row identity is by PK value, not position).

Each test uses a fresh function-scoped ``tmp_path`` with its own small writable
base + catalog (mirrors ``test_unique.py``'s ``u_dir``), so declared PKs never
leak into the shared ``data_dir`` catalog file.
"""

from __future__ import annotations

import os

import cudf
import duckdb
import pandas as pd
import pytest

from ryudb import Catalog, Engine

# A tiny writable base: t(k BIGINT, b BIGINT, label nullable str). Seed rows
# k=1,2,3 with labels 'A','B',NULL -- enough to exercise WHERE, delete-all,
# composite-PK partial overlap, and NULL-label survival.
_BASE = [
    (1, 10, "A"),
    (2, 20, "B"),
    (3, 30, None),
]


@pytest.fixture
def d_dir(tmp_path) -> str:
    d = tmp_path
    (d / "t").mkdir()
    cudf.DataFrame(
        {
            "k": [r[0] for r in _BASE],
            "b": [r[1] for r in _BASE],
            "label": pd.array([r[2] for r in _BASE], dtype=object),
        }
    ).to_pandas().to_parquet(d / "t" / "0.parquet")
    return str(d)


def _engine(d_dir: str) -> Engine:
    cat = Catalog(d_dir)
    cat.register("t", os.path.join(d_dir, "t"))
    return Engine(cat)


def _count(eng: Engine) -> int:
    return int(eng.sql("SELECT count(*) AS n FROM t").to_pandas()["n"].iloc[0])


def _keys(eng: Engine) -> list[int]:
    return list(eng.sql("SELECT k FROM t ORDER BY k").to_pandas()["k"])


# --------------------------------------------------------- delete WHERE autocommit


def test_delete_where_autocommit(d_dir):
    eng = _engine(d_dir)
    eng.catalog.set_primary_key("t", ["k"])
    base_n = _count(eng)
    n = eng.sql("DELETE FROM t WHERE k = 2")
    assert n == 1
    assert _count(eng) == base_n - 1
    assert _keys(eng) == [1, 3]
    # a tombstone was written; no insert batch.
    assert eng.delta.has_tombstones("t")
    assert not eng.delta.batches_with_ts("t")
    # a second DELETE with the same predicate sees fewer rows (tombstone applied).
    n2 = eng.sql("DELETE FROM t WHERE k = 2")
    assert n2 == 0


def test_delete_all_autocommit(d_dir):
    eng = _engine(d_dir)
    eng.catalog.set_primary_key("t", ["k"])
    base_n = _count(eng)
    n = eng.sql("DELETE FROM t")
    assert n == base_n
    assert _count(eng) == 0
    assert _keys(eng) == []


# --------------------------------------------- in-txn read-your-writes + rollback/commit


def test_delete_in_txn_read_your_writes_rollback(d_dir):
    eng = _engine(d_dir)
    eng.catalog.set_primary_key("t", ["k"])
    base_n = _count(eng)
    eng.sql("BEGIN")
    n = eng.sql("DELETE FROM t WHERE k = 1")
    assert n == 1
    # read-your-writes: the deletion is visible inside the txn.
    assert _count(eng) == base_n - 1
    assert 1 not in _keys(eng)
    eng.sql("ROLLBACK")
    # rollback undoes the buffered tombstone.
    assert _count(eng) == base_n
    assert 1 in _keys(eng)
    assert not eng.delta.has_tombstones("t")


def test_delete_in_txn_read_your_writes_commit(d_dir):
    eng = _engine(d_dir)
    eng.catalog.set_primary_key("t", ["k"])
    base_n = _count(eng)
    eng.sql("BEGIN")
    eng.sql("DELETE FROM t WHERE k = 1")
    eng.sql("COMMIT")
    assert _count(eng) == base_n - 1
    assert 1 not in _keys(eng)
    # committed -> durable tombstone in the shared delta.
    assert eng.delta.has_tombstones("t")


# --------------------------------------------- delete-then-reinsert-same-PK composition


def test_delete_then_reinsert_same_pk(d_dir):
    """A DELETE of PK=k tombstones the old row; a later INSERT of the same PK is
    ALLOWED by _enforce_unique (the tombstoned PK is invisible to its _scan) AND
    visible (the reinsert's ins_ts exceeds the tombstone's ts, so the timestamp-
    aware anti-join keeps it)."""
    eng = _engine(d_dir)
    eng.catalog.set_primary_key("t", ["k"])
    base_n = _count(eng)
    eng.sql("DELETE FROM t WHERE k = 2")
    assert _count(eng) == base_n - 1
    # reinsert the same PK with new values -> succeeds (no UNIQUE violation).
    eng.sql("INSERT INTO t (k, b, label) VALUES (2, 99, 'Z')")
    assert _count(eng) == base_n  # back to base size
    df = eng.sql("SELECT k, b, label FROM t WHERE k = 2").to_pandas()
    assert int(df["k"].iloc[0]) == 2
    assert int(df["b"].iloc[0]) == 99
    assert df["label"].iloc[0] == "Z"


# ------------------------------------------------------------- delete requires PK


def test_delete_requires_pk(d_dir):
    eng = _engine(d_dir)  # no declared PK
    with pytest.raises(RuntimeError, match="requires a declared PRIMARY KEY"):
        eng.sql("DELETE FROM t WHERE k = 1")
    # nothing written.
    assert _count(eng) == len(_BASE)
    assert not eng.delta.has_tombstones("t")


# --------------------------------------------------- no-match returns zero, no write


def test_delete_no_match_returns_zero(d_dir):
    eng = _engine(d_dir)
    eng.catalog.set_primary_key("t", ["k"])
    base_n = _count(eng)
    n = eng.sql("DELETE FROM t WHERE k = 999")
    assert n == 0
    assert _count(eng) == base_n
    assert not eng.delta.has_tombstones("t")


# -------------------------------------------------------- WAL durability on restart


def test_delete_survives_restart(d_dir):
    eng = _engine(d_dir)
    eng.catalog.set_primary_key("t", ["k"])
    eng.sql("DELETE FROM t WHERE k = 3")
    assert 3 not in _keys(eng)
    # a fresh Engine replays the WAL -> the tombstone is reapplied.
    eng2 = Engine(Catalog(d_dir))
    eng2.catalog.set_primary_key("t", ["k"])
    assert 3 not in _keys(eng2)
    assert _count(eng2) == len(_BASE) - 1


# ----------------------------------------- checkpoint materializes the deletion into base


def test_delete_then_checkpoint(d_dir):
    eng = _engine(d_dir)
    eng.catalog.set_primary_key("t", ["k"])
    eng.sql("DELETE FROM t WHERE k = 3")
    assert eng.delta.has_tombstones("t")
    rep = eng.checkpoint()
    assert rep == {"t": len(_BASE) - 1}
    # tombstone channel + insert delta cleared; WAL truncated to 0.
    assert not eng.delta.has_tombstones("t")
    assert not eng.delta.has_unflushed("t")
    wal_size = os.path.getsize(os.path.join(d_dir, "ryudb.wal")) if os.path.exists(
        os.path.join(d_dir, "ryudb.wal")
    ) else 0
    assert wal_size == 0
    # the deletion is materialized into the new base.
    assert _count(eng) == len(_BASE) - 1
    assert 3 not in _keys(eng)
    # restart-from-base sees the deletion (no WAL replay needed).
    eng2 = Engine(Catalog(d_dir))
    assert _count(eng2) == len(_BASE) - 1
    assert 3 not in _keys(eng2)
    # after checkpoint, reinserting the deleted PK works cleanly (no live tombstone).
    eng2.catalog.set_primary_key("t", ["k"])
    eng2.sql("INSERT INTO t (k, b, label) VALUES (3, 333, 'back')")
    assert 3 in _keys(eng2)


# ------------------------------------- mixed INSERT+DELETE in one atomic commit (WAL)


def test_delete_in_txn_mixed_commit(d_dir):
    """BEGIN, INSERT a new row, DELETE an existing row, COMMIT -> one WAL record
    carries both an insert and a tombstone batch; the row is not visible; restart
    confirms atomicity (both replayed or neither)."""
    eng = _engine(d_dir)
    eng.catalog.set_primary_key("t", ["k"])
    base_n = _count(eng)
    eng.sql("BEGIN")
    eng.sql("INSERT INTO t (k, b, label) VALUES (77, 770, 'new')")
    eng.sql("DELETE FROM t WHERE k = 1")
    eng.sql("COMMIT")
    assert _count(eng) == base_n  # -1 (delete) +1 (insert) == base
    assert _keys(eng) == [2, 3, 77]
    # restart: the single WAL record replays both batches atomically.
    eng2 = Engine(Catalog(d_dir))
    eng2.catalog.set_primary_key("t", ["k"])
    assert _count(eng2) == base_n
    assert _keys(eng2) == [2, 3, 77]


# --------------------------------------------------------------- composite PK


def test_delete_composite_pk(d_dir):
    eng = _engine(d_dir)
    eng.catalog.set_primary_key("t", ["k", "b"])
    base_n = _count(eng)
    # delete exactly (k=2, b=20); (k=1,b=10) and (k=3,b=30) survive.
    n = eng.sql("DELETE FROM t WHERE k = 2 AND b = 20")
    assert n == 1
    assert _count(eng) == base_n - 1
    assert _keys(eng) == [1, 3]
    # a row sharing k but differing b is NOT deleted (composite key not matched).
    eng.sql("INSERT INTO t (k, b, label) VALUES (2, 99, 'other')")
    assert _count(eng) == base_n


# --------------------------------------------------------------- CLI smoke


def test_cli_delete_output(d_dir, capsys):
    from ryudb import cli

    eng = _engine(d_dir)
    eng.catalog.set_primary_key("t", ["k"])
    base_n = _count(eng)
    rc = cli._run_statement(eng, "DELETE FROM t WHERE k = 1", quiet=False)
    assert rc == 0
    assert "deleted 1 rows" in capsys.readouterr().out
    assert _count(eng) == base_n - 1
    # a following SELECT through the CLI confirms the row is gone.
    rc = cli._run_statement(eng, "SELECT count(*) AS n FROM t", quiet=False)
    assert rc == 0
    out = capsys.readouterr().out
    assert str(base_n - 1) in out


# ------------------------------------------------------- DuckDB oracle round-trip


def test_delete_duckdb_oracle(d_dir):
    """Mirror INSERTs + a DELETE into a writable DuckDB temp table and compare
    the full visible result (content, not just count)."""
    eng = _engine(d_dir)
    eng.catalog.set_primary_key("t", ["k"])
    con = duckdb.connect()
    con.execute(f"CREATE TABLE t_w AS SELECT * FROM read_parquet('{d_dir}/t/0.parquet')")
    # add a couple of rows on both sides, then delete a range. Non-null labels
    # keep the comparison clean (conftest _clean normalizes float NaN->None but
    # not string NA).
    for ins in (
        "INSERT INTO t (k, b, label) VALUES (4, 40, 'D')",
        "INSERT INTO t (k, b, label) VALUES (5, 50, 'E')",
    ):
        eng.sql(ins)
        con.execute(ins.replace("t (", "t_w ("))
    eng.sql("DELETE FROM t WHERE k >= 2 AND k <= 4")
    con.execute("DELETE FROM t_w WHERE k >= 2 AND k <= 4")
    q = "SELECT k, b, label FROM t ORDER BY k"
    ryu = eng.sql(q)
    dft = con.execute(q.replace(" t ", " t_w ")).fetchdf()
    from .conftest import assert_same
    assert_same(ryu, dft)