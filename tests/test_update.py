"""Phase 2 step 10: UPDATE DML via PK-tombstone-then-reinsert.

``UPDATE t SET col = expr [, ...] [WHERE pred]`` evaluates the predicate against
the currently-visible snapshot, collects the **primary-key values** of the
matched rows, and replaces them: an UPDATE tombstone (PK values, the
``exclude_same_ts`` flag set) is flushed together with a fresh INSERT batch of
the post-SET rows **under one commit_ts**. The read path
(``Engine._merge_delta``) anti-joins the tombstone timestamp-aware but treats an
UPDATE tombstone as *strict* (``tomb_upd <= ins_ts`` keeps the row), so the
re-inserted row at the *same* commit ts survives its own tombstone -- this is the
load-bearing invariant that lets a single-ts UPDATE be both durable and correct
(avoids the non-durable ``restore_to`` wart the two-ts split would introduce).

v1 scope: autocommit only. An UPDATE inside an explicit transaction raises
``NotImplementedError`` (correct per-row MVCC versioning would need the two-ts
split this version deliberately avoids). Requires a declared PRIMARY KEY (row
identity is by PK value, mirroring DELETE). Atomicity + read-your-writes
enforcement come from running the autocommit UPDATE inside an *implicit*
transaction: the old rows' UPDATE tombstone is visible to ``_enforce_unique``
(the old PKs are gone from the scan -> no false self-collision when SET keeps
the PK), and the tombstone + reinsert flush as one durable WAL record.

Each test uses a fresh function-scoped ``tmp_path`` with its own small writable
base + catalog (mirrors ``test_delete.py``'s ``d_dir``), so declared PKs never
leak into the shared ``data_dir`` catalog file.
"""

from __future__ import annotations

import os

import cudf
import pandas as pd
import pytest

from ryudb import Catalog, Engine

# A tiny writable base: t(k BIGINT, b BIGINT, label nullable str). Seed rows
# k=1,2,3 with labels 'A','B',NULL -- enough to exercise WHERE, update-all,
# composite-PK partial overlap, NULL-label handling, and PK change/collision.
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


def _rows(eng: Engine) -> list[tuple]:
    """All rows as (k, b, label) sorted by k, for value-level assertions."""
    df = eng.sql("SELECT k, b, label FROM t ORDER BY k").to_pandas()
    return [(int(r.k), int(r.b), None if pd.isna(r.label) else r.label) for r in df.itertuples()]


# --------------------------------------------------------- update WHERE autocommit


def test_update_where_autocommit(d_dir):
    eng = _engine(d_dir)
    eng.catalog.set_primary_key("t", ["k"])
    n = eng.sql("UPDATE t SET b = 99, label = 'X' WHERE k = 2")
    assert n == 1
    rows = {r[0]: r for r in _rows(eng)}
    assert rows[2] == (2, 99, "X")
    # untouched rows survive unchanged.
    assert rows[1] == (1, 10, "A")
    assert rows[3] == (3, 30, None)
    assert _count(eng) == len(_BASE)


def test_update_all_autocommit(d_dir):
    eng = _engine(d_dir)
    eng.catalog.set_primary_key("t", ["k"])
    n = eng.sql("UPDATE t SET label = 'Z'")
    assert n == len(_BASE)
    rows = _rows(eng)
    # every row's label is now 'Z' (the NULL label is overwritten too).
    assert all(r[2] == "Z" for r in rows)
    # b is untouched.
    assert {r[1] for r in rows} == {10, 20, 30}


# ------------------------------------------------------- SET expression shapes


def test_update_set_literal_broadcast(d_dir):
    """A literal RHS broadcasts to every matched row (scalar -> per-row value)."""
    eng = _engine(d_dir)
    eng.catalog.set_primary_key("t", ["k"])
    eng.sql("UPDATE t SET b = 7 WHERE k <> 2")
    rows = {r[0]: r for r in _rows(eng)}
    assert rows[1][1] == 7
    assert rows[3][1] == 7
    assert rows[2][1] == 20  # unmatched, untouched


def test_update_multiple_set(d_dir):
    """Multiple comma-separated SET assignments apply in one UPDATE."""
    eng = _engine(d_dir)
    eng.catalog.set_primary_key("t", ["k"])
    eng.sql("UPDATE t SET b = b + 5, label = 'U' WHERE k = 1")
    rows = {r[0]: r for r in _rows(eng)}
    assert rows[1] == (1, 15, "U")
    # only k=1 changed.
    assert rows[2] == (2, 20, "B")


# ------------------------------- the load-bearing invariant: SET keeps PK survives


def test_update_keeps_pk_survives_own_tombstone(d_dir):
    """A SET that does NOT change the PK is the central correctness case: the
    matched rows are tombstoned (by PK) AND re-inserted (same PK) under one
    commit_ts. The UPDATE tombstone is strict (``tomb_upd <= ins_ts``) so the
    re-inserted row at the same ts survives its own tombstone -- the row must
    NOT vanish. A naive same-ts DELETE-then-INSERT would delete the new row."""
    eng = _engine(d_dir)
    eng.catalog.set_primary_key("t", ["k"])
    base_n = _count(eng)
    n = eng.sql("UPDATE t SET b = b * 2")  # touches every row, keeps every PK
    assert n == base_n
    # every row still present (PKs unchanged), with doubled b.
    rows = {r[0]: r for r in _rows(eng)}
    assert rows[1] == (1, 20, "A")
    assert rows[2] == (2, 40, "B")
    assert rows[3] == (3, 60, None)
    assert _count(eng) == base_n  # no rows lost
    # a second identical UPDATE compounds (proves the rows truly survive, not a
    # one-shot fluke): b doubles again.
    eng.sql("UPDATE t SET b = b * 2")
    rows = {r[0]: r for r in _rows(eng)}
    assert rows[1][1] == 40
    assert rows[2][1] == 80
    assert rows[3][1] == 120


# --------------------------------------------------------------- PK-changing UPDATE


def test_update_changes_pk(d_dir):
    """A SET that changes the PK to an unused value succeeds (old PK tombstoned,
    new PK inserted; no collision with survivors)."""
    eng = _engine(d_dir)
    eng.catalog.set_primary_key("t", ["k"])
    n = eng.sql("UPDATE t SET k = 4 WHERE k = 1")
    assert n == 1
    assert _keys(eng) == [2, 3, 4]
    rows = {r[0]: r for r in _rows(eng)}
    assert rows[4] == (4, 10, "A")  # the non-PK columns carried over


def test_update_pk_collision_raises(d_dir):
    """A SET that changes the PK to a value already present (in a survivor) is
    rejected before any durable state is written (UNIQUE violation), and the
    table is left unchanged."""
    eng = _engine(d_dir)
    eng.catalog.set_primary_key("t", ["k"])
    base_rows = _rows(eng)
    with pytest.raises(RuntimeError, match="UNIQUE violation"):
        eng.sql("UPDATE t SET k = 2 WHERE k = 1")  # 2 already exists
    # nothing applied: rows unchanged, no live tombstone/insert left behind.
    assert _rows(eng) == base_rows
    assert not eng.delta.has_tombstones("t")
    assert not eng.delta.batches_with_ts("t")


# --------------------------------------------------------------- UPDATE requires PK


def test_update_requires_pk(d_dir):
    eng = _engine(d_dir)  # no declared PK
    with pytest.raises(RuntimeError, match="requires a declared PRIMARY KEY"):
        eng.sql("UPDATE t SET b = 1 WHERE k = 1")
    # nothing written.
    assert _count(eng) == len(_BASE)
    assert not eng.delta.has_tombstones("t")


# ------------------------------------------------- explicit-txn UPDATE is NotImplementedError


def test_update_in_txn_not_implemented(d_dir):
    eng = _engine(d_dir)
    eng.catalog.set_primary_key("t", ["k"])
    eng.sql("BEGIN")
    with pytest.raises(NotImplementedError):
        eng.sql("UPDATE t SET b = 1 WHERE k = 1")
    # the explicit txn is still active (UPDATE did not corrupt txn state) and no
    # buffered write leaked; ROLLBACK returns to the base.
    eng.sql("ROLLBACK")
    assert _count(eng) == len(_BASE)
    assert not eng.delta.has_tombstones("t")


# --------------------------------------------------- no-match returns zero, no write


def test_update_no_match_returns_zero(d_dir):
    eng = _engine(d_dir)
    eng.catalog.set_primary_key("t", ["k"])
    base_rows = _rows(eng)
    n = eng.sql("UPDATE t SET b = 1 WHERE k = 999")
    assert n == 0
    assert _rows(eng) == base_rows
    assert not eng.delta.has_tombstones("t")
    assert not eng.delta.batches_with_ts("t")


# -------------------------------------------------------- WAL durability on restart


def test_update_survives_restart(d_dir):
    eng = _engine(d_dir)
    eng.catalog.set_primary_key("t", ["k"])
    eng.sql("UPDATE t SET b = 555, label = 'R' WHERE k = 3")
    rows_before = {r[0]: r for r in _rows(eng)}
    assert rows_before[3] == (3, 555, "R")
    # a fresh Engine replays the WAL -> the UPDATE tombstone + reinsert replay.
    eng2 = Engine(Catalog(d_dir))
    eng2.catalog.set_primary_key("t", ["k"])
    rows_after = {r[0]: r for r in _rows(eng2)}
    assert rows_after[3] == (3, 555, "R")
    assert _count(eng2) == len(_BASE)
    assert _keys(eng2) == [1, 2, 3]


# ----------------------------------------- checkpoint materializes the update into base


def test_update_then_checkpoint(d_dir):
    eng = _engine(d_dir)
    eng.catalog.set_primary_key("t", ["k"])
    eng.sql("UPDATE t SET b = b + 1 WHERE k = 1")
    assert eng.delta.has_tombstones("t")  # UPDATE left a live tombstone + insert
    assert eng.delta.has_unflushed("t")
    rep = eng.checkpoint()
    assert rep == {"t": len(_BASE)}  # net row count unchanged by an UPDATE
    # tombstone channel + insert delta cleared; WAL truncated to 0.
    assert not eng.delta.has_tombstones("t")
    assert not eng.delta.has_unflushed("t")
    wal_size = os.path.getsize(os.path.join(d_dir, "ryudb.wal")) if os.path.exists(
        os.path.join(d_dir, "ryudb.wal")
    ) else 0
    assert wal_size == 0
    # the updated value is materialized into the new base.
    rows = {r[0]: r for r in _rows(eng)}
    assert rows[1] == (1, 11, "A")
    # restart-from-base sees the update (no WAL replay needed).
    eng2 = Engine(Catalog(d_dir))
    rows2 = {r[0]: r for r in _rows(eng2)}
    assert rows2[1] == (1, 11, "A")


# ------------------------------------- restore_to boundary: single-ts is durable


def test_update_restore_to_boundary(d_dir):
    """Restoring to the UPDATE's own commit_ts shows the UPDATE applied (both the
    reinsert AND its UPDATE tombstone share that ts, so the reinsert survives);
    restoring to one-before drops both -> the pre-UPDATE state. This is the case
    the two-ts split got wrong (tombstone at T, insert at T+1 -> restore_to(T)
    kept the tombstone but dropped the WAL insert -> deleted row reappeared on
    restart). Single-ts keeps delta and WAL in lockstep."""
    eng = _engine(d_dir)
    eng.catalog.set_primary_key("t", ["k"])
    eng.sql("UPDATE t SET b = 100 WHERE k = 1")  # ts=1
    ts_after = eng._commit_ts
    assert ts_after == 1
    # restore to ts=1 -> the UPDATE is still applied (reinsert at ts=1 survives
    # its own strict tombstone at ts=1).
    eng.restore_to(1)
    rows = {r[0]: r for r in _rows(eng)}
    assert rows[1] == (1, 100, "A")
    assert _count(eng) == len(_BASE)
    # restore to ts=0 -> before the UPDATE: the base row is back, both the
    # tombstone and the reinsert are discarded (they share ts=1 > 0).
    eng.restore_to(0)
    rows = {r[0]: r for r in _rows(eng)}
    assert rows[1] == (1, 10, "A")
    assert _count(eng) == len(_BASE)
    # and it survives a restart after the restore (WAL truncated consistently).
    eng2 = Engine(Catalog(d_dir))
    rows2 = {r[0]: r for r in _rows(eng2)}
    assert rows2[1] == (1, 10, "A")


# ------------------------------------- update then update the same row again


def test_update_then_update_same_row(d_dir):
    """Two sequential UPDATEs of the same PK compound: each is its own commit_ts,
    each tombstone+reinsert pair. The second UPDATE's tombstone must remove the
    first UPDATE's reinserted row (its ins_ts = ts1 < ts2) and the second
    reinsert (ins_ts = ts2) survives its own ts2 tombstone."""
    eng = _engine(d_dir)
    eng.catalog.set_primary_key("t", ["k"])
    eng.sql("UPDATE t SET b = 11 WHERE k = 1")  # ts=1
    eng.sql("UPDATE t SET b = 12 WHERE k = 1")  # ts=2
    rows = {r[0]: r for r in _rows(eng)}
    assert rows[1] == (1, 12, "A")
    assert _count(eng) == len(_BASE)  # exactly one row for k=1, not two


# --------------------------------------------------------------- composite PK


def test_update_composite_pk(d_dir):
    eng = _engine(d_dir)
    eng.catalog.set_primary_key("t", ["k", "b"])
    # update exactly (k=2, b=20); a row sharing k but differing b is unaffected.
    n = eng.sql("UPDATE t SET label = 'C' WHERE k = 2 AND b = 20")
    assert n == 1
    rows = {r[0]: r for r in _rows(eng)}
    assert rows[2] == (2, 20, "C")
    assert rows[1] == (1, 10, "A")
    assert rows[3] == (3, 30, None)
    # changing one PK column to a FRESH composite value (no survivor has it)
    # succeeds: (k=2, b=21) is unused, so no UNIQUE violation.
    eng.sql("UPDATE t SET b = 21 WHERE k = 2")
    rows = {r[0]: r for r in _rows(eng)}
    assert rows[2] == (2, 21, "C")


def test_update_composite_pk_collision(d_dir):
    """Composite-PK UPDATE that would create a duplicate composite key is
    rejected (separates the collision case from the non-collision above)."""
    eng = _engine(d_dir)
    eng.catalog.set_primary_key("t", ["k", "b"])
    base_rows = _rows(eng)
    # move (k=2,b=20) onto (k=1,b=10)'s composite key -> collision.
    with pytest.raises(RuntimeError, match="UNIQUE violation"):
        eng.sql("UPDATE t SET k = 1, b = 10 WHERE k = 2")
    # unchanged, nothing left behind.
    assert _rows(eng) == base_rows
    assert not eng.delta.has_tombstones("t")


# --------------------------------------------------------------- CLI smoke


def test_cli_update_output(d_dir, capsys):
    from ryudb import cli

    eng = _engine(d_dir)
    eng.catalog.set_primary_key("t", ["k"])
    rc = cli._run_statement(eng, "UPDATE t SET b = 42 WHERE k = 1", quiet=False)
    assert rc == 0
    assert "updated 1 rows" in capsys.readouterr().out
    rows = {r[0]: r for r in _rows(eng)}
    assert rows[1][1] == 42