"""Phase 2 step 8: declared PRIMARY KEY / UNIQUE enforcement on INSERT.

``_insert`` now calls ``_enforce_unique`` BEFORE the WAL write / buffer append,
rejecting any INSERT row that duplicates an existing visible value of a declared
PK/UNIQUE key (or duplicates within the batch). Declared-constraints-only: the
data-uniqueness ``is_unique_key``/``_pk_cache`` facts (the fused star-join's
dimension-PK eligibility gate) are NOT consulted, so a duplicate insert into a
column that merely happens to hold unique data -- with no declared constraint --
still succeeds (test_no_constraint_noop). PK columns are NOT NULL; UNIQUE
columns are nullable and NULLs are exempt (NULL != NULL).

Each test uses a fresh function-scoped ``tmp_path`` with its own small writable
base + catalog, so the declared constraints never leak into the shared
``data_dir`` catalog file (mirrors ``test_insert.py``'s ``_fresh_engine``).
"""

from __future__ import annotations

import os

import cudf
import pandas as pd
import pytest

from ryudb import Catalog, Engine

# A tiny writable base: t(k BIGINT, b BIGINT, label nullable str). Seed rows
# k=1,2,3 with labels 'A','B',NULL -- enough to exercise collision, NULL-exempt
# UNIQUE, and composite-PK partial overlap.
_BASE = [
    (1, 10, "A"),
    (2, 20, "B"),
    (3, 30, None),
]


@pytest.fixture
def u_dir(tmp_path) -> str:
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


def _engine(u_dir: str) -> Engine:
    cat = Catalog(u_dir)
    cat.register("t", os.path.join(u_dir, "t"))
    return Engine(cat)


def _count(eng: Engine) -> int:
    return int(eng.sql("SELECT count(*) AS n FROM t").to_pandas()["n"].iloc[0])


# --------------------------------------------------------------- PK autocommit


def test_pk_duplicate_rejected_autocommit(u_dir):
    eng = _engine(u_dir)
    eng.catalog.set_primary_key("t", ["k"])
    base_n = _count(eng)
    # k=1 already exists in the base -> rejected.
    with pytest.raises(RuntimeError, match="UNIQUE violation"):
        eng.sql("INSERT INTO t (k, b, label) VALUES (1, 99, 'dup')")
    # All-or-nothing: nothing was written.
    assert _count(eng) == base_n
    assert not eng.delta.has_unflushed("t")
    # A fresh key is accepted.
    eng.sql("INSERT INTO t (k, b, label) VALUES (999, 99, 'new')")
    assert _count(eng) == base_n + 1


def test_pk_duplicate_rejected_in_txn(u_dir):
    eng = _engine(u_dir)
    eng.catalog.set_primary_key("t", ["k"])
    base_n = _count(eng)
    eng.sql("BEGIN")
    with pytest.raises(RuntimeError, match="UNIQUE violation"):
        eng.sql("INSERT INTO t (k, b, label) VALUES (1, 99, 'dup')")
    # No partial buffer: the txn's write set is empty.
    assert not eng._txn.has("t")
    eng.sql("ROLLBACK")
    assert _count(eng) == base_n


# --------------------------------------------------------------------- UNIQUE


def test_unique_duplicate_rejected(u_dir):
    eng = _engine(u_dir)
    eng.catalog.set_unique("t", ["label"])
    base_n = _count(eng)
    # 'A' already exists -> rejected.
    with pytest.raises(RuntimeError, match="UNIQUE violation"):
        eng.sql("INSERT INTO t (k, b, label) VALUES (999, 99, 'A')")
    assert _count(eng) == base_n
    # NULLs are exempt: two NULL labels are both accepted (NULL != NULL).
    eng.sql("INSERT INTO t (k, b, label) VALUES (1001, 1, NULL)")
    eng.sql("INSERT INTO t (k, b, label) VALUES (1002, 2, NULL)")
    assert _count(eng) == base_n + 2
    # A fresh non-null label is accepted.
    eng.sql("INSERT INTO t (k, b, label) VALUES (1003, 3, 'Z')")
    assert _count(eng) == base_n + 3


# ---------------------------------------------------------------- composite PK


def test_composite_pk(u_dir):
    eng = _engine(u_dir)
    eng.catalog.set_primary_key("t", ["k", "b"])
    base_n = _count(eng)
    # (1, 10) already exists -> rejected.
    with pytest.raises(RuntimeError, match="UNIQUE violation"):
        eng.sql("INSERT INTO t (k, b, label) VALUES (1, 10, 'dup')")
    assert _count(eng) == base_n
    # Only k collides, b differs -> accepted (composite key not duplicated).
    eng.sql("INSERT INTO t (k, b, label) VALUES (1, 99, 'ok')")
    assert _count(eng) == base_n + 1


# ------------------------------------------------------- internal batch dup


def test_internal_dup_in_batch_rejected(u_dir):
    eng = _engine(u_dir)
    eng.catalog.set_primary_key("t", ["k"])
    base_n = _count(eng)
    # Two rows in one INSERT with the same key -> internal-dup check fires.
    with pytest.raises(RuntimeError, match="duplicate within INSERT batch"):
        eng.sql(
            "INSERT INTO t (k, b, label) VALUES (9001, 1, 'x'), (9001, 2, 'y')"
        )
    assert _count(eng) == base_n
    assert not eng.delta.has_unflushed("t")


# --------------------------------------------------- declared-constraints-only


def test_no_constraint_noop(u_dir):
    """No declared PK/UNIQUE -> a duplicate of a base value is accepted (the
    data-uniqueness is_unique_key/_pk_cache facts are NOT consulted)."""
    eng = _engine(u_dir)
    base_n = _count(eng)
    eng.sql("INSERT INTO t (k, b, label) VALUES (1, 99, 'A')")  # k=1, label 'A' both dup
    assert _count(eng) == base_n + 1


# ----------------------------------------------- read-your-writes buffer collision


def test_read_your_writes_catches_buffer_collision(u_dir):
    eng = _engine(u_dir)
    eng.catalog.set_primary_key("t", ["k"])
    base_n = _count(eng)
    eng.sql("BEGIN")
    eng.sql("INSERT INTO t (k, b, label) VALUES (5000, 1, 'first')")  # buffered
    # A 2nd in-txn INSERT colliding with the buffered row is caught via
    # read-your-writes (_scan sees the buffer).
    with pytest.raises(RuntimeError, match="UNIQUE violation"):
        eng.sql("INSERT INTO t (k, b, label) VALUES (5000, 2, 'second')")
    eng.sql("ROLLBACK")
    assert _count(eng) == base_n


# --------------------------------------------- PK NOT NULL composes with enforce


def test_pk_not_null_still_enforced(u_dir):
    """A NULL PK value is rejected by the NOT NULL path (PK cols are NOT NULL),
    not by _enforce_unique -- the two checks compose."""
    eng = _engine(u_dir)
    eng.catalog.set_primary_key("t", ["k"])
    with pytest.raises(RuntimeError, match="NOT NULL"):
        eng.sql("INSERT INTO t (k, b, label) VALUES (NULL, 1, 'x')")


# --------------------------------------------------------------- CLI smoke


def test_cli_alter_pk_then_duplicate_rejected(u_dir, capsys):
    from ryudb import cli

    eng = _engine(u_dir)
    base_n = _count(eng)
    # Declare the PK via the :alter dot-command (the CLI surface).
    assert cli._dot_command("alter t pk k", eng, eng.catalog) is False
    out = capsys.readouterr().out
    assert "primary key set to ['k']" in out
    # A duplicate-PK INSERT through the CLI statement runner errors out.
    rc = cli._run_statement(
        eng, "INSERT INTO t (k, b, label) VALUES (1, 99, 'dup')", quiet=False
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "UNIQUE violation" in err
    # A fresh key is accepted and prints the row count.
    rc = cli._run_statement(
        eng, "INSERT INTO t (k, b, label) VALUES (7777, 1, 'new')", quiet=False
    )
    assert rc == 0
    assert "inserted 1 rows" in capsys.readouterr().out
    assert _count(eng) == base_n + 1