"""MERGE DML (UPSERT): ``MERGE INTO target USING source ON ... WHEN MATCHED ...
/ WHEN NOT MATCHED ...``.

A MERGE joins the visible target to the materialized source on the ON equi-keys
(plus an optional non-equi residual), splits matched vs not-matched, and applies
per-row arms: a ``WHEN MATCHED THEN UPDATE`` is ``_update``'s one-``commit_ts``
tombstone-old + insert-new pattern (the ``exclude_same_ts`` tombstone lets each
reinserted row survive its own tombstone via ``_merge_delta``'s
``_tomb_upd <= ins_ts`` rule); a ``WHEN MATCHED THEN DELETE`` is ``_delete``'s PK
tombstone; a ``WHEN NOT MATCHED THEN INSERT`` is ``_insert``'s typed-batch path on
the unmatched source rows. The whole statement flushes atomically under ONE
``commit_ts`` via the implicit-txn + ``_write_commit`` seam, and ``_enforce_unique``
runs on the combined new frame after the tombstones are buffered (read-your-writes
sees the old matched rows gone -> no false self-collision when SET keeps the PK).

v1 scope: at most one ``WHEN MATCHED`` arm (UPDATE/DELETE) + one ``WHEN NOT
MATCHED [BY TARGET]`` arm (INSERT); autocommit only; ON is equi-conjunctive.
DuckDB is the oracle. Each test uses a fresh ``tmp_path`` with its own writable
base + catalog (mirrors ``test_update.py``'s ``d_dir``), so the declared PK never
leaks into the shared ``data_dir`` catalog.
"""

from __future__ import annotations

import os

import cudf
import duckdb
import pandas as pd
import pytest

from ryudb import Catalog, Engine

from .conftest import assert_same

# Target t(k BIGINT, b BIGINT, label nullable str): k=1,2,3 with labels A,B,NULL.
# Source s(k, b, label): k=2,3 match (update), k=4,5 do not (insert); k=4 has a
# NULL label to exercise NULL round-trip on the INSERT arm.
_BASE_T = [
    (1, 10, "A"),
    (2, 20, "B"),
    (3, 30, None),
]
_BASE_S = [
    (2, 200, "BB"),
    (3, 300, "CC"),
    (4, 400, None),
    (5, 500, "EE"),
]


@pytest.fixture
def m_dir(tmp_path) -> str:
    d = tmp_path
    (d / "t").mkdir()
    (d / "s").mkdir()
    cudf.DataFrame(
        {
            "k": [r[0] for r in _BASE_T],
            "b": [r[1] for r in _BASE_T],
            "label": pd.array([r[2] for r in _BASE_T], dtype=object),
        }
    ).to_pandas().to_parquet(d / "t" / "0.parquet")
    cudf.DataFrame(
        {
            "k": [r[0] for r in _BASE_S],
            "b": [r[1] for r in _BASE_S],
            "label": pd.array([r[2] for r in _BASE_S], dtype=object),
        }
    ).to_pandas().to_parquet(d / "s" / "0.parquet")
    return str(d)


def _engine(m_dir: str) -> Engine:
    cat = Catalog(m_dir)
    cat.register("t", os.path.join(m_dir, "t"))
    cat.register("s", os.path.join(m_dir, "s"))
    return Engine(cat)


def _duck(m_dir: str) -> duckdb.DuckDBPyConnection:
    """Writable DuckDB oracle: ``t_w`` / ``s_w`` materialized from the parquet so
    MERGE can mutate them (the read_parquet views are read-only)."""
    con = duckdb.connect()
    con.execute(f"CREATE TABLE t_w AS SELECT * FROM read_parquet('{m_dir}/t/0.parquet')")
    con.execute(f"CREATE TABLE s_w AS SELECT * FROM read_parquet('{m_dir}/s/0.parquet')")
    return con


def _count(eng: Engine) -> int:
    return int(eng.sql("SELECT count(*) AS n FROM t").to_pandas()["n"].iloc[0])


def _keys(eng: Engine) -> list[int]:
    return list(eng.sql("SELECT k FROM t ORDER BY k").to_pandas()["k"])


def _rows(eng: Engine) -> list[tuple]:
    """All rows as (k, b, label) sorted by k, for value-level assertions."""
    df = eng.sql("SELECT k, b, label FROM t ORDER BY k").to_pandas()
    return [(int(r.k), int(r.b), None if pd.isna(r.label) else r.label) for r in df.itertuples()]


def _rew(sql: str) -> str:
    """Rewrite a bare ``MERGE INTO t USING s ON t.k=s.k ...`` to the DuckDB
    ``t_w`` / ``s_w`` oracle (qualified refs t.*/s.* -> t_w.*/s_w.*)."""
    return (
        sql.replace("INTO t ", "INTO t_w ")
        .replace("USING s ", "USING s_w ")
        .replace("t.k = s.k", "t_w.k = s_w.k")
        .replace("s.b", "s_w.b")
        .replace("s.label", "s_w.label")
        .replace("s.k", "s_w.k")
    )


def _cmp(m_dir, ryu_sql: str, duck_sql: str | None = None) -> int:
    """Run a MERGE on both engines and assert the full visible tables match."""
    eng = _engine(m_dir)
    eng.catalog.set_primary_key("t", ["k"])
    duck = _duck(m_dir)
    n = eng.sql(ryu_sql)
    duck.execute(duck_sql if duck_sql is not None else _rew(ryu_sql))
    q = "SELECT k, b, label FROM t ORDER BY k"
    assert_same(eng.sql(q), duck.execute(q.replace(" t ", " t_w ")).fetchdf())
    return n


# --------------------------------------------------------------------------- #
# The canonical UPSERT: WHEN MATCHED UPDATE + WHEN NOT MATCHED INSERT
# --------------------------------------------------------------------------- #


def test_merge_upsert(m_dir):
    """Update the matched rows (k=2,3) and insert the unmatched ones (k=4,5),
    including a NULL label on the inserted k=4. Matches DuckDB on the full table."""
    sql = (
        "MERGE INTO t USING s ON t.k = s.k "
        "WHEN MATCHED THEN UPDATE SET b = s.b, label = s.label "
        "WHEN NOT MATCHED THEN INSERT (k, b, label) VALUES (s.k, s.b, s.label)"
    )
    n = _cmp(m_dir, sql)
    assert n == 4  # 2 updated + 2 inserted


def test_merge_upsert_returns_affected_count(m_dir):
    eng = _engine(m_dir)
    eng.catalog.set_primary_key("t", ["k"])
    sql = (
        "MERGE INTO t USING s ON t.k = s.k "
        "WHEN MATCHED THEN UPDATE SET b = s.b "
        "WHEN NOT MATCHED THEN INSERT (k, b) VALUES (s.k, s.b)"
    )
    assert eng.sql(sql) == 4


def test_merge_upsert_value_level(m_dir):
    """Pin the exact post-MERGE rows (the NULL label on k=4 survives the round-trip)."""
    eng = _engine(m_dir)
    eng.catalog.set_primary_key("t", ["k"])
    eng.sql(
        "MERGE INTO t USING s ON t.k = s.k "
        "WHEN MATCHED THEN UPDATE SET b = s.b, label = s.label "
        "WHEN NOT MATCHED THEN INSERT (k, b, label) VALUES (s.k, s.b, s.label)"
    )
    rows = {r[0]: r for r in _rows(eng)}
    assert rows[1] == (1, 10, "A")       # untouched
    assert rows[2] == (2, 200, "BB")     # updated
    assert rows[3] == (3, 300, "CC")     # updated
    assert rows[4] == (4, 400, None)     # inserted, NULL label
    assert rows[5] == (5, 500, "EE")     # inserted
    assert _count(eng) == 5


# --------------------------------------------------------------------------- #
# Individual arms
# --------------------------------------------------------------------------- #


def test_merge_matched_update_only(m_dir):
    sql = "MERGE INTO t USING s ON t.k = s.k WHEN MATCHED THEN UPDATE SET b = s.b"
    n = _cmp(m_dir, sql)
    assert n == 2  # only the 2 matched rows
    eng = _engine(m_dir)
    eng.catalog.set_primary_key("t", ["k"])
    eng.sql(sql)
    assert _count(eng) == 3  # no inserts -> row count unchanged


def test_merge_not_matched_insert_only(m_dir):
    sql = (
        "MERGE INTO t USING s ON t.k = s.k "
        "WHEN NOT MATCHED THEN INSERT (k, b, label) VALUES (s.k, s.b, s.label)"
    )
    n = _cmp(m_dir, sql)
    assert n == 2  # only the 2 unmatched source rows
    eng = _engine(m_dir)
    eng.catalog.set_primary_key("t", ["k"])
    eng.sql(sql)
    assert _count(eng) == 5
    assert not eng.delta.has_tombstones("t")  # pure insert -> no tombstones


def test_merge_matched_delete(m_dir):
    sql = "MERGE INTO t USING s ON t.k = s.k WHEN MATCHED THEN DELETE"
    n = _cmp(m_dir, sql)
    assert n == 2
    eng = _engine(m_dir)
    eng.catalog.set_primary_key("t", ["k"])
    eng.sql(sql)
    assert _count(eng) == 1  # k=2,3 deleted; k=1 survives
    assert _keys(eng) == [1]
    assert eng.delta.has_tombstones("t")
    assert not eng.delta.batches_with_ts("t")  # pure delete -> no insert batches


# --------------------------------------------------------------------------- #
# AND conditions on the WHEN clauses
# --------------------------------------------------------------------------- #


def test_merge_matched_and_condition(m_dir):
    """Only matched rows whose source b > 250 are updated (k=3, not k=2)."""
    sql = (
        "MERGE INTO t USING s ON t.k = s.k "
        "WHEN MATCHED AND s.b > 250 THEN UPDATE SET b = s.b"
    )
    n = _cmp(m_dir, sql)
    assert n == 1
    eng = _engine(m_dir)
    eng.catalog.set_primary_key("t", ["k"])
    eng.sql(sql)
    rows = {r[0]: r for r in _rows(eng)}
    assert rows[2] == (2, 20, "B")    # NOT updated (s.b=200 not > 250)
    assert rows[3] == (3, 300, None)  # b updated (s.b=300), label untouched (NULL)


def test_merge_not_matched_and_condition(m_dir):
    """Only unmatched source rows with k > 3 are inserted (k=4,5; k=2,3 match)."""
    sql = (
        "MERGE INTO t USING s ON t.k = s.k "
        "WHEN NOT MATCHED AND s.k > 3 THEN INSERT (k, b) VALUES (s.k, s.b)"
    )
    n = _cmp(m_dir, sql)
    assert n == 2


# --------------------------------------------------------------------------- #
# ON residual (non-equi AND) + subquery source + aliases
# --------------------------------------------------------------------------- #


def test_merge_on_residual(m_dir):
    """An ON with an extra non-equi conjunct (s.b > 250) narrows the match: only
    k=3 matches (s.b=300); k=2 (s.b=200) does NOT match -> left untouched (no
    NOT MATCHED arm, so no insert / no key collision)."""
    sql = (
        "MERGE INTO t USING s ON t.k = s.k AND s.b > 250 "
        "WHEN MATCHED THEN UPDATE SET b = s.b"
    )
    eng = _engine(m_dir)
    eng.catalog.set_primary_key("t", ["k"])
    duck = _duck(m_dir)
    n = eng.sql(sql)
    duck.execute(_rew(sql))
    q = "SELECT k, b, label FROM t ORDER BY k"
    assert_same(eng.sql(q), duck.execute(q.replace(" t ", " t_w ")).fetchdf())
    assert n == 1  # only k=3 matched
    rows = {r[0]: r for r in _rows(eng)}
    assert rows[2] == (2, 20, "B")   # NOT matched (s.b=200) -> unchanged
    assert rows[3][1] == 300         # matched -> b updated


def test_merge_subquery_source(m_dir):
    """USING a FROM-subquery (a Derived subplan) as the source."""
    ryu = (
        "MERGE INTO t AS tgt "
        "USING (SELECT k, b, label FROM s WHERE k IN (4, 5)) AS src "
        "ON tgt.k = src.k "
        "WHEN MATCHED THEN UPDATE SET b = src.b "
        "WHEN NOT MATCHED THEN INSERT (k, b, label) VALUES (src.k, src.b, src.label)"
    )
    duck_sql = (
        "MERGE INTO t_w AS tgt "
        "USING (SELECT k, b, label FROM s_w WHERE k IN (4, 5)) AS src "
        "ON tgt.k = src.k "
        "WHEN MATCHED THEN UPDATE SET b = src.b "
        "WHEN NOT MATCHED THEN INSERT (k, b, label) VALUES (src.k, src.b, src.label)"
    )
    n = _cmp(m_dir, ryu, duck_sql)
    assert n == 2  # both unmatched -> 2 inserts


def test_merge_aliases(m_dir):
    """Target/source correlation names (tgt/src) route via {alias}__col."""
    ryu = (
        "MERGE INTO t AS tgt USING s AS src ON tgt.k = src.k "
        "WHEN MATCHED THEN UPDATE SET b = src.b, label = src.label "
        "WHEN NOT MATCHED THEN INSERT (k, b, label) VALUES (src.k, src.b, src.label)"
    )
    duck_sql = (
        "MERGE INTO t_w AS tgt USING s_w AS src ON tgt.k = src.k "
        "WHEN MATCHED THEN UPDATE SET b = src.b, label = src.label "
        "WHEN NOT MATCHED THEN INSERT (k, b, label) VALUES (src.k, src.b, src.label)"
    )
    assert _cmp(m_dir, ryu, duck_sql) == 4


# --------------------------------------------------------------------------- #
# Invariants: idempotency, PK survival, SET keeps PK
# --------------------------------------------------------------------------- #


def test_merge_idempotent(m_dir):
    """Running the same UPSERT twice is stable: the 2nd run matches all 4 source
    rows (now present) -> 4 updates, 0 inserts; row count stays 5."""
    eng = _engine(m_dir)
    eng.catalog.set_primary_key("t", ["k"])
    sql = (
        "MERGE INTO t USING s ON t.k = s.k "
        "WHEN MATCHED THEN UPDATE SET b = s.b, label = s.label "
        "WHEN NOT MATCHED THEN INSERT (k, b, label) VALUES (s.k, s.b, s.label)"
    )
    assert eng.sql(sql) == 4
    assert _count(eng) == 5
    assert eng.sql(sql) == 4  # 4 updates, 0 inserts
    assert _count(eng) == 5


def test_merge_update_keeps_pk_survives_own_tombstone(m_dir):
    """A WHEN MATCHED UPDATE that does NOT change the PK is the central case: the
    matched rows are tombstoned (by PK) AND re-inserted (same PK) under one
    commit_ts. The UPDATE tombstone is strict (tomb_upd <= ins_ts) so the
    re-inserted row survives -- it must NOT vanish (mirrors
    test_update_keeps_pk_survives_own_tombstone)."""
    eng = _engine(m_dir)
    eng.catalog.set_primary_key("t", ["k"])
    base_n = _count(eng)
    n = eng.sql(
        "MERGE INTO t USING s ON t.k = s.k "
        "WHEN MATCHED THEN UPDATE SET b = b * 2"  # keeps PK, mutates b
    )
    assert n == 2
    assert _count(eng) == base_n  # no rows lost
    rows = {r[0]: r for r in _rows(eng)}
    assert rows[2] == (2, 40, "B")  # 20 * 2
    assert rows[3] == (3, 60, None)  # 30 * 2, NULL label preserved
    assert rows[1] == (1, 10, "A")  # untouched


# --------------------------------------------------------------------------- #
# Rejections: PK/UNIQUE violations leave no partial state
# --------------------------------------------------------------------------- #


def test_merge_insert_collision_with_surviving_row(m_dir):
    """A WHEN NOT MATCHED INSERT whose key collides with a SURVIVING (unmatched)
    target row raises UNIQUE violation and leaves no partial state. Source s2 has
    only k=4 (unmatched); INSERT produces k=1, which collides with the surviving
    t.k=1."""
    eng = _engine(m_dir)
    eng.catalog.set_primary_key("t", ["k"])
    # Custom source with a single unmatched row.
    import tempfile

    d2 = tempfile.mkdtemp()
    os.makedirs(os.path.join(d2, "s2"))
    cudf.DataFrame({"k": [4], "b": [400], "label": ["DD"]}).to_pandas().to_parquet(
        os.path.join(d2, "s2", "0.parquet")
    )
    eng.catalog.register("s2", os.path.join(d2, "s2"))
    base_rows = _rows(eng)
    with pytest.raises(RuntimeError, match="UNIQUE violation"):
        eng.sql(
            "MERGE INTO t USING s2 ON t.k = s2.k "
            "WHEN NOT MATCHED THEN INSERT (k, b) VALUES (1, 99)"
        )
    assert _rows(eng) == base_rows  # unchanged
    assert not eng.delta.has_tombstones("t")
    assert not eng.delta.batches_with_ts("t")


def test_merge_insert_internal_duplicate(m_dir):
    """Two unmatched source rows whose INSERT produces the SAME key raise
    'duplicate within INSERT batch' (the new-frame internal-dup check)."""
    eng = _engine(m_dir)
    eng.catalog.set_primary_key("t", ["k"])
    import tempfile

    d2 = tempfile.mkdtemp()
    os.makedirs(os.path.join(d2, "s2"))
    # k=4,5 are both unmatched (t has 1,2,3); INSERT VALUES (1, 99) for both -> dup.
    cudf.DataFrame({"k": [4, 5], "b": [40, 50], "label": ["D", "E"]}).to_pandas().to_parquet(
        os.path.join(d2, "s2", "0.parquet")
    )
    eng.catalog.register("s2", os.path.join(d2, "s2"))
    with pytest.raises(RuntimeError, match="duplicate within INSERT batch"):
        eng.sql(
            "MERGE INTO t USING s2 ON t.k = s2.k "
            "WHEN NOT MATCHED THEN INSERT (k, b) VALUES (1, 99)"
        )
    assert not eng.delta.has_tombstones("t")
    assert not eng.delta.batches_with_ts("t")


def test_merge_update_pk_collision(m_dir):
    """A WHEN MATCHED UPDATE that changes the PK to a surviving row's PK raises
    UNIQUE violation and leaves no partial state (mirrors test_update.py:179)."""
    eng = _engine(m_dir)
    eng.catalog.set_primary_key("t", ["k"])
    import tempfile

    d2 = tempfile.mkdtemp()
    os.makedirs(os.path.join(d2, "s2"))
    # k=1 matches t.k=1; SET k=2 collides with the surviving t.k=2.
    cudf.DataFrame({"k": [1], "b": [100], "label": ["Z"]}).to_pandas().to_parquet(
        os.path.join(d2, "s2", "0.parquet")
    )
    eng.catalog.register("s2", os.path.join(d2, "s2"))
    base_rows = _rows(eng)
    with pytest.raises(RuntimeError, match="UNIQUE violation"):
        eng.sql(
            "MERGE INTO t USING s2 ON t.k = s2.k "
            "WHEN MATCHED THEN UPDATE SET k = 2"
        )
    assert _rows(eng) == base_rows
    assert not eng.delta.has_tombstones("t")
    assert not eng.delta.batches_with_ts("t")


def test_merge_cardinality_violation(m_dir):
    """A target row matched by >1 source row is a cardinality violation (matches
    DuckDB) and leaves no partial state."""
    eng = _engine(m_dir)
    eng.catalog.set_primary_key("t", ["k"])
    import tempfile

    d2 = tempfile.mkdtemp()
    os.makedirs(os.path.join(d2, "s2"))
    # Two source rows with k=2 both match t.k=2.
    cudf.DataFrame({"k": [2, 2], "b": [1, 2], "label": ["x", "y"]}).to_pandas().to_parquet(
        os.path.join(d2, "s2", "0.parquet")
    )
    eng.catalog.register("s2", os.path.join(d2, "s2"))
    with pytest.raises(RuntimeError, match="cardinality violation"):
        eng.sql(
            "MERGE INTO t USING s2 ON t.k = s2.k "
            "WHEN MATCHED THEN UPDATE SET b = s2.b"
        )
    assert not eng.delta.has_tombstones("t")
    assert not eng.delta.batches_with_ts("t")


def test_merge_requires_pk(m_dir):
    """A WHEN MATCHED UPDATE/DELETE arm requires a declared PRIMARY KEY."""
    eng = _engine(m_dir)  # no PK declared
    with pytest.raises(RuntimeError, match="requires a declared PRIMARY KEY"):
        eng.sql("MERGE INTO t USING s ON t.k = s.k WHEN MATCHED THEN UPDATE SET b = s.b")
    with pytest.raises(RuntimeError, match="requires a declared PRIMARY KEY"):
        eng.sql("MERGE INTO t USING s ON t.k = s.k WHEN MATCHED THEN DELETE")


def test_merge_explicit_txn_rejected(m_dir):
    """MERGE inside an explicit transaction is not supported in v1 (mirrors UPDATE)."""
    eng = _engine(m_dir)
    eng.catalog.set_primary_key("t", ["k"])
    eng.sql("BEGIN")
    with pytest.raises(NotImplementedError):
        eng.sql(
            "MERGE INTO t USING s ON t.k = s.k "
            "WHEN MATCHED THEN UPDATE SET b = s.b"
        )
    eng.sql("ROLLBACK")


# --------------------------------------------------------------------------- #
# Parse rejections (deferred forms)
# --------------------------------------------------------------------------- #


def test_merge_parse_non_equi_on():
    from ryudb.sql.parse import parse

    schema = {"t": ["k", "b"], "s": ["k", "b"]}
    with pytest.raises(NotImplementedError, match="equi-predicates"):
        parse("MERGE INTO t USING s ON t.k > s.k WHEN MATCHED THEN DELETE", schema)


def test_merge_parse_unqualified_on():
    from ryudb.sql.parse import parse

    schema = {"t": ["k", "b"], "s": ["k", "b"]}
    with pytest.raises(NotImplementedError, match="qualified"):
        parse("MERGE INTO t USING s ON k = k WHEN MATCHED THEN DELETE", schema)


def test_merge_parse_returning_rejected():
    from ryudb.sql.parse import parse

    schema = {"t": ["k", "b"], "s": ["k", "b"]}
    with pytest.raises(NotImplementedError, match="RETURNING"):
        parse(
            "MERGE INTO t USING s ON t.k = s.k WHEN MATCHED THEN DELETE RETURNING *",
            schema,
        )


def test_merge_parse_not_matched_by_source():
    from ryudb.sql.parse import parse

    schema = {"t": ["k", "b"], "s": ["k", "b"]}
    with pytest.raises(NotImplementedError, match="BY SOURCE"):
        parse(
            "MERGE INTO t USING s ON t.k = s.k "
            "WHEN NOT MATCHED BY SOURCE THEN DELETE",
            schema,
        )


def test_merge_parse_two_matched_arms():
    from ryudb.sql.parse import parse

    schema = {"t": ["k", "b"], "s": ["k", "b"]}
    with pytest.raises(NotImplementedError, match="multiple WHEN MATCHED"):
        parse(
            "MERGE INTO t USING s ON t.k = s.k "
            "WHEN MATCHED AND s.b > 5 THEN DELETE "
            "WHEN MATCHED THEN UPDATE SET b = s.b",
            schema,
        )


def test_merge_parse_matched_insert_cross():
    from ryudb.sql.parse import ParseError, parse

    schema = {"t": ["k", "b"], "s": ["k", "b"]}
    with pytest.raises(ParseError, match="WHEN MATCHED THEN INSERT"):
        parse(
            "MERGE INTO t USING s ON t.k = s.k "
            "WHEN MATCHED THEN INSERT (k) VALUES (s.k)",
            schema,
        )
    with pytest.raises(ParseError, match="WHEN NOT MATCHED THEN DELETE"):
        parse("MERGE INTO t USING s ON t.k = s.k WHEN NOT MATCHED THEN DELETE", schema)


def test_merge_parse_no_when():
    # sqlglot's grammar requires >=1 WHEN clause, so a WHEN-less MERGE is rejected
    # by sqlglot.parse itself (sqlglot.errors.ParseError, not RyuDB's ParseError)
    # before _build_merge runs -- the no-WHEN guard there is defensive.
    import sqlglot.errors

    from ryudb.sql.parse import parse

    schema = {"t": ["k", "b"], "s": ["k", "b"]}
    with pytest.raises(sqlglot.errors.ParseError, match="Whens"):
        parse("MERGE INTO t USING s ON t.k = s.k", schema)


# --------------------------------------------------------------------------- #
# CLI smoke
# --------------------------------------------------------------------------- #


def test_cli_merge_output(m_dir, capsys):
    from ryudb import cli

    eng = _engine(m_dir)
    eng.catalog.set_primary_key("t", ["k"])
    rc = cli._run_statement(
        eng,
        "MERGE INTO t USING s ON t.k = s.k "
        "WHEN MATCHED THEN UPDATE SET b = s.b "
        "WHEN NOT MATCHED THEN INSERT (k, b) VALUES (s.k, s.b)",
        quiet=False,
    )
    assert rc == 0
    assert "merged 4 rows" in capsys.readouterr().out
    rows = {r[0]: r for r in _rows(eng)}
    assert rows[2][1] == 200
    assert rows[4][1] == 400


def test_explain_merge(m_dir):
    """EXPLAIN on a MERGE pretty-prints the Merge node + its USING source."""
    eng = _engine(m_dir)
    eng.catalog.set_primary_key("t", ["k"])
    out = eng.explain(
        "MERGE INTO t USING s ON t.k = s.k "
        "WHEN MATCHED THEN UPDATE SET b = s.b "
        "WHEN NOT MATCHED THEN INSERT (k, b) VALUES (s.k, s.b)"
    )
    assert "Merge(t on k=k" in out
    assert "WHEN MATCHED" in out
    assert "WHEN NOT MATCHED" in out
    assert "Scan(s" in out


# --------------------------------------------------------------------------- #
# Regression: INSERT / UPDATE / DELETE still work on the same fixture
# --------------------------------------------------------------------------- #


def test_regression_insert_still_works(m_dir):
    eng = _engine(m_dir)
    eng.catalog.set_primary_key("t", ["k"])
    n = eng.sql("INSERT INTO t (k, b, label) VALUES (6, 60, 'F')")
    assert n == 1
    assert 6 in _keys(eng)


def test_regression_update_still_works(m_dir):
    eng = _engine(m_dir)
    eng.catalog.set_primary_key("t", ["k"])
    n = eng.sql("UPDATE t SET b = 99 WHERE k = 2")
    assert n == 1
    rows = {r[0]: r for r in _rows(eng)}
    assert rows[2] == (2, 99, "B")


def test_regression_delete_still_works(m_dir):
    eng = _engine(m_dir)
    eng.catalog.set_primary_key("t", ["k"])
    n = eng.sql("DELETE FROM t WHERE k = 2")
    assert n == 1
    assert 2 not in _keys(eng)