"""Composite-equi-correlated subqueries: ``EXISTS`` / ``NOT EXISTS`` whose
correlation is on more than one equality (a composite correlation key), e.g.

    SELECT ... FROM t tx
    WHERE EXISTS (SELECT 1 FROM s WHERE s.a = tx.a AND s.b = tx.b)

Phase E-3 originally lowered only a *single* equality correlation to a semi/
anti join. This extends it to a composite key: ``_classify_correlation`` now
collects every equi-correlation conjunct, the inner projection emits all of the
inner keys, and the join carries a multi-element ``on_left`` / ``on_right``.

The fused star-join kernel defers on semi/anti joins and on multi-key joins
(``fused.py``: ``how not in (inner,left,right,full) or len(n.on_left) != 1``),
so a composite correlation always routes to the cuDF fallback, where
``_semi_anti_join`` inner-merges on the equi-keys and keeps the left rows whose
row-id appears (semi) / does not appear (anti) among the matched pairs. NULL
semantics match DuckDB: a NULL in any key matches nothing (``=`` on NULL is
unknown), so a NULL-keyed outer row is excluded by EXISTS and kept by NOT
EXISTS. The right side of a decorrelated EXISTS projects only the equi-keys, so
no cross-input column collision arises.

Deferred (raise ``NotImplementedError``): a non-equi correlation
(``s.w <> tx.v``), and a correlated scalar with a composite key (multiple
equi-correlations). DuckDB is the oracle.
"""

from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from ryudb import Catalog, Engine
from ryudb.sql.parse import parse

from .conftest import as_sorted


@pytest.fixture
def cengine(tmp_path) -> tuple[Engine, duckdb.DuckDBPyConnection]:
    """``t(a, b, v)`` and ``s(a, b, w)`` sharing the composite key (a, b). The
    key sets overlap partially so EXISTS / NOT EXISTS partition ``t``."""
    d = tmp_path
    (d / "t").mkdir()
    (d / "s").mkdir()
    pd.DataFrame(
        {
            "a": [1, 1, 2, 2, 3, None],
            "b": [10, 20, 10, 20, 30, 40],
            "v": [100, 200, 300, 400, 500, 600],
        }
    ).to_parquet(str(d / "t" / "0.parquet"))
    pd.DataFrame(
        {
            "a": [1, 1, 2, 4, None],
            "b": [10, 20, 10, 40, 40],
            "w": [7, 8, 9, 11, 12],
        }
    ).to_parquet(str(d / "s" / "0.parquet"))
    cat = Catalog(str(d))
    cat.register("t", str(d / "t"))
    cat.register("s", str(d / "s"))
    eng = Engine(cat)
    duck = duckdb.connect()
    duck.execute(f"CREATE VIEW t AS SELECT * FROM read_parquet('{d}/t/*.parquet')")
    duck.execute(f"CREATE VIEW s AS SELECT * FROM read_parquet('{d}/s/*.parquet')")
    return eng, duck


def _ryu(eng: Engine, sql: str):
    return as_sorted(eng.sql(sql))


def _duck(duck, sql: str):
    return as_sorted(duck.execute(sql).fetchdf())


# --------------------------------------------------------------------------- #
# Composite-key correlated EXISTS / NOT EXISTS
# --------------------------------------------------------------------------- #


def test_composite_exists(cengine):
    eng, duck = cengine
    sql = (
        "SELECT a, b, v FROM t tx "
        "WHERE EXISTS (SELECT 1 FROM s WHERE s.a = tx.a AND s.b = tx.b) "
        "ORDER BY a, b"
    )
    assert _ryu(eng, sql) == _duck(duck, sql)


def test_composite_not_exists(cengine):
    eng, duck = cengine
    sql = (
        "SELECT a, b, v FROM t tx "
        "WHERE NOT EXISTS (SELECT 1 FROM s WHERE s.a = tx.a AND s.b = tx.b) "
        "ORDER BY a, b"
    )
    assert _ryu(eng, sql) == _duck(duck, sql)


def test_composite_exists_local_conjunct(cengine):
    """A local (non-correlation) conjunct inside the subquery is preserved as
    the inner WHERE and applied before the semi-join."""
    eng, duck = cengine
    sql = (
        "SELECT a, b, v FROM t tx "
        "WHERE EXISTS (SELECT 1 FROM s WHERE s.a = tx.a AND s.b = tx.b "
        "AND s.w > 8) ORDER BY a, b"
    )
    assert _ryu(eng, sql) == _duck(duck, sql)


def test_composite_not_exists_local_conjunct(cengine):
    eng, duck = cengine
    sql = (
        "SELECT a, b, v FROM t tx "
        "WHERE NOT EXISTS (SELECT 1 FROM s WHERE s.a = tx.a AND s.b = tx.b "
        "AND s.w > 8) ORDER BY a, b"
    )
    assert _ryu(eng, sql) == _duck(duck, sql)


def test_composite_exists_with_outer_local_conjunct(cengine):
    """A regular (non-subquery) WHERE conjunct on the outer query composes with
    the correlated EXISTS."""
    eng, duck = cengine
    sql = (
        "SELECT a, b, v FROM t tx "
        "WHERE v > 150 AND EXISTS (SELECT 1 FROM s WHERE s.a = tx.a AND s.b = tx.b) "
        "ORDER BY a, b"
    )
    assert _ryu(eng, sql) == _duck(duck, sql)


def test_composite_three_keys(cengine):
    """A three-column composite correlation key (uses w on s as a third key by
    correlating it to v on t)."""
    eng, duck = cengine
    sql = (
        "SELECT a, b, v FROM t tx "
        "WHERE EXISTS (SELECT 1 FROM s WHERE s.a = tx.a AND s.b = tx.b "
        "AND s.w = tx.v) ORDER BY a, b"
    )
    assert _ryu(eng, sql) == _duck(duck, sql)


# --------------------------------------------------------------------------- #
# NULL-key semantics: a NULL in any key matches nothing
# --------------------------------------------------------------------------- #


def test_composite_exists_null_outer_key(cengine):
    """The NULL-a row of t (a=NULL, b=40) matches nothing under EXISTS (NULL =
    NULL is unknown) -> excluded. Columns sort to (a, b, v); the NULL-a row is
    ``(None, 40, 600)`` and must be absent from the EXISTS result."""
    eng, duck = cengine
    sql = (
        "SELECT a, b, v FROM t tx "
        "WHERE EXISTS (SELECT 1 FROM s WHERE s.a = tx.a AND s.b = tx.b) "
        "ORDER BY a, b"
    )
    r = _ryu(eng, sql)
    assert r == _duck(duck, sql)
    assert (None, 40, 600) not in r


def test_composite_not_exists_null_outer_key(cengine):
    """The NULL-a row of t is kept by NOT EXISTS (it matched nothing)."""
    eng, duck = cengine
    sql = (
        "SELECT a, b, v FROM t tx "
        "WHERE NOT EXISTS (SELECT 1 FROM s WHERE s.a = tx.a AND s.b = tx.b) "
        "ORDER BY a, b"
    )
    r = _ryu(eng, sql)
    assert r == _duck(duck, sql)
    assert (None, 40, 600) in r


def test_composite_not_exists_null_inner_key(cengine):
    """A NULL inner key (s has a=NULL, b=40) never matches an outer row, so it
    does not suppress any outer row under NOT EXISTS (a NULL in the NOT-IN-style
    set would be dangerous for IN, but EXISTS/NOT EXISTS only care about row
    existence on the equality match, which a NULL key never satisfies)."""
    eng, duck = cengine
    sql = (
        "SELECT a, b, v FROM t tx "
        "WHERE NOT EXISTS (SELECT 1 FROM s WHERE s.a = tx.a AND s.b = tx.b) "
        "ORDER BY a, b"
    )
    assert _ryu(eng, sql) == _duck(duck, sql)


# --------------------------------------------------------------------------- #
# Single-key regression (the composite path must not break the single-key path)
# --------------------------------------------------------------------------- #


def test_single_key_exists_still_works(cengine):
    eng, duck = cengine
    sql = "SELECT a, v FROM t tx WHERE EXISTS (SELECT 1 FROM s WHERE s.a = tx.a) ORDER BY a, v"
    assert _ryu(eng, sql) == _duck(duck, sql)


def test_single_key_not_exists_still_works(cengine):
    eng, duck = cengine
    sql = (
        "SELECT a, v FROM t tx "
        "WHERE NOT EXISTS (SELECT 1 FROM s WHERE s.a = tx.a) ORDER BY a, v"
    )
    assert _ryu(eng, sql) == _duck(duck, sql)


# --------------------------------------------------------------------------- #
# Rejections (deferred shapes)
# --------------------------------------------------------------------------- #


def test_non_equi_correlation_rejected(cengine):
    """A non-equi correlation (``s.w <> tx.v``) is not lowered to a join."""
    eng, _duck = cengine
    with pytest.raises(NotImplementedError):
        eng.sql(
            "SELECT a, b, v FROM t tx "
            "WHERE EXISTS (SELECT 1 FROM s WHERE s.a = tx.a AND s.w <> tx.v)"
        )


def test_composite_correlated_scalar_rejected(cengine):
    """A correlated scalar subquery with a composite correlation key is not
    supported (only a single equi-correlation lowers to a LEFT join onto a
    grouped aggregate)."""
    eng, _duck = cengine
    with pytest.raises(NotImplementedError):
        eng.sql(
            "SELECT a, b, (SELECT MAX(w) FROM s WHERE s.a = t.a AND s.b = t.b) AS m "
            "FROM t"
        )


# --------------------------------------------------------------------------- #
# Parse shape: composite correlation lowers to a multi-key semi/anti join
# --------------------------------------------------------------------------- #


def test_composite_exists_parses_to_multi_key_semi():
    from ryudb.sql.plan import Join, Project

    plan = parse("SELECT a, b, v FROM t tx WHERE EXISTS (SELECT 1 FROM s WHERE s.a = tx.a AND s.b = tx.b)")
    proj = plan
    assert isinstance(proj, Project)
    join = proj.input
    assert isinstance(join, Join)
    assert join.how == "semi"
    assert list(join.on_left) == ["a", "b"]
    assert list(join.on_right) == ["a", "b"]


def test_composite_not_exists_parses_to_multi_key_anti():
    from ryudb.sql.plan import Join, Project

    plan = parse(
        "SELECT a, b, v FROM t tx "
        "WHERE NOT EXISTS (SELECT 1 FROM s WHERE s.a = tx.a AND s.b = tx.b)"
    )
    proj = plan
    assert isinstance(proj, Project)
    join = proj.input
    assert isinstance(join, Join)
    assert join.how == "anti"
    assert list(join.on_left) == ["a", "b"]
    assert list(join.on_right) == ["a", "b"]