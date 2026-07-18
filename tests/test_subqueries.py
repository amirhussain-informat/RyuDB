"""SQL surface, Phase E-1: uncorrelated IN / NOT IN subqueries -- RyuDB vs DuckDB.

``x IN (SELECT ...)`` / ``x NOT IN (SELECT ...)`` in WHERE fold into semi/anti
joins (parse._apply_where_subqueries): the subquery becomes the Join's right
child, a normal subtree the optimizer recurses into. The executor lowers semi/
anti via cuDF ``isin`` (executor._join); ``dropna()`` on the key set makes IN
NULL-safe (a NULL left key never matches; NULLs in the subquery set never spuriously
match). NOT IN is only correct for non-NULL keys on both sides -- the NOT IN cases
filter NULLs. EXISTS, scalar subqueries, correlated subqueries, and IN under OR
are deferred (raise NotImplementedError). Comparison is via conftest.as_sorted.

The fused CUDA kernels are untouched -- a semi/anti join is cuDF ``isin``, and a
join/aggregate *inside* the subquery still fuses normally (test_in_subquery_union
of a UNION, test_in_subquery_agg of an aggregate).
"""

from __future__ import annotations

import cudf
import pytest

from ryudb import Catalog, Engine
from ryudb.sql.parse import parse

from .conftest import as_sorted

# a and b share k in {2, 3} and both have a NULL-k row. The shared NULL exercises
# IN's NULL-safety (dropna on the key set); the NOT IN cases filter NULLs since
# NOT IN with a NULL in the set is not reproduced by ~isin.
_A = [
    (1, 10),
    (2, 20),
    (3, 30),
    (4, 40),
    (None, 50),
]
_B = [
    (2, 200),
    (3, 300),
    (5, 500),
    (None, 600),
]


@pytest.fixture
def sdir(tmp_path):
    d = tmp_path
    for name, cols, rows in [("a", ["k", "v"], _A), ("b", ["k", "w"], _B)]:
        (d / name).mkdir()
        cudf.DataFrame({c: [row[i] for row in rows] for i, c in enumerate(cols)}) \
            .to_pandas().to_parquet(d / name / "0.parquet")
    return d


@pytest.fixture
def sengine(sdir) -> Engine:
    cat = Catalog(str(sdir))
    for name in ("a", "b"):
        cat.register(name, str(sdir / name))
    return Engine(cat)


@pytest.fixture
def sduck(sdir):
    import duckdb

    con = duckdb.connect()
    for name in ("a", "b"):
        con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{sdir}/{name}/*.parquet')")
    return con


def _ryu(engine: Engine, sql: str):
    return as_sorted(engine.sql(sql))


def _duck(con, sql: str):
    return as_sorted(con.execute(sql).fetchdf())


# --------------------------------------------------------------------------- #
# IN (SELECT ...)
# --------------------------------------------------------------------------- #


def test_in_subquery(sengine, sduck):
    # NULL left key and a NULL in the subquery set both drop via dropna -- the
    # kept/dropped outcome matches DuckDB.
    sql = "SELECT k, v FROM a WHERE k IN (SELECT k FROM b)"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_in_subquery_alias(sengine, sduck):
    # The subquery projects an aliased single column -> on_right is the alias.
    sql = "SELECT k FROM a WHERE k IN (SELECT k AS c FROM b)"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_in_with_filter(sengine, sduck):
    # AND of an IN-subquery conjunct (-> semi join) and a regular conjunct
    # (-> residual Filter).
    sql = "SELECT k, v FROM a WHERE k IN (SELECT k FROM b) AND v > 15"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_in_subquery_agg(sengine, sduck):
    # The subquery is an aggregate (one row, one col) -> on_right is the agg
    # output name; MIN ignores the NULL in b.
    sql = "SELECT k, v FROM a WHERE k IN (SELECT MIN(k) FROM b)"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_in_subquery_union(sengine, sduck):
    # The subquery is a one-column UNION (a SetOp) -> on_right from the left arm.
    sql = "SELECT k, v FROM a WHERE k IN (SELECT k FROM b UNION SELECT k FROM a)"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_in_empty(sengine, sduck):
    # Empty subquery set, no NULL in it -> IN keeps nothing.
    sql = "SELECT k, v FROM a WHERE k IN (SELECT k FROM b WHERE k > 100)"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


# --------------------------------------------------------------------------- #
# NOT IN (SELECT ...) -- non-NULL keys on both sides (NULLs filtered)
# --------------------------------------------------------------------------- #


def test_not_in_subquery(sengine, sduck):
    sql = ("SELECT k, v FROM a WHERE k IS NOT NULL "
           "AND k NOT IN (SELECT k FROM b WHERE k IS NOT NULL)")
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_not_in_empty(sengine, sduck):
    # Empty subquery set -> NOT IN keeps all (non-NULL) rows.
    sql = ("SELECT k, v FROM a WHERE k IS NOT NULL "
           "AND k NOT IN (SELECT k FROM b WHERE k > 100)")
    assert _ryu(sengine, sql) == _duck(sduck, sql)


# --------------------------------------------------------------------------- #
# Parsing: IN/NOT IN subqueries lower to semi/anti joins
# --------------------------------------------------------------------------- #


def test_in_subquery_parses_to_semi():
    from ryudb.sql.plan import Join

    plan = parse("SELECT * FROM a WHERE k IN (SELECT k FROM b)")
    assert isinstance(plan, Join)
    assert plan.how == "semi"
    assert plan.on_left == ["k"]
    assert plan.on_right == ["k"]


def test_not_in_subquery_parses_to_anti():
    from ryudb.sql.plan import Join

    plan = parse("SELECT * FROM a WHERE k NOT IN (SELECT k FROM b)")
    assert isinstance(plan, Join)
    assert plan.how == "anti"


def test_in_with_filter_parses_to_semi_then_filter():
    from ryudb.sql.plan import Filter, Join, Project

    plan = parse("SELECT k, v FROM a WHERE k IN (SELECT k FROM b) AND v > 15")
    # SELECT k,v wraps the WHERE in a Project; the residual v>15 is a Filter over
    # the semi join.
    assert isinstance(plan, Project)
    assert isinstance(plan.input, Filter)
    assert isinstance(plan.input.input, Join)
    assert plan.input.input.how == "semi"


# --------------------------------------------------------------------------- #
# Deferred forms (raise NotImplementedError)
# --------------------------------------------------------------------------- #


def test_correlated_rejected(sengine):
    # The subquery references outer alias a -> correlated (deferred to E-3).
    with pytest.raises(NotImplementedError):
        sengine.sql("SELECT k, v FROM a WHERE k IN (SELECT k FROM b WHERE b.k = a.k)")


def test_in_under_or_rejected(sengine):
    # Semi/anti are not distributive over OR (deferred).
    with pytest.raises(NotImplementedError):
        sengine.sql("SELECT k, v FROM a WHERE k IN (SELECT k FROM b) OR v > 100")


def test_in_star_subquery_rejected(sengine):
    # SELECT * in an IN-subquery is ambiguous (must project one column).
    with pytest.raises(NotImplementedError):
        sengine.sql("SELECT k FROM a WHERE k IN (SELECT * FROM b)")


def test_in_expr_key_rejected(sengine):
    # IN-subquery key must be a bare column (expression keys are deferred).
    with pytest.raises(NotImplementedError):
        sengine.sql("SELECT k, v FROM a WHERE v + 1 IN (SELECT k FROM b)")


def test_exists_rejected(sengine):
    # EXISTS is deferred to E-2.
    with pytest.raises(NotImplementedError):
        sengine.sql("SELECT k, v FROM a WHERE EXISTS (SELECT 1 FROM b)")


def test_scalar_subquery_rejected(sengine):
    # Scalar subqueries in projection are deferred to E-2.
    with pytest.raises(NotImplementedError):
        sengine.sql("SELECT k, (SELECT COUNT(*) FROM b) AS c FROM a")