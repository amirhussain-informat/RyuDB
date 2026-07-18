"""SQL surface, Phase F-2b: FROM-subqueries (derived tables) -- RyuDB vs DuckDB.

A derived table is ``FROM (SELECT ...) [AS] t`` -- a parenthesized subquery used
as a relation in the FROM clause, optionally joined with other relations. It
lowers to a ``Derived(input, alias)`` plan node (plan.py): a row-preserving
source whose execution returns the subplan's output frame verbatim. The design
exploits the flat-column model -- ``Col`` carries only a column name (the table
qualifier is dropped at parse time), so the outer query references the derived
table's output columns as ordinary flat names; no binding/namespace layer is
needed. The derived table is a *scope barrier* (like ``Window``): outer
predicates/projections do not push across it, but the optimizer recurses into
``input`` so the subplan is optimized (predicate pushdown, projection pruning,
join-side selection) within its own scope.

Anonymous derived tables (``FROM (SELECT ...)`` with no alias) are rejected
(DuckDB requires an alias too). Lateral/correlated references to a derived
alias fall through to the existing correlated-subquery classifier and raise
``NotImplementedError`` (out of scope for this phase).
"""

from __future__ import annotations

import cudf
import pytest

from ryudb import Catalog, Engine
from ryudb.sql.parse import parse, ParseError
from ryudb.sql.plan import Aggregate, Derived, Join, Project, Scan

from .conftest import as_sorted

# A: (k, a, b) -- a join key with a NULL group and a duplicate (k,a,b) row, for
# derived + DISTINCT/HAVING/aggregate cases and NULL-key safety.
_A = [
    (1, 10, 100),
    (1, 10, 100),
    (1, 20, 200),
    (2, 30, 300),
    (2, 40, 400),
    (3, 50, 500),
    (None, 60, 600),
    (None, 60, 600),
]

# B: (k, c) -- overlaps A on k=1,2 and adds k=4 (no match in A). Deliberately
# has NO NULL-k row: cuDF ``merge`` matches NULL==NULL on equi keys, which is a
# pre-existing join limitation (DuckDB treats NULL!=NULL), so a NULL key on both
# sides would diverge. A keeps its NULL-k rows for the non-join NULL tests.
_B = [
    (1, 1000),
    (2, 2000),
    (2, 2500),
    (4, 4000),
]


@pytest.fixture
def sdir(tmp_path):
    d = tmp_path
    (d / "a").mkdir()
    (d / "b").mkdir()
    cudf.DataFrame({"k": [r[0] for r in _A], "a": [r[1] for r in _A], "b": [r[2] for r in _A]}) \
        .to_pandas().to_parquet(d / "a" / "0.parquet")
    cudf.DataFrame({"k": [r[0] for r in _B], "c": [r[1] for r in _B]}) \
        .to_pandas().to_parquet(d / "b" / "0.parquet")
    return d


@pytest.fixture
def sengine(sdir) -> Engine:
    cat = Catalog(str(sdir))
    cat.register("a", str(sdir / "a"))
    cat.register("b", str(sdir / "b"))
    return Engine(cat)


@pytest.fixture
def sduck(sdir):
    import duckdb

    con = duckdb.connect()
    con.execute(f"CREATE VIEW a AS SELECT * FROM read_parquet('{sdir}/a/*.parquet')")
    con.execute(f"CREATE VIEW b AS SELECT * FROM read_parquet('{sdir}/b/*.parquet')")
    return con


def _ryu(engine: Engine, sql: str):
    return as_sorted(engine.sql(sql))


def _duck(con, sql: str):
    return as_sorted(con.execute(sql).fetchdf())


# --------------------------------------------------------------------------- #
# Basic derived tables (base relation)
# --------------------------------------------------------------------------- #


def test_derived_select_star(sengine, sduck):
    sql = "SELECT * FROM (SELECT a, b FROM a) t"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_derived_rename(sengine, sduck):
    sql = "SELECT t.x FROM (SELECT a AS x FROM a) t"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_derived_outer_where(sengine, sduck):
    sql = "SELECT t.x FROM (SELECT a AS x, b FROM a) t WHERE t.x > 20"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_derived_inner_where(sengine, sduck):
    sql = "SELECT * FROM (SELECT a, b FROM a WHERE b > 200) t"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_derived_inner_group_by(sengine, sduck):
    # Aggregation inside the derived table; outer WHERE on the aggregate output.
    sql = "SELECT t.cnt FROM (SELECT k, COUNT(*) AS cnt FROM a GROUP BY k) t WHERE t.cnt > 1"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_derived_outer_count(sengine, sduck):
    sql = "SELECT COUNT(*) FROM (SELECT a FROM a) t"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_derived_outer_sum(sengine, sduck):
    sql = "SELECT SUM(t.x) FROM (SELECT a AS x FROM a) t"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_derived_outer_group_by(sengine, sduck):
    sql = "SELECT t.k, COUNT(*) FROM (SELECT a AS k FROM a) t GROUP BY t.k"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


# --------------------------------------------------------------------------- #
# Derived tables as join partners
# --------------------------------------------------------------------------- #


def test_derived_join_real(sengine, sduck):
    sql = "SELECT t.x, b.c FROM (SELECT a AS x, k FROM a) t JOIN b ON t.k = b.k"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_derived_join_real_left(sengine, sduck):
    sql = "SELECT t.x, b.c FROM (SELECT a AS x, k FROM a) t LEFT JOIN b ON t.k = b.k"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_two_derived_join(sengine, sduck):
    sql = (
        "SELECT t1.x, t2.y "
        "FROM (SELECT a AS x, k FROM a) t1 "
        "JOIN (SELECT c AS y, k FROM b) t2 ON t1.k = t2.k"
    )
    assert _ryu(sengine, sql) == _duck(sduck, sql)


# --------------------------------------------------------------------------- #
# Derived tables with DISTINCT / ORDER BY / LIMIT / HAVING
# --------------------------------------------------------------------------- #


def test_derived_distinct(sengine, sduck):
    sql = "SELECT DISTINCT t.k FROM (SELECT k FROM a) t"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_derived_order_limit(sengine, sduck):
    sql = "SELECT t.x FROM (SELECT a AS x FROM a) t ORDER BY t.x LIMIT 3"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_derived_outer_having(sengine, sduck):
    sql = (
        "SELECT t.k, COUNT(*) FROM (SELECT a AS k FROM a) t "
        "GROUP BY t.k HAVING COUNT(*) > 1"
    )
    assert _ryu(sengine, sql) == _duck(sduck, sql)


# --------------------------------------------------------------------------- #
# NULL safety, set-ops inside a derived table, nesting
# --------------------------------------------------------------------------- #


def test_derived_null_key(sengine, sduck):
    sql = "SELECT t.k FROM (SELECT k FROM a) t WHERE t.k IS NULL"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_derived_setop_inside(sengine, sduck):
    sql = "SELECT * FROM (SELECT a FROM a UNION SELECT c FROM b) t"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_derived_nested(sengine, sduck):
    sql = "SELECT * FROM (SELECT x FROM (SELECT a AS x FROM a) u) t"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


# --------------------------------------------------------------------------- #
# Structural / parse-shape tests (raw plan, pre-optimization)
# --------------------------------------------------------------------------- #


def test_derived_parses_to_derived_node():
    plan = parse("SELECT t.x FROM (SELECT a AS x FROM a) t", {"a": ["k", "a", "b"]})
    assert isinstance(plan, Project)
    assert isinstance(plan.input, Derived)
    assert plan.input.alias == "t"
    assert isinstance(plan.input.input, Project)
    assert isinstance(plan.input.input.input, Scan)


def test_derived_join_parses_to_join_with_derived_right():
    plan = parse(
        "SELECT t.x, b.c FROM (SELECT a AS x, k FROM a) t JOIN b ON t.k = b.k",
        {"a": ["k", "a", "b"], "b": ["k", "c"]},
    )
    assert isinstance(plan, Project)
    assert isinstance(plan.input, Join)
    assert isinstance(plan.input.left, Derived)
    assert plan.input.left.alias == "t"
    assert isinstance(plan.input.right, Scan)


def test_derived_aggregate_inside_parses_to_aggregate_in_subplan():
    plan = parse(
        "SELECT t.cnt FROM (SELECT k, COUNT(*) AS cnt FROM a GROUP BY k) t",
        {"a": ["k", "a", "b"]},
    )
    assert isinstance(plan, Project)
    assert isinstance(plan.input, Derived)
    sub = plan.input.input
    assert isinstance(sub, Aggregate)


# --------------------------------------------------------------------------- #
# Rejected forms
# --------------------------------------------------------------------------- #


def test_anonymous_derived_rejected(sengine):
    with pytest.raises(ParseError):
        sengine.sql("SELECT * FROM (SELECT a FROM a)")


def test_lateral_derived_reference_rejected(sengine):
    # A scalar subquery in the projection referencing the derived alias `t` is a
    # lateral/correlated reference; the existing correlated-subquery classifier
    # only handles a single equi-correlation, so this raises NotImplementedError.
    with pytest.raises(NotImplementedError):
        sengine.sql("SELECT (SELECT 1 WHERE t.a > 0) FROM (SELECT a FROM a) t")