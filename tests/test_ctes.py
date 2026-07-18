"""SQL surface, Phase F-2c: CTEs / WITH clauses -- RyuDB vs DuckDB.

A CTE (``WITH name AS (SELECT ...) SELECT ... FROM name``) is a named, reusable
subquery visible throughout its query's scope. It reuses the Phase F-2b
``Derived(input, alias)`` node directly: a CTE reference ``FROM cte1`` is a plain
``exp.Table`` (indistinguishable by type from a base table) that ``_table_ref``
resolves -- via a threaded CTE name -> built-subplan map -- to
``Derived(ctes["cte1"], alias)``, exactly like a named derived table. So no new
plan node, executor path, or optimizer rule is added; the only work is threading
the CTE map through the parse pipeline so a CTE reference resolves wherever it
appears (main FROM, join partner, derived-table body, IN/EXISTS/scalar subquery,
and later CTE bodies).

The CTE map is built eagerly, in order: a later CTE sees the earlier ones, and
each CTE body's subplan is built once and shared across references (the optimizer
rebuilds rather than mutates, so every reference gets its own optimized copy).
``WITH RECURSIVE`` and column-list CTEs (``name(col,...) AS (...)``) are
rejected; ``MATERIALIZED`` hints are ignored (RyuDB always inlines). Forward /
self CTE references fall through to a ``Scan`` and fail at execution (an
acceptable limitation).
"""

from __future__ import annotations

import cudf
import pytest

from ryudb import Catalog, Engine
from ryudb.sql.parse import parse
from ryudb.sql.plan import Derived, Join, Project, Scan

from .conftest import as_sorted

# A: (k, a, b) -- a join key with a NULL group and a duplicate (k,a,b) row.
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

# B: (k, c) -- overlaps A on k=1,2 and adds k=4 (no match in A). No NULL-k: cuDF
# ``merge`` matches NULL==NULL on equi keys (DuckDB treats NULL!=NULL), so a NULL
# key on both sides would diverge. A keeps its NULL-k rows for the non-join NULL
# tests.
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
# Basic CTEs
# --------------------------------------------------------------------------- #


def test_cte_basic(sengine, sduck):
    sql = "WITH c AS (SELECT a, b FROM a) SELECT * FROM c"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_cte_rename(sengine, sduck):
    sql = "WITH c AS (SELECT a AS x FROM a) SELECT c.x FROM c"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_cte_outer_where(sengine, sduck):
    sql = "WITH c AS (SELECT a AS x, b FROM a) SELECT c.x FROM c WHERE c.x > 20"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_cte_inner_where(sengine, sduck):
    sql = "WITH c AS (SELECT a, b FROM a WHERE b > 200) SELECT * FROM c"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


# --------------------------------------------------------------------------- #
# Chained CTEs and a CTE referenced multiple times
# --------------------------------------------------------------------------- #


def test_cte_chained(sengine, sduck):
    # A later CTE references an earlier one.
    sql = (
        "WITH c1 AS (SELECT a, k FROM a), "
        "c2 AS (SELECT a FROM c1) SELECT * FROM c2"
    )
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_cte_referenced_twice(sengine, sduck):
    # The same CTE referenced twice -- both references share the CTE's built
    # subplan (the optimizer rebuilds per reference, so each is optimized
    # independently). The self-join selects only the (single, merged) join key
    # ``k`` plus an aggregate: a self-join that selects a same-named non-key
    # column from both refs would collide (cuDF suffixes the duplicate to
    # ``a_x``/``a_y`` and the flat-column model -- qualifiers dropped -- cannot
    # recover ``a``). That collision is a pre-existing flat-model limitation,
    # not CTE-specific; test_two_derived_join mirrors this avoidance by renaming.
    sql = (
        "WITH c AS (SELECT a, k FROM a WHERE k IS NOT NULL) "
        "SELECT c1.k, COUNT(*) FROM c c1 JOIN c c2 ON c1.k = c2.k GROUP BY c1.k"
    )
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_cte_explicit_alias(sengine, sduck):
    sql = "WITH c AS (SELECT a AS x, k FROM a) SELECT t.x FROM c AS t WHERE t.x > 20"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


# --------------------------------------------------------------------------- #
# CTE as a join partner + aggregates + DISTINCT / ORDER BY / LIMIT
# --------------------------------------------------------------------------- #


def test_cte_join_real(sengine, sduck):
    sql = "WITH c AS (SELECT a AS x, k FROM a) SELECT c.x, b.c AS c2 FROM c JOIN b ON c.k = b.k"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_cte_outer_group_by(sengine, sduck):
    sql = "WITH c AS (SELECT k, a FROM a) SELECT c.k, COUNT(*) FROM c GROUP BY c.k"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_cte_outer_count(sengine, sduck):
    sql = "WITH c AS (SELECT a FROM a) SELECT COUNT(*) FROM c"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_cte_distinct(sengine, sduck):
    sql = "WITH c AS (SELECT k FROM a) SELECT DISTINCT c.k FROM c"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_cte_order_limit(sengine, sduck):
    sql = "WITH c AS (SELECT a AS x FROM a) SELECT c.x FROM c ORDER BY c.x LIMIT 2"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


# --------------------------------------------------------------------------- #
# CTE referenced in nested subqueries + nested WITH + NULL safety
# --------------------------------------------------------------------------- #


def test_cte_in_in_subquery(sengine, sduck):
    # The CTE map threads into the IN-subquery's own _build_query.
    sql = "WITH c AS (SELECT k FROM a) SELECT a.a FROM a WHERE a.k IN (SELECT k FROM c)"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_cte_in_correlated_exists(sengine, sduck):
    sql = (
        "WITH c AS (SELECT k FROM a) "
        "SELECT a.a FROM a WHERE EXISTS (SELECT 1 FROM c WHERE c.k = a.k)"
    )
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_cte_inside_derived(sengine, sduck):
    sql = "WITH c AS (SELECT a FROM a) SELECT * FROM (SELECT * FROM c) t"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_cte_nested_with(sengine, sduck):
    # An outer CTE is visible inside a nested WITH: `i` references `o`.
    sql = (
        "WITH o AS (SELECT a FROM a) "
        "SELECT * FROM (WITH i AS (SELECT a FROM o) SELECT * FROM i) t"
    )
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_cte_null_key(sengine, sduck):
    sql = "WITH c AS (SELECT k FROM a) SELECT c.k FROM c WHERE c.k IS NULL"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


# --------------------------------------------------------------------------- #
# Structural / parse-shape tests (raw plan, pre-optimization)
# --------------------------------------------------------------------------- #


def test_cte_parses_to_derived_not_scan():
    plan = parse("WITH c AS (SELECT a AS x FROM a) SELECT c.x FROM c",
                 {"a": ["k", "a", "b"]})
    assert isinstance(plan, Project)
    assert isinstance(plan.input, Derived)
    assert plan.input.alias == "c"
    assert isinstance(plan.input.input, Project)
    assert isinstance(plan.input.input.input, Scan)


def test_cte_referenced_twice_parses_to_join_with_two_derived():
    plan = parse(
        "WITH c AS (SELECT a, k FROM a) SELECT c1.a, c2.a FROM c c1 JOIN c c2 ON c1.k = c2.k",
        {"a": ["k", "a", "b"]},
    )
    assert isinstance(plan, Project)
    assert isinstance(plan.input, Join)
    assert isinstance(plan.input.left, Derived)
    assert isinstance(plan.input.right, Derived)
    # Both references lower to Derived, NOT Scan.
    assert not isinstance(plan.input.left, Scan)
    assert not isinstance(plan.input.right, Scan)


# --------------------------------------------------------------------------- #
# Rejected forms
# --------------------------------------------------------------------------- #


def test_recursive_cte_rejected(sengine):
    with pytest.raises(NotImplementedError):
        sengine.sql("WITH RECURSIVE c AS (SELECT a FROM a) SELECT * FROM c")


def test_column_list_cte_rejected(sengine):
    with pytest.raises(NotImplementedError):
        sengine.sql("WITH c(x) AS (SELECT a FROM a) SELECT c.x FROM c")