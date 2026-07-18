"""SQL surface, Phase F-2a: SELECT DISTINCT and HAVING -- RyuDB vs DuckDB.

Two small, high-leverage gaps filled:

* **``SELECT DISTINCT``** lowers to a ``Distinct`` plan node (a row-preserving
  node like ``Sort``/``Limit``) that runs cuDF ``drop_duplicates`` on its input.
  It wraps the projection *before* ``ORDER BY``/``LIMIT`` (``_build_tail``), so
  SQL's DISTINCT-then-ORDER BY-then-LIMIT order is preserved. cuDF
  ``drop_duplicates`` treats NaN as equal, so DISTINCT is NULL-correct and
  matches DuckDB (a NULL row dedups against another NULL row). ``DISTINCT ON
  (...)`` is a different feature (keep-first-per-key) and is rejected.

* **``HAVING``** lowers to a ``Filter`` over the ``Aggregate`` -- no new
  executor path. The ``Aggregate`` emits one column per group key / selected
  aggregate (named by their output aliases), and HAVING is evaluated against
  that frame. Each aggregate in the HAVING expression rewrites to a ``Col`` of
  its matching SELECT-list aggregate's alias (matched by sqlglot ``.sql()``,
  so an unaliased SELECT aggregate and a bare HAVING aggregate of the same
  shape match automatically); a HAVING aggregate NOT in the SELECT list is
  added as a synthetic ``_hvN`` aggregate (computed during grouping, then
  pruned from the output by a wrapping ``Project``). A HAVING reference to a
  column that is neither a group key nor an aggregate is rejected (standard SQL
  requires one or the other); subqueries in HAVING are deferred.

  ``push_predicates`` never pushes a ``Filter`` past an ``Aggregate`` (only past
  a ``Join``), so the HAVING ``Filter`` stays above the group -- exactly right.
"""

from __future__ import annotations

import cudf
import pytest

from ryudb import Catalog, Engine
from ryudb.sql.parse import parse

from .conftest import as_sorted

# a: duplicates on (k,w), a NULL-k group, and NULL-w rows -- exercises dedup and
# NULL-equal DISTINCT, plus grouped aggregates with ties and NULLs for HAVING.
_A = [
    (1, 10),
    (1, 10),
    (1, 20),
    (2, 30),
    (2, 40),
    (3, 30),
    (None, 50),
    (None, 50),
]


@pytest.fixture
def sdir(tmp_path):
    d = tmp_path
    (d / "a").mkdir()
    cudf.DataFrame({"k": [r[0] for r in _A], "w": [r[1] for r in _A]}) \
        .to_pandas().to_parquet(d / "a" / "0.parquet")
    return d


@pytest.fixture
def sengine(sdir) -> Engine:
    cat = Catalog(str(sdir))
    cat.register("a", str(sdir / "a"))
    return Engine(cat)


@pytest.fixture
def sduck(sdir):
    import duckdb

    con = duckdb.connect()
    con.execute(f"CREATE VIEW a AS SELECT * FROM read_parquet('{sdir}/a/*.parquet')")
    return con


def _ryu(engine: Engine, sql: str):
    return as_sorted(engine.sql(sql))


def _duck(con, sql: str):
    return as_sorted(con.execute(sql).fetchdf())


# --------------------------------------------------------------------------- #
# DISTINCT
# --------------------------------------------------------------------------- #


def test_distinct_single_col(sengine, sduck):
    sql = "SELECT DISTINCT k FROM a"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_distinct_multi_col(sengine, sduck):
    sql = "SELECT DISTINCT k, w FROM a"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_distinct_star(sengine, sduck):
    sql = "SELECT DISTINCT * FROM a"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_distinct_with_where(sengine, sduck):
    sql = "SELECT DISTINCT k FROM a WHERE w > 10"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_distinct_with_order_by(sengine, sduck):
    sql = "SELECT DISTINCT k FROM a ORDER BY k"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_distinct_with_order_by_and_limit(sengine, sduck):
    # ORDER BY makes the LIMIT deterministic (which distinct rows are kept).
    sql = "SELECT DISTINCT k FROM a ORDER BY k DESC LIMIT 2"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_distinct_with_group_by(sengine, sduck):
    # DISTINCT after GROUP BY is a no-op (groups are already unique); must still
    # match DuckDB exactly.
    sql = "SELECT DISTINCT k, COUNT(*) AS c FROM a GROUP BY k"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_distinct_null_dedup(sengine, sduck):
    # The two (None, 50) rows dedup to one; a NULL dedups against another NULL
    # (cuDF drop_duplicates treats NaN as equal, matching DuckDB).
    sql = "SELECT DISTINCT k, w FROM a ORDER BY k, w"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


# --------------------------------------------------------------------------- #
# HAVING
# --------------------------------------------------------------------------- #


def test_having_alias(sengine, sduck):
    # HAVING references the SELECT alias c.
    sql = "SELECT k, COUNT(*) AS c FROM a GROUP BY k HAVING c > 1"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_having_bare_agg_matches_select(sengine, sduck):
    # HAVING COUNT(*) matches the SELECT COUNT(*) AS c -> rewrites to Col("c").
    sql = "SELECT k, COUNT(*) AS c FROM a GROUP BY k HAVING COUNT(*) > 1"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_having_agg_not_in_select(sengine, sduck):
    # COUNT(*) is in HAVING but NOT in SELECT -> synthetic _hv1 aggregate, pruned.
    sql = "SELECT k FROM a GROUP BY k HAVING COUNT(*) > 1 ORDER BY k"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_having_sum_not_in_select(sengine, sduck):
    sql = "SELECT k FROM a GROUP BY k HAVING SUM(w) > 50 ORDER BY k"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_having_multiple_aggs_mixed(sengine, sduck):
    # AVG(w) selected (aliased a); MIN(w)/MAX(w) only in HAVING (synthetic).
    sql = ("SELECT k, AVG(w) AS a FROM a GROUP BY k "
           "HAVING MIN(w) > 10 AND MAX(w) < 60 ORDER BY k")
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_having_group_key_ref(sengine, sduck):
    sql = "SELECT k, COUNT(*) AS c FROM a GROUP BY k HAVING k > 1 ORDER BY k"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_having_global_aggregate(sengine, sduck):
    # No GROUP BY: a single global aggregate, filtered by HAVING.
    sql = "SELECT COUNT(*) AS c FROM a HAVING c > 2"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_having_global_aggregate_bare(sengine, sduck):
    sql = "SELECT COUNT(*) FROM a HAVING COUNT(*) > 2"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_having_with_order_by(sengine, sduck):
    sql = "SELECT k, COUNT(*) AS c FROM a GROUP BY k HAVING c > 1 ORDER BY c DESC"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_having_and_distinct(sengine, sduck):
    sql = ("SELECT DISTINCT k, COUNT(*) AS c FROM a GROUP BY k HAVING c > 1 "
           "ORDER BY k")
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_having_with_min_max_avg(sengine, sduck):
    sql = ("SELECT k, MIN(w) AS mn, MAX(w) AS mx, AVG(w) AS av FROM a GROUP BY k "
           "HAVING MAX(w) - MIN(w) > 5 ORDER BY k")
    assert _ryu(sengine, sql) == _duck(sduck, sql)


# --------------------------------------------------------------------------- #
# Parse shape
# --------------------------------------------------------------------------- #


def test_distinct_parses_to_distinct_project():
    from ryudb.sql.plan import Distinct, Project, Scan

    plan = parse("SELECT DISTINCT k FROM a")
    assert isinstance(plan, Distinct)
    assert isinstance(plan.input, Project)
    assert isinstance(plan.input.input, Scan)


def test_distinct_star_parses_to_distinct_scan():
    from ryudb.sql.plan import Distinct, Scan

    plan = parse("SELECT DISTINCT * FROM a")
    assert isinstance(plan, Distinct)
    assert isinstance(plan.input, Scan)


def test_distinct_group_by_parses_to_distinct_aggregate():
    from ryudb.sql.plan import Aggregate, Distinct

    plan = parse("SELECT DISTINCT k, COUNT(*) AS c FROM a GROUP BY k")
    assert isinstance(plan, Distinct)
    assert isinstance(plan.input, Aggregate)


def test_having_parses_to_filter_aggregate():
    from ryudb.sql.plan import Aggregate, BinOp, Col, Filter, Lit

    plan = parse("SELECT k, COUNT(*) AS c FROM a GROUP BY k HAVING c > 1")
    assert isinstance(plan, Filter)
    assert isinstance(plan.input, Aggregate)
    pred = plan.predicate
    assert isinstance(pred, BinOp)
    assert pred.op == ">"
    assert isinstance(pred.left, Col) and pred.left.name == "c"
    assert isinstance(pred.right, Lit)


def test_having_synthetic_parses_to_project_filter_aggregate():
    from ryudb.sql.plan import Aggregate, Col, Filter, Project

    plan = parse("SELECT k FROM a GROUP BY k HAVING COUNT(*) > 1")
    # Project prunes the synthetic _hv1; Filter is the HAVING; Aggregate has _hv1.
    assert isinstance(plan, Project)
    assert isinstance(plan.input, Filter)
    agg = plan.input.input
    assert isinstance(agg, Aggregate)
    agg_names = [n for _, n in agg.aggs]
    assert agg_names == ["_hv1"]
    # The output Project keeps only the user's column (k), not _hv1.
    out_names = [a for _, a in plan.items]
    assert out_names == ["k"]
    assert all(isinstance(e, Col) for e, _ in plan.items)


def test_having_with_order_by_keeps_sort_on_top():
    from ryudb.sql.plan import Filter, Sort

    plan = parse("SELECT k, COUNT(*) AS c FROM a GROUP BY k HAVING c > 1 ORDER BY c")
    # Sort sits above the HAVING Filter (DISTINCT-then-ORDER BY ordering).
    assert isinstance(plan, Sort)
    assert isinstance(plan.input, Filter)


# --------------------------------------------------------------------------- #
# Deferred / rejected
# --------------------------------------------------------------------------- #


def _rej(engine: Engine, sql: str):
    with pytest.raises(NotImplementedError):
        engine.sql(sql)


def test_distinct_on_rejected(sengine):
    _rej(sengine, "SELECT DISTINCT ON (k) w FROM a")


def test_having_non_grouped_column_rejected(sengine):
    # v is neither a group key nor an aggregate -> rejected (matches SQL semantics).
    _rej(sengine, "SELECT k, COUNT(*) AS c FROM a GROUP BY k HAVING w > 5")


def test_having_without_group_by_nonaggregate_rejected(sengine):
    # No GROUP BY + a non-aggregate SELECT column + HAVING -> invalid SQL.
    _rej(sengine, "SELECT k FROM a HAVING COUNT(*) > 5")


def test_having_subquery_rejected(sengine):
    _rej(sengine, "SELECT k, COUNT(*) AS c FROM a GROUP BY k "
                  "HAVING (SELECT MAX(w) FROM a) > 5")


def test_window_with_having_rejected(sengine):
    # F-1 guard: window functions with HAVING are still deferred.
    _rej(sengine, "SELECT k, ROW_NUMBER() OVER (PARTITION BY k ORDER BY w) AS rn "
                  "FROM a HAVING rn > 1")