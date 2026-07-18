"""SQL surface, Phase E: uncorrelated subqueries -- RyuDB vs DuckDB.

Phase E-1 (``x IN (SELECT ...)`` / ``x NOT IN (SELECT ...)`` in WHERE) folds into
semi/anti joins (``_apply_where_subqueries``): the subquery is the Join's right
child, a normal subtree the optimizer recurses into. The executor lowers semi/
anti via cuDF ``isin``; ``dropna()`` on the key set makes IN NULL-safe. NOT IN is
only correct for non-NULL keys on both sides -- the NOT IN cases filter NULLs.

Phase E-2 (uncorrelated scalar subqueries and ``EXISTS``) flattens each subquery
into a cross-join of a 1-row relation onto the outer plan (``_flatten_outer_subqueries``):

* A scalar subquery must be a single-row aggregate (``SELECT COUNT(*)/MAX(...)
  FROM ...``); the aggregate's one output row is broadcast to every outer row via
  a cross-join, and the subquery node is replaced by a Column ref to the
  broadcast column. Works in projection and in WHERE comparisons.
* ``EXISTS (SELECT ...)`` becomes ``(SELECT COUNT(*) FROM (subq) LIMIT 1) > 0``:
  1 iff the subquery has any row, 0 otherwise. Replacing the ``exp.Exists`` node
  with ``col > 0`` (not a dedicated node) means EXISTS works under AND/OR/NOT and
  in projection, not just as a top-level WHERE conjunct (unlike IN). NOT EXISTS
  falls out via ``Not`` (the count is a non-NULL int, so ``Not`` inverts it).

Phase E-3 (correlated subqueries with a single equality correlation
``inner.k = outer.k``) decorrelates at parse time into the same join machinery:

* Correlated ``EXISTS``/``NOT EXISTS`` -> a semi/anti join on the correlation key
  (the correlation predicate *is* the join key; the subquery's WHERE is reduced to
  its local conjuncts and its projection is dropped). NOT EXISTS -> anti-join is
  NULL-correct: a NULL outer key matches nothing -> NOT EXISTS true -> anti-join
  keeps the row (matches DuckDB).
* Correlated scalar ``(SELECT MAX(w) FROM b WHERE b.k = a.k)`` -> a LEFT join of
  the outer plan onto a grouped aggregate (the correlation key becomes the GROUP
  BY key and the left-join key); the subquery node is replaced by a Column ref to
  the aggregate output, null-padded for unmatched outer rows. The grouped
  aggregate's NULL-key group is dropped before the join so the LEFT join is
  NULL-safe (SQL ``=`` matches nothing on NULL; cuDF merge would match NULL==NULL).
  ``COUNT`` over an empty correlation is 0 (COALESCE), not NULL.

Non-equi / multi-equi correlation, correlated IN, and an outer ref in the
aggregate argument / SELECT list are rejected (deferred). Non-aggregate / GROUP
BY scalar subqueries are rejected (no 1-row guarantee). The fused CUDA kernels
are untouched -- a cross/semi/anti join and a join/aggregate *inside* the
subquery still fuse normally.
"""

from __future__ import annotations

import cudf
import pytest

from ryudb import Catalog, Engine
from ryudb.sql.parse import parse

from .conftest import as_sorted

# a and b share k in {2, 3} and both have a NULL-k row. The shared NULL exercises
# IN's NULL-safety (dropna on the key set) and EXISTS's row-count semantics
# (EXISTS counts NULL rows too -- only row existence matters). The NOT IN cases
# filter NULLs since NOT IN with a NULL in the set is not reproduced by ~isin.
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
# c shares k with a/b and has a duplicate k=2 (for the correlated-scalar-with-join
# case: MAX over the joined b+c rows is a grouped aggregate over multiple rows).
_C = [
    (2, "x"),
    (2, "y"),
    (3, "z"),
]


@pytest.fixture
def sdir(tmp_path):
    d = tmp_path
    for name, cols, rows in [
        ("a", ["k", "v"], _A),
        ("b", ["k", "w"], _B),
        ("c", ["k", "t"], _C),
    ]:
        (d / name).mkdir()
        cudf.DataFrame({c: [row[i] for row in rows] for i, c in enumerate(cols)}) \
            .to_pandas().to_parquet(d / name / "0.parquet")
    return d


@pytest.fixture
def sengine(sdir) -> Engine:
    cat = Catalog(str(sdir))
    for name in ("a", "b", "c"):
        cat.register(name, str(sdir / name))
    return Engine(cat)


@pytest.fixture
def sduck(sdir):
    import duckdb

    con = duckdb.connect()
    for name in ("a", "b", "c"):
        con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{sdir}/{name}/*.parquet')")
    return con


def _ryu(engine: Engine, sql: str):
    return as_sorted(engine.sql(sql))


def _duck(con, sql: str):
    return as_sorted(con.execute(sql).fetchdf())


# --------------------------------------------------------------------------- #
# IN (SELECT ...)  -- Phase E-1
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
# NOT IN (SELECT ...) -- non-NULL keys on both sides (NULLs filtered) -- E-1
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
# Scalar subqueries -- Phase E-2 (cross-join broadcast of a 1-row aggregate)
# --------------------------------------------------------------------------- #


def test_scalar_count_projection(sengine, sduck):
    # COUNT(*) of b broadcast to every row of a (incl. the NULL-k row).
    sql = "SELECT k, (SELECT COUNT(*) FROM b) AS c FROM a"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_scalar_max_projection(sengine, sduck):
    sql = "SELECT k, (SELECT MAX(w) FROM b) AS c FROM a"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_scalar_min_in_where(sengine, sduck):
    sql = "SELECT k, v FROM a WHERE v > (SELECT MIN(w) FROM b)"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_scalar_max_in_where(sengine, sduck):
    sql = "SELECT k, v FROM a WHERE v < (SELECT MAX(w) FROM b)"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_scalar_filtered_subquery(sengine, sduck):
    # The scalar subquery has its own WHERE; MIN over a filtered b.
    sql = "SELECT k, v FROM a WHERE v > (SELECT MIN(w) FROM b WHERE w > 250)"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_scalar_empty_returns_null(sengine, sduck):
    # MAX over an empty set is NULL; ``v > NULL`` is NULL -> every row dropped
    # (DuckDB's three-valued WHERE).
    sql = "SELECT k, v FROM a WHERE v > (SELECT MAX(w) FROM b WHERE w > 99999)"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_scalar_and_in_combined(sengine, sduck):
    # A scalar subquery (-> cross join) and an IN subquery (-> semi join) in the
    # same WHERE, plus a regular conjunct.
    sql = ("SELECT k, v FROM a "
           "WHERE k IN (SELECT k FROM b) AND v > (SELECT MIN(w) FROM b)")
    assert _ryu(sengine, sql) == _duck(sduck, sql)


# --------------------------------------------------------------------------- #
# EXISTS / NOT EXISTS -- Phase E-2 (count(*) > 0 cross-join)
# --------------------------------------------------------------------------- #


def test_exists_uncorrelated(sengine, sduck):
    # EXISTS over a non-empty filtered b.
    sql = "SELECT k, v FROM a WHERE EXISTS (SELECT 1 FROM b WHERE w > 250)"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_not_exists_uncorrelated(sengine, sduck):
    sql = "SELECT k, v FROM a WHERE NOT EXISTS (SELECT 1 FROM b WHERE w > 250)"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_exists_empty(sengine, sduck):
    # EXISTS over an empty subquery -> false for every row -> nothing kept.
    sql = "SELECT k, v FROM a WHERE EXISTS (SELECT 1 FROM b WHERE w > 99999)"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_not_exists_empty(sengine, sduck):
    # NOT EXISTS over an empty subquery -> true for every row -> all kept.
    sql = "SELECT k, v FROM a WHERE NOT EXISTS (SELECT 1 FROM b WHERE w > 99999)"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_exists_counts_null_rows(sengine, sduck):
    # EXISTS counts the NULL-k row of b too (only row existence matters);
    # ``SELECT 1 FROM b`` (no filter) is non-empty -> every a row kept.
    sql = "SELECT k, v FROM a WHERE EXISTS (SELECT 1 FROM b)"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_exists_under_or(sengine, sduck):
    # EXISTS works under OR (unlike IN, which is restricted to AND-conjuncts):
    # the ``col > 0`` replacement is a normal boolean operand.
    sql = "SELECT k, v FROM a WHERE EXISTS (SELECT 1 FROM b WHERE w > 250) OR v < 15"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_exists_and_scalar(sengine, sduck):
    # EXISTS and a scalar subquery together.
    sql = ("SELECT k, v FROM a "
           "WHERE EXISTS (SELECT 1 FROM b WHERE w > 250) "
           "AND v > (SELECT MIN(w) FROM b)")
    assert _ryu(sengine, sql) == _duck(sduck, sql)


# --------------------------------------------------------------------------- #
# Parsing: subquery lowering shapes
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


def test_scalar_subquery_parses_to_cross_join():
    from ryudb.sql.plan import Aggregate, Join, Project

    plan = parse("SELECT k, (SELECT COUNT(*) FROM b) AS c FROM a")
    # SELECT k, (scalar) AS c -> Project over a cross-join of Scan(a) and the
    # broadcast aggregate (renamed to _sq1).
    assert isinstance(plan, Project)
    join = plan.input
    assert isinstance(join, Join)
    assert join.how == "cross"
    assert isinstance(join.right, Project)  # the rename Project over Aggregate
    assert isinstance(join.right.input, Aggregate)
    assert not join.right.input.group_keys  # global aggregate (1 row)


def test_exists_parses_to_cross_join_then_gt():
    from ryudb.sql.plan import Aggregate, BinOp, Filter, Join, Limit, Lit, Project

    plan = parse("SELECT k, v FROM a WHERE EXISTS (SELECT 1 FROM b)")
    # Project over Filter((_sq1 > 0)) over cross-join(Scan a, Aggregate count(*)).
    assert isinstance(plan, Project)
    filt = plan.input
    assert isinstance(filt, Filter)
    pred = filt.predicate
    assert isinstance(pred, BinOp)
    assert pred.op == ">"
    assert isinstance(pred.right, Lit)
    join = filt.input
    assert isinstance(join, Join)
    assert join.how == "cross"
    assert isinstance(join.right, Aggregate)
    assert isinstance(join.right.input, Limit)  # LIMIT 1 on the subquery


# --------------------------------------------------------------------------- #
# Correlated EXISTS / NOT EXISTS -- Phase E-3 (semi/anti join on the corr key)
# --------------------------------------------------------------------------- #


def test_correlated_exists(sengine, sduck):
    # NULL a.k matches nothing under ``=`` -> the NULL-k row of a drops (matches
    # DuckDB); b's own NULL-k row does not join to a non-NULL outer key.
    sql = "SELECT k, v FROM a WHERE EXISTS (SELECT 1 FROM b WHERE b.k = a.k)"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_correlated_exists_local(sengine, sduck):
    # The subquery has a local predicate (b.w > 250) on top of the correlation.
    sql = "SELECT k, v FROM a WHERE EXISTS (SELECT 1 FROM b WHERE b.k = a.k AND b.w > 250)"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_correlated_not_exists(sengine, sduck):
    # NOT EXISTS -> anti-join is NULL-correct: a NULL a.k matches nothing -> NOT
    # EXISTS true -> the NULL-k row of a is kept (matches DuckDB).
    sql = "SELECT k, v FROM a WHERE NOT EXISTS (SELECT 1 FROM b WHERE b.k = a.k)"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_correlated_not_exists_local(sengine, sduck):
    sql = ("SELECT k, v FROM a "
           "WHERE NOT EXISTS (SELECT 1 FROM b WHERE b.k = a.k AND b.w > 250)")
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_correlated_exists_with_subquery_join(sengine, sduck):
    # The subquery has its own join (b JOIN c) plus the correlation on b.k; the
    # join is preserved inside the subquery and the correlation key stays b.k.
    sql = ("SELECT k, v FROM a "
           "WHERE EXISTS (SELECT 1 FROM b JOIN c ON b.k = c.k WHERE b.k = a.k)")
    assert _ryu(sengine, sql) == _duck(sduck, sql)


# --------------------------------------------------------------------------- #
# Correlated scalar -- Phase E-3 (LEFT join onto a grouped aggregate)
# --------------------------------------------------------------------------- #


def test_correlated_scalar_projection(sengine, sduck):
    # Unmatched a.k (1, 4, and NULL) -> MAX over empty -> NULL m.
    sql = "SELECT k, (SELECT MAX(w) FROM b WHERE b.k = a.k) AS m FROM a"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_correlated_scalar_in_where(sengine, sduck):
    sql = "SELECT k, v FROM a WHERE v > (SELECT MAX(w) FROM b WHERE b.k = a.k)"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_correlated_scalar_count(sengine, sduck):
    # COUNT over an empty correlation is 0 (COALESCE), not NULL; a NULL a.k also
    # yields 0 (the NULL-key group is dropped, so the NULL outer key null-pads).
    sql = "SELECT k, (SELECT COUNT(*) FROM b WHERE b.k = a.k) AS c FROM a"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_correlated_scalar_local(sengine, sduck):
    # The subquery has a local predicate (b.w > 250) on top of the correlation.
    sql = ("SELECT k, (SELECT MAX(w) FROM b WHERE b.k = a.k AND b.w > 250) AS m "
           "FROM a")
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_correlated_scalar_with_subquery_join(sengine, sduck):
    # The subquery joins b and c (duplicate k=2 in c) and aggregates MAX(c.t);
    # grouped by the correlation key b.k over the joined rows.
    sql = ("SELECT k, (SELECT MAX(c.t) FROM b JOIN c ON b.k = c.k "
           "WHERE b.k = a.k) AS m FROM a")
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_correlated_scalar_and_uncorrelated(sengine, sduck):
    # A correlated scalar (-> LEFT join) and an uncorrelated scalar (-> cross
    # join broadcast) in the same SELECT.
    sql = ("SELECT k, (SELECT MAX(w) FROM b WHERE b.k = a.k) AS m, "
           "(SELECT COUNT(*) FROM b) AS t FROM a")
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_correlated_exists_and_scalar(sengine, sduck):
    # A correlated EXISTS (-> semi join) and an uncorrelated scalar (-> cross
    # join broadcast) in the same WHERE.
    sql = ("SELECT k, v FROM a "
           "WHERE EXISTS (SELECT 1 FROM b WHERE b.k = a.k) "
           "AND v > (SELECT MIN(w) FROM b)")
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_correlated_exists_and_local_conjunct(sengine, sduck):
    # A correlated EXISTS conjunct plus a regular (non-subquery) conjunct.
    sql = "SELECT k, v FROM a WHERE v > 15 AND EXISTS (SELECT 1 FROM b WHERE b.k = a.k)"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


# --------------------------------------------------------------------------- #
# Parsing: correlated subquery lowering shapes -- Phase E-3
# --------------------------------------------------------------------------- #


def test_correlated_exists_parses_to_semi():
    from ryudb.sql.plan import Filter, Join, Project

    plan = parse("SELECT k, v FROM a WHERE EXISTS (SELECT 1 FROM b WHERE b.k = a.k)")
    # Project over the semi join (no residual Filter -- the correlation is the
    # join key, not a predicate above the join). b's WHERE is reduced to empty
    # (no local conjuncts), so the right side is a bare Scan(b).
    assert isinstance(plan, Project)
    join = plan.input
    assert isinstance(join, Join)
    assert join.how == "semi"
    assert join.on_left == ["k"]
    assert join.on_right == ["k"]
    assert not isinstance(join.right, Filter)  # no local predicate to wrap


def test_correlated_not_exists_parses_to_anti():
    from ryudb.sql.plan import Join, Project

    plan = parse("SELECT k, v FROM a WHERE NOT EXISTS (SELECT 1 FROM b WHERE b.k = a.k)")
    assert isinstance(plan, Project)
    join = plan.input
    assert isinstance(join, Join)
    assert join.how == "anti"


def test_correlated_exists_local_parses_to_semi_with_filter():
    from ryudb.sql.plan import Filter, Join, Project, Scan

    plan = parse("SELECT k, v FROM a WHERE EXISTS (SELECT 1 FROM b "
                 "WHERE b.k = a.k AND b.w > 250)")
    assert isinstance(plan, Project)
    join = plan.input
    assert isinstance(join, Join)
    assert join.how == "semi"
    # The local conjunct b.w > 250 stays as a Filter over Scan(b) in the right,
    # under the projection of the inner key column (SELECT k).
    assert isinstance(join.right, Project)
    assert isinstance(join.right.input, Filter)
    assert isinstance(join.right.input.input, Scan)


def test_correlated_scalar_parses_to_left_join():
    from ryudb.sql.plan import Aggregate, Filter, Join, Project

    plan = parse("SELECT k, (SELECT MAX(w) FROM b WHERE b.k = a.k) AS m FROM a")
    # Project over a LEFT join of Scan(a) onto a grouped aggregate (group key
    # = correlation key, renamed to _sq1_k), with a NULL-key-dropping Filter over
    # the rename Project.
    assert isinstance(plan, Project)
    join = plan.input
    assert isinstance(join, Join)
    assert join.how == "left"
    assert join.on_left == ["k"]
    assert join.on_right == ["_sq1_k"]
    filt = join.right
    assert isinstance(filt, Filter)  # drops the NULL-key group (NULL-safe join)
    renamed = filt.input
    assert isinstance(renamed, Project)  # rename MAX->_sq1, _sq1_k->_sq1_k
    agg = renamed.input
    assert isinstance(agg, Aggregate)
    assert len(agg.group_keys) == 1  # grouped by the correlation key


def test_correlated_scalar_count_parses_to_coalesce():
    from ryudb.sql.plan import Coalesce, Col, Project

    plan = parse("SELECT k, (SELECT COUNT(*) FROM b WHERE b.k = a.k) AS c FROM a")
    # The COUNT subquery is replaced by COALESCE(_sq1, 0), not a bare Col ref.
    assert isinstance(plan, Project)
    item = next(e for e, alias in plan.items if alias == "c")
    assert isinstance(item, Coalesce)
    assert isinstance(item.args[0], Col)
    assert item.args[0].name == "_sq1"


# --------------------------------------------------------------------------- #
# Deferred forms (raise NotImplementedError)
# --------------------------------------------------------------------------- #


def test_correlated_in_rejected(sengine):
    # Correlated IN is deferred (a residual non-equi correlation on the
    # subquery side -- here b.w > a.v -- has no clean equi-key).
    with pytest.raises(NotImplementedError):
        sengine.sql("SELECT k, v FROM a WHERE k IN (SELECT k FROM b WHERE b.w > a.v)")


def test_non_equi_correlation_rejected(sengine):
    # A non-equi correlation (b.w > a.v) is deferred (no equi join key).
    with pytest.raises(NotImplementedError):
        sengine.sql("SELECT k, v FROM a WHERE EXISTS (SELECT 1 FROM b WHERE b.w > a.v)")


def test_multi_equi_correlation_rejected(sengine):
    # Two equi correlations (b.k = a.k AND b.w = a.v) are deferred (semi/anti and
    # the decorrelated left join are single-key only).
    with pytest.raises(NotImplementedError):
        sengine.sql("SELECT k, v FROM a WHERE EXISTS (SELECT 1 FROM b "
                    "WHERE b.k = a.k AND b.w = a.v)")


def test_outer_ref_in_agg_arg_rejected(sengine):
    # An outer reference in the aggregate argument cannot be pre-grouped.
    with pytest.raises(NotImplementedError):
        sengine.sql("SELECT k, (SELECT MAX(b.w + a.v) FROM b WHERE b.k = a.k) AS m "
                    "FROM a")


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


def test_nonaggregate_scalar_rejected(sengine):
    # A non-aggregate scalar subquery (0+ rows) has no 1-row guarantee.
    with pytest.raises(NotImplementedError):
        sengine.sql("SELECT k, (SELECT w FROM b) AS c FROM a")


def test_groupby_scalar_rejected(sengine):
    # A GROUP BY scalar subquery yields N rows (one per group).
    with pytest.raises(NotImplementedError):
        sengine.sql("SELECT k, (SELECT k FROM b GROUP BY k) AS c FROM a")