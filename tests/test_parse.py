"""Parser tests: SQL -> logical plan shape."""

from __future__ import annotations

from ryudb.sql.parse import parse
from ryudb.sql.plan import (
    AggFunc,
    BinOp,
    Join,
)


def _types(plan):
    from ryudb.sql.plan import walk
    return [type(n).__name__ for n in walk(plan)]


def test_simple_scan_project():
    plan = parse("SELECT a, b FROM t")
    assert _types(plan) == ["Project", "Scan"]


def test_filter():
    plan = parse("SELECT a FROM t WHERE a > 5")
    assert _types(plan) == ["Project", "Filter", "Scan"]
    assert isinstance(plan.input.predicate, BinOp)


def test_join():
    plan = parse("SELECT a FROM t1 JOIN t2 ON t1.k = t2.k")
    assert _types(plan) == ["Project", "Join", "Scan", "Scan"]
    join = plan.input
    assert join.on_left == ["k"]
    assert join.on_right == ["k"]


def test_groupby_aggregate():
    plan = parse("SELECT k, count(*) AS c FROM t GROUP BY k")
    assert _types(plan) == ["Aggregate", "Scan"]
    assert plan.group_keys[0][1] == "k"
    assert isinstance(plan.aggs[0][0], AggFunc)
    assert plan.aggs[0][1] == "c"


def test_order_limit():
    plan = parse("SELECT a FROM t ORDER BY a DESC LIMIT 10")
    assert _types(plan) == ["Limit", "Sort", "Project", "Scan"]


def test_star_no_project():
    plan = parse("SELECT * FROM t")
    # SELECT * passes through with no explicit Project node
    assert _types(plan) == ["Scan"]


def test_in_subquery_lowers_to_semi_join():
    # Uncorrelated IN (SELECT ...) in WHERE folds to a semi join (Phase E-1).
    plan = parse("SELECT * FROM t WHERE a IN (SELECT b FROM u)")
    assert isinstance(plan, Join)
    assert plan.how == "semi"
    assert plan.on_left == ["a"]
    assert plan.on_right == ["b"]