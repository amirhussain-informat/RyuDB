"""Optimizer tests: predicate pushdown, projection pruning, join-side selection."""

from __future__ import annotations

from ryudb.sql.optimize import optimize
from ryudb.sql.parse import parse
from ryudb.sql.plan import Filter, Join, Scan, walk

SCHEMA = {
    "orders": ["o_orderkey", "o_custkey", "o_totalprice", "o_orderdate"],
    "lineitem": ["l_orderkey", "l_quantity", "l_extendedprice", "l_shipdate"],
}
STATS = {"orders": 5, "lineitem": 8}


def _scans(plan):
    return [n for n in walk(plan) if isinstance(n, Scan)]


def test_predicate_pushdown_below_join():
    plan = parse(
        "SELECT o_custkey FROM orders JOIN lineitem ON o_orderkey = l_orderkey "
        "WHERE o_totalprice > 75"
    )
    opt = optimize(plan, SCHEMA, STATS)
    # The filter must sit directly above the orders scan, not above the join.
    orders_scan = next(s for s in _scans(opt) if s.table == "orders")
    parent = _parent(opt, orders_scan)
    assert isinstance(parent, Filter), "expected Filter directly above orders scan"


def _parent(root, target):
    for node in walk(root):
        for child in _children(node):
            if child is target:
                return node
    return None


def _children(node):
    if isinstance(node, Filter):
        return [node.input]
    if isinstance(node, Join):
        return [node.left, node.right]
    if hasattr(node, "input"):
        return [node.input]
    return []


def test_projection_pruning():
    plan = parse(
        "SELECT o_custkey, sum(l_extendedprice) AS rev "
        "FROM orders JOIN lineitem ON o_orderkey = l_orderkey "
        "GROUP BY o_custkey"
    )
    opt = optimize(plan, SCHEMA, STATS)
    orders_scan = next(s for s in _scans(opt) if s.table == "orders")
    lineitem_scan = next(s for s in _scans(opt) if s.table == "lineitem")
    assert orders_scan.columns == {"o_custkey", "o_orderkey"}
    assert lineitem_scan.columns == {"l_orderkey", "l_extendedprice"}


def test_star_disables_pruning():
    plan = parse("SELECT * FROM orders WHERE o_totalprice > 50")
    opt = optimize(plan, SCHEMA, STATS)
    orders_scan = _scans(opt)[0]
    assert orders_scan.columns is None


def test_join_side_selection_smaller_build():
    plan = parse(
        "SELECT o_custkey FROM orders JOIN lineitem ON o_orderkey = l_orderkey"
    )
    opt = optimize(plan, SCHEMA, STATS)
    join = next(n for n in walk(opt) if isinstance(n, Join))
    # orders (5) < lineitem (8) => orders on the left (build) side
    assert isinstance(join.left, Scan) and join.left.table == "orders"
    assert join.on_left == ["o_orderkey"]


def test_multitable_filter_stays_above_join():
    # l_quantity references lineitem only -> pushed down; o_totalprice -> orders.
    plan = parse(
        "SELECT o_custkey FROM orders JOIN lineitem ON o_orderkey = l_orderkey "
        "WHERE o_totalprice > 75 AND l_quantity >= 5"
    )
    opt = optimize(plan, SCHEMA, STATS)
    orders_scan = next(s for s in _scans(opt) if s.table == "orders")
    lineitem_scan = next(s for s in _scans(opt) if s.table == "lineitem")
    assert isinstance(_parent(opt, orders_scan), Filter)
    assert isinstance(_parent(opt, lineitem_scan), Filter)