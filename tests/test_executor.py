"""Executor correctness tests: RyuDB (GPU) vs DuckDB (CPU) on the same SQL."""

from __future__ import annotations

import pytest

from .conftest import assert_same

QUERIES = [
    "SELECT o_orderkey, o_totalprice FROM orders",
    "SELECT * FROM orders WHERE o_custkey = 10",
    "SELECT o_orderkey FROM orders WHERE o_totalprice > 75",
    "SELECT count(*) AS n FROM lineitem",
    "SELECT sum(l_extendedprice) AS total FROM lineitem",
    "SELECT avg(l_quantity) AS qavg, min(l_quantity) AS qmin, max(l_quantity) AS qmax FROM lineitem",
    "SELECT l_orderkey, sum(l_quantity) AS qty FROM lineitem GROUP BY l_orderkey",
    "SELECT l_orderkey, count(*) AS n FROM lineitem GROUP BY l_orderkey ORDER BY l_orderkey",
    "SELECT o_custkey, sum(l_extendedprice) AS rev "
    "FROM orders JOIN lineitem ON o_orderkey = l_orderkey "
    "WHERE o_totalprice > 75 GROUP BY o_custkey ORDER BY o_custkey",
    "SELECT o_custkey, sum(l_extendedprice) AS rev "
    "FROM orders JOIN lineitem ON o_orderkey = l_orderkey "
    "WHERE o_totalprice > 75 AND l_quantity >= 5 GROUP BY o_custkey",
    "SELECT l_orderkey, l_quantity FROM lineitem ORDER BY l_orderkey, l_quantity DESC LIMIT 3",
    "SELECT n_name, sum(o_totalprice) AS spend "
    "FROM orders JOIN nation ON o_custkey = n_nationkey GROUP BY n_name ORDER BY n_name",
    "SELECT o_orderkey, l_quantity FROM orders JOIN lineitem ON o_orderkey = l_orderkey "
    "WHERE l_quantity >= 5 ORDER BY o_orderkey, l_quantity DESC",
    "SELECT l_shipdate, count(*) AS c FROM lineitem GROUP BY l_shipdate ORDER BY l_shipdate",
    "SELECT o_orderkey, o_totalprice * 0.1 AS tax FROM orders ORDER BY o_orderkey",
]


@pytest.mark.parametrize("sql", QUERIES)
def test_vs_duckdb(engine, duck, sql):
    ryu = engine.sql(sql)
    d = duck.execute(sql).fetchdf()
    assert_same(ryu, d)


def test_empty_result(engine, duck):
    ryu = engine.sql("SELECT * FROM orders WHERE o_totalprice > 99999")
    d = duck.execute("SELECT * FROM orders WHERE o_totalprice > 99999").fetchdf()
    assert len(ryu) == 0
    assert len(d) == 0


def test_date_filter(engine, duck):
    sql = ("SELECT l_orderkey, l_quantity FROM lineitem "
           "WHERE l_shipdate <= date '1998-09-01' ORDER BY l_orderkey")
    ryu = engine.sql(sql)
    d = duck.execute(sql).fetchdf()
    assert_same(ryu, d)


def test_explain_returns_string(engine):
    plan = engine.explain("SELECT count(*) FROM lineitem")
    assert isinstance(plan, str)
    assert "Aggregate" in plan


# --------------------------------------------------------------------------- #
# finer-grained cooperative cancel (PR #98)
# --------------------------------------------------------------------------- #

class _FlipEvent:
    """A stand-in for ``threading.Event`` whose ``is_set()`` returns False for
    the first ``skip`` calls then True forever after, counting how many times
    it was polled. Used to prove a cancel that flips *partway through* a long
    inner loop is honored before the node completes — i.e. the loop-level
    cancel check fires, not just the per-node boundary check.

    With only the per-node boundary check, a plan like Scan -> Project runs
    exactly 2 ``is_set()`` polls; a ``skip`` > 2 would never trip and the query
    would complete normally. The finer-grained loop checks add one poll per
    loop iteration, so the event flips mid-loop and ``CancelledByUser`` raises
    — and ``n_polls`` ends up > 2 (the boundary count), proving the loop-level
    checks ran."""

    def __init__(self, skip: int) -> None:
        self._skip = skip
        self.n_polls = 0

    def is_set(self) -> bool:
        self.n_polls += 1
        return self.n_polls > self._skip

    def set(self) -> None:  # noqa: D401 -- satisfy any isinstance/attr checks
        self._skip = -1


def test_cancel_mid_project_loop(engine):
    """A wide Project (30 distinct arithmetic columns) runs a 30-iteration
    ``eval_expr`` loop; a flip-after-5 cancel raises mid-loop, and the poll
    count (>2 boundary checks) proves the per-column loop check fired."""
    from ryudb.exec.executor import CancelledByUser

    cols = ", ".join(f"l_quantity + {i} AS x{i}" for i in range(30))
    sql = f"SELECT {cols} FROM lineitem"
    # Control: without a cancel event the query completes and returns 30 cols.
    assert engine.cancel_event is None
    out = engine.sql(sql)
    assert len(out.columns) == 30

    ev = _FlipEvent(skip=5)
    engine.cancel_event = ev  # type: ignore[assignment]
    try:
        with pytest.raises(CancelledByUser):
            engine.sql(sql)
    finally:
        engine.cancel_event = None
    # 2 node-boundary polls (Project, Scan) + the per-column loop polls; >2
    # proves the loop-level check fired, and <32 (boundary + 30 cols) proves it
    # raised before completing the loop.
    assert 2 < ev.n_polls < 32


def test_cancel_mid_scalar_global_agg_loop(engine):
    """A no-WHERE/no-GROUP-BY global aggregate with many aggregates runs the
    ``_scalar_global_agg`` per-aggregate reduction loop; a mid-loop cancel
    raises before the node completes (loop-level check, not just boundary)."""
    from ryudb.exec.executor import CancelledByUser

    funcs = ["sum", "avg", "min", "max"]
    cols = []
    for i in range(30):
        f = funcs[i % 4]
        c = "l_quantity" if i % 2 == 0 else "l_extendedprice"
        cols.append(f"{f}({c}) AS a{i}")
    sql = f"SELECT {', '.join(cols)} FROM lineitem"
    # No cancel -> completes, one row.
    assert engine.sql(sql).shape[0] == 1

    ev = _FlipEvent(skip=5)
    engine.cancel_event = ev  # type: ignore[assignment]
    try:
        with pytest.raises(CancelledByUser):
            engine.sql(sql)
    finally:
        engine.cancel_event = None
    assert 2 < ev.n_polls < 32


def test_cancel_event_none_is_noop(engine):
    """Sanity: ``cancel_event`` left at None (the CLI/in-process default) never
    raises — the ``None`` check short-circuits every poll. A many-aggregate
    query runs to completion."""
    assert engine.cancel_event is None
    cols = ", ".join(f"sum(l_quantity) AS a{i}" for i in range(20))
    out = engine.sql(f"SELECT {cols} FROM lineitem")
    assert out.shape[0] == 1 and len(out.columns) == 20