"""Executor correctness tests: RyuDB (GPU) vs DuckDB (CPU) on the same SQL."""

from __future__ import annotations

import pandas as pd
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