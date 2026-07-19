"""The SQL aggregate ``FILTER (WHERE ...)`` clause.

A per-aggregate predicate restricts which rows contribute to *that one*
aggregate, leaving sibling aggregates and group membership untouched. It is the
canonical TPC-H Q1 staple (``sum(l_quantity) filter (where l_returnflag='R')``)
and the cleanest conditional-aggregate form.

Implementation: sqlglot's ``exp.Filter(this=AggFunc, expression=Where(pred))``
lowers to an ``AggFunc`` carrying a ``filter`` predicate. The cuDF aggregate
paths (``_fused_agg`` / ``_scalar_global_agg``) null the agg's contributing
column where the predicate is False/NA (``df_where`` = ``series.where(mask)``,
which nulls on False OR NA -- so a NULL/unknown predicate excludes the row,
exactly SQL FILTER semantics); cuDF reductions skip nulls. The C++ fused kernels
read only ``af.func``/``af.arg`` and would silently ignore a filter, so any
per-agg filter forces the cuDF fallback (correctness never depends on the fused
paths). FILTER composes with ``ROLLUP/CUBE/GROUPING SETS`` for free: the G-1
desugar carries each ``AggFunc`` (filter attached) into every per-set branch.

The NULL-predicate cases use a bespoke table (the shared fixtures are NULL-free)
so a real NULL in the filter column produces an unknown predicate. DuckDB is the
correctness oracle.
"""

from __future__ import annotations

import duckdb
import pytest

from ryudb import Catalog, Engine
from ryudb.sql.parse import parse
from ryudb.sql.plan import AggFunc, Aggregate, BinOp

from .conftest import assert_same


# --------------------------------------------------------------------- helpers


@pytest.fixture
def null_engine(tmp_path) -> tuple[Engine, duckdb.DuckDBPyConnection]:
    """A small ``g(a INT, b INT, x DOUBLE)`` table with NULLs in both ``a`` and
    ``b`` so a FILTER predicate over a NULLable column yields an unknown (NULL)
    predicate -- the case that proves NULL predicates exclude the row (not
    include it, and not error)."""
    d = tmp_path / "g"
    d.mkdir()
    con = duckdb.connect()
    con.execute("CREATE TABLE g (a INT, b INT, x DOUBLE)")
    con.execute(
        "INSERT INTO g VALUES "
        "(1, 10, 1.0), (1, 11, 2.0), (1, 11, 3.0), "
        "(2, 10, 4.0), (NULL, 10, 5.0), (2, NULL, 6.0), (3, NULL, 7.0)"
    )
    con.execute(f"COPY (SELECT * FROM g) TO '{d}/0.parquet' (FORMAT PARQUET)")
    cat = Catalog(str(tmp_path))
    cat.register("g", str(d))
    eng = Engine(cat)
    oracle = duckdb.connect()
    oracle.execute(f"CREATE VIEW g AS SELECT * FROM read_parquet('{d}/0.parquet')")
    return eng, oracle


def _cmp(eng: Engine, oracle: duckdb.DuckDBPyConnection, sql: str, *, swap: str = "g"):
    """Run ``sql`` on the engine and on the oracle (rewriting ``FROM <swap>``
    to the oracle's view of the same name) and assert equality."""
    ryu = eng.sql(sql)
    duck = oracle.execute(sql).fetchdf()
    assert_same(ryu, duck)


# --------------------------------------------------------------------- parse


def test_parse_filter_attaches_to_aggfunc():
    plan = parse("SELECT sum(x) FILTER (WHERE a>1) AS s FROM t", {"t": ["a", "x"]})
    assert isinstance(plan, Aggregate)
    af = plan.aggs[0][0]
    assert isinstance(af, AggFunc)
    assert af.func == "SUM"
    assert af.filter is not None
    assert isinstance(af.filter, BinOp)
    # The filter's columns are projected (load-bearing for eval_expr).
    assert af.columns() == {"x", "a"}


def test_parse_count_star_filter():
    plan = parse("SELECT count(*) FILTER (WHERE a>1) AS c FROM t", {"t": ["a", "x"]})
    af = plan.aggs[0][0]
    assert af.func == "COUNT"
    assert af.filter is not None
    # COUNT(*) projects only the filter's columns (the * adds none).
    assert af.columns() == {"a"}


def test_parse_no_filter_is_none():
    plan = parse("SELECT sum(x) AS s FROM t", {"t": ["a", "x"]})
    assert plan.aggs[0][0].filter is None


# ------------------------------------------------------- global (no GROUP BY)


def test_global_filtered_sum(null_engine):
    eng, oracle = null_engine
    _cmp(eng, oracle, "SELECT sum(x) FILTER (WHERE a>1) AS s FROM g")


def test_global_null_predicate_excludes(null_engine):
    """``b`` is NULLable; ``b>5`` is unknown on NULL-b rows -> those rows are
    excluded from the filtered sum (not included, not an error)."""
    eng, oracle = null_engine
    _cmp(eng, oracle, "SELECT sum(x) FILTER (WHERE b>5) AS s FROM g")


def test_global_count_star_filter(null_engine):
    eng, oracle = null_engine
    _cmp(eng, oracle, "SELECT count(*) FILTER (WHERE b>5) AS c FROM g")


def test_global_mixed_filtered_and_unfiltered(null_engine):
    """TPC-H Q1 shape: unfiltered aggregates alongside filtered ones in one
    query -- the per-agg mask touches only the filtered aggs."""
    eng, oracle = null_engine
    _cmp(
        eng, oracle,
        "SELECT sum(x) AS all_s, sum(x) FILTER (WHERE a>1) AS f_s, "
        "count(*) AS all_c, count(*) FILTER (WHERE b=10) AS f_c FROM g",
    )


def test_global_avg_min_max_filter(null_engine):
    eng, oracle = null_engine
    _cmp(
        eng, oracle,
        "SELECT avg(x) FILTER (WHERE a>1) AS av, min(x) FILTER (WHERE a>1) AS mn, "
        "max(x) FILTER (WHERE a>1) AS mx FROM g",
    )


def test_global_count_col_filter(null_engine):
    """``COUNT(b) FILTER (WHERE a>1)`` counts non-null ``b`` where the filter
    passes (NULL-b rows excluded twice over)."""
    eng, oracle = null_engine
    _cmp(eng, oracle, "SELECT count(b) FILTER (WHERE a>1) AS c FROM g")


# ------------------------------------------------------- grouped (GROUP BY)


def test_grouped_filtered_sum(null_engine):
    eng, oracle = null_engine
    _cmp(eng, oracle, "SELECT a, sum(x) FILTER (WHERE b>5) AS s FROM g GROUP BY a")


def test_grouped_count_star_filter_is_not_null(null_engine):
    eng, oracle = null_engine
    _cmp(eng, oracle, "SELECT a, count(*) FILTER (WHERE b IS NOT NULL) AS c FROM g GROUP BY a")


def test_grouped_filtered_and_unfiltered(null_engine):
    eng, oracle = null_engine
    _cmp(
        eng, oracle,
        "SELECT a, sum(x) AS s, sum(x) FILTER (WHERE b>5) AS fs, "
        "count(*) AS c, count(*) FILTER (WHERE b=10) AS fc FROM g GROUP BY a",
    )


# ------------------------------------------------------- FILTER + WHERE


def test_filter_with_where(null_engine):
    """FILTER is independent of WHERE: WHERE restricts the whole input (all
    aggs + group membership); FILTER restricts one agg's contributing rows."""
    eng, oracle = null_engine
    _cmp(eng, oracle, "SELECT sum(x) FILTER (WHERE a>1) AS s FROM g WHERE b<20")


def test_filter_with_where_grouped(null_engine):
    eng, oracle = null_engine
    _cmp(
        eng, oracle,
        "SELECT a, sum(x) FILTER (WHERE b>5) AS s FROM g WHERE a IS NOT NULL GROUP BY a",
    )


# ------------------------------------------------------- on the shared fixtures


def test_engine_grouped_filtered(engine, duck):
    _cmp(engine, duck,
         "SELECT l_orderkey, sum(l_quantity) FILTER (WHERE l_quantity>3) AS s, "
         "count(*) FILTER (WHERE l_quantity>3) AS c FROM lineitem GROUP BY l_orderkey")


def test_engine_global_filtered(engine, duck):
    _cmp(engine, duck,
         "SELECT sum(l_extendedprice) AS all_s, "
         "sum(l_extendedprice) FILTER (WHERE l_orderkey>2) AS f_s, "
         "count(*) FILTER (WHERE l_quantity>=5) AS f_c FROM lineitem")


def test_engine_filter_with_join(engine, duck):
    """FILTER over a join+WHERE shape (forces the cuDF fallback past the fused
    star-join+aggregate kernel)."""
    _cmp(engine, duck,
         "SELECT o_custkey, sum(l_extendedprice) FILTER (WHERE l_quantity>3) AS s "
         "FROM orders JOIN lineitem ON o_orderkey=l_orderkey "
         "WHERE o_totalprice>60 GROUP BY o_custkey")


# ------------------------------------------------------- typed DECIMAL/DATE


def test_typed_filtered_decimal(typed_engine, typed_duck):
    typed_engine.clear_scan_cache()
    ryu = typed_engine.sql(
        "SELECT sum(l_extendedprice) FILTER (WHERE l_discount>0.05) AS s, "
        "count(*) FILTER (WHERE l_tax>0.02) AS c FROM lineitem"
    )
    duck = typed_duck.execute(
        "SELECT sum(l_extendedprice) FILTER (WHERE l_discount>0.05) AS s, "
        "count(*) FILTER (WHERE l_tax>0.02) AS c FROM lineitem"
    ).fetchdf()
    assert_same(ryu, duck)


def test_typed_filtered_date(typed_engine, typed_duck):
    typed_engine.clear_scan_cache()
    ryu = typed_engine.sql(
        "SELECT l_orderkey, sum(l_quantity) FILTER (WHERE l_shipdate < date '1995-01-01') AS s "
        "FROM lineitem GROUP BY l_orderkey ORDER BY l_orderkey"
    )
    duck = typed_duck.execute(
        "SELECT l_orderkey, sum(l_quantity) FILTER (WHERE l_shipdate < date '1995-01-01') AS s "
        "FROM lineitem GROUP BY l_orderkey ORDER BY l_orderkey"
    ).fetchdf()
    assert_same(ryu, duck)


# ------------------------------------------------------- FILTER + ROLLUP (G-1)


def test_filter_with_rollup(null_engine):
    """FILTER composes with ROLLUP for free: the G-1 desugar carries the
    filter-bearing AggFunc into every per-set branch."""
    eng, oracle = null_engine
    _cmp(eng, oracle, "SELECT a, sum(x) FILTER (WHERE a>1) AS s FROM g GROUP BY ROLLUP(a)")


def test_filter_with_rollup_two_dims(null_engine):
    eng, oracle = null_engine
    _cmp(
        eng, oracle,
        "SELECT a, b, sum(x) FILTER (WHERE a>1) AS s, "
        "count(*) FILTER (WHERE b=10) AS c FROM g GROUP BY ROLLUP(a, b)",
    )


def test_filter_with_cube(null_engine):
    eng, oracle = null_engine
    _cmp(eng, oracle, "SELECT a, b, sum(x) FILTER (WHERE b>5) AS s FROM g GROUP BY CUBE(a, b)")


# ------------------------------------------------------- rejections


def test_reject_filter_on_non_aggregate():
    # ``x FILTER (WHERE ...)`` on a non-aggregate is invalid SQL (DuckDB rejects
    # it too); sqlglot itself rejects it at parse time before our _expr runs.
    with pytest.raises(Exception):
        parse("SELECT x FILTER (WHERE a>1) FROM t", {"t": ["a", "x"]})


# ----------------------------------------------------------------- CLI smoke


def test_cli_filter_output(engine, capsys):
    from ryudb import cli

    cli._run_statement(
        engine,
        "SELECT l_orderkey, sum(l_quantity) AS qty, "
        "sum(l_quantity) FILTER (WHERE l_quantity>3) AS big_qty, "
        "count(*) FILTER (WHERE l_quantity>3) AS big_n "
        "FROM lineitem GROUP BY l_orderkey ORDER BY l_orderkey",
        quiet=False,
    )
    out = capsys.readouterr().out
    # The result is non-empty and contains the aliased filtered column header.
    assert "big_qty" in out
    assert "big_n" in out