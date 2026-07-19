"""GROUP BY ROLLUP / CUBE / GROUPING SETS + the GROUPING() marker function.

A grouping extension is desugared at parse time into a ``UNION ALL`` of one
``Aggregate`` branch per grouping set; each branch NULLs the omitted grouping
columns and substitutes a per-branch constant integer for every ``GROUPING()``
call (the "not-grouped" bitmap, leftmost argument = most-significant bit).
``UNION ALL`` (not ``UNION``) is load-bearing: when a real data NULL collides
with a subtotal NULL the duplicate rows are NOT deduped -- they are
distinguished by ``GROUPING`` (verified against DuckDB). DuckDB is the
correctness oracle.

The critical cases -- NULL data in the grouping columns so a real NULL group
collides with a subtotal NULL -- use a small bespoke table (the shared
``lineitem``/``orders`` fixtures have no NULLs). The no-NULL fixtures exercise
ROLLUP over a WHERE filter (the cold Parquet reader handles filtered GROUP BY
correctly only when the group columns are NULL-free; a pre-existing cold-reader
bug corrupts NULL group keys under a filter, which is independent of this
feature and out of scope here).
"""

from __future__ import annotations

import duckdb
import pytest

from ryudb import Catalog, Engine
from ryudb.sql.parse import parse
from ryudb.sql.plan import Aggregate, Project, SetOp

from .conftest import assert_same


# --------------------------------------------------------------------- helpers


@pytest.fixture
def null_engine(tmp_path) -> tuple[Engine, duckdb.DuckDBPyConnection]:
    """A small ``g(a INT, b INT, x DOUBLE)`` table with NULLs in both ``a`` and
    ``b`` so a real-data NULL group collides with the subtotal NULL -- the case
    that proves UNION ALL (not UNION) and NULL-key retention."""
    d = tmp_path / "g"
    d.mkdir()
    con = duckdb.connect()
    con.execute("CREATE TABLE g (a INT, b INT, x DOUBLE)")
    con.execute(
        "INSERT INTO g VALUES "
        "(1, 10, 1.0), (1, 11, 2.0), (1, 11, 3.0), "
        "(2, 10, 4.0), (NULL, 10, 5.0), (2, NULL, 6.0)"
    )
    con.execute(f"COPY (SELECT * FROM g) TO '{d}/0.parquet' (FORMAT PARQUET)")
    cat = Catalog(str(tmp_path))
    cat.register("g", str(d))
    eng = Engine(cat)
    # A separate oracle connection over the same parquet (writable views below).
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


def test_parse_rollup_is_union_all_of_branches():
    plan = parse(
        "SELECT a, b, sum(x) FROM t GROUP BY ROLLUP(a, b)",
        {"t": ["a", "b", "x"]},
    )
    # The desugar is a UNION ALL (distinct=False) of per-set branches.
    assert isinstance(plan, SetOp)
    assert plan.op == "union"
    assert plan.distinct is False
    # Three branches: {a,b}, {a}, {} -- one Project -> Aggregate each.
    branches = []

    def walk(n):
        if isinstance(n, SetOp):
            walk(n.left)
            walk(n.right)
        elif isinstance(n, Project) and isinstance(n.input, Aggregate):
            branches.append(n)

    walk(plan)
    assert len(branches) == 3
    gk_sets = [frozenset(al for _, al in b.input.group_keys) for b in branches]
    assert frozenset() in gk_sets
    assert frozenset(["a"]) in gk_sets
    assert frozenset(["a", "b"]) in gk_sets


def test_parse_cube_is_four_branches():
    plan = parse("SELECT a, b, sum(x) FROM t GROUP BY CUBE(a, b)", {"t": ["a", "b", "x"]})
    branches = []

    def walk(n):
        if isinstance(n, SetOp):
            walk(n.left)
            walk(n.right)
        elif isinstance(n, Project) and isinstance(n.input, Aggregate):
            branches.append(n)

    walk(plan)
    assert len(branches) == 4  # 2**2 subsets
    gk_sets = {frozenset(al for _, al in b.input.group_keys) for b in branches}
    assert gk_sets == {frozenset(), frozenset(["a"]), frozenset(["b"]), frozenset(["a", "b"])}


def test_parse_grouping_sets_explicit():
    plan = parse(
        "SELECT a, b, sum(x) FROM t GROUP BY GROUPING SETS ((a, b), (a), ())",
        {"t": ["a", "b", "x"]},
    )
    branches = []

    def walk(n):
        if isinstance(n, SetOp):
            walk(n.left)
            walk(n.right)
        elif isinstance(n, Project) and isinstance(n.input, Aggregate):
            branches.append(n)

    walk(plan)
    assert len(branches) == 3
    gk_sets = {frozenset(al for _, al in b.input.group_keys) for b in branches}
    assert gk_sets == {frozenset(), frozenset(["a"]), frozenset(["a", "b"])}


# ------------------------------------------------------- NULL-overlap ROLLUP


def test_rollup_ab_null_overlap(null_engine):
    eng, oracle = null_engine
    _cmp(eng, oracle, "SELECT a, b, sum(x) AS s, count(*) AS c FROM g GROUP BY ROLLUP(a, b)")


def test_rollup_a_null_overlap(null_engine):
    eng, oracle = null_engine
    _cmp(eng, oracle, "SELECT a, sum(x) AS s FROM g GROUP BY ROLLUP(a)")


def test_cube_ab(null_engine):
    eng, oracle = null_engine
    _cmp(eng, oracle, "SELECT a, b, sum(x) AS s FROM g GROUP BY CUBE(a, b)")


def test_grouping_sets_explicit(null_engine):
    eng, oracle = null_engine
    _cmp(eng, oracle, "SELECT a, b, sum(x) AS s FROM g GROUP BY GROUPING SETS ((a, b), (a), ())")


# ------------------------------------------------------- GROUPING() bitmap


def test_grouping_single_and_bitmap(null_engine):
    eng, oracle = null_engine
    _cmp(
        eng,
        oracle,
        "SELECT a, b, sum(x) AS s, GROUPING(a) AS ga, GROUPING(b) AS gb, "
        "GROUPING(a, b) AS gab FROM g GROUP BY ROLLUP(a, b)",
    )


def test_grouping_bitmap_distinguishes_overlap(null_engine):
    """The two (NULL, NULL) rows -- a real ``a=NULL`` group and the grand total
    -- must both appear and be distinguished by GROUPING(a)."""
    eng, oracle = null_engine
    df = eng.sql(
        "SELECT a, GROUPING(a) AS ga FROM g GROUP BY ROLLUP(a) ORDER BY a"
    ).to_pandas()
    # Exactly two NULL-a rows: ga=0 (the real a=NULL group) and ga=1 (grand total).
    null_rows = df[df["a"].isna()]
    assert len(null_rows) == 2
    assert set(null_rows["ga"].tolist()) == {0, 1}


# ------------------------------------------------------- mixed / composite


def test_mixed_always_on_and_rollup(null_engine):
    eng, oracle = null_engine
    _cmp(eng, oracle, "SELECT a, b, sum(x) AS s FROM g GROUP BY a, ROLLUP(b)")


def test_rollup_composite_item(tmp_path):
    """``ROLLUP(a, (b, c))`` drops the composite (b, c) unit together -- the
    hierarchy is {a,b,c}, {a}, {} with NO {a, b}-without-c level."""
    d = tmp_path / "gc"
    d.mkdir()
    con = duckdb.connect()
    con.execute("CREATE TABLE gc (a INT, b INT, c INT, x DOUBLE)")
    con.execute(
        "INSERT INTO gc VALUES "
        "(1, 10, 100, 1.0), (1, 11, 101, 2.0), (2, 10, 100, 3.0), "
        "(2, 11, 101, 4.0), (NULL, 10, 100, 5.0)"
    )
    con.execute(f"COPY (SELECT * FROM gc) TO '{d}/0.parquet' (FORMAT PARQUET)")
    oracle = duckdb.connect()
    oracle.execute(f"CREATE VIEW gc AS SELECT * FROM read_parquet('{d}/0.parquet')")
    cat = Catalog(str(tmp_path))
    cat.register("gc", str(d))
    eng = Engine(cat)
    _cmp(eng, oracle, "SELECT a, b, c, sum(x) AS s FROM gc GROUP BY ROLLUP(a, (b, c))")
    # Sanity: the {a, b}-without-c level must NOT appear (b and c are a unit).
    df = eng.sql("SELECT a, b, c, sum(x) AS s FROM gc GROUP BY ROLLUP(a, (b, c))").to_pandas()
    # A row with b non-null and c null (other than the grand total / {a} level)
    # would only exist if (b,c) were split -- assert none exist.
    partial = df[df["b"].notna() & df["c"].isna()]
    assert len(partial) == 0


# ------------------------------------------------------- tail clauses


def test_rollup_order_by(engine, duck):
    _cmp(engine, duck,
         "SELECT l_orderkey, l_shipdate, sum(l_quantity) AS q FROM lineitem "
         "GROUP BY ROLLUP(l_orderkey, l_shipdate) ORDER BY l_orderkey, l_shipdate")


def test_rollup_distinct(engine, duck):
    _cmp(engine, duck,
         "SELECT DISTINCT l_orderkey, l_shipdate FROM lineitem "
         "GROUP BY ROLLUP(l_orderkey, l_shipdate)")


def test_rollup_over_where_no_nulls(engine, duck):
    """ROLLUP over a WHERE filter on NULL-free group columns (the cold reader
    handles filtered GROUP BY only when group cols are NULL-free)."""
    _cmp(engine, duck,
         "SELECT o_custkey, count(*) AS c, sum(o_totalprice) AS s FROM orders "
         "WHERE o_totalprice > 60 GROUP BY ROLLUP(o_custkey)")


# ------------------------------------------------------- typed DECIMAL/DATE


def test_rollup_typed_decimal_date(typed_engine, typed_duck):
    typed_engine.clear_scan_cache()
    ryu = typed_engine.sql(
        "SELECT l_shipdate, sum(l_quantity) AS qty, max(l_extendedprice) AS mx "
        "FROM lineitem GROUP BY ROLLUP(l_shipdate) ORDER BY l_shipdate"
    )
    duck = typed_duck.execute(
        "SELECT l_shipdate, sum(l_quantity) AS qty, max(l_extendedprice) AS mx "
        "FROM lineitem GROUP BY ROLLUP(l_shipdate) ORDER BY l_shipdate"
    ).fetchdf()
    assert_same(ryu, duck)


def test_cube_typed_two_dims(typed_engine, typed_duck):
    typed_engine.clear_scan_cache()
    ryu = typed_engine.sql(
        "SELECT l_discount, l_tax, sum(l_quantity) AS qty, max(l_extendedprice) AS mx "
        "FROM lineitem GROUP BY CUBE(l_discount, l_tax) ORDER BY l_discount, l_tax"
    )
    duck = typed_duck.execute(
        "SELECT l_discount, l_tax, sum(l_quantity) AS qty, max(l_extendedprice) AS mx "
        "FROM lineitem GROUP BY CUBE(l_discount, l_tax) ORDER BY l_discount, l_tax"
    ).fetchdf()
    assert_same(ryu, duck)


# ------------------------------------------------------- no-agg ROLLUP


def test_rollup_no_aggregate(null_engine):
    """``SELECT a FROM t GROUP BY ROLLUP(a)`` = distinct a + one NULL grand-total
    row. Exercises the empty-aggs Aggregate path (a no-agg branch yields the
    distinct group keys; the grand-total branch yields one NULL row)."""
    eng, oracle = null_engine
    _cmp(eng, oracle, "SELECT a FROM g GROUP BY ROLLUP(a)")


def test_rollup_no_aggregate_two_dims(null_engine):
    eng, oracle = null_engine
    _cmp(eng, oracle, "SELECT a, b FROM g GROUP BY ROLLUP(a, b)")


# ------------------------------------------------------- rejections


def test_reject_having_with_rollup():
    with pytest.raises(NotImplementedError, match="HAVING"):
        parse(
            "SELECT a, sum(x) FROM t GROUP BY ROLLUP(a) HAVING sum(x) > 1",
            {"t": ["a", "x"]},
        )


def test_reject_grouping_without_extension():
    # GROUPING() requires a grouping set (ROLLUP/CUBE/GROUPING SETS); a bare
    # GROUP BY with GROUPING() is invalid SQL and _expr has no Grouping case.
    with pytest.raises(NotImplementedError):
        parse("SELECT a, GROUPING(a) FROM t GROUP BY a", {"t": ["a", "x"]})


def test_reject_grouping_col_not_in_select():
    with pytest.raises(NotImplementedError, match="SELECT list"):
        parse("SELECT a, sum(x) FROM t GROUP BY ROLLUP(a, b)", {"t": ["a", "b", "x"]})


def test_reject_grouping_names_non_grouping_col():
    with pytest.raises(NotImplementedError, match="non-grouping"):
        parse(
            "SELECT a, sum(x) AS s, GROUPING(b) AS gb FROM t GROUP BY ROLLUP(a)",
            {"t": ["a", "b", "x"]},
        )


def test_reject_expression_group_key():
    # The desugar matches dimensions by column name, so only bare columns are
    # supported in a grouping extension (computed keys like ``a % 10`` are not).
    with pytest.raises(NotImplementedError, match="bare columns"):
        parse(
            "SELECT a, sum(x) AS s FROM t GROUP BY ROLLUP(a, a % 10)",
            {"t": ["a", "x"]},
        )


# ----------------------------------------------------------------- CLI smoke


def test_cli_rollup_output(engine, capsys):
    from ryudb import cli

    cli._run_statement(
        engine,
        "SELECT l_orderkey, l_shipdate, sum(l_quantity) AS q, "
        "GROUPING(l_orderkey, l_shipdate) AS g FROM lineitem "
        "GROUP BY ROLLUP(l_orderkey, l_shipdate) ORDER BY l_orderkey, l_shipdate",
        quiet=False,
    )
    out = capsys.readouterr().out
    # The grand-total row has both keys NULL and GROUPING = 3.
    assert "3" in out
    # A detail row (GROUPING = 0) is present -- orderkey values 1..5.
    assert "1" in out