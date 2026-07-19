"""Self-join and cross-table column-name collision resolution.

A self-join (``FROM t a JOIN t b``) or any join whose two inputs share a column
name used to fail: the plan's ``Col`` is flat (no table qualifier), cuDF
``merge`` suffixes colliding columns ``_x``/``_y``, and ``eval_expr`` resolved
``Col`` by bare name -- so ``a.k``/``b.k`` could not address ``k_x``/``k_y``,
and same-named equi-join keys collapsed to one column. The fix tracks the
sqlglot column qualifier (``Col.table``) and the FROM/JOIN alias (``Scan.alias``)
through parse, renames each side's colliding columns to ``{alias}__{name}`` in
``_join`` (so qualified refs resolve and both join keys survive, matching
DuckDB), dedups same-named projection outputs (``v``/``v_1``), and keeps
self-join WHERE conjuncts above the join (optimizer pushdown would route them to
the wrong alias). DuckDB is the oracle.

TPC-H has zero cross-input column-name collisions, so the rename path is inert on
the bench; these tests pin the additive self-join / same-named-column behavior.
"""

from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from ryudb import Catalog, Engine

from .conftest import assert_same


# --------------------------------------------------------------------------- #
# Fixtures: a single table ``t(k, v)`` for self-joins, and two tables ``t1`` /
# ``t2`` both with columns ``(k, v)`` for the cross-table same-named-column case.
# --------------------------------------------------------------------------- #


def _write_t(path, rows):
    pd.DataFrame(rows, columns=["k", "v"]).to_parquet(str(path))


@pytest.fixture
def sengine(tmp_path) -> tuple[Engine, duckdb.DuckDBPyConnection]:
    """``t(k INT, v INT)`` with a few rows; self-join ``a JOIN t b`` shares both
    ``k`` and ``v`` across the two sides."""
    d = tmp_path / "t"
    d.mkdir()
    _write_t(d / "0.parquet", {"k": [1, 2, 3, 4], "v": [10, 20, 30, 40]})
    cat = Catalog(str(tmp_path))
    cat.register("t", str(d))
    eng = Engine(cat)
    duck = duckdb.connect()
    duck.execute(f"CREATE VIEW t AS SELECT * FROM read_parquet('{d}/0.parquet')")
    return eng, duck


@pytest.fixture
def xengine(tmp_path) -> tuple[Engine, duckdb.DuckDBPyConnection]:
    """Two tables ``t1`` and ``t2`` both with columns ``(k, v)`` so a join
    ``t1 a JOIN t2 b`` collides on both ``k`` and ``v`` (cross-table)."""
    d1 = tmp_path / "t1"
    d1.mkdir()
    _write_t(d1 / "0.parquet", {"k": [1, 2, 3], "v": [100, 200, 300]})
    d2 = tmp_path / "t2"
    d2.mkdir()
    _write_t(d2 / "0.parquet", {"k": [1, 2, 4], "v": [7, 8, 9]})
    cat = Catalog(str(tmp_path))
    cat.register("t1", str(d1))
    cat.register("t2", str(d2))
    eng = Engine(cat)
    duck = duckdb.connect()
    duck.execute(f"CREATE VIEW t1 AS SELECT * FROM read_parquet('{d1}/0.parquet')")
    duck.execute(f"CREATE VIEW t2 AS SELECT * FROM read_parquet('{d2}/0.parquet')")
    return eng, duck


def _ryu(eng: Engine, sql: str):
    return eng.sql(sql)


def _duck(duck, sql: str):
    return duck.execute(sql).fetchdf()


# --------------------------------------------------------------------------- #
# CROSS self-join (the original repro)
# --------------------------------------------------------------------------- #


def test_cross_self_join_where_eq(sengine):
    eng, duck = sengine
    sql = (
        "SELECT a.v, b.v FROM t a CROSS JOIN t b WHERE a.k = b.k ORDER BY a.k"
    )
    assert_same(_ryu(eng, sql), _duck(duck, sql))


def test_cross_self_join_aliased_outputs(sengine):
    eng, duck = sengine
    sql = (
        "SELECT a.v AS av, b.v AS bv FROM t a CROSS JOIN t b "
        "WHERE a.k = b.k ORDER BY a.k"
    )
    assert_same(_ryu(eng, sql), _duck(duck, sql))


def test_cross_self_join_unqualified_ambiguous_raises(sengine):
    """An unqualified colliding column is ambiguous; DuckDB errors and so must
    RyuDB (it should not silently pick one side)."""
    eng, _duck = sengine
    with pytest.raises(Exception):
        eng.sql("SELECT v FROM t a CROSS JOIN t b WHERE a.k = b.k")


# --------------------------------------------------------------------------- #
# INNER self-join
# --------------------------------------------------------------------------- #


def test_inner_self_join_aliased(sengine):
    eng, duck = sengine
    sql = "SELECT a.v AS av, b.v AS bv FROM t a JOIN t b ON a.k = b.k ORDER BY a.k"
    assert_same(_ryu(eng, sql), _duck(duck, sql))


def test_inner_self_join_same_key_both_retained(sengine):
    """DuckDB keeps both ``a.k`` and ``b.k`` as two columns for ``ON a.k=b.k``;
    RyuDB must too (the join key no longer collapses to one column)."""
    eng, duck = sengine
    sql = "SELECT a.k AS ak, b.k AS bk FROM t a JOIN t b ON a.k = b.k ORDER BY a.k"
    assert_same(_ryu(eng, sql), _duck(duck, sql))


# --------------------------------------------------------------------------- #
# OUTER self-joins (pure-equi, no residual)
# --------------------------------------------------------------------------- #


def test_left_self_join(sengine):
    eng, duck = sengine
    sql = "SELECT a.k AS ak, b.v AS bv FROM t a LEFT JOIN t b ON a.k = b.k ORDER BY a.k"
    assert_same(_ryu(eng, sql), _duck(duck, sql))


def test_right_self_join(sengine):
    eng, duck = sengine
    sql = "SELECT a.k AS ak, b.v AS bv FROM t a RIGHT JOIN t b ON a.k = b.k ORDER BY a.k"
    assert_same(_ryu(eng, sql), _duck(duck, sql))


def test_full_self_join(sengine):
    eng, duck = sengine
    sql = "SELECT a.k AS ak, b.k AS bk FROM t a FULL JOIN t b ON a.k = b.k ORDER BY a.k, b.k"
    assert_same(_ryu(eng, sql), _duck(duck, sql))


# --------------------------------------------------------------------------- #
# Self-join + aggregate / GROUP BY / ORDER BY qualified
# --------------------------------------------------------------------------- #


def test_self_join_aggregate(sengine):
    eng, duck = sengine
    sql = (
        "SELECT a.k AS k, count(*) AS c, sum(b.v) AS s "
        "FROM t a JOIN t b ON a.k = b.k GROUP BY a.k ORDER BY a.k"
    )
    assert_same(_ryu(eng, sql), _duck(duck, sql))


def test_self_join_order_by_second_collision(sengine):
    """``ORDER BY b.v`` where both ``a.v`` and ``b.v`` are selected: the second
    colliding column is deduped to ``v_1`` in the output, so the sort must
    resolve ``b.v`` against the pre-projection frame (not the bare ``v`` = a.v)."""
    eng, duck = sengine
    sql = "SELECT a.v, b.v FROM t a CROSS JOIN t b WHERE a.k = b.k ORDER BY b.v"
    assert_same(_ryu(eng, sql), _duck(duck, sql))


def test_self_join_group_by_composite(sengine, tmp_path):
    """Composite equi-key self-join + GROUP BY a composite of qualified columns."""
    d = tmp_path / "tc"
    d.mkdir()
    pd.DataFrame(
        {"k": [1, 1, 2, 2, 3], "j": [10, 20, 10, 20, 10], "v": [5, 6, 7, 8, 9]}
    ).to_parquet(str(d / "0.parquet"))
    cat = Catalog(str(tmp_path))
    cat.register("tc", str(d))
    eng = Engine(cat)
    duck = duckdb.connect()
    duck.execute(f"CREATE VIEW tc AS SELECT * FROM read_parquet('{d}/0.parquet')")
    sql = (
        "SELECT a.k, a.j, sum(b.v) AS s "
        "FROM tc a JOIN tc b ON a.k = b.k AND a.j = b.j "
        "GROUP BY a.k, a.j ORDER BY a.k, a.j"
    )
    assert_same(eng.sql(sql), duck.execute(sql).fetchdf())


# --------------------------------------------------------------------------- #
# Cross-table same-named column
# --------------------------------------------------------------------------- #


def test_cross_table_same_named_column(xengine):
    eng, duck = xengine
    sql = "SELECT a.v AS av, b.v AS bv FROM t1 a JOIN t2 b ON a.k = b.k ORDER BY a.k"
    assert_same(_ryu(eng, sql), _duck(duck, sql))


def test_cross_table_same_named_key_both_retained(xengine):
    eng, duck = xengine
    sql = "SELECT a.k AS ak, b.k AS bk FROM t1 a JOIN t2 b ON a.k = b.k ORDER BY a.k"
    assert_same(_ryu(eng, sql), _duck(duck, sql))


def test_cross_table_qualified_where(xengine):
    """A qualified WHERE on the cross-table collision (``a.v > b.v``) resolves
    each side by alias."""
    eng, duck = xengine
    sql = (
        "SELECT a.v AS av, b.v AS bv FROM t1 a JOIN t2 b ON a.k = b.k "
        "WHERE a.v > b.v ORDER BY a.k"
    )
    assert_same(_ryu(eng, sql), _duck(duck, sql))


# --------------------------------------------------------------------------- #
# Output-name dedup (no explicit aliases)
# --------------------------------------------------------------------------- #


def test_select_two_colliding_cols_unaliased(sengine):
    """``SELECT a.v, b.v`` without aliases: DuckDB names them ``v``/``v_1``;
    ``as_sorted`` ignores names so the value sets must match."""
    eng, duck = sengine
    sql = "SELECT a.v, b.v FROM t a JOIN t b ON a.k = b.k ORDER BY a.v, b.v"
    assert_same(_ryu(eng, sql), _duck(duck, sql))


# --------------------------------------------------------------------------- #
# USING / NATURAL
# --------------------------------------------------------------------------- #


def test_using_no_nonkey_collision(tmp_path):
    """Plain ``USING(k)`` where the two sides share ONLY the key (disjoint
    non-key columns) is unchanged: the key coalesces into one column. The
    alias-rename path must NOT fire for USING (it would un-coalesce the key)."""
    d1 = tmp_path / "u1"
    d1.mkdir()
    pd.DataFrame({"k": [1, 2, 3], "x": [10, 20, 30]}).to_parquet(str(d1 / "0.parquet"))
    d2 = tmp_path / "u2"
    d2.mkdir()
    pd.DataFrame({"k": [1, 2, 4], "y": [100, 200, 400]}).to_parquet(str(d2 / "0.parquet"))
    cat = Catalog(str(tmp_path))
    cat.register("u1", str(d1))
    cat.register("u2", str(d2))
    eng = Engine(cat)
    duck = duckdb.connect()
    duck.execute(f"CREATE VIEW u1 AS SELECT * FROM read_parquet('{d1}/0.parquet')")
    duck.execute(f"CREATE VIEW u2 AS SELECT * FROM read_parquet('{d2}/0.parquet')")
    sql = "SELECT k, x, y FROM u1 JOIN u2 USING(k) ORDER BY k"
    assert_same(eng.sql(sql), duck.execute(sql).fetchdf())


def test_using_with_nonkey_collision_raises(sengine):
    """``USING(k)`` where the two sides also share non-key ``v`` -> deferred
    (the USING key coalesces but the non-key collision would suffix and break
    qualified refs)."""
    eng, _duck = sengine
    with pytest.raises(NotImplementedError):
        eng.sql("SELECT a.v, b.v FROM t a JOIN t b USING(k)")


# --------------------------------------------------------------------------- #
# Deferred / rejection cases
# --------------------------------------------------------------------------- #


def test_outer_self_join_with_residual_raises(sengine):
    """Outer join with a residual non-equi ON predicate + collision is deferred."""
    eng, _duck = sengine
    with pytest.raises(NotImplementedError):
        eng.sql(
            "SELECT a.v AS av, b.v AS bv FROM t a LEFT JOIN t b ON a.k = b.k AND a.v > b.v"
        )


def test_colliding_non_simple_side_raises(sengine):
    """``_side_alias`` returns the alias for Scan, Filter->Scan, and Derived
    (CTE/FROM-subquery) sides, and None for a Project/Join side (which has no
    single-table provenance -> a collision on such a side is deferred)."""
    from ryudb.exec.executor import Engine
    import ryudb.sql.plan as P

    eng, _duck = sengine
    scan_a = P.Scan("t", alias="a")
    assert Engine._side_alias(scan_a) == "a"
    assert Engine._side_alias(P.Filter(scan_a, P.Col("k"))) == "a"
    assert Engine._side_alias(P.Derived(scan_a, "c")) == "c"
    # A Project or Join side has no single-table provenance -> None (deferred).
    assert Engine._side_alias(P.Project(scan_a, [(P.Col("k"), "k")])) is None
    join = P.Join(scan_a, P.Scan("t", alias="b"), ["k"], ["k"], "inner")
    assert Engine._side_alias(join) is None


# --------------------------------------------------------------------------- #
# Regression: a 2-table join with NO collision still matches DuckDB (the rename
# path must not fire on no-collision joins).
# --------------------------------------------------------------------------- #


def test_no_collision_join_regression(sengine, tmp_path):
    """``t1 JOIN t2`` where column names are disjoint: rename path is inert."""
    d1 = tmp_path / "r1"
    d1.mkdir()
    pd.DataFrame({"k": [1, 2, 3], "a": [10, 20, 30]}).to_parquet(str(d1 / "0.parquet"))
    d2 = tmp_path / "r2"
    d2.mkdir()
    pd.DataFrame({"k": [1, 2, 4], "b": [100, 200, 400]}).to_parquet(str(d2 / "0.parquet"))
    cat = Catalog(str(tmp_path))
    cat.register("r1", str(d1))
    cat.register("r2", str(d2))
    eng = Engine(cat)
    duck = duckdb.connect()
    duck.execute(f"CREATE VIEW r1 AS SELECT * FROM read_parquet('{d1}/0.parquet')")
    duck.execute(f"CREATE VIEW r2 AS SELECT * FROM read_parquet('{d2}/0.parquet')")
    sql = "SELECT r1.k AS k, a, b FROM r1 JOIN r2 ON r1.k = r2.k ORDER BY r1.k"
    assert_same(eng.sql(sql), duck.execute(sql).fetchdf())


# --------------------------------------------------------------------------- #
# Parse-shape: Col.table and Scan.alias are populated; ORDER BY keeps the
# qualifier.
# --------------------------------------------------------------------------- #


def test_parse_shape_col_table_and_scan_alias(sengine):
    from ryudb.sql.parse import parse
    import ryudb.sql.plan as P

    eng, _duck = sengine
    plan = parse(
        "SELECT a.v, b.v FROM t a CROSS JOIN t b WHERE a.k = b.k",
        eng.catalog.schema_dict(),
    )
    scans = [n for n in P.walk(plan) if isinstance(n, P.Scan)]
    assert sorted(s.alias for s in scans) == ["a", "b"]
    # The projection items carry the qualifier.
    proj = next(n for n in P.walk(plan) if isinstance(n, P.Project))
    tables = sorted(e.table for e, _ in proj.items if isinstance(e, P.Col))
    assert tables == ["a", "b"]


def test_parse_shape_order_by_keeps_qualifier(sengine):
    from ryudb.sql.parse import parse
    import ryudb.sql.plan as P

    eng, _duck = sengine
    plan = parse(
        "SELECT a.v AS av, b.v AS bv FROM t a JOIN t b ON a.k = b.k ORDER BY b.v",
        eng.catalog.schema_dict(),
    )
    sort = next(n for n in P.walk(plan) if isinstance(n, P.Sort))
    (key, _asc), = sort.keys
    assert isinstance(key, P.Col)
    assert key.table == "b"