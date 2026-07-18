"""Phase 1 join-type tests: LEFT/RIGHT/FULL OUTER, CROSS, NATURAL (+ USING).

Three layers, mirroring the existing test layout:

  * parse     -- ``parse()``/``optimize()`` shape assertions (how derivation,
                 on_predicate capture, pushdown how-guard, side-swap rewrite).
  * executor  -- RyuDB (GPU, cuDF merge) vs DuckDB on the same SQL, including
                 the ON-vs-WHERE regression for outer joins.
  * fallback  -- outer-join aggregates defer to cuDF (the fused gate is inner-
                 only), so they must match DuckDB without any C++ kernel.

Fixture tables (intentionally unmatched rows so null-padding is exercised):

  * ``r(rid, rv)`` / ``s(sid, sv)``  -- DIFFERENT key names. cuDF keeps both key
    columns on an outer merge, so ``SELECT r.rid`` is NULL for right-only rows --
    matching DuckDB's standard-SQL ``ON r.rid = s.sid`` (which keeps both keys;
    only USING/NATURAL coalesce). Used for outer / cross / on-residual / chain.
  * ``p(id, pv)`` / ``q(id, qv)``    -- SAME key name. Used for NATURAL / USING,
    where both RyuDB and DuckDB coalesce the single ``id`` column.
  * ``t(tcol)``                      -- no column in common with r; a NATURAL
    join degrades to a cross product (SQL standard; DuckDB refuses NATURAL with
    no common columns, so that case is compared against an explicit CROSS JOIN).

The shared ``orders``/``lineitem``/``nation`` fixture is all-matching, so it
cannot test null-padding; everything needing unmatched rows lives here.
"""

from __future__ import annotations

import cudf
import pytest

from ryudb import Catalog, Engine
from ryudb.sql.optimize import optimize
from ryudb.sql.parse import parse
from ryudb.sql.plan import Filter, Join, Scan, walk

from .conftest import as_sorted

# r: rid 1,2,3,5 ; s: sid 2,3,4,6 -> LEFT keeps 1,5 (null); RIGHT keeps 4,6.
_R = [(1, 100), (2, 200), (3, 300), (5, 500)]
_S = [(2, 22), (3, 33), (4, 44), (6, 66)]
# Same-named-key pair for NATURAL / USING (both engines coalesce id).
_P = [(1, "a"), (2, "b"), (3, "c"), (5, "e")]
_Q = [(2, "x"), (3, "y"), (4, "z"), (6, "w")]
# No common columns with r -> NATURAL JOIN degrades to CROSS.
_T = [(10,), (20,)]


@pytest.fixture
def join_dir(tmp_path):
    d = tmp_path
    for name, cols, rows in [
        ("r", ["rid", "rv"], _R),
        ("s", ["sid", "sv"], _S),
        ("p", ["id", "pv"], _P),
        ("q", ["id", "qv"], _Q),
        ("t", ["tcol"], _T),
    ]:
        (d / name).mkdir()
        cudf.DataFrame({c: [row[i] for row in rows] for i, c in enumerate(cols)}) \
            .to_pandas().to_parquet(d / name / "0.parquet")
    return d


@pytest.fixture
def jengine(join_dir) -> Engine:
    cat = Catalog(str(join_dir))
    for name in ("r", "s", "p", "q", "t"):
        cat.register(name, str(join_dir / name))
    return Engine(cat)


@pytest.fixture
def jduck(join_dir):
    import duckdb
    con = duckdb.connect()
    for name in ("r", "s", "p", "q", "t"):
        con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{join_dir}/{name}/*.parquet')")
    return con


# Schema used only by the parse/optimize shape tests (no engine / parquet).
_SCHEMA = {"r": ["id", "rv"], "s": ["id", "sv"], "t": ["tcol"]}


def _first_join(plan) -> Join:
    return next(n for n in walk(plan) if isinstance(n, Join))


# --------------------------------------------------------------------------- #
# Parse: how derivation + on_predicate capture
# --------------------------------------------------------------------------- #


def test_parse_how_inner():
    j = _first_join(parse("SELECT * FROM r JOIN s ON r.id = s.id", _SCHEMA))
    assert j.how == "inner"
    assert j.on_predicate is None


def test_parse_how_outer_sides():
    assert _first_join(parse("SELECT * FROM r LEFT JOIN s ON r.id = s.id", _SCHEMA)).how == "left"
    assert _first_join(parse("SELECT * FROM r RIGHT JOIN s ON r.id = s.id", _SCHEMA)).how == "right"
    assert _first_join(parse("SELECT * FROM r FULL JOIN s ON r.id = s.id", _SCHEMA)).how == "full"
    assert _first_join(parse("SELECT * FROM r LEFT OUTER JOIN s ON r.id = s.id", _SCHEMA)).how == "left"
    assert _first_join(parse("SELECT * FROM r FULL OUTER JOIN s ON r.id = s.id", _SCHEMA)).how == "full"


def test_parse_cross():
    j = _first_join(parse("SELECT * FROM r CROSS JOIN s", _SCHEMA))
    assert j.how == "cross"
    assert j.on_left == [] and j.on_right == []
    assert j.on_predicate is None


def test_parse_on_residual_lands_on_join_not_filter():
    # The central outer-join fix: a non-equi ON residual is captured on the Join
    # (applied with outer semantics), NOT wrapped in a top Filter (which would
    # wrongly drop null-padded rows).
    plan = parse("SELECT * FROM r LEFT JOIN s ON r.id = s.id AND s.sv > 30", _SCHEMA)
    j = _first_join(plan)
    assert j.how == "left"
    assert j.on_predicate is not None
    assert not [
        n for n in walk(plan)
        if isinstance(n, Filter) and isinstance(n.input, Join)
    ], "ON residual must not become a Filter above the Join"


def test_parse_natural_common_columns():
    j = _first_join(parse("SELECT * FROM r NATURAL JOIN s", _SCHEMA))
    assert j.how == "inner"
    assert j.on_left == ["id"] and j.on_right == ["id"]
    assert j.on_predicate is None


def test_parse_natural_with_side():
    assert _first_join(parse("SELECT * FROM r NATURAL LEFT JOIN s", _SCHEMA)).how == "left"
    assert _first_join(parse("SELECT * FROM r NATURAL RIGHT JOIN s", _SCHEMA)).how == "right"
    assert _first_join(parse("SELECT * FROM r NATURAL FULL JOIN s", _SCHEMA)).how == "full"


def test_parse_natural_no_common_is_cross():
    j = _first_join(parse("SELECT * FROM r NATURAL JOIN t", _SCHEMA))
    assert j.how == "cross"
    assert j.on_left == [] and j.on_right == []


def test_parse_natural_requires_schema():
    with pytest.raises(NotImplementedError, match="schema"):
        parse("SELECT * FROM r NATURAL JOIN s")  # no schema argument


def test_parse_using_same_named_keys():
    j = _first_join(parse("SELECT * FROM r JOIN s USING (id)", _SCHEMA))
    assert j.how == "inner"
    assert j.on_left == ["id"] and j.on_right == ["id"]
    assert j.on_predicate is None


# --------------------------------------------------------------------------- #
# Optimize: outer-join-aware pushdown + side-swap how rewrite
# --------------------------------------------------------------------------- #


def _has_filter_above_scan(plan, table) -> bool:
    return any(
        isinstance(n, Filter) and isinstance(n.input, Scan) and n.input.table == table
        for n in walk(plan)
    )


def test_pushdown_left_keeps_null_supplying_side():
    # A WHERE on the null-supplying side (s) of a LEFT join must NOT push below
    # it (it would drop the null-padded rows the LEFT join must keep).
    p = parse("SELECT * FROM r LEFT JOIN s ON r.id = s.id WHERE s.sv > 30", _SCHEMA)
    opt = optimize(p, _SCHEMA, {})
    assert not _has_filter_above_scan(opt, "s")


def test_pushdown_left_pushes_preserved_side():
    p = parse("SELECT * FROM r LEFT JOIN s ON r.id = s.id WHERE r.rv > 150", _SCHEMA)
    opt = optimize(p, _SCHEMA, {})
    assert _has_filter_above_scan(opt, "r")


def test_pushdown_right_keeps_null_supplying_side():
    p = parse("SELECT * FROM r RIGHT JOIN s ON r.id = s.id WHERE r.rv > 150", _SCHEMA)
    opt = optimize(p, _SCHEMA, {})
    assert not _has_filter_above_scan(opt, "r")


def test_pushdown_right_pushes_preserved_side():
    p = parse("SELECT * FROM r RIGHT JOIN s ON r.id = s.id WHERE s.sv > 30", _SCHEMA)
    opt = optimize(p, _SCHEMA, {})
    assert _has_filter_above_scan(opt, "s")


def test_pushdown_full_pushes_neither():
    p = parse("SELECT * FROM r FULL JOIN s ON r.id = s.id WHERE r.rv > 150 AND s.sv > 30", _SCHEMA)
    opt = optimize(p, _SCHEMA, {})
    assert not _has_filter_above_scan(opt, "r")
    assert not _has_filter_above_scan(opt, "s")


def test_pushdown_inner_pushes_both():
    p = parse("SELECT * FROM r JOIN s ON r.id = s.id WHERE r.rv > 150 AND s.sv > 30", _SCHEMA)
    opt = optimize(p, _SCHEMA, {})
    assert _has_filter_above_scan(opt, "r")
    assert _has_filter_above_scan(opt, "s")


def test_side_swap_rewrites_left_to_right():
    # Declare r huge so the optimizer flips sides; a LEFT join becomes RIGHT.
    p = parse("SELECT * FROM r LEFT JOIN s ON r.id = s.id", _SCHEMA)
    opt = optimize(p, _SCHEMA, {"r": 10_000, "s": 10})
    j = _first_join(opt)
    assert j.how == "right"
    assert j.on_left == ["id"] and j.on_right == ["id"]


def test_side_swap_inner_stays_inner():
    p = parse("SELECT * FROM r JOIN s ON r.id = s.id", _SCHEMA)
    opt = optimize(p, _SCHEMA, {"r": 10_000, "s": 10})
    assert _first_join(opt).how == "inner"


def test_on_predicate_survives_optimizer():
    p = parse("SELECT r.id FROM r LEFT JOIN s ON r.id = s.id AND s.sv > 30", _SCHEMA)
    opt = optimize(p, _SCHEMA, {"r": 10_000, "s": 10})
    assert _first_join(opt).on_predicate is not None


# --------------------------------------------------------------------------- #
# Executor: RyuDB vs DuckDB
# --------------------------------------------------------------------------- #


def _ryu(engine: Engine, sql: str):
    return as_sorted(engine.sql(sql))


def _duck(con, sql: str):
    return as_sorted(con.execute(sql).fetchdf())


def test_left_join_vs_duckdb(jengine, jduck):
    sql = "SELECT r.rid, r.rv, s.sv FROM r LEFT JOIN s ON r.rid = s.sid ORDER BY r.rid"
    assert _ryu(jengine, sql) == _duck(jduck, sql)


def test_right_join_vs_duckdb(jengine, jduck):
    sql = "SELECT r.rid, r.rv, s.sv FROM r RIGHT JOIN s ON r.rid = s.sid ORDER BY s.sv"
    assert _ryu(jengine, sql) == _duck(jduck, sql)


def test_full_join_vs_duckdb(jengine, jduck):
    sql = "SELECT r.rid, r.rv, s.sv FROM r FULL JOIN s ON r.rid = s.sid ORDER BY r.rid"
    assert _ryu(jengine, sql) == _duck(jduck, sql)


def test_cross_join_vs_duckdb(jengine, jduck):
    sql = "SELECT r.rv, s.sv FROM r CROSS JOIN s ORDER BY r.rv, s.sv"
    assert _ryu(jengine, sql) == _duck(jduck, sql)
    n = int(jengine.sql("SELECT count(*) AS n FROM r CROSS JOIN s").to_pandas()["n"].iloc[0])
    assert n == len(_R) * len(_S)


def test_left_join_on_residual_vs_where(jengine, jduck):
    """The regression: an ON residual filters only matched rows (unmatched left
    rows survive null-padded); the same predicate in WHERE drops null-padded
    rows. The two must differ, and each must match DuckDB."""
    on_sql = ("SELECT r.rid, r.rv, s.sv FROM r LEFT JOIN s ON r.rid = s.sid AND s.sv > 30 "
              "ORDER BY r.rid")
    where_sql = ("SELECT r.rid, r.rv, s.sv FROM r LEFT JOIN s ON r.rid = s.sid WHERE s.sv > 30 "
                 "ORDER BY r.rid")
    on_res, where_res = _ryu(jengine, on_sql), _ryu(jengine, where_sql)
    assert on_res == _duck(jduck, on_sql)
    assert where_res == _duck(jduck, where_sql)
    assert on_res != where_res, "ON and WHERE must differ for a LEFT join"
    # The ON result keeps the unmatched left rows (rid 1 and 5) with sv NULL.
    assert {1, 5} <= {row[0] for row in on_res}


def test_full_join_on_residual_vs_duckdb(jengine, jduck):
    sql = ("SELECT r.rid, r.rv, s.sv FROM r FULL JOIN s ON r.rid = s.sid AND s.sv > 30 "
           "ORDER BY r.rid")
    assert _ryu(jengine, sql) == _duck(jduck, sql)


def test_right_join_on_residual_vs_duckdb(jengine, jduck):
    sql = ("SELECT r.rid, r.rv, s.sv FROM r RIGHT JOIN s ON r.rid = s.sid AND r.rv > 150 "
           "ORDER BY s.sv")
    assert _ryu(jengine, sql) == _duck(jduck, sql)


def test_using_coalesces_key_vs_duckdb(jengine, jduck):
    # USING(id) coalesces the two id columns into one; bare Col(id) must resolve
    # and the single-column output must match DuckDB (which coalesces too).
    sql = "SELECT id, pv, qv FROM p JOIN q USING (id) ORDER BY id"
    ryu = jengine.sql(sql).to_pandas()
    assert "id" in ryu.columns
    assert "id_x" not in ryu.columns and "id_y" not in ryu.columns
    assert _ryu(jengine, sql) == _duck(jduck, sql)


def test_natural_join_vs_duckdb(jengine, jduck):
    sql = "SELECT id, pv, qv FROM p NATURAL JOIN q ORDER BY id"
    assert _ryu(jengine, sql) == _duck(jduck, sql)


def test_natural_left_vs_duckdb(jengine, jduck):
    sql = "SELECT id, pv, qv FROM p NATURAL LEFT JOIN q ORDER BY id"
    assert _ryu(jengine, sql) == _duck(jduck, sql)


def test_natural_full_vs_duckdb(jengine, jduck):
    sql = "SELECT id, pv, qv FROM p NATURAL FULL JOIN q ORDER BY id"
    assert _ryu(jengine, sql) == _duck(jduck, sql)


def test_natural_no_common_is_cross(jengine, jduck):
    # DuckDB refuses NATURAL JOIN with no common columns; RyuDB follows the SQL
    # standard and degrades to a cross product -- compare against DuckDB's
    # explicit CROSS JOIN of the same tables.
    ryu = _ryu(jengine, "SELECT r.rid, t.tcol FROM r NATURAL JOIN t ORDER BY r.rid, t.tcol")
    duck = _duck(jduck, "SELECT r.rid, t.tcol FROM r CROSS JOIN t ORDER BY r.rid, t.tcol")
    assert ryu == duck
    n = int(jengine.sql("SELECT count(*) AS n FROM r NATURAL JOIN t").to_pandas()["n"].iloc[0])
    assert n == len(_R) * len(_T)


def test_left_join_aggregate_vs_duckdb(jengine, jduck):
    # Aggregate over a LEFT join: unmatched left rows count toward COUNT(*) but
    # contribute NULL to SUM(s.sv). Exercises the cuDF path -- the fused gate is
    # inner-only, so this must NOT use the C++ kernel.
    sql = ("SELECT r.rid, count(*) AS c, sum(s.sv) AS tot "
           "FROM r LEFT JOIN s ON r.rid = s.sid GROUP BY r.rid ORDER BY r.rid")
    assert _ryu(jengine, sql) == _duck(jduck, sql)


def test_3way_left_chain_vs_duckdb(jengine, jduck):
    sql = ("SELECT r.rid, s.sv, t.tcol FROM r LEFT JOIN s ON r.rid = s.sid "
           "CROSS JOIN t ORDER BY r.rid, s.sv, t.tcol")
    assert _ryu(jengine, sql) == _duck(jduck, sql)


# --------------------------------------------------------------------------- #
# Fallback: outer joins never use the fused kernel
# --------------------------------------------------------------------------- #


def test_fused_gate_defers_outer_join(jengine, jduck, monkeypatch):
    """A LEFT-join aggregate must defer to cuDF (the fused gate is inner-only).
    Stubbing the fused entrypoint to None (what the real gate returns for outer)
    must leave RyuDB's output identical -- proving the kernel never runs for
    outer joins."""
    from ryudb.exec import fused

    monkeypatch.setattr(fused, "fused_join_aggregate", lambda *a, **k: None)
    sql = ("SELECT r.rid, count(*) AS c FROM r LEFT JOIN s ON r.rid = s.sid "
           "GROUP BY r.rid ORDER BY r.rid")
    assert _ryu(jengine, sql) == _duck(jduck, sql)