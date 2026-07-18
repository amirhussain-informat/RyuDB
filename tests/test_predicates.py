"""SQL surface, Phase A: predicates & expressions (IS NULL, IN, BETWEEN, LIKE,
CASE, COALESCE, CAST) -- RyuDB vs DuckDB oracle.

These are table-stakes SQL that the parser/executor used to reject with
NotImplementedError. Each is lowered sqlglot -> RyuDB Expr (plan.py) -> cuDF op
(ops.py); the fused CUDA kernels are untouched (any query using one of these in
a WHERE/agg simply defers to the cuDF path -- see test_predicate_defers_fused).

The fixture deliberately includes NULLs so IS NULL / COALESCE / CASE-no-ELSE /
NULL-in-IN three-valued logic are exercised. Comparison is via conftest.as_sorted
(NULL -> None, floats rounded) because _match casts to float and cannot hold
strings/NULLs.
"""

from __future__ import annotations

import cudf
import pytest

from ryudb import Catalog, Engine
from ryudb.exec import fused
from ryudb.sql.optimize import optimize
from ryudb.sql.parse import parse
from ryudb.sql.plan import Aggregate, walk

from .conftest import as_sorted

# t: k has a NULL, s has a NULL, v has a NULL -> every NULL path is exercised.
_T = [
    (1, "abc", 5.0),
    (2, "abd", 15.0),
    (3, "xbc", 25.0),
    (4, "ab", 100.0),
    (5, None, None),
    (None, "zzz", 3.0),
]
# u: a dim to join against t for the fused-defer case.
_U = [
    (1, "A"),
    (2, "B"),
    (3, "A"),
]


@pytest.fixture
def pred_dir(tmp_path):
    d = tmp_path
    for name, cols, rows in [("t", ["k", "s", "v"], _T), ("u", ["k", "grp"], _U)]:
        (d / name).mkdir()
        cudf.DataFrame({c: [row[i] for row in rows] for i, c in enumerate(cols)}) \
            .to_pandas().to_parquet(d / name / "0.parquet")
    return d


@pytest.fixture
def pengine(pred_dir) -> Engine:
    cat = Catalog(str(pred_dir))
    for name in ("t", "u"):
        cat.register(name, str(pred_dir / name))
    return Engine(cat)


@pytest.fixture
def pduck(pred_dir):
    import duckdb
    con = duckdb.connect()
    for name in ("t", "u"):
        con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{pred_dir}/{name}/*.parquet')")
    return con


def _ryu(engine: Engine, sql: str):
    return as_sorted(engine.sql(sql))


def _duck(con, sql: str):
    return as_sorted(con.execute(sql).fetchdf())


# --------------------------------------------------------------------------- #
# IS [NOT] NULL
# --------------------------------------------------------------------------- #


def test_is_null(pengine, pduck):
    sql = "SELECT k FROM t WHERE k IS NULL"
    assert _ryu(pengine, sql) == _duck(pduck, sql)


def test_is_not_null(pengine, pduck):
    sql = "SELECT k FROM t WHERE k IS NOT NULL ORDER BY k"
    assert _ryu(pengine, sql) == _duck(pduck, sql)
    assert all(r[0] is not None for r in _ryu(pengine, sql))


def test_is_null_as_expr(pengine, pduck):
    # IS NULL used as a SELECT expression (boolean column), not just a filter.
    sql = "SELECT k, k IS NULL AS is_null FROM t ORDER BY k"
    assert _ryu(pengine, sql) == _duck(pduck, sql)


# --------------------------------------------------------------------------- #
# [NOT] IN (list)
# --------------------------------------------------------------------------- #


def test_in_list(pengine, pduck):
    assert _ryu(pengine, "SELECT k FROM t WHERE k IN (1, 3, 5)") == _duck(pduck, "SELECT k FROM t WHERE k IN (1, 3, 5)")


def test_not_in_list(pengine, pduck):
    sql = "SELECT k FROM t WHERE k NOT IN (1, 2) ORDER BY k"
    assert _ryu(pengine, sql) == _duck(pduck, sql)
    # NULL k is NOT IN (1,2) -> NULL -> dropped (SQL three-valued logic).
    assert all(r[0] is not None for r in _ryu(pengine, sql))


def test_in_string_list(pengine, pduck):
    sql = "SELECT s FROM t WHERE s IN ('abc', 'xbc') ORDER BY s"
    assert _ryu(pengine, sql) == _duck(pduck, sql)


# --------------------------------------------------------------------------- #
# [NOT] BETWEEN
# --------------------------------------------------------------------------- #


def test_between(pengine, pduck):
    sql = "SELECT k FROM t WHERE v BETWEEN 10 AND 30 ORDER BY k"
    assert _ryu(pengine, sql) == _duck(pduck, sql)


def test_not_between(pengine, pduck):
    sql = "SELECT k FROM t WHERE v NOT BETWEEN 10 AND 30 ORDER BY k"
    assert _ryu(pengine, sql) == _duck(pduck, sql)


# --------------------------------------------------------------------------- #
# [NOT] LIKE / ILIKE
# --------------------------------------------------------------------------- #


def test_like_prefix(pengine, pduck):
    sql = "SELECT s FROM t WHERE s LIKE 'a%' ORDER BY s"
    assert _ryu(pengine, sql) == _duck(pduck, sql)


def test_like_underscore(pengine, pduck):
    sql = "SELECT s FROM t WHERE s LIKE '_b%' ORDER BY s"
    assert _ryu(pengine, sql) == _duck(pduck, sql)


def test_not_like(pengine, pduck):
    sql = "SELECT s FROM t WHERE s NOT LIKE 'a%' ORDER BY s"
    assert _ryu(pengine, sql) == _duck(pduck, sql)
    # NULL s NOT LIKE 'a%' -> NULL -> dropped.
    assert all(r[0] is not None for r in _ryu(pengine, sql))


def test_ilike(pengine, pduck):
    sql = "SELECT s FROM t WHERE s ILIKE 'A%' ORDER BY s"
    assert _ryu(pengine, sql) == _duck(pduck, sql)


def test_like_literal_meta(pengine, pduck):
    # A pattern containing a regex metachar ('.') must be escaped -- it matches
    # the literal '.', not 'any char'. (No '.' in the data -> empty result both.)
    sql = "SELECT s FROM t WHERE s LIKE 'a.c'"
    assert _ryu(pengine, sql) == _duck(pduck, sql)


# --------------------------------------------------------------------------- #
# CASE
# --------------------------------------------------------------------------- #


def test_case_searched(pengine, pduck):
    sql = ("SELECT CASE WHEN v > 20 THEN 'big' WHEN v > 10 THEN 'mid' "
           "ELSE 'small' END AS c, k FROM t ORDER BY k")
    assert _ryu(pengine, sql) == _duck(pduck, sql)


def test_case_simple(pengine, pduck):
    sql = "SELECT CASE k WHEN 1 THEN 'one' WHEN 2 THEN 'two' ELSE 'other' END AS c, k FROM t ORDER BY k"
    assert _ryu(pengine, sql) == _duck(pduck, sql)


def test_case_no_else(pengine, pduck):
    # No ELSE -> NULL for unmatched rows (incl. the NULL-operand row).
    sql = "SELECT CASE WHEN v > 50 THEN 'big' END AS c, k FROM t ORDER BY k"
    assert _ryu(pengine, sql) == _duck(pduck, sql)
    assert any(r[0] is None for r in _ryu(pengine, sql))


def test_case_in_where(pengine, pduck):
    sql = "SELECT k FROM t WHERE CASE WHEN v > 20 THEN 1 ELSE 0 END = 1 ORDER BY k"
    assert _ryu(pengine, sql) == _duck(pduck, sql)


# --------------------------------------------------------------------------- #
# COALESCE
# --------------------------------------------------------------------------- #


def test_coalesce_two(pengine, pduck):
    sql = "SELECT COALESCE(s, '<<NULL>>') AS c, k FROM t ORDER BY k"
    assert _ryu(pengine, sql) == _duck(pduck, sql)


def test_coalesce_multi(pengine, pduck):
    # v NULL -> fall to k -> k NULL -> fall to 0.
    sql = "SELECT COALESCE(v, k, 0) AS c, k FROM t ORDER BY k"
    assert _ryu(pengine, sql) == _duck(pduck, sql)


# --------------------------------------------------------------------------- #
# CAST
# --------------------------------------------------------------------------- #


def test_cast_float(pengine, pduck):
    sql = "SELECT CAST(k AS FLOAT) AS kf, k FROM t ORDER BY k"
    assert _ryu(pengine, sql) == _duck(pduck, sql)


def test_cast_int_rounds(pengine, pduck):
    # DuckDB CAST(double AS int) rounds to nearest (5.x -> 6); the impl must
    # round, not truncate. v values are integers here, so this also covers NULL->NULL.
    sql = "SELECT CAST(v AS INT) AS vi, k FROM t ORDER BY k"
    assert _ryu(pengine, sql) == _duck(pduck, sql)


def test_cast_int_truncation_matches_duckdb(pengine, pduck):
    # v is integral; adding 0.9 yields fractional values (5.9, 15.9, ...) so
    # rounding (5.9 -> 6) vs truncation (5.9 -> 5) is observable, and the NULL v
    # row -> NULL. This goes through the non-literal Cast path (ops._cast int).
    sql = "SELECT CAST(v + 0.9 AS INT) AS vi, k FROM t ORDER BY k"
    assert _ryu(pengine, sql) == _duck(pduck, sql)


def test_cast_str(pengine, pduck):
    sql = "SELECT CAST(k AS VARCHAR) AS ks, k FROM t ORDER BY k"
    assert _ryu(pengine, sql) == _duck(pduck, sql)


def test_cast_bool(pengine, pduck):
    sql = "SELECT CAST(k AS BOOLEAN) AS kb, k FROM t ORDER BY k"
    assert _ryu(pengine, sql) == _duck(pduck, sql)


# --------------------------------------------------------------------------- #
# Combos + fused-defer
# --------------------------------------------------------------------------- #


def test_predicate_combo(pengine, pduck):
    sql = ("SELECT s FROM t WHERE s LIKE 'a%' AND v BETWEEN 1 AND 100 "
           "AND k IN (1, 2, 3) ORDER BY s")
    assert _ryu(pengine, sql) == _duck(pduck, sql)


def test_predicate_defers_fused_but_correct(pengine, pduck):
    """A LIKE in the WHERE of an aggregate-over-join: the fused kernel cannot
    tokenize a LIKE predicate, so fused_join_aggregate defers (returns None) and
    the cuDF path runs -- and still matches DuckDB. Correctness never depends on
    the extension. (The Filter under the Aggregate also makes the fused gate
    defer even without the LIKE, but the LIKE is the Phase-A-relevant reason.)"""
    sql = ("SELECT grp, sum(v) AS tot FROM u JOIN t ON u.k = t.k "
           "WHERE t.s LIKE 'a%' GROUP BY grp ORDER BY grp")
    if fused._kernels.is_available:
        agg = _agg_node(sql, pengine)
        assert fused.fused_join_aggregate(agg, pengine) is None, (
            "a LIKE predicate under an aggregate-over-join must defer to cuDF")
    assert _ryu(pengine, sql) == _duck(pduck, sql)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _agg_node(sql: str, engine: Engine):
    plan = optimize(parse(sql, engine.catalog.schema_dict()),
                    engine.catalog.schema_dict(), engine.catalog.stats_dict())
    return next(n for n in walk(plan) if isinstance(n, Aggregate))