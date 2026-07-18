"""SQL surface, Phase B: string & scalar functions -- RyuDB vs DuckDB oracle.

UPPER/LOWER/LENGTH/SUBSTR/TRIM/CONCAT/||/REPLACE/POSITION/LEFT/RIGHT/INITCAP/
REVERSE and ABS/ROUND/CEIL/FLOOR/MOD. All previously raised NotImplementedError
in parse._expr (no exp.Func dispatch). Each maps to a generic ``Func(tag, args)``
Expr (plan.py) lowered tag-by-tag to a cuDF op (ops.py); the fused CUDA kernels
are untouched -- a Func in a WHERE/agg-arg makes the fused gate defer to cuDF
(see test_func_defers_fused_but_correct).

The fixture includes NULLs so NULL propagation (str accessors / ||) and the
NULL-ignoring CONCAT are exercised. Comparison is via conftest.as_sorted, which
sorts both sides and normalizes NULL (NaN/NA->None) and int/float (1==1.0), so
dtype-only differences (cuDF LENGTH returns int, POSITION returns int) compare
equal. INITCAP has no DuckDB oracle (this build lacks it) -> asserted against a
known expected value instead.
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

# t: NULLs in k, s, v so every NULL path is exercised. s has mixed case, spaces
# (for TRIM), an 'x' (for TRIM 'x' FROM s) and an 'a' (for REPLACE).
_T = [
    (1, "abc", 2.5),
    (2, "AbCd", 0.125),
    (3, "  x  ", 3.5),
    (4, "xax", -2.5),
    (5, None, None),
    (None, "zzz", 1.0),
]
_U = [
    (1, "A"),
    (2, "B"),
    (3, "A"),
]


@pytest.fixture
def fdir(tmp_path):
    d = tmp_path
    for name, cols, rows in [("t", ["k", "s", "v"], _T), ("u", ["k", "grp"], _U)]:
        (d / name).mkdir()
        cudf.DataFrame({c: [row[i] for row in rows] for i, c in enumerate(cols)}) \
            .to_pandas().to_parquet(d / name / "0.parquet")
    return d


@pytest.fixture
def fengine(fdir) -> Engine:
    cat = Catalog(str(fdir))
    for name in ("t", "u"):
        cat.register(name, str(fdir / name))
    return Engine(cat)


@pytest.fixture
def fduck(fdir):
    import duckdb
    con = duckdb.connect()
    for name in ("t", "u"):
        con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{fdir}/{name}/*.parquet')")
    return con


def _ryu(engine: Engine, sql: str):
    return as_sorted(engine.sql(sql))


def _duck(con, sql: str):
    return as_sorted(con.execute(sql).fetchdf())


# --------------------------------------------------------------------------- #
# String functions
# --------------------------------------------------------------------------- #


def test_upper(fengine, fduck):
    assert _ryu(fengine, "SELECT UPPER(s) AS x FROM t") == _duck(fduck, "SELECT UPPER(s) AS x FROM t")


def test_lower(fengine, fduck):
    assert _ryu(fengine, "SELECT LOWER(s) AS x FROM t") == _duck(fduck, "SELECT LOWER(s) AS x FROM t")


def test_length(fengine, fduck):
    # LENGTH(NULL) -> NULL on both engines.
    assert _ryu(fengine, "SELECT LENGTH(s) AS x FROM t") == _duck(fduck, "SELECT LENGTH(s) AS x FROM t")


def test_substr_3arg(fengine, fduck):
    assert _ryu(fengine, "SELECT SUBSTR(s, 2, 2) AS x FROM t") == \
        _duck(fduck, "SELECT SUBSTR(s, 2, 2) AS x FROM t")


def test_substr_2arg(fengine, fduck):
    assert _ryu(fengine, "SELECT SUBSTR(s, 2) AS x FROM t") == _duck(fduck, "SELECT SUBSTR(s, 2) AS x FROM t")


def test_trim(fengine, fduck):
    assert _ryu(fengine, "SELECT TRIM(s) AS x FROM t") == _duck(fduck, "SELECT TRIM(s) AS x FROM t")


def test_ltrim(fengine, fduck):
    assert _ryu(fengine, "SELECT LTRIM(s) AS x FROM t") == _duck(fduck, "SELECT LTRIM(s) AS x FROM t")


def test_rtrim(fengine, fduck):
    assert _ryu(fengine, "SELECT RTRIM(s) AS x FROM t") == _duck(fduck, "SELECT RTRIM(s) AS x FROM t")


def test_trim_chars(fengine, fduck):
    # TRIM('x' FROM s) strips the 'x' char, not whitespace.
    assert _ryu(fengine, "SELECT TRIM('x' FROM s) AS x FROM t") == \
        _duck(fduck, "SELECT TRIM('x' FROM s) AS x FROM t")


def test_trim_leading_chars(fengine, fduck):
    assert _ryu(fengine, "SELECT TRIM(LEADING 'x' FROM s) AS x FROM t") == \
        _duck(fduck, "SELECT TRIM(LEADING 'x' FROM s) AS x FROM t")


def test_concat(fengine, fduck):
    # CONCAT mixes a string and an int (cast to str).
    assert _ryu(fengine, "SELECT CONCAT(s, '_', k) AS x FROM t") == \
        _duck(fduck, "SELECT CONCAT(s, '_', k) AS x FROM t")


def test_concat_null(fengine, fduck):
    # CONCAT ignores NULLs (the NULL s row -> just '_'||k), unlike ||.
    assert _ryu(fengine, "SELECT CONCAT(s, k) AS x FROM t") == _duck(fduck, "SELECT CONCAT(s, k) AS x FROM t")


def test_pipe(fengine, fduck):
    # || propagates NULL (the NULL s row -> NULL).
    assert _ryu(fengine, "SELECT s || '_' AS x FROM t") == _duck(fduck, "SELECT s || '_' AS x FROM t")


def test_replace(fengine, fduck):
    assert _ryu(fengine, "SELECT REPLACE(s, 'a', 'Z') AS x FROM t") == \
        _duck(fduck, "SELECT REPLACE(s, 'a', 'Z') AS x FROM t")


def test_position(fengine, fduck):
    # 1-based; not found -> 0; NULL -> NULL.
    assert _ryu(fengine, "SELECT POSITION('a' IN s) AS x FROM t") == \
        _duck(fduck, "SELECT POSITION('a' IN s) AS x FROM t")


def test_strpos_func(fengine, fduck):
    # STRPOS is the same node (exp.StrPosition) as POSITION.
    assert _ryu(fengine, "SELECT STRPOS(s, 'z') AS x FROM t") == \
        _duck(fduck, "SELECT STRPOS(s, 'z') AS x FROM t")


def test_left(fengine, fduck):
    assert _ryu(fengine, "SELECT LEFT(s, 2) AS x FROM t") == _duck(fduck, "SELECT LEFT(s, 2) AS x FROM t")


def test_right(fengine, fduck):
    assert _ryu(fengine, "SELECT RIGHT(s, 2) AS x FROM t") == _duck(fduck, "SELECT RIGHT(s, 2) AS x FROM t")


def test_reverse(fengine, fduck):
    assert _ryu(fengine, "SELECT REVERSE(s) AS x FROM t") == _duck(fduck, "SELECT REVERSE(s) AS x FROM t")


def test_initcap(fengine):
    # DuckDB (this build) has no INITCAP, so assert against known expected
    # values: first letter of each word uppercased, rest lowercased; NULL->NULL.
    rows = _ryu(fengine, "SELECT INITCAP(s) AS x FROM t")
    expected = as_sorted(cudf.DataFrame({"x": ["Abc", "Abcd", "  X  ", "Xax", None, "Zzz"]}))
    assert rows == expected


# --------------------------------------------------------------------------- #
# Numeric functions
# --------------------------------------------------------------------------- #


def test_abs(fengine, fduck):
    assert _ryu(fengine, "SELECT ABS(v) AS x FROM t") == _duck(fduck, "SELECT ABS(v) AS x FROM t")


def test_round_default(fengine, fduck):
    # ROUND(v) -> 0 decimals, half-away-from-zero (2.5->3, -2.5->-3, 0.125->0).
    assert _ryu(fengine, "SELECT ROUND(v) AS x FROM t") == _duck(fduck, "SELECT ROUND(v) AS x FROM t")


def test_round_decimals(fengine, fduck):
    # 0.125 -> 0.13 (half-away), 0.125*100=12.5 -> 13. cuDF banker's would give 0.12.
    assert _ryu(fengine, "SELECT ROUND(v, 2) AS x FROM t") == _duck(fduck, "SELECT ROUND(v, 2) AS x FROM t")


def test_ceil(fengine, fduck):
    assert _ryu(fengine, "SELECT CEIL(v) AS x FROM t") == _duck(fduck, "SELECT CEIL(v) AS x FROM t")


def test_floor(fengine, fduck):
    assert _ryu(fengine, "SELECT FLOOR(v) AS x FROM t") == _duck(fduck, "SELECT FLOOR(v) AS x FROM t")


def test_mod_func(fengine, fduck):
    assert _ryu(fengine, "SELECT MOD(k, 3) AS x FROM t") == _duck(fduck, "SELECT MOD(k, 3) AS x FROM t")


def test_mod_op(fengine, fduck):
    # v % 3 via the % operator (exp.Mod -> BinOp "%").
    assert _ryu(fengine, "SELECT v % 3 AS x FROM t") == _duck(fduck, "SELECT v % 3 AS x FROM t")


# --------------------------------------------------------------------------- #
# Combos + fused-defer
# --------------------------------------------------------------------------- #


def test_func_in_where(fengine, fduck):
    sql = "SELECT s FROM t WHERE LENGTH(s) > 3"
    assert _ryu(fengine, sql) == _duck(fduck, sql)


def test_func_in_agg(fengine, fduck):
    # SUM(LENGTH(s)) -- a Func inside an agg arg; fused defers, cuDF handles it.
    sql = "SELECT SUM(LENGTH(s)) AS tot FROM t"
    assert _ryu(fengine, sql) == _duck(fduck, sql)


def test_func_combo(fengine, fduck):
    sql = "SELECT UPPER(SUBSTR(s, 1, 2)) || '!' AS x FROM t"
    assert _ryu(fengine, sql) == _duck(fduck, sql)


def test_func_defers_fused_but_correct(fengine, fduck):
    """A scalar Func (LENGTH) inside an aggregate arg over a join: the fused
    kernel's tokenizer cannot handle a Func, so fused_join_aggregate defers
    (returns None) and the cuDF path runs -- and still matches DuckDB.
    Correctness never depends on the extension."""
    sql = ("SELECT grp, SUM(LENGTH(t.s)) AS tot FROM u JOIN t ON u.k = t.k "
           "GROUP BY grp ORDER BY grp")
    if fused._kernels.is_available:
        agg = _agg_node(sql, fengine)
        assert fused.fused_join_aggregate(agg, fengine) is None, (
            "a scalar Func under an aggregate-over-join must defer to cuDF")
    assert _ryu(fengine, sql) == _duck(fduck, sql)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _agg_node(sql: str, engine: Engine):
    plan = optimize(parse(sql, engine.catalog.schema_dict()),
                    engine.catalog.schema_dict(), engine.catalog.stats_dict())
    return next(n for n in walk(plan) if isinstance(n, Aggregate))