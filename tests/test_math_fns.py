"""SQL surface, Phase G-8: numeric math functions -- RyuDB vs DuckDB oracle.

POWER / SQRT / CBRT / EXP / LN / LOG / LOG10 / LOG2 / SIN / COS / TAN / ASIN /
ACOS / ATAN / ATAN2 / DEGREES / RADIANS / PI / TRUNC. These previously raised
NotImplementedError (no exp.Func dispatch in parse._SCALAR_FUNC_BUILDERS, and
TRUNC parses as exp.Anonymous). Each maps to a generic ``Func(tag, args)`` Expr
(plan.py) lowered tag-by-tag to a numpy ufunc on a cuDF Series (ops.py); the
fused CUDA kernels are untouched (a Func in a WHERE/agg-arg makes the fused gate
defer to cuDF).

DuckDB semantics being matched:
- ``LOG(x)`` == ``LOG10(x)`` (base-10). ``LN(x)`` is natural. ``LOG(b, x)`` is
  log base b. sqlglot collapses LOG/LOG10/LOG2/LOG(b,x) all to ``exp.Log``; the
  parser distinguishes by arg count (1-arg -> log10, 2-arg -> ln(value)/ln(base)).
- ``TRUNC(x)`` truncates toward zero; ``TRUNC(x, n)`` to n decimal places.
- ``PI()`` is a no-arg scalar (broadcast per row).
- NULL in -> NULL out.

Domain divergence (documented, NOT matched): DuckDB RAISES on SQRT of a
negative, LN/LOG of <=0, ASIN/ACOS outside [-1,1], and LOG with base 1 (ln(1)=0
-> division by zero). cuDF / numpy return NaN/inf silently. The functions here
are correct for in-domain inputs, so tests use in-domain data only (positives
for SQRT/LN/LOG, [-1,1] for ASIN/ACOS, base != 1 for 2-arg LOG). as_sorted
normalizes NULL (NaN/NA->None) and int/float, so the float-vs-DECIMAL dtype
difference on TRUNC (DuckDB returns DECIMAL) compares equal.
"""

from __future__ import annotations

import cudf
import pytest

from ryudb import Catalog, Engine

from .conftest import as_sorted

# a: positives (sqrt/ln/log domain). b: second operand. x: [-1,1] (asin/acos).
# ang: radians for trig. NULLs in every column exercise NULL propagation.
_T = [
    (0.5, 1.0, 0.0, 0.0),
    (2.0, 3.0, 0.25, 0.5),
    (10.0, 2.0, 0.5, 1.0),
    (1.0, 0.5, 1.0, 1.5),
    (None, None, None, None),
]


@pytest.fixture
def fdir(tmp_path):
    d = tmp_path
    (d / "t").mkdir()
    cudf.DataFrame(
        {c: [row[i] for row in _T] for i, c in enumerate(("a", "b", "x", "ang"))}
    ).to_pandas().to_parquet(d / "t" / "0.parquet")
    return d


@pytest.fixture
def fengine(fdir) -> Engine:
    cat = Catalog(str(fdir))
    cat.register("t", str(fdir / "t"))
    return Engine(cat)


@pytest.fixture
def fduck(fdir):
    import duckdb
    con = duckdb.connect()
    con.execute(f"CREATE VIEW t AS SELECT * FROM read_parquet('{fdir}/t/*.parquet')")
    return con


def _ryu(engine: Engine, sql: str):
    return as_sorted(engine.sql(sql))


def _duck(con, sql: str):
    return as_sorted(con.execute(sql).fetchdf())


# --------------------------------------------------------------------------- #
# Power
# --------------------------------------------------------------------------- #


def test_power_columns(fengine, fduck):
    assert _ryu(fengine, "SELECT a, b, POWER(a, b) AS p FROM t") == \
        _duck(fduck, "SELECT a, b, POWER(a, b) AS p FROM t")


def test_power_literal_exp(fengine, fduck):
    assert _ryu(fengine, "SELECT a, POWER(a, 2) AS p FROM t") == \
        _duck(fduck, "SELECT a, POWER(a, 2) AS p FROM t")


def test_power_literal_base(fengine, fduck):
    assert _ryu(fengine, "SELECT a, POWER(2, a) AS p FROM t") == \
        _duck(fduck, "SELECT a, POWER(2, a) AS p FROM t")


def test_power_zero_exp(fengine, fduck):
    # POWER(x, 0) = 1 for any x (incl POWER(0,0)=1); NULL a -> NULL.
    assert _ryu(fengine, "SELECT a, POWER(a, 0) AS p FROM t") == \
        _duck(fduck, "SELECT a, POWER(a, 0) AS p FROM t")


# --------------------------------------------------------------------------- #
# Roots / exp / logs
# --------------------------------------------------------------------------- #


def test_sqrt(fengine, fduck):
    assert _ryu(fengine, "SELECT a, SQRT(a) AS s FROM t") == \
        _duck(fduck, "SELECT a, SQRT(a) AS s FROM t")


def test_cbrt(fengine, fduck):
    assert _ryu(fengine, "SELECT a, CBRT(a) AS s FROM t") == \
        _duck(fduck, "SELECT a, CBRT(a) AS s FROM t")


def test_exp(fengine, fduck):
    assert _ryu(fengine, "SELECT a, EXP(a) AS s FROM t") == \
        _duck(fduck, "SELECT a, EXP(a) AS s FROM t")


def test_ln(fengine, fduck):
    assert _ryu(fengine, "SELECT a, LN(a) AS s FROM t") == \
        _duck(fduck, "SELECT a, LN(a) AS s FROM t")


def test_log_is_log10(fengine, fduck):
    # DuckDB LOG(x) == LOG10(x).
    assert _ryu(fengine, "SELECT a, LOG(a) AS s FROM t") == \
        _duck(fduck, "SELECT a, LOG(a) AS s FROM t")


def test_log10(fengine, fduck):
    assert _ryu(fengine, "SELECT a, LOG10(a) AS s FROM t") == \
        _duck(fduck, "SELECT a, LOG10(a) AS s FROM t")


def test_log2(fengine, fduck):
    assert _ryu(fengine, "SELECT a, LOG2(a) AS s FROM t") == \
        _duck(fduck, "SELECT a, LOG2(a) AS s FROM t")


def test_log_base_value(fengine, fduck):
    # LOG(b, x) = log base b of x. Use base 2 (a has no 1 to avoid div-by-zero).
    assert _ryu(fengine, "SELECT a, LOG(2, a) AS s FROM t") == \
        _duck(fduck, "SELECT a, LOG(2, a) AS s FROM t")


def test_log_base_column(fengine, fduck):
    # LOG(a, 10) = log base a of 10; a=1 row -> DuckDB div-by-zero, so filter it.
    sql = "SELECT a, LOG(a, 10) AS s FROM t WHERE a <> 1 ORDER BY a"
    assert _ryu(fengine, sql) == _duck(fduck, sql)


# --------------------------------------------------------------------------- #
# Trig
# --------------------------------------------------------------------------- #


def test_sin(fengine, fduck):
    assert _ryu(fengine, "SELECT ang, SIN(ang) AS s FROM t") == \
        _duck(fduck, "SELECT ang, SIN(ang) AS s FROM t")


def test_cos(fengine, fduck):
    assert _ryu(fengine, "SELECT ang, COS(ang) AS s FROM t") == \
        _duck(fduck, "SELECT ang, COS(ang) AS s FROM t")


def test_tan(fengine, fduck):
    assert _ryu(fengine, "SELECT ang, TAN(ang) AS s FROM t") == \
        _duck(fduck, "SELECT ang, TAN(ang) AS s FROM t")


def test_asin(fengine, fduck):
    assert _ryu(fengine, "SELECT x, ASIN(x) AS s FROM t") == \
        _duck(fduck, "SELECT x, ASIN(x) AS s FROM t")


def test_acos(fengine, fduck):
    assert _ryu(fengine, "SELECT x, ACOS(x) AS s FROM t") == \
        _duck(fduck, "SELECT x, ACOS(x) AS s FROM t")


def test_atan(fengine, fduck):
    assert _ryu(fengine, "SELECT a, ATAN(a) AS s FROM t") == \
        _duck(fduck, "SELECT a, ATAN(a) AS s FROM t")


def test_atan2(fengine, fduck):
    assert _ryu(fengine, "SELECT a, b, ATAN2(a, b) AS s FROM t") == \
        _duck(fduck, "SELECT a, b, ATAN2(a, b) AS s FROM t")


# --------------------------------------------------------------------------- #
# Degree conversions + PI
# --------------------------------------------------------------------------- #


def test_degrees(fengine, fduck):
    assert _ryu(fengine, "SELECT a, DEGREES(a) AS s FROM t") == \
        _duck(fduck, "SELECT a, DEGREES(a) AS s FROM t")


def test_radians(fengine, fduck):
    assert _ryu(fengine, "SELECT a, RADIANS(a) AS s FROM t") == \
        _duck(fduck, "SELECT a, RADIANS(a) AS s FROM t")


def test_pi(fengine, fduck):
    assert _ryu(fengine, "SELECT PI() AS p, a FROM t") == \
        _duck(fduck, "SELECT PI() AS p, a FROM t")


def test_pi_in_expr(fengine, fduck):
    assert _ryu(fengine, "SELECT a, a * PI() AS s FROM t") == \
        _duck(fduck, "SELECT a, a * PI() AS s FROM t")


# --------------------------------------------------------------------------- #
# TRUNC
# --------------------------------------------------------------------------- #


def test_trunc_no_decimals(fengine, fduck):
    assert _ryu(fengine, "SELECT a, TRUNC(a) AS s FROM t") == \
        _duck(fduck, "SELECT a, TRUNC(a) AS s FROM t")


def test_trunc_decimals(fengine, fduck):
    assert _ryu(fengine, "SELECT a, TRUNC(a, 2) AS s FROM t") == \
        _duck(fduck, "SELECT a, TRUNC(a, 2) AS s FROM t")


def test_trunc_one_decimal(fengine, fduck):
    assert _ryu(fengine, "SELECT a, TRUNC(a, 1) AS s FROM t") == \
        _duck(fduck, "SELECT a, TRUNC(a, 1) AS s FROM t")


def test_trunc_negative(fengine, fduck):
    # TRUNC truncates toward zero (TRUNC(-2.7) = -2, not -3).
    assert _ryu(fengine, "SELECT TRUNC(-2.7) AS s, a FROM t") == \
        _duck(fduck, "SELECT TRUNC(-2.7) AS s, a FROM t")


def test_trunc_neg_operand(fengine, fduck):
    assert _ryu(fengine, "SELECT a, TRUNC(-a, 1) AS s FROM t") == \
        _duck(fduck, "SELECT a, TRUNC(-a, 1) AS s FROM t")


# --------------------------------------------------------------------------- #
# Compositions
# --------------------------------------------------------------------------- #


def test_nested_sqrt_power(fengine, fduck):
    assert _ryu(fengine, "SELECT a, SQRT(POWER(a, 2)) AS s FROM t") == \
        _duck(fduck, "SELECT a, SQRT(POWER(a, 2)) AS s FROM t")


def test_in_arithmetic(fengine, fduck):
    assert _ryu(fengine, "SELECT a, ROUND(SQRT(a) * 100, 2) AS s FROM t") == \
        _duck(fduck, "SELECT a, ROUND(SQRT(a) * 100, 2) AS s FROM t")


def test_in_where(fengine, fduck):
    assert _ryu(fengine, "SELECT a FROM t WHERE SQRT(a) > 1.0 ORDER BY a") == \
        _duck(fduck, "SELECT a FROM t WHERE SQRT(a) > 1.0 ORDER BY a")


def test_group_by_floor_sqrt(fengine, fduck):
    sql = "SELECT FLOOR(SQRT(a)) AS g, COUNT(*) AS n FROM t GROUP BY g ORDER BY g"
    assert _ryu(fengine, sql) == _duck(fduck, sql)


def test_agg_sum_power(fengine, fduck):
    assert _ryu(fengine, "SELECT SUM(POWER(a, 2)) AS s FROM t") == \
        _duck(fduck, "SELECT SUM(POWER(a, 2)) AS s FROM t")


def test_log_in_case(fengine, fduck):
    sql = "SELECT a, CASE WHEN LOG(a) > 0.5 THEN 1 ELSE 0 END AS c FROM t"
    assert _ryu(fengine, sql) == _duck(fduck, sql)


def test_trig_in_having(fengine, fduck):
    sql = ("SELECT ang, COUNT(*) AS n FROM t GROUP BY ang "
           "HAVING SUM(SIN(ang)) IS NOT NULL ORDER BY ang")
    assert _ryu(fengine, sql) == _duck(fduck, sql)