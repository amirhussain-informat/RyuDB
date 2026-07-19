"""SQL surface, Phase G-7: misc scalar expressions -- RyuDB vs DuckDB oracle.

NULLIF / GREATEST / LEAST / SIGN. These previously raised NotImplementedError
(no exp.Func dispatch in parse._SCALAR_FUNC_BUILDERS). Each maps to a generic
``Func(tag, args)`` Expr (plan.py) lowered tag-by-tag to a cuDF op (ops.py); the
fused CUDA kernels are untouched (a Func in a WHERE/agg-arg makes the fused gate
defer to cuDF, see test_func_defers_fused_but_correct in test_functions.py).

DuckDB semantics being matched:
- NULLIF(a, b) = CASE WHEN a = b THEN NULL ELSE a. Three-valued: a=NULL -> NULL,
  b=NULL -> a, a==b -> NULL.
- GREATEST/LEAST skip NULLs (NULL is not the min/max); all-NULL -> NULL. Mixed
  numeric+string is a BinderException (type mismatch) -> we reject with
  NotImplementedError (see test_greatest_mixed_numeric_string_rejected).
- SIGN(x) returns -1/0/1 (TINYINT in DuckDB); NULL -> NULL.

The fixture has NULLs in k, v, s so every NULL path is exercised. ``s`` is a
string column for the string-literal GREATEST/LEAST cases. Comparison is via
conftest.as_sorted, which normalizes NULL (NaN/NA->None) and int/float
(1==1.0), so dtype-only differences (SIGN returns float -1.0/0.0/1.0 vs
DuckDB TINYINT) compare equal.
"""

from __future__ import annotations

import cudf
import pytest

from ryudb import Catalog, Engine

from .conftest import as_sorted

# k: int with a NULL; v: float with a NULL; s: string with a NULL. NULLs land in
# different rows so the GREATEST/LEAST NULL-skip and NULLIF NULL paths are
# exercised independently.
_T = [
    (1, 10.0, "ab"),
    (2, 20.0, "cd"),
    (3, 30.0, "ef"),
    (4, None, "gh"),
    (5, 50.0, None),
    (None, 60.0, "zz"),
]


@pytest.fixture
def fdir(tmp_path):
    d = tmp_path
    (d / "t").mkdir()
    cudf.DataFrame(
        {c: [row[i] for row in _T] for i, c in enumerate(("k", "v", "s"))}
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
# NULLIF
# --------------------------------------------------------------------------- #


def test_nullif_scalar(fengine, fduck):
    # k=2 -> NULL, others keep k; k=NULL -> NULL.
    assert _ryu(fengine, "SELECT k, NULLIF(k, 2) AS n FROM t") == \
        _duck(fduck, "SELECT k, NULLIF(k, 2) AS n FROM t")


def test_nullif_columns(fengine, fduck):
    # No row has k == v, so all keep k.
    assert _ryu(fengine, "SELECT k, v, NULLIF(k, v) AS n FROM t") == \
        _duck(fduck, "SELECT k, v, NULLIF(k, v) AS n FROM t")


def test_nullif_null_literal(fengine, fduck):
    # NULLIF(k, NULL) -> b is NULL -> a (k) preserved (incl k=NULL -> NULL).
    assert _ryu(fengine, "SELECT k, NULLIF(k, NULL) AS n FROM t") == \
        _duck(fduck, "SELECT k, NULLIF(k, NULL) AS n FROM t")


def test_nullif_equal_expr(fengine, fduck):
    # NULLIF(k, k) -> NULL for non-NULL k, NULL for NULL k.
    assert _ryu(fengine, "SELECT k, NULLIF(k, k) AS n FROM t") == \
        _duck(fduck, "SELECT k, NULLIF(k, k) AS n FROM t")


def test_nullif_in_arithmetic(fengine, fduck):
    assert _ryu(fengine, "SELECT k, NULLIF(k, 2) + 100 AS x FROM t") == \
        _duck(fduck, "SELECT k, NULLIF(k, 2) + 100 AS x FROM t")


def test_nullif_in_where(fengine, fduck):
    # NULLIF(k, 2) IS NULL keeps k=2 and k=NULL.
    assert _ryu(fengine, "SELECT k FROM t WHERE NULLIF(k, 2) IS NULL ORDER BY k") == \
        _duck(fduck, "SELECT k FROM t WHERE NULLIF(k, 2) IS NULL ORDER BY k")


# --------------------------------------------------------------------------- #
# GREATEST / LEAST
# --------------------------------------------------------------------------- #


def test_greatest_numeric(fengine, fduck):
    assert _ryu(fengine, "SELECT k, v, GREATEST(k, v) AS g FROM t") == \
        _duck(fduck, "SELECT k, v, GREATEST(k, v) AS g FROM t")


def test_greatest_with_literal(fengine, fduck):
    assert _ryu(fengine, "SELECT k, v, GREATEST(k, v, 15) AS g FROM t") == \
        _duck(fduck, "SELECT k, v, GREATEST(k, v, 15) AS g FROM t")


def test_greatest_null_arg(fengine, fduck):
    # NULL arg is skipped (DuckDB GREATEST ignores NULLs).
    assert _ryu(fengine, "SELECT k, GREATEST(k, NULL) AS g FROM t") == \
        _duck(fduck, "SELECT k, GREATEST(k, NULL) AS g FROM t")


def test_greatest_all_null(fengine, fduck):
    # All-NULL -> NULL.
    assert _ryu(fengine, "SELECT k, GREATEST(NULL, NULL) AS g FROM t") == \
        _duck(fduck, "SELECT k, GREATEST(NULL, NULL) AS g FROM t")


def test_least_numeric(fengine, fduck):
    assert _ryu(fengine, "SELECT k, v, LEAST(k, v) AS g FROM t") == \
        _duck(fduck, "SELECT k, v, LEAST(k, v) AS g FROM t")


def test_least_with_literal(fengine, fduck):
    assert _ryu(fengine, "SELECT k, v, LEAST(k, v, 5) AS g FROM t") == \
        _duck(fduck, "SELECT k, v, LEAST(k, v, 5) AS g FROM t")


def test_greatest_strings(fengine, fduck):
    # String column vs string literal.
    assert _ryu(fengine, "SELECT s, GREATEST(s, 'zz') AS g FROM t") == \
        _duck(fduck, "SELECT s, GREATEST(s, 'zz') AS g FROM t")


def test_least_strings(fengine, fduck):
    assert _ryu(fengine, "SELECT s, LEAST(s, '00') AS g FROM t") == \
        _duck(fduck, "SELECT s, LEAST(s, '00') AS g FROM t")


def test_greatest_three_string_args(fengine, fduck):
    assert _ryu(fengine, "SELECT GREATEST(s, 'zz', 'aa') AS g FROM t") == \
        _duck(fduck, "SELECT GREATEST(s, 'zz', 'aa') AS g FROM t")


def test_greatest_in_case(fengine, fduck):
    sql = "SELECT k, v, CASE WHEN GREATEST(k, v) > 20 THEN 1 ELSE 0 END AS c FROM t"
    assert _ryu(fengine, sql) == _duck(fduck, sql)


def test_greatest_group_by(fengine, fduck):
    sql = "SELECT GREATEST(k, v) AS g, COUNT(*) AS n FROM t GROUP BY g ORDER BY g"
    assert _ryu(fengine, sql) == _duck(fduck, sql)


def test_greatest_mixed_numeric_string_rejected(fengine):
    # DuckDB rejects this at bind time (BinderException: type mismatch). We
    # reject with NotImplementedError (clean error, not a CUDA fault).
    with pytest.raises(NotImplementedError):
        fengine.sql("SELECT GREATEST(k, v, s) AS g FROM t")


# --------------------------------------------------------------------------- #
# SIGN
# --------------------------------------------------------------------------- #


def test_sign_int(fengine, fduck):
    assert _ryu(fengine, "SELECT k, SIGN(k - 2) AS s FROM t") == \
        _duck(fduck, "SELECT k, SIGN(k - 2) AS s FROM t")


def test_sign_float(fengine, fduck):
    assert _ryu(fengine, "SELECT v, SIGN(v / 10.0) AS s FROM t") == \
        _duck(fduck, "SELECT v, SIGN(v / 10.0) AS s FROM t")


def test_sign_zero(fengine, fduck):
    assert _ryu(fengine, "SELECT SIGN(0) AS s, k FROM t") == \
        _duck(fduck, "SELECT SIGN(0) AS s, k FROM t")


def test_sign_null(fengine, fduck):
    # k=NULL -> NULL.
    assert _ryu(fengine, "SELECT k, SIGN(k) AS s FROM t") == \
        _duck(fduck, "SELECT k, SIGN(k) AS s FROM t")


# --------------------------------------------------------------------------- #
# Compositions with later phases
# --------------------------------------------------------------------------- #


def test_nullif_in_having(fengine, fduck):
    sql = ("SELECT k, COUNT(*) AS n FROM t GROUP BY k "
           "HAVING NULLIF(COUNT(*), 1) IS NOT NULL ORDER BY k")
    assert _ryu(fengine, sql) == _duck(fduck, sql)


def test_greatest_in_window_order(fengine, fduck):
    sql = ("SELECT k, v, ROW_NUMBER() OVER (ORDER BY GREATEST(k, v)) AS rn FROM t")
    assert _ryu(fengine, sql) == _duck(fduck, sql)


def test_nullif_in_qualify(fengine, fduck):
    sql = ("SELECT k, v FROM t QUALIFY NULLIF(ROW_NUMBER() OVER (ORDER BY k), 1) IS NULL")
    assert _ryu(fengine, sql) == _duck(fduck, sql)