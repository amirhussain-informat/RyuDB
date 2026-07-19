"""SQL surface, Phase G-9: hyperbolic + string functions -- RyuDB vs DuckDB.

SINH / COSH / TANH (hyperbolic), REPEAT, LPAD / RPAD, REGEXP_REPLACE,
REGEXP_MATCHES, SPLIT_PART, CONCAT_WS. These previously raised
NotImplementedError (no exp.Func dispatch in parse._SCALAR_FUNC_BUILDERS;
REGEXP_MATCHES parses as exp.Anonymous). Each maps to a generic
``Func(tag, args)`` Expr (plan.py) lowered to a cuDF ``.str`` accessor or
numpy ufunc (ops.py); the fused CUDA kernels are untouched (a Func in a
WHERE/agg-arg makes the fused gate defer to cuDF).

DuckDB semantics being matched:
- LPAD / RPAD(s, width, fill): pad to `width`; if the string is LONGER than
  `width` it is truncated to `width` chars (kept from the left). DuckDB has no
  2-arg LPAD (the fill string is required); RyuDB accepts it (default space) but
  that form is not DuckDB-comparable, so it is not tested here.
- REGEXP_REPLACE(s, pat, repl) (3-arg) replaces the FIRST match only (DuckDB
  occurrence=1; the 'g' flag form is out of scope).
- REGEXP_MATCHES(s, pat) -> boolean (NULL input -> NULL).
- SPLIT_PART(s, delim, n): literal (non-regex) delimiter; n=1 first part,
  n>count -> '', n=0 -> '', n<0 counts from the end (-1 = last), NULL -> NULL.
- CONCAT_WS(sep, a, b, ...): join non-NULL args per row with sep; all-NULL -> ''.
- REPEAT(s, n): n=0 -> ''.

Comparison is via conftest.as_sorted (normalizes NULL and int/float).
"""

from __future__ import annotations

import cudf
import pytest

from ryudb import Catalog, Engine

from .conftest import as_sorted

# s: strings with a comma (for SPLIT_PART), mixed lengths (for pad/truncate),
# an 'a' (for regexp), and a NULL. a: in-domain floats for hyperbolic. k: int
# for CONCAT_WS (stringifies cleanly). NULLs in every column.
_T = [
    ("ab", 0.5, 1),
    ("cde", 1.0, 2),
    ("fg,hij", 2.0, 3),
    (None, None, 4),
    ("aaxa", 3.0, 5),
]


@pytest.fixture
def fdir(tmp_path):
    d = tmp_path
    (d / "t").mkdir()
    cudf.DataFrame(
        {c: [row[i] for row in _T] for i, c in enumerate(("s", "a", "k"))}
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
# Hyperbolic
# --------------------------------------------------------------------------- #


def test_sinh(fengine, fduck):
    assert _ryu(fengine, "SELECT a, SINH(a) AS x FROM t") == \
        _duck(fduck, "SELECT a, SINH(a) AS x FROM t")


def test_cosh(fengine, fduck):
    assert _ryu(fengine, "SELECT a, COSH(a) AS x FROM t") == \
        _duck(fduck, "SELECT a, COSH(a) AS x FROM t")


def test_tanh(fengine, fduck):
    assert _ryu(fengine, "SELECT a, TANH(a) AS x FROM t") == \
        _duck(fduck, "SELECT a, TANH(a) AS x FROM t")


# --------------------------------------------------------------------------- #
# REPEAT
# --------------------------------------------------------------------------- #


def test_repeat(fengine, fduck):
    assert _ryu(fengine, "SELECT s, REPEAT(s, 3) AS x FROM t") == \
        _duck(fduck, "SELECT s, REPEAT(s, 3) AS x FROM t")


def test_repeat_zero(fengine, fduck):
    # REPEAT(s, 0) -> '' (NULL s -> NULL).
    assert _ryu(fengine, "SELECT s, REPEAT(s, 0) AS x FROM t") == \
        _duck(fduck, "SELECT s, REPEAT(s, 0) AS x FROM t")


# --------------------------------------------------------------------------- #
# LPAD / RPAD
# --------------------------------------------------------------------------- #


def test_lpad(fengine, fduck):
    assert _ryu(fengine, "SELECT s, LPAD(s, 5, 'x') AS x FROM t") == \
        _duck(fduck, "SELECT s, LPAD(s, 5, 'x') AS x FROM t")


def test_rpad(fengine, fduck):
    assert _ryu(fengine, "SELECT s, RPAD(s, 5, 'x') AS x FROM t") == \
        _duck(fduck, "SELECT s, RPAD(s, 5, 'x') AS x FROM t")


def test_lpad_truncate(fengine, fduck):
    # 'fg,hij' (6 chars) LPAD to 4 -> 'fg,h' (truncated to width, kept from left).
    assert _ryu(fengine, "SELECT s, LPAD(s, 4, 'x') AS x FROM t") == \
        _duck(fduck, "SELECT s, LPAD(s, 4, 'x') AS x FROM t")


def test_rpad_truncate(fengine, fduck):
    assert _ryu(fengine, "SELECT s, RPAD(s, 4, 'x') AS x FROM t") == \
        _duck(fduck, "SELECT s, RPAD(s, 4, 'x') AS x FROM t")


# --------------------------------------------------------------------------- #
# REGEXP_REPLACE / REGEXP_MATCHES
# --------------------------------------------------------------------------- #


def test_regexp_replace_first(fengine, fduck):
    # 3-arg form replaces the FIRST match only: 'aaxa' -> 'Zaxa'.
    assert _ryu(fengine, "SELECT s, REGEXP_REPLACE(s, 'a', 'Z') AS x FROM t") == \
        _duck(fduck, "SELECT s, REGEXP_REPLACE(s, 'a', 'Z') AS x FROM t")


def test_regexp_replace_class(fengine, fduck):
    assert _ryu(fengine, "SELECT s, REGEXP_REPLACE(s, '[ae]', 'X') AS x FROM t") == \
        _duck(fduck, "SELECT s, REGEXP_REPLACE(s, '[ae]', 'X') AS x FROM t")


def test_regexp_matches(fengine, fduck):
    assert _ryu(fengine, "SELECT s, REGEXP_MATCHES(s, 'a') AS x FROM t") == \
        _duck(fduck, "SELECT s, REGEXP_MATCHES(s, 'a') AS x FROM t")


def test_regexp_matches_anchor(fengine, fduck):
    assert _ryu(fengine, "SELECT s, REGEXP_MATCHES(s, '^[a-c]') AS x FROM t") == \
        _duck(fduck, "SELECT s, REGEXP_MATCHES(s, '^[a-c]') AS x FROM t")


# --------------------------------------------------------------------------- #
# SPLIT_PART
# --------------------------------------------------------------------------- #


def test_split_part_first(fengine, fduck):
    assert _ryu(fengine, "SELECT s, SPLIT_PART(s, ',', 1) AS x FROM t") == \
        _duck(fduck, "SELECT s, SPLIT_PART(s, ',', 1) AS x FROM t")


def test_split_part_second(fengine, fduck):
    assert _ryu(fengine, "SELECT s, SPLIT_PART(s, ',', 2) AS x FROM t") == \
        _duck(fduck, "SELECT s, SPLIT_PART(s, ',', 2) AS x FROM t")


def test_split_part_out_of_range(fengine, fduck):
    # part > count -> '' (NULL s -> NULL).
    assert _ryu(fengine, "SELECT s, SPLIT_PART(s, ',', 5) AS x FROM t") == \
        _duck(fduck, "SELECT s, SPLIT_PART(s, ',', 5) AS x FROM t")


def test_split_part_zero(fengine, fduck):
    # part = 0 -> '' (NULL s -> NULL).
    assert _ryu(fengine, "SELECT s, SPLIT_PART(s, ',', 0) AS x FROM t") == \
        _duck(fduck, "SELECT s, SPLIT_PART(s, ',', 0) AS x FROM t")


def test_split_part_negative(fengine, fduck):
    # -1 = last part: 'fg,hij' -> 'hij'; 'ab' (no delim) -> 'ab'.
    assert _ryu(fengine, "SELECT s, SPLIT_PART(s, ',', -1) AS x FROM t") == \
        _duck(fduck, "SELECT s, SPLIT_PART(s, ',', -1) AS x FROM t")


def test_split_part_negative_two(fengine, fduck):
    assert _ryu(fengine, "SELECT s, SPLIT_PART(s, ',', -2) AS x FROM t") == \
        _duck(fduck, "SELECT s, SPLIT_PART(s, ',', -2) AS x FROM t")


def test_split_part_literal_delim(fengine, fduck):
    # A regex-special delimiter ('.') is treated as a LITERAL, not "any char".
    assert _ryu(fengine, "SELECT SPLIT_PART('a.b.c', '.', 2) AS x, s FROM t") == \
        _duck(fduck, "SELECT SPLIT_PART('a.b.c', '.', 2) AS x, s FROM t")


# --------------------------------------------------------------------------- #
# CONCAT_WS
# --------------------------------------------------------------------------- #


def test_concat_ws(fengine, fduck):
    assert _ryu(fengine, "SELECT s, k, CONCAT_WS('-', s, k) AS x FROM t") == \
        _duck(fduck, "SELECT s, k, CONCAT_WS('-', s, k) AS x FROM t")


def test_concat_ws_skip_null(fengine, fduck):
    # NULL s -> just k; NULL k -> just s; both NULL -> ''.
    assert _ryu(fengine, "SELECT s, k, CONCAT_WS('|', s, k) AS x FROM t") == \
        _duck(fduck, "SELECT s, k, CONCAT_WS('|', s, k) AS x FROM t")


def test_concat_ws_three_args(fengine, fduck):
    assert _ryu(fengine, "SELECT s, k, a, CONCAT_WS('|', s, k, a) AS x FROM t") == \
        _duck(fduck, "SELECT s, k, a, CONCAT_WS('|', s, k, a) AS x FROM t")


def test_concat_ws_all_null(fengine, fduck):
    assert _ryu(fengine, "SELECT CONCAT_WS('-', NULL, NULL) AS x, k FROM t") == \
        _duck(fduck, "SELECT CONCAT_WS('-', NULL, NULL) AS x, k FROM t")


# --------------------------------------------------------------------------- #
# Compositions
# --------------------------------------------------------------------------- #


def test_sinh_in_arithmetic(fengine, fduck):
    assert _ryu(fengine, "SELECT a, ROUND(SINH(a) * 100, 2) AS x FROM t") == \
        _duck(fduck, "SELECT a, ROUND(SINH(a) * 100, 2) AS x FROM t")


def test_lpad_in_where(fengine, fduck):
    assert _ryu(fengine, "SELECT s FROM t WHERE LPAD(s, 5, 'x') = 'xxxab'") == \
        _duck(fduck, "SELECT s FROM t WHERE LPAD(s, 5, 'x') = 'xxxab'")


def test_regexp_in_where(fengine, fduck):
    assert _ryu(fengine, "SELECT s FROM t WHERE REGEXP_MATCHES(s, 'a') ORDER BY s") == \
        _duck(fduck, "SELECT s FROM t WHERE REGEXP_MATCHES(s, 'a') ORDER BY s")


def test_split_part_group_by(fengine, fduck):
    sql = "SELECT SPLIT_PART(s, ',', 1) AS p, COUNT(*) AS n FROM t GROUP BY p ORDER BY p"
    assert _ryu(fengine, sql) == _duck(fduck, sql)


def test_repeat_agg(fengine, fduck):
    assert _ryu(fengine, "SELECT MAX(LENGTH(REPEAT(s, 2))) AS x FROM t") == \
        _duck(fduck, "SELECT MAX(LENGTH(REPEAT(s, 2))) AS x FROM t")


def test_concat_ws_in_having(fengine, fduck):
    sql = ("SELECT k, COUNT(*) AS n FROM t GROUP BY k "
           "HAVING LENGTH(CONCAT_WS(',', k)) > 0 ORDER BY k")
    assert _ryu(fengine, sql) == _duck(fduck, sql)