"""Logical aggregates: ``bool_and(x)`` / ``bool_or(x)`` (a.k.a. ``logical_and``
/ ``logical_or`` in sqlglot's parse, though DuckDB only spells them
``bool_and`` / ``bool_or``).

``bool_and(p)`` is true iff every non-null ``p`` is true (a logical AND across
the group); ``bool_or(p)`` is true iff any non-null ``p`` is true. Both lower to
cuDF ``min`` / ``max`` on the boolean arg series: cuDF reductions skip nulls, so
a NULL predicate (unknown) is skipped -- not treated as false -- and an all-NULL
or empty group is NULL, matching DuckDB. cuDF preserves NULL in comparisons
(``NULL > 5`` is ``<NA>``, not ``False``), so the common predicate-arg form
``bool_and(v > 5)`` is NULL-correct too. The same ``_AGG_METHOD`` entry serves
the global reduction (``_scalar_agg`` -> ``getattr(series, method)()``) and the
grouped single-pass (``_fused_agg`` -> ``groupby.agg({col: [method]})``, which
supports ``"min"`` / ``"max"`` as func strings on a bool column). The fused
C++/CUDA kernel only handles COUNT/SUM/AVG/MIN/MAX, so a logical aggregate
always defers to cuDF (no fused change). DISTINCT dedupes the arg before the
reduction; FILTER nulls failing rows' arg first; HAVING matches the agg by its
sqlglot ``.sql()``. DuckDB is the oracle.
"""

from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from ryudb import Catalog, Engine

from .conftest import assert_same


@pytest.fixture
def lengine(tmp_path) -> tuple[Engine, duckdb.DuckDBPyConnection]:
    """``t(g STR, v INT)`` with six groups exercising every bool_and/bool_or
    outcome, including NULL-skip and all-NULL groups:

    - ``a`` {6,7}    -> all true      -> bool_and=T, bool_or=T
    - ``b`` {1,2}    -> all false     -> bool_and=F, bool_or=F
    - ``c`` {6,1}    -> mixed         -> bool_and=F, bool_or=T
    - ``d`` {6,NULL} -> true + NULL   -> bool_and=T, bool_or=T (NULL skipped)
    - ``e`` {NULL,NULL} -> all NULL   -> bool_and=NULL, bool_or=NULL
    - ``f`` {1,NULL} -> false + NULL   -> bool_and=F, bool_or=F (NULL skipped)
    """
    d = tmp_path
    (d / "t").mkdir()
    pd.DataFrame(
        {
            "g": ["a", "a", "b", "b", "c", "c", "d", "d", "e", "e", "f", "f"],
            "v": [6, 7, 1, 2, 6, 1, 6, None, None, None, 1, None],
        }
    ).to_parquet(str(d / "t" / "0.parquet"))
    cat = Catalog(str(d))
    cat.register("t", str(d / "t"))
    eng = Engine(cat)
    duck = duckdb.connect()
    duck.execute(f"CREATE VIEW t AS SELECT * FROM read_parquet('{d}/t/*.parquet')")
    return eng, duck


@pytest.fixture
def bengine(tmp_path) -> tuple[Engine, duckdb.DuckDBPyConnection]:
    """``tb(g STR, flag BOOL)`` with NULLs, to test the direct-boolean-column
    form ``bool_and(flag)`` (not just a predicate arg). Same group shape as
    ``lengine`` but the boolean is a stored column."""
    d = tmp_path
    (d / "tb").mkdir()
    pd.DataFrame(
        {
            "g": ["a", "a", "b", "b", "c", "c", "d", "d", "e", "e", "f", "f"],
            "flag": pd.array(
                [True, True, False, False, True, False, True, None, None, None, False, None],
                dtype="boolean",
            ),
        }
    ).to_parquet(str(d / "tb" / "0.parquet"))
    cat = Catalog(str(d))
    cat.register("tb", str(d / "tb"))
    eng = Engine(cat)
    duck = duckdb.connect()
    duck.execute(f"CREATE VIEW tb AS SELECT * FROM read_parquet('{d}/tb/*.parquet')")
    return eng, duck


def _ryu(eng: Engine, sql: str):
    return eng.sql(sql)


def _duck(duck, sql: str):
    return duck.execute(sql).fetchdf()


# --------------------------------------------------------------------------- #
# Global logical aggregates
# --------------------------------------------------------------------------- #


def test_global_bool_and_or(lengine):
    eng, duck = lengine
    sql = "SELECT bool_and(v>5), bool_or(v>5) FROM t"
    assert_same(_ryu(eng, sql), _duck(duck, sql))


def test_global_bool_and_or_empty_is_null(lengine):
    """An empty set (WHERE matches nothing) -> bool_and/bool_or are NULL."""
    eng, duck = lengine
    sql = "SELECT bool_and(v>5), bool_or(v>5) FROM t WHERE v > 100"
    assert_same(_ryu(eng, sql), _duck(duck, sql))


def test_global_mixed_with_basic_agg(lengine):
    """Logical aggregates alongside COUNT/SUM/AVG in one global SELECT."""
    eng, duck = lengine
    sql = "SELECT count(*), sum(v), avg(v), bool_and(v>5), bool_or(v>5) FROM t"
    assert_same(_ryu(eng, sql), _duck(duck, sql))


# --------------------------------------------------------------------------- #
# Grouped logical aggregates (NULL-skip + all-NULL group)
# --------------------------------------------------------------------------- #


def test_grouped_bool_and_or(lengine):
    """All six groups at once: the all-true / all-false / mixed / true+NULL /
    all-NULL / false+NULL outcomes, including NULL-skip and the NULL all-NULL
    group."""
    eng, duck = lengine
    sql = "SELECT g, bool_and(v>5), bool_or(v>5) FROM t GROUP BY g ORDER BY g"
    assert_same(_ryu(eng, sql), _duck(duck, sql))


def test_grouped_mixed_with_basic(lengine):
    """Logical + basic aggregates, grouped (exercises the single-pass
    ``groupby.agg`` multi-func spec with a bool arg column)."""
    eng, duck = lengine
    sql = "SELECT g, count(*), sum(v), bool_and(v>5), bool_or(v>5) FROM t GROUP BY g ORDER BY g"
    assert_same(_ryu(eng, sql), _duck(duck, sql))


# --------------------------------------------------------------------------- #
# Direct boolean column (not a predicate arg)
# --------------------------------------------------------------------------- #


def test_grouped_bool_column(bengine):
    """``bool_and(flag)`` / ``bool_or(flag)`` over a stored boolean column with
    NULLs (the arg is a plain ``Col``, not a ``BinOp``)."""
    eng, duck = bengine
    sql = "SELECT g, bool_and(flag), bool_or(flag) FROM tb GROUP BY g ORDER BY g"
    assert_same(_ryu(eng, sql), _duck(duck, sql))


def test_global_bool_column(bengine):
    eng, duck = bengine
    sql = "SELECT bool_and(flag), bool_or(flag) FROM tb"
    assert_same(_ryu(eng, sql), _duck(duck, sql))


# --------------------------------------------------------------------------- #
# DISTINCT-qualified logical aggregates
# --------------------------------------------------------------------------- #


def test_global_distinct_bool_and(lengine):
    """``bool_and(DISTINCT p)`` dedupes the boolean arg before the reduction."""
    eng, duck = lengine
    sql = "SELECT bool_and(DISTINCT (v>5)) FROM t"
    assert_same(_ryu(eng, sql), _duck(duck, sql))


def test_grouped_distinct_bool_or(lengine):
    eng, duck = lengine
    sql = "SELECT g, bool_or(DISTINCT (v>5)) FROM t GROUP BY g ORDER BY g"
    assert_same(_ryu(eng, sql), _duck(duck, sql))


# --------------------------------------------------------------------------- #
# Composition: WHERE / HAVING / FILTER
# --------------------------------------------------------------------------- #


def test_bool_and_with_where(lengine):
    eng, duck = lengine
    sql = "SELECT bool_and(v>5), bool_or(v>5) FROM t WHERE v > 0"
    assert_same(_ryu(eng, sql), _duck(duck, sql))


def test_grouped_bool_with_where(lengine):
    eng, duck = lengine
    sql = "SELECT g, bool_and(v>5), bool_or(v>5) FROM t WHERE v > 1 GROUP BY g ORDER BY g"
    assert_same(_ryu(eng, sql), _duck(duck, sql))


def test_bool_or_in_having(lengine):
    """HAVING references the logical aggregate by its full shape (matched to
    the SELECT alias via sqlglot .sql()). Keeps groups with any true value."""
    eng, duck = lengine
    sql = "SELECT g, bool_and(v>5) AS ba FROM t GROUP BY g HAVING bool_or(v>5) ORDER BY g"
    assert_same(_ryu(eng, sql), _duck(duck, sql))


def test_bool_and_in_having_not_in_select(lengine):
    """A logical aggregate in HAVING that is NOT in the SELECT list is added as
    a synthetic aggregate and pruned from the output."""
    eng, duck = lengine
    sql = "SELECT g, count(*) AS c FROM t GROUP BY g HAVING bool_and(v>5) ORDER BY g"
    assert_same(_ryu(eng, sql), _duck(duck, sql))


def test_bool_and_with_filter(lengine):
    """``bool_and(v>5) FILTER (WHERE p)``: reduce the arg of passing rows only.
    A NULL filter predicate excludes the row (so the all-NULL group ``e`` has no
    passing rows -> NULL)."""
    eng, duck = lengine
    sql = "SELECT bool_and(v>5) FILTER (WHERE v > 0) FROM t"
    assert_same(_ryu(eng, sql), _duck(duck, sql))


def test_grouped_bool_or_with_filter(lengine):
    eng, duck = lengine
    sql = "SELECT g, bool_or(v>5) FILTER (WHERE v > 0) FROM t GROUP BY g ORDER BY g"
    assert_same(_ryu(eng, sql), _duck(duck, sql))


# --------------------------------------------------------------------------- #
# Alias: logical_and / logical_or (sqlglot parses these to the same exp class;
# DuckDB does NOT support the alias, so this is a RyuDB-internal acceptance test
# that the alias produces the same result as bool_and / bool_or).
# --------------------------------------------------------------------------- #


def test_logical_alias_matches_bool(lengine):
    """``logical_and`` / ``logical_or`` parse to the same ``exp.LogicalAnd`` /
    ``exp.LogicalOr`` as ``bool_and`` / ``bool_or`` and produce identical
    results. (DuckDB lacks the alias, so compare RyuDB-against-RyuDB.)"""
    eng, _duck = lengine
    assert_same(_ryu(eng, "SELECT bool_and(v>5), bool_or(v>5) FROM t"),
                _ryu(eng, "SELECT logical_and(v>5), logical_or(v>5) FROM t"))
    assert_same(_ryu(eng, "SELECT g, bool_and(v>5), bool_or(v>5) FROM t GROUP BY g ORDER BY g"),
                _ryu(eng, "SELECT g, logical_and(v>5), logical_or(v>5) FROM t GROUP BY g ORDER BY g"))


# --------------------------------------------------------------------------- #
# Rejections (deferred forms)
# --------------------------------------------------------------------------- #


def test_window_bool_and_rejected(lengine):
    """A logical aggregate as a window function is not supported yet -- the
    window builder only accepts COUNT/SUM/AVG/MIN/MAX as aggregate windows."""
    eng, _duck = lengine
    with pytest.raises(NotImplementedError):
        eng.sql("SELECT g, bool_and(v>5) OVER (PARTITION BY g) FROM t")


# --------------------------------------------------------------------------- #
# Regression: basic aggregates are unchanged
# --------------------------------------------------------------------------- #


def test_basic_aggs_unchanged(lengine):
    eng, duck = lengine
    assert_same(_ryu(eng, "SELECT count(*), sum(v), avg(v), min(v), max(v) FROM t"),
                _duck(duck, "SELECT count(*), sum(v), avg(v), min(v), max(v) FROM t"))
    assert_same(_ryu(eng, "SELECT g, count(*), sum(v) FROM t GROUP BY g ORDER BY g"),
                _duck(duck, "SELECT g, count(*), sum(v) FROM t GROUP BY g ORDER BY g"))