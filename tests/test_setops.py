"""SQL surface, Phase C: set operators (UNION [ALL] / INTERSECT / EXCEPT) --
RyuDB vs DuckDB oracle.

These compose two SELECTs into a ``SetOp`` plan node (plan.py) lowered to cuDF
``concat`` / ``drop_duplicates`` / ``merge`` (executor._setop). The fused CUDA
kernels are untouched -- a set op calls ``_exec`` on each branch, so any
aggregate-over-join inside a branch still fuses normally.

The fixture deliberately shares NULL-bearing rows across ``a`` and ``b`` so the
NULL semantics of the DISTINCT set ops are exercised: cuDF ``drop_duplicates``
treats NaN as equal (UNION-distinct is NULL-correct) and cuDF ``merge`` matches
nulls (INTERSECT/EXCEPT are NULL-correct with no sentinel trick). INTERSECT ALL
/ EXCEPT ALL (the multiset variants) are intentionally unsupported and raise
``NotImplementedError``. Comparison is via conftest.as_sorted (NULL -> None,
floats rounded, order-independent).
"""

from __future__ import annotations

import cudf
import pytest

from ryudb import Catalog, Engine
from ryudb.sql.parse import ParseError, parse

from .conftest import as_sorted

# a and b share three identical rows (incl. a NULL-s row (3,None,3.0) and a
# NULL-k+NULL-v row (None,"z",None)) so INTERSECT/EXCEPT exercise NULL matching
# across every column. The non-shared rows make UNION/EXCEPT non-trivial.
_A = [
    (1, "x", 1.0),
    (2, "y", 2.0),
    (3, None, 3.0),
    (None, "z", None),
]
_B = [
    (1, "x", 1.0),
    (2, "w", 8.0),
    (3, None, 3.0),
    (4, "q", 6.0),
    (None, "z", None),
]


@pytest.fixture
def sdir(tmp_path):
    d = tmp_path
    for name, cols, rows in [("a", ["k", "s", "v"], _A), ("b", ["k", "s", "v"], _B)]:
        (d / name).mkdir()
        cudf.DataFrame({c: [row[i] for row in rows] for i, c in enumerate(cols)}) \
            .to_pandas().to_parquet(d / name / "0.parquet")
    return d


@pytest.fixture
def sengine(sdir) -> Engine:
    cat = Catalog(str(sdir))
    for name in ("a", "b"):
        cat.register(name, str(sdir / name))
    return Engine(cat)


@pytest.fixture
def sduck(sdir):
    import duckdb
    con = duckdb.connect()
    for name in ("a", "b"):
        con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{sdir}/{name}/*.parquet')")
    return con


def _ryu(engine: Engine, sql: str):
    return as_sorted(engine.sql(sql))


def _duck(con, sql: str):
    return as_sorted(con.execute(sql).fetchdf())


# --------------------------------------------------------------------------- #
# UNION / UNION ALL
# --------------------------------------------------------------------------- #


def test_union(sengine, sduck):
    sql = "SELECT k, s, v FROM a UNION SELECT k, s, v FROM b"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_union_all(sengine, sduck):
    sql = "SELECT k, s, v FROM a UNION ALL SELECT k, s, v FROM b"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_union_star(sengine, sduck):
    sql = "SELECT * FROM a UNION SELECT * FROM b"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_union_orderby(sengine, sduck):
    # ORDER BY attaches to the set-op node; as_sorted makes order irrelevant,
    # but this still exercises the Sort-wraps-SetOp path.
    sql = "SELECT k, s, v FROM a UNION SELECT k, s, v FROM b ORDER BY k"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_union_limit(sengine, sduck):
    # Filter NULLs so the LIMIT pick is not at the mercy of engine-specific
    # NULL ordering (DuckDB and cuDF differ on NULLS FIRST/LAST for DESC).
    sql = ("SELECT k FROM a WHERE k IS NOT NULL UNION "
           "SELECT k FROM b WHERE k IS NOT NULL ORDER BY k DESC LIMIT 2")
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_union_orderby_limit(sengine, sduck):
    sql = ("SELECT k FROM a WHERE k IS NOT NULL UNION "
           "SELECT k FROM b WHERE k IS NOT NULL ORDER BY k LIMIT 3")
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_union_nested(sengine, sduck):
    # The parenthesized right arm arrives as an exp.Subquery wrapping a Union;
    # _build_query unwraps it.
    sql = "SELECT k FROM a UNION (SELECT k FROM b UNION SELECT k FROM a)"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_union_left_assoc(sengine, sduck):
    sql = "SELECT k FROM a UNION SELECT k FROM b UNION SELECT k FROM a"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_mixed_types(sengine, sduck):
    # int (a.k) + float (b.v) -> float under UNION; cuDF concat auto-promotes,
    # matching DuckDB's type coercion. NULLs collapse (NaN treated as equal).
    sql = "SELECT k FROM a UNION SELECT v FROM b"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_union_left_names(sengine, sduck):
    # SQL names a set op's outputs from the LEFT projection; the right side is
    # renamed positionally. Values still match the DuckDB oracle.
    sql = "SELECT k AS x FROM a UNION SELECT k FROM b"
    assert _ryu(sengine, sql) == _duck(sduck, sql)
    # The output column is the left arm's alias, not the right's bare name.
    assert list(sengine.sql(sql).columns) == ["x"]


def test_union_with_where(sengine, sduck):
    sql = "SELECT k FROM a WHERE k > 1 UNION SELECT k FROM b WHERE k < 4"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_union_of_aggregates(sengine, sduck):
    # Each arm is an aggregate; the fused kernel runs inside each branch.
    sql = ("SELECT k, COUNT(*) AS c FROM a GROUP BY k UNION "
           "SELECT k, COUNT(*) AS c FROM b GROUP BY k")
    assert _ryu(sengine, sql) == _duck(sduck, sql)


# --------------------------------------------------------------------------- #
# INTERSECT / EXCEPT (DISTINCT only; NULLs compare equal)
# --------------------------------------------------------------------------- #


def test_intersect(sengine, sduck):
    # The shared NULL-s row (3,None,3.0) and NULL-k row (None,"z",None) must be
    # intersected -- cuDF merge matches nulls, so these survive.
    sql = "SELECT * FROM a INTERSECT SELECT * FROM b"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_except(sengine, sduck):
    sql = "SELECT * FROM a EXCEPT SELECT * FROM b"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_except_reverse(sengine, sduck):
    sql = "SELECT * FROM b EXCEPT SELECT * FROM a"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_intersect_single_col(sengine, sduck):
    # NULL-k appears in both -> intersected (NULL matches NULL in merge).
    sql = "SELECT k FROM a INTERSECT SELECT k FROM b"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_intersect_all_unsupported(sengine):
    sql = "SELECT k FROM a INTERSECT ALL SELECT k FROM b"
    with pytest.raises(NotImplementedError):
        sengine.sql(sql)


def test_except_all_unsupported(sengine):
    sql = "SELECT k FROM a EXCEPT ALL SELECT k FROM b"
    with pytest.raises(NotImplementedError):
        sengine.sql(sql)


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


def test_column_count_mismatch(sengine):
    sql = "SELECT k, s FROM a UNION SELECT k FROM b"
    # Raised at execute time (the parser cannot know arm widths without schema).
    with pytest.raises(ParseError):
        sengine.sql(sql)


def test_setop_parses_to_setop():
    # A union lowers to a SetOp root (not a Select), so the optimizer runs and
    # the executor dispatches to _setop.
    plan = parse("SELECT k FROM a UNION SELECT k FROM b")
    from ryudb.sql.plan import SetOp
    assert isinstance(plan, SetOp)
    assert plan.op == "union"
    assert plan.distinct is True


def test_union_all_parses_distinct_false():
    plan = parse("SELECT k FROM a UNION ALL SELECT k FROM b")
    from ryudb.sql.plan import SetOp
    assert isinstance(plan, SetOp)
    assert plan.distinct is False