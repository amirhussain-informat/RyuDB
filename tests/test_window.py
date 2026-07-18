"""SQL surface, Phase F-1: window functions -- RyuDB vs DuckDB.

Phase F-1 adds a ``Window`` plan node (a row-preserving "compute" node -- it
emits every input row plus one column per window function) and a ``WindowFunc``
expression. Three families are supported:

* **Ranking** (``ROW_NUMBER`` / ``RANK`` / ``DENSE_RANK``) over ``PARTITION BY ..
  ORDER BY ..`` -- requires an ORDER BY. Ties give equal RANK (gaps after) /
  DENSE_RANK (no gaps); ROW_NUMBER is unique per row. NULLs in the ORDER BY key
  sort LAST regardless of ASC/DESC (DuckDB's default for both directions), which
  the executor reproduces with a single ``na_position="last"``.
* **Offset** (``LAG`` / ``LEAD``) with an optional integer ``offset`` (default 1)
  and a ``default`` value (default NULL). The first/last ``offset`` rows of each
  partition get NULL (or the default), reached by computing per-partition
  boundaries on global sorted positions (cuDF's ``groupby.ngroup``/``cumcount``
  break on NULL keys, so the executor avoids groupby for this).
* **Aggregate broadcast** (``SUM`` / ``COUNT`` / ``AVG`` / ``MIN`` / ``MAX``)
  with NO ORDER BY -- the whole-partition value is broadcast to every row via
  cuDF ``groupby(dropna=False).transform`` (NULL partition keys form their own
  partition, matching DuckDB). ``COUNT(*)`` is the partition row count;
  ``COUNT(expr)`` is the non-NULL count; an empty partition is 0 (COALESCE) not
  NULL.

Deferred (raise ``NotImplementedError``): running/cumulative aggregates (an
ORDER BY on an aggregate window), explicit frames (``ROWS``/``RANGE
BETWEEN``), ``QUALIFY``, expression ``PARTITION BY``/``ORDER BY`` keys (only
bare columns), rank functions without an ORDER BY, and window functions mixed
with ``GROUP BY``/``HAVING``.

A window function in arithmetic (``w - LAG(w) OVER (...)``) and a plain
projection alongside a window output are supported (the ``Window`` node passes
input columns through verbatim). ``WHERE`` is built below the ``Window`` (it
filters rows *before* the window frames them) and ``ORDER BY`` on a window
output works (the outer ``Sort`` reads the projected alias).

Note: cuDF's equi-join matches ``NULL == NULL`` (RyuDB's pre-existing join
behavior, unlike DuckDB/SQL-standard), which is orthogonal to window functions.
The window tests therefore avoid NULL join keys.
"""

from __future__ import annotations

import cudf
import pytest

from ryudb import Catalog, Engine
from ryudb.sql.parse import parse

from .conftest import as_sorted

# a: partition key k (with a NULL-k partition), order key w with ties, a NULL
# w, and a (NULL, NULL) row -- exercises NULL-in-order-key sorting (NULLS LAST
# both directions), tie handling for RANK/DENSE_RANK, and the NULL partition.
_A = [
    (1, 30),
    (1, 10),
    (1, 10),
    (1, None),
    (2, 50),
    (2, 50),
    (2, 40),
    (None, 20),
    (None, None),
]
# b: for join-then-window. No NULL k (cuDF equi-join would match NULL==NULL and
# diverge from DuckDB -- a pre-existing join behavior, not a window bug), and a
# duplicate k=1 so the join multiplies rows before the window frames them.
_B = [
    (1, 100),
    (1, 150),
    (2, 200),
    (3, 300),
]


@pytest.fixture
def sdir(tmp_path):
    d = tmp_path
    for name, cols, rows in [("a", ["k", "w"], _A), ("b", ["k", "v"], _B)]:
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
# Ranking -- ROW_NUMBER / RANK / DENSE_RANK
# --------------------------------------------------------------------------- #


def test_row_number_partition_order(sengine, sduck):
    sql = "SELECT k, w, ROW_NUMBER() OVER (PARTITION BY k ORDER BY w) AS rn FROM a"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_row_number_desc(sengine, sduck):
    # NULLs LAST in both directions (DuckDB default) -- the NULL w sorts last
    # even with DESC.
    sql = "SELECT k, w, ROW_NUMBER() OVER (PARTITION BY k ORDER BY w DESC) AS rn FROM a"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_rank_partition_order(sengine, sduck):
    # Ties in w give equal RANK with a gap after (10,10 -> 1,1 then 3).
    sql = "SELECT k, w, RANK() OVER (PARTITION BY k ORDER BY w) AS r FROM a"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_dense_rank_partition_order(sengine, sduck):
    # DENSE_RANK has no gap after ties (10,10 -> 1,1 then 2).
    sql = "SELECT k, w, DENSE_RANK() OVER (PARTITION BY k ORDER BY w) AS dr FROM a"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_rank_no_partition(sengine, sduck):
    sql = "SELECT k, w, RANK() OVER (ORDER BY w) AS r FROM a"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_row_number_multikey_order(sengine, sduck):
    # Multi-key ORDER BY (no NULLs in the tiebreaker here -- k has NULLs but the
    # combination still sorts deterministically; DuckDB NULLS LAST on k too).
    sql = "SELECT k, w, ROW_NUMBER() OVER (ORDER BY k, w) AS rn FROM a"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_dense_rank_desc(sengine, sduck):
    sql = "SELECT k, w, DENSE_RANK() OVER (PARTITION BY k ORDER BY w DESC) AS dr FROM a"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


# --------------------------------------------------------------------------- #
# Offset -- LAG / LEAD
# --------------------------------------------------------------------------- #


def test_lag_default(sengine, sduck):
    sql = "SELECT k, w, LAG(w) OVER (PARTITION BY k ORDER BY w) AS lw FROM a"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_lead_default(sengine, sduck):
    sql = "SELECT k, w, LEAD(w) OVER (PARTITION BY k ORDER BY w) AS ld FROM a"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_lag_offset_and_default(sengine, sduck):
    # offset=2, default=-1 (parses through exp.Neg -> BinOp("-", 0, 1)).
    sql = "SELECT k, w, LAG(w, 2, -1) OVER (PARTITION BY k ORDER BY w) AS lw FROM a"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_lead_offset(sengine, sduck):
    sql = "SELECT k, w, LEAD(w, 2) OVER (PARTITION BY k ORDER BY w) AS ld FROM a"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_window_in_arithmetic(sengine, sduck):
    sql = "SELECT k, w, w - LAG(w) OVER (PARTITION BY k ORDER BY w) AS d FROM a"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


# --------------------------------------------------------------------------- #
# Aggregate broadcast -- SUM / COUNT / AVG / MIN / MAX (no ORDER BY)
# --------------------------------------------------------------------------- #


def test_sum_over_partition(sengine, sduck):
    sql = "SELECT k, w, SUM(w) OVER (PARTITION BY k) AS s FROM a"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_count_star_over_partition(sengine, sduck):
    sql = "SELECT k, w, COUNT(*) OVER (PARTITION BY k) AS c FROM a"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_count_expr_over_partition(sengine, sduck):
    # COUNT(w) ignores the NULL w.
    sql = "SELECT k, w, COUNT(w) OVER (PARTITION BY k) AS c FROM a"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_avg_min_max_over_partition(sengine, sduck):
    sql = ("SELECT k, w, AVG(w) OVER (PARTITION BY k) AS av, "
           "MIN(w) OVER (PARTITION BY k) AS mn, "
           "MAX(w) OVER (PARTITION BY k) AS mx FROM a")
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_sum_over_whole_frame(sengine, sduck):
    # No PARTITION BY -> the whole table is one frame.
    sql = "SELECT SUM(w) OVER () AS s FROM a"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_count_star_over_whole_frame(sengine, sduck):
    sql = "SELECT COUNT(*) OVER () AS c FROM a"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


# --------------------------------------------------------------------------- #
# Window + WHERE / ORDER BY on output / plain projection
# --------------------------------------------------------------------------- #


def test_window_with_where(sengine, sduck):
    # WHERE filters rows BEFORE the window frames them.
    sql = "SELECT k, w, ROW_NUMBER() OVER (PARTITION BY k ORDER BY w) AS rn FROM a WHERE w > 10"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_order_by_window_output(sengine, sduck):
    sql = "SELECT k, w, ROW_NUMBER() OVER (PARTITION BY k ORDER BY w) AS rn FROM a ORDER BY rn DESC"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_window_alongside_plain_projection(sengine, sduck):
    sql = "SELECT k, w, ROW_NUMBER() OVER (PARTITION BY k ORDER BY w) AS rn, w * 2 AS d2 FROM a"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_multiple_windows(sengine, sduck):
    # Two independent windows in one projection.
    sql = ("SELECT k, w, ROW_NUMBER() OVER (PARTITION BY k ORDER BY w) AS rn, "
           "SUM(w) OVER (PARTITION BY k) AS s FROM a")
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_join_then_window(sengine, sduck):
    # The join multiplies rows (a.k=1 x b.k=1 twice), then the window frames the
    # joined set. No NULL join keys (b has none) -- cuDF equi-join matches NULLs.
    sql = ("SELECT a.k, a.w, b.v, "
           "ROW_NUMBER() OVER (PARTITION BY a.k ORDER BY b.v) AS rn "
           "FROM a JOIN b ON a.k = b.k")
    assert _ryu(sengine, sql) == _duck(sduck, sql)


# --------------------------------------------------------------------------- #
# Parse shape -- the Window node and WindowFunc attributes
# --------------------------------------------------------------------------- #


def _window_of(plan):
    """Walk a parsed plan and return the (single) Window node."""
    from ryudb.sql.plan import Window

    windows = [n for n in _walk(plan) if isinstance(n, Window)]
    assert len(windows) == 1, f"expected one Window node, found {len(windows)}"
    return windows[0]


def _walk(node):
    yield node
    for attr in ("input", "left", "right"):
        child = getattr(node, attr, None)
        if child is not None:
            yield from _walk(child)


def test_window_node_under_project():
    from ryudb.sql.plan import Project, Scan

    plan = parse("SELECT k, w, ROW_NUMBER() OVER (PARTITION BY k ORDER BY w) AS rn FROM a")
    assert isinstance(plan, Project)
    win = _window_of(plan)
    assert isinstance(win.input, Scan)
    assert win.input.table == "a"
    assert len(win.funcs) == 1
    wf, name = win.funcs[0]
    # The Window node emits under an internal _wfN name; the outer Project renames
    # it to the user's alias (rn).
    assert name == "_wf1"
    assert wf.func == "ROW_NUMBER"
    assert wf.arg is None
    assert [p.name for p in wf.partition_keys] == ["k"]
    assert [(e.name, asc) for e, asc in wf.order_keys] == [("w", True)]


def test_window_func_rank_desc_attrs():
    plan = parse("SELECT k, RANK() OVER (PARTITION BY k ORDER BY w DESC) AS r FROM a")
    wf, _ = _window_of(plan).funcs[0]
    assert wf.func == "RANK"
    assert [(e.name, asc) for e, asc in wf.order_keys] == [("w", False)]


def test_window_func_lag_attrs():
    from ryudb.sql.plan import BinOp, Lit

    plan = parse("SELECT k, LAG(w, 2, -1) OVER (PARTITION BY k ORDER BY w) AS lw FROM a")
    wf, _ = _window_of(plan).funcs[0]
    assert wf.func == "LAG"
    assert wf.arg is not None  # Col(w)
    # offset is a Lit whose value is the sqlglot literal string "2".
    assert isinstance(wf.offset, Lit) and int(wf.offset.value) == 2
    # default -1 parses through exp.Neg -> BinOp("-", 0, 1).
    assert isinstance(wf.default, BinOp)
    assert wf.default.op == "-"


def test_window_func_count_star_attrs():
    plan = parse("SELECT k, COUNT(*) OVER (PARTITION BY k) AS c FROM a")
    wf, _ = _window_of(plan).funcs[0]
    assert wf.func == "COUNT"
    from ryudb.sql.plan import Star

    assert isinstance(wf.arg, Star)
    assert wf.order_keys == ()


def test_window_node_passes_input_columns():
    # The Window node is row-preserving: the outer Project references both input
    # columns (k, w) and the window output (via the _wfN rename).
    from ryudb.sql.plan import Col, Project

    plan = parse("SELECT k, w, ROW_NUMBER() OVER (PARTITION BY k ORDER BY w) AS rn FROM a")
    assert isinstance(plan, Project)
    ref_names = [e.name for e, _ in plan.items if isinstance(e, Col)]
    assert ref_names == ["k", "w", "_wf1"]


def test_window_replaced_in_projection_item():
    # The exp.Window in the projection item is rewritten to a bare Column ref to
    # the window's internal output name (_wfN), so the item is a plain Col that
    # the outer Project aliases to the user's name (rn).
    from ryudb.sql.plan import Col, Project

    plan = parse("SELECT k, ROW_NUMBER() OVER (PARTITION BY k ORDER BY w) AS rn FROM a")
    assert isinstance(plan, Project)
    rn_item, rn_alias = next((e, alias) for e, alias in plan.items if alias == "rn")
    assert isinstance(rn_item, Col)
    assert rn_item.name == "_wf1"


def test_multiple_windows_two_funcs():
    plan = parse(
        "SELECT k, w, "
        "ROW_NUMBER() OVER (PARTITION BY k ORDER BY w) AS rn, "
        "SUM(w) OVER (PARTITION BY k) AS s FROM a"
    )
    win = _window_of(plan)
    assert len(win.funcs) == 2
    # Internal names _wf1/_wf2; the outer Project maps them to rn/s.
    funcs = {name: wf.func for wf, name in win.funcs}
    assert funcs == {"_wf1": "ROW_NUMBER", "_wf2": "SUM"}


# --------------------------------------------------------------------------- #
# Deferred forms -- raise NotImplementedError
# --------------------------------------------------------------------------- #


def _rej(engine: Engine, sql: str):
    with pytest.raises(NotImplementedError):
        engine.sql(sql)


def test_running_aggregate_rejected(sengine):
    # ORDER BY on an aggregate window -> running/cumulative aggregate (deferred).
    _rej(sengine, "SELECT k, w, SUM(w) OVER (PARTITION BY k ORDER BY w) AS s FROM a")


def test_explicit_frame_rejected(sengine):
    _rej(sengine,
         "SELECT k, w, SUM(w) OVER (PARTITION BY k ORDER BY w ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS s FROM a")


def test_rank_without_order_rejected(sengine):
    _rej(sengine, "SELECT k, w, ROW_NUMBER() OVER (PARTITION BY k) AS rn FROM a")


def test_expr_partition_key_rejected(sengine):
    _rej(sengine, "SELECT k, w, ROW_NUMBER() OVER (PARTITION BY k + 1 ORDER BY w) AS rn FROM a")


def test_expr_order_key_rejected(sengine):
    _rej(sengine, "SELECT k, w, ROW_NUMBER() OVER (PARTITION BY k ORDER BY w + 1) AS rn FROM a")


def test_window_with_group_by_rejected(sengine):
    _rej(sengine, "SELECT k, MAX(w) FROM a GROUP BY k, ROW_NUMBER() OVER (PARTITION BY k ORDER BY w)")


def test_unsupported_window_func_rejected(sengine):
    # NTILE / PERCENT_RANK / CUME_DIST / FIRST_VALUE / LAST_VALUE / NTH_VALUE are
    # not in the F-1 whitelist.
    _rej(sengine, "SELECT k, NTILE(2) OVER (PARTITION BY k ORDER BY w) AS nt FROM a")