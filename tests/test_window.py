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
* **Running/cumulative aggregates** (``SUM`` / ``COUNT`` / ``AVG`` / ``MIN`` /
  ``MAX``) WITH an ORDER BY -- the SQL default frame (``RANGE BETWEEN UNBOUNDED
  PRECEDING AND CURRENT ROW``, peer-group cumulative: rows with equal order
  keys share the cumulative value, matching DuckDB) or an explicit ``ROWS`` /
  ``RANGE`` frame. ``ROWS`` frames use per-partition prefix sums (``SUM`` /
  ``COUNT`` / ``AVG``, O(n), any bounds incl. FOLLOWING) or ``cummin`` /
  ``cummax`` (``MIN`` / ``MAX``, cumulative only). NULLs in the agg arg are
  skipped (SQL): an all-null or empty window yields NULL for ``SUM`` / ``AVG``,
  0 for ``COUNT``; ``MIN`` / ``MAX`` skip nulls via ffill of the cumulative.

Deferred (raise ``NotImplementedError``): rank functions without an ORDER BY,
window functions mixed with ``GROUP BY``/``HAVING``, ``RANGE`` frames with
value offsets (``RANGE BETWEEN N PRECEDING``), ``EXCLUDE``, a frame on an
aggregate with no ORDER BY, and ``MIN``/``MAX`` with a non-cumulative frame
(trailing ``ROWS N PRECEDING AND CURRENT ROW`` or any FOLLOWING bound), named
window chaining (``WINDOW w2 AS (w1 ...)``) and partial overrides
(``OVER (w ORDER BY ...)``).
``QUALIFY`` (window-function filtering) is supported (Phase G-4) -- see the
QUALIFY section below. Named window definitions (``WINDOW w AS (...)``;
``OVER w``) are supported (Phase G-5) -- inlined at parse time, see the
Named-window section below. Expression ``PARTITION BY`` / ``ORDER BY`` keys are
supported (Phase G-6) -- materialized into synthetic sort columns, see the
Expression-keys section below.

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
# c: for QUALIFY top-N-with-payload. ``w`` is UNIQUE within each ``k`` partition
# (no ties), so ROW_NUMBER is deterministic and a payload column ``v`` can be
# selected alongside rn without non-deterministic tie-breaking making the
# RyuDB-vs-DuckDB row comparison flaky.
_C = [
    (1, 10, 100),
    (1, 20, 200),
    (1, 30, 300),
    (2, 5, 400),
    (2, 9, 500),
    (2, 12, 600),
]


@pytest.fixture
def sdir(tmp_path):
    d = tmp_path
    for name, cols, rows in [
        ("a", ["k", "w"], _A),
        ("b", ["k", "v"], _B),
        ("c", ["k", "w", "v"], _C),
    ]:
        (d / name).mkdir()
        cudf.DataFrame({c: [row[i] for row in rows] for i, c in enumerate(cols)}) \
            .to_pandas().to_parquet(d / name / "0.parquet")
    return d


@pytest.fixture
def sengine(sdir) -> Engine:
    cat = Catalog(str(sdir))
    for name in ("a", "b", "c"):
        cat.register(name, str(sdir / name))
    return Engine(cat)


@pytest.fixture
def sduck(sdir):
    import duckdb

    con = duckdb.connect()
    for name in ("a", "b", "c"):
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


# --------------------------------------------------------------------------- #
# Running/cumulative aggregates + frames (Phase G-3)
# --------------------------------------------------------------------------- #


def test_running_sum_default_frame(sengine, sduck):
    # The SQL default frame (RANGE UNBOUNDED PRECEDING TO CURRENT ROW) is
    # peer-group cumulative: the tied w=10 rows share the cumulative sum.
    sql = "SELECT k, w, SUM(w) OVER (PARTITION BY k ORDER BY w) AS s FROM a"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_running_count_star(sengine, sduck):
    sql = "SELECT k, w, COUNT(*) OVER (PARTITION BY k ORDER BY w) AS c FROM a"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_running_count_expr(sengine, sduck):
    # COUNT(w) running ignores the NULL w.
    sql = "SELECT k, w, COUNT(w) OVER (PARTITION BY k ORDER BY w) AS c FROM a"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_running_avg(sengine, sduck):
    sql = "SELECT k, w, AVG(w) OVER (PARTITION BY k ORDER BY w) AS av FROM a"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_running_min_max(sengine, sduck):
    sql = ("SELECT k, w, MIN(w) OVER (PARTITION BY k ORDER BY w) AS mn, "
           "MAX(w) OVER (PARTITION BY k ORDER BY w) AS mx FROM a")
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_default_range_differs_from_rows_on_ties(sengine, sduck):
    # The default (RANGE, peer) and an explicit ROWS cumulative diverge on the
    # tied w=10 rows -- and BOTH match DuckDB. This is the load-bearing proof
    # that peer-group semantics are implemented, not positional cumsum.
    sql_range = "SELECT k, w, SUM(w) OVER (PARTITION BY k ORDER BY w) AS s FROM a"
    sql_rows = ("SELECT k, w, SUM(w) OVER (PARTITION BY k ORDER BY w "
                "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS s FROM a")
    assert _ryu(sengine, sql_range) == _duck(sduck, sql_range)
    assert _ryu(sengine, sql_rows) == _duck(sduck, sql_rows)
    # And they actually differ (so the test is meaningful).
    assert _ryu(sengine, sql_range) != _ryu(sengine, sql_rows)


def test_rows_cumulative(sengine, sduck):
    sql = ("SELECT k, w, SUM(w) OVER (PARTITION BY k ORDER BY w "
           "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS s FROM a")
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_rows_trailing_one(sengine, sduck):
    sql = ("SELECT k, w, SUM(w) OVER (PARTITION BY k ORDER BY w "
           "ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS s FROM a")
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_rows_trailing_two_avg(sengine, sduck):
    sql = ("SELECT k, w, AVG(w) OVER (PARTITION BY k ORDER BY w "
           "ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS av FROM a")
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_rows_centered(sengine, sduck):
    # A FOLLOWING bound: SUM/COUNT/AVG via prefix sums.
    sql = ("SELECT k, w, SUM(w) OVER (PARTITION BY k ORDER BY w "
           "ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING) AS s FROM a")
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_rows_current_to_following(sengine, sduck):
    sql = ("SELECT k, w, SUM(w) OVER (PARTITION BY k ORDER BY w "
           "ROWS BETWEEN CURRENT ROW AND 1 FOLLOWING) AS s FROM a")
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_rows_current_to_unbounded_following(sengine, sduck):
    # "Remaining" sum: every row sees the sum from itself to the partition end.
    sql = ("SELECT k, w, SUM(w) OVER (PARTITION BY k ORDER BY w "
           "ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) AS s FROM a")
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_rows_unbounded_to_unbounded(sengine, sduck):
    # Whole-partition frame via an explicit ROWS frame = the broadcast value.
    sql = ("SELECT k, w, SUM(w) OVER (PARTITION BY k ORDER BY w "
           "ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS s FROM a")
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_rows_count_star_centered(sengine, sduck):
    sql = ("SELECT k, w, COUNT(*) OVER (PARTITION BY k ORDER BY w "
           "ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING) AS c FROM a")
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_range_default_explicit(sengine, sduck):
    # Explicit RANGE UNBOUNDED PRECEDING TO CURRENT ROW == the default frame.
    sql = ("SELECT k, w, SUM(w) OVER (PARTITION BY k ORDER BY w "
           "RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS s FROM a")
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_range_unbounded_to_unbounded(sengine, sduck):
    sql = ("SELECT k, w, SUM(w) OVER (PARTITION BY k ORDER BY w "
           "RANGE BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS s FROM a")
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_min_max_cumulative_with_following(sengine, sduck):
    # MIN/MAX with start UNBOUNDED PRECEDING and a FOLLOWING end is cumulative
    # (cummin/cummax at hi) -- supported.
    sql = ("SELECT k, w, MAX(w) OVER (PARTITION BY k ORDER BY w "
           "ROWS BETWEEN UNBOUNDED PRECEDING AND 1 FOLLOWING) AS mx FROM a")
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_min_max_whole_partition_frame(sengine, sduck):
    sql = ("SELECT k, w, MIN(w) OVER (PARTITION BY k ORDER BY w "
           "ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS mn FROM a")
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_running_no_partition(sengine, sduck):
    # Single partition (no PARTITION BY) running aggregate.
    sql = "SELECT k, w, SUM(w) OVER (ORDER BY w) AS s FROM a"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_running_multikey_order(sengine, sduck):
    # Peer = equal on ALL order keys.
    sql = "SELECT k, w, SUM(w) OVER (PARTITION BY k ORDER BY k, w) AS s FROM a"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_running_in_arithmetic(sengine, sduck):
    sql = "SELECT k, w, w - SUM(w) OVER (PARTITION BY k ORDER BY w) AS d FROM a"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_running_with_where(sengine, sduck):
    sql = ("SELECT k, w, SUM(w) OVER (PARTITION BY k ORDER BY w) AS s "
           "FROM a WHERE w > 10")
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_running_mixed_with_broadcast(sengine, sduck):
    # A running aggregate alongside a whole-partition broadcast in one query.
    sql = ("SELECT k, w, SUM(w) OVER (PARTITION BY k ORDER BY w) AS run_s, "
           "SUM(w) OVER (PARTITION BY k) AS all_s FROM a")
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_running_join_then_window(sengine, sduck):
    sql = ("SELECT a.k, a.w, b.v, "
           "SUM(b.v) OVER (PARTITION BY a.k ORDER BY b.v) AS s "
           "FROM a JOIN b ON a.k = b.k")
    assert _ryu(sengine, sql) == _duck(sduck, sql)


# --------------------------------------------------------------------------- #
# Parse shape -- the frame on the WindowFunc
# --------------------------------------------------------------------------- #


def test_parse_default_frame_synthesized():
    from ryudb.sql.plan import Frame, FrameBound

    plan = parse("SELECT SUM(w) OVER (PARTITION BY k ORDER BY w) AS s FROM a")
    wf, _ = _window_of(plan).funcs[0]
    assert wf.frame == Frame(
        "RANGE", FrameBound("UNBOUNDED_PRECEDING"), FrameBound("CURRENT_ROW")
    )


def test_parse_explicit_rows_frame():
    from ryudb.sql.plan import Frame, FrameBound

    plan = parse(
        "SELECT SUM(w) OVER (PARTITION BY k ORDER BY w "
        "ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS s FROM a")
    wf, _ = _window_of(plan).funcs[0]
    assert wf.frame == Frame(
        "ROWS", FrameBound("PRECEDING", 1), FrameBound("CURRENT_ROW")
    )


def test_parse_explicit_rows_centered_frame():
    from ryudb.sql.plan import Frame, FrameBound

    plan = parse(
        "SELECT SUM(w) OVER (PARTITION BY k ORDER BY w "
        "ROWS BETWEEN 1 PRECEDING AND 2 FOLLOWING) AS s FROM a")
    wf, _ = _window_of(plan).funcs[0]
    assert wf.frame == Frame(
        "ROWS", FrameBound("PRECEDING", 1), FrameBound("FOLLOWING", 2)
    )


def test_parse_broadcast_frame_none():
    plan = parse("SELECT SUM(w) OVER (PARTITION BY k) AS s FROM a")
    wf, _ = _window_of(plan).funcs[0]
    assert wf.frame is None


def test_parse_ranking_ignores_frame():
    # A frame on ROW_NUMBER is ignored (DuckDB does too); frame stays None.
    plan = parse(
        "SELECT ROW_NUMBER() OVER (PARTITION BY k ORDER BY w "
        "ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS rn FROM a")
    wf, _ = _window_of(plan).funcs[0]
    assert wf.func == "ROW_NUMBER"
    assert wf.frame is None


# --------------------------------------------------------------------------- #
# Deferred forms -- raise NotImplementedError
# --------------------------------------------------------------------------- #


def _rej(engine: Engine, sql: str):
    with pytest.raises(NotImplementedError):
        engine.sql(sql)


def test_rank_without_order_rejected(sengine):
    _rej(sengine, "SELECT k, w, ROW_NUMBER() OVER (PARTITION BY k) AS rn FROM a")


def test_expr_partition_key(sengine, sduck):
    # An expression PARTITION BY key (G-6): the executor materializes k+1 into a
    # synthetic sort column. _A has k in {1,2,None} -> k+1 in {2,3,NULL}.
    sql = "SELECT k, w, ROW_NUMBER() OVER (PARTITION BY k + 1 ORDER BY w) AS rn FROM a"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_expr_order_key(sengine, sduck):
    # An expression ORDER BY key (G-6): w+1 is materialized into a synthetic sort
    # column. Ties on w+1 are exactly ties on w, so ROW_NUMBER on _A (tied w) is
    # only deterministic when selecting (k, w, rn) -- tied rows are then
    # indistinguishable in the output multiset.
    sql = "SELECT k, w, ROW_NUMBER() OVER (PARTITION BY k ORDER BY w + 1) AS rn FROM a"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_window_with_group_by_rejected(sengine):
    _rej(sengine, "SELECT k, MAX(w) FROM a GROUP BY k, ROW_NUMBER() OVER (PARTITION BY k ORDER BY w)")


def test_unsupported_window_func_rejected(sengine):
    # NTILE / PERCENT_RANK / CUME_DIST / FIRST_VALUE / LAST_VALUE / NTH_VALUE are
    # not in the F-1 whitelist.
    _rej(sengine, "SELECT k, NTILE(2) OVER (PARTITION BY k ORDER BY w) AS nt FROM a")


def test_range_value_offset_rejected(sengine):
    # RANGE with a value offset needs value-based (not positional) scanning.
    _rej(sengine,
         "SELECT k, w, SUM(w) OVER (PARTITION BY k ORDER BY w "
         "RANGE BETWEEN 1 PRECEDING AND CURRENT ROW) AS s FROM a")


def test_min_max_trailing_rejected(sengine):
    # MIN/MAX with a non-cumulative frame (trailing ROWS) is deferred.
    _rej(sengine,
         "SELECT k, w, MIN(w) OVER (PARTITION BY k ORDER BY w "
         "ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS mn FROM a")


def test_min_max_following_rejected(sengine):
    # MIN/MAX with a FOLLOWING bound and a non-UNBOUNDED start is deferred.
    _rej(sengine,
         "SELECT k, w, MIN(w) OVER (PARTITION BY k ORDER BY w "
         "ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING) AS mn FROM a")


def test_frame_without_order_rejected(sengine):
    # A frame on an aggregate with no ORDER BY is ambiguous (deferred).
    _rej(sengine,
         "SELECT k, w, SUM(w) OVER (PARTITION BY k "
         "ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS s FROM a")


def test_window_exclude_rejected(sengine):
    _rej(sengine,
         "SELECT k, w, SUM(w) OVER (PARTITION BY k ORDER BY w "
         "ROWS BETWEEN 1 PRECEDING AND CURRENT ROW EXCLUDE CURRENT ROW) AS s FROM a")


# --------------------------------------------------------------------------- #
# CLI smoke -- a running total + moving average
# --------------------------------------------------------------------------- #


def test_cli_running_output(sengine, capsys):
    from ryudb import cli

    cli._run_statement(
        sengine,
        "SELECT k, w, "
        "sum(w) OVER (PARTITION BY k ORDER BY w) AS run_s, "
        "avg(w) OVER (PARTITION BY k ORDER BY w "
        "ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS ma "
        "FROM a ORDER BY k, w",
        quiet=False,
    )
    out = capsys.readouterr().out
    assert "run_s" in out
    assert "ma" in out


# --------------------------------------------------------------------------- #
# QUALIFY -- filter on window-function results (Phase G-4)
#
# QUALIFY lowers to a Filter directly above the Window node (below the outer
# Project), mirroring HAVING's Aggregate -> Filter -> Project shape. The
# predicate references the Window's _wfN/_qfN output columns: a SELECT-list
# window alias is resolved to its _wfN, and an inline window not in the SELECT
# list is synthesized as a new _qfN on the Window node. Determinism note: tests
# that select a payload column (v) use _C, whose order key w is UNIQUE per
# partition, so ROW_NUMBER assignment (and thus which row survives QUALIFY) is
# deterministic. Tests on _A (which has tied w) select only (k, w, rn) so tied
# rows are indistinguishable in the output multiset.
# --------------------------------------------------------------------------- #


def test_qualify_row_number_top1(sengine, sduck):
    sql = ("SELECT k, w, v, ROW_NUMBER() OVER (PARTITION BY k ORDER BY w) AS rn "
           "FROM c QUALIFY rn = 1")
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_qualify_row_number_top2(sengine, sduck):
    sql = ("SELECT k, w, v, ROW_NUMBER() OVER (PARTITION BY k ORDER BY w) AS rn "
           "FROM c QUALIFY rn <= 2")
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_qualify_inline_window_not_in_select(sengine, sduck):
    # The window function lives only in QUALIFY (not projected) -> synthesized
    # as a hidden _qfN on the Window node; the output has no rn column.
    sql = ("SELECT k, w, v FROM c "
           "QUALIFY ROW_NUMBER() OVER (PARTITION BY k ORDER BY w) = 1")
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_qualify_inline_window_top2(sengine, sduck):
    sql = ("SELECT k, w, v FROM c "
           "QUALIFY ROW_NUMBER() OVER (PARTITION BY k ORDER BY w) <= 2")
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_qualify_rank_ties_keeps_all_top(sengine, sduck):
    # RANK is deterministic on ties: both tied-top rows get rk=1 and survive.
    # _A's k=1 has w=10,10 (ties) -> both kept; k=2 has w=40 (unique min) -> one.
    sql = ("SELECT k, w, RANK() OVER (PARTITION BY k ORDER BY w) AS rk "
           "FROM a QUALIFY rk = 1")
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_qualify_dense_rank_ties(sengine, sduck):
    sql = ("SELECT k, w, DENSE_RANK() OVER (PARTITION BY k ORDER BY w) AS dr "
           "FROM a QUALIFY dr = 1")
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_qualify_row_number_ties_top1(sengine, sduck):
    # _A has tied w; selecting only (k, w, rn) makes tied rows indistinguishable
    # in the output multiset, so the result is deterministic despite ROW_NUMBER
    # tie-breaking being implementation-defined.
    sql = ("SELECT k, w, ROW_NUMBER() OVER (PARTITION BY k ORDER BY w) AS rn "
           "FROM a QUALIFY rn = 1")
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_qualify_row_number_ties_top2(sengine, sduck):
    sql = ("SELECT k, w, ROW_NUMBER() OVER (PARTITION BY k ORDER BY w) AS rn "
           "FROM a QUALIFY rn <= 2")
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_qualify_lag_first_row_per_partition(sengine, sduck):
    # LAG of the first row per partition is NULL -> QUALIFY lv IS NULL keeps
    # exactly the first row of each partition.
    sql = ("SELECT k, w, v, LAG(v) OVER (PARTITION BY k ORDER BY w) AS lv "
           "FROM c QUALIFY lv IS NULL")
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_qualify_broadcast_aggregate(sengine, sduck):
    # Broadcast aggregate (no ORDER BY) feeds QUALIFY. sum(v) per k:
    # k=1 -> 600, k=2 -> 1500. QUALIFY s > 1000 keeps only k=2's rows.
    sql = ("SELECT k, v, SUM(v) OVER (PARTITION BY k) AS s FROM c "
           "QUALIFY s > 1000 ORDER BY k, v")
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_qualify_count_star(sengine, sduck):
    sql = ("SELECT k, v, COUNT(*) OVER (PARTITION BY k) AS n FROM c "
           "QUALIFY n >= 3 ORDER BY k, v")
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_qualify_with_base_column(sengine, sduck):
    # A base column in the QUALIFY predicate must NOT push below the Window
    # (it would corrupt ROW_NUMBER). k=1 rn=1 is v=100 (fails v>200); k=2 rn=1
    # is v=400 (passes) -> only (2, 5, 400, 1) survives.
    sql = ("SELECT k, w, v, ROW_NUMBER() OVER (PARTITION BY k ORDER BY w) AS rn "
           "FROM c QUALIFY rn = 1 AND v > 200")
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_qualify_with_order_by(sengine, sduck):
    sql = ("SELECT k, w, v, ROW_NUMBER() OVER (PARTITION BY k ORDER BY w) AS rn "
           "FROM c QUALIFY rn = 1 ORDER BY k DESC")
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_qualify_with_distinct(sengine, sduck):
    # Each partition has a rn=1 row, so DISTINCT k yields every partition key.
    sql = "SELECT DISTINCT k FROM c QUALIFY ROW_NUMBER() OVER (PARTITION BY k ORDER BY w) = 1"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_qualify_with_limit(sengine, sduck):
    sql = ("SELECT k, w, v, ROW_NUMBER() OVER (PARTITION BY k ORDER BY w) AS rn "
           "FROM c QUALIFY rn = 1 ORDER BY k LIMIT 1")
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_qualify_arithmetic_in_predicate(sengine, sduck):
    sql = ("SELECT k, w, v, ROW_NUMBER() OVER (PARTITION BY k ORDER BY w) AS rn "
           "FROM c QUALIFY rn + 1 <= 2")
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_qualify_same_window_no_duplicate(sengine, sduck):
    # The window is projected (AS rn) AND referenced inline in QUALIFY with the
    # same OVER -> matched by sql() to the single _wf1 (not computed twice).
    sql = ("SELECT k, w, ROW_NUMBER() OVER (PARTITION BY k ORDER BY w) AS rn "
           "FROM c QUALIFY ROW_NUMBER() OVER (PARTITION BY k ORDER BY w) = 1")
    plan = parse(sql)
    win = _window_of(plan)
    assert len(win.funcs) == 1, "inline QUALIFY window must reuse the projected _wf1"
    assert win.funcs[0][1] == "_wf1"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_qualify_two_different_windows(sengine, sduck):
    # A projected window (rn) and a different inline QUALIFY window (the SUM
    # broadcast) -> two funcs on the Window node.
    sql = ("SELECT k, w, ROW_NUMBER() OVER (PARTITION BY k ORDER BY w) AS rn "
           "FROM c QUALIFY ROW_NUMBER() OVER (PARTITION BY k ORDER BY w) = 1 "
           "AND SUM(v) OVER (PARTITION BY k) > 1000")
    plan = parse(sql)
    win = _window_of(plan)
    funcs = {name: wf.func for wf, name in win.funcs}
    assert funcs == {"_wf1": "ROW_NUMBER", "_qf1": "SUM"}
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_qualify_no_partition(sengine, sduck):
    # Single partition (no PARTITION BY): rn=1 keeps the single first row.
    sql = ("SELECT k, w, v, ROW_NUMBER() OVER (ORDER BY w) AS rn "
           "FROM c QUALIFY rn = 1")
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_qualify_typed_decimal_date(typed_engine):
    # DECIMAL + DATE columns flow through the QUALIFY filter. l_orderkey is
    # unique, so ROW_NUMBER ORDER BY l_orderkey is deterministic; QUALIFY rn=1
    # keeps the first row per l_discount partition (4 partitions -> 4 rows).
    import duckdb

    d = typed_engine.catalog.data_dir
    con = duckdb.connect()
    con.execute(f"CREATE VIEW lineitem AS SELECT * FROM read_parquet('{d}/lineitem/*.parquet')")
    sql = ("SELECT l_orderkey, l_quantity, l_shipdate, "
           "ROW_NUMBER() OVER (PARTITION BY l_discount ORDER BY l_orderkey) AS rn "
           "FROM lineitem QUALIFY rn = 1 ORDER BY l_orderkey")
    assert _ryu(typed_engine, sql) == _duck(con, sql)


# --- QUALIFY parse shape --- #


def test_qualify_lowers_to_filter_above_window():
    from ryudb.sql.plan import Filter, Project, Window

    plan = parse("SELECT k, w, ROW_NUMBER() OVER (PARTITION BY k ORDER BY w) AS rn "
                 "FROM a QUALIFY rn = 1")
    assert isinstance(plan, Project)            # outer projection
    flt = plan.input
    assert isinstance(flt, Filter)              # QUALIFY -> Filter
    assert isinstance(flt.input, Window)        # directly above the Window node
    # The predicate references the Window's _wf1 output (rn alias -> _wf1).
    assert flt.predicate.columns() == {"_wf1"}


def test_qualify_inline_window_appends_qf_to_window():
    from ryudb.sql.plan import Filter, Project

    plan = parse("SELECT k, w FROM a "
                 "QUALIFY ROW_NUMBER() OVER (PARTITION BY k ORDER BY w) = 1")
    assert isinstance(plan, Project)
    flt = plan.input
    assert isinstance(flt, Filter)
    win = flt.input
    # No projected window -> the only func is the qualify-only _qf1.
    funcs = {name: wf.func for wf, name in win.funcs}
    assert funcs == {"_qf1": "ROW_NUMBER"}
    assert flt.predicate.columns() == {"_qf1"}


# --- QUALIFY rejections --- #


def test_qualify_without_window_rejected(sengine):
    # QUALIFY with no window function anywhere (projection or predicate).
    _rej(sengine, "SELECT k, w FROM a QUALIFY k > 5")


def test_qualify_with_group_by_rejected(sengine):
    # QUALIFY requires a window; window + GROUP BY is unsupported -> rejected.
    _rej(sengine, "SELECT k, MAX(w) FROM a GROUP BY k "
                  "QUALIFY ROW_NUMBER() OVER (PARTITION BY k ORDER BY w) = 1")


def test_qualify_subquery_rejected(sengine):
    _rej(sengine, "SELECT k, w FROM a QUALIFY k IN (SELECT k FROM a)")


def test_qualify_with_named_window(sengine, sduck):
    # QUALIFY + a named window def: the QUALIFY inline window references the
    # named def, which is inlined at parse time before _build_qualify_predicate.
    sql = ("SELECT k, w, v FROM c WINDOW w AS (PARTITION BY k ORDER BY w) "
           "QUALIFY ROW_NUMBER() OVER w = 1")
    assert _ryu(sengine, sql) == _duck(sduck, sql)


# --------------------------------------------------------------------------- #
# Named window definitions (Phase G-5) -- WINDOW w AS (...) inlined into OVER w
# --------------------------------------------------------------------------- #
#
# A named window definition (``WINDOW w AS (PARTITION BY .. ORDER BY ..)``) is
# inlined at parse time into every ``OVER w`` reference (in the projection and
# in QUALIFY), so each reference becomes a plain inline ``exp.Window`` that the
# existing ``_build_window`` / ``_build_one_window`` lower directly -- zero
# executor/optimizer change (pure parse-time rewriting, like QUALIFY). A pure
# reference ``OVER w`` parses to an ``exp.Window`` whose ``alias`` arg holds the
# referenced name (an ``exp.Identifier``) with no ``partition_by`` / ``order`` /
# ``spec``; the parser copies those from the matching def and drops the
# ``alias``. Chaining defs, partial overrides, duplicate defs, and unknown
# references are rejected.


def test_named_window_row_number(sengine, sduck):
    sql = ("SELECT k, w, v, ROW_NUMBER() OVER w AS rn FROM c "
           "WINDOW w AS (PARTITION BY k ORDER BY w)")
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_named_window_two_funcs_same_def(sengine, sduck):
    # Two different functions referencing the same named window.
    sql = ("SELECT k, w, ROW_NUMBER() OVER w AS rn, RANK() OVER w AS rk FROM c "
           "WINDOW w AS (PARTITION BY k ORDER BY w)")
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_named_window_desc_order(sengine, sduck):
    sql = ("SELECT k, w, v, ROW_NUMBER() OVER w AS rn FROM c "
           "WINDOW w AS (PARTITION BY k ORDER BY w DESC)")
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_named_window_lag(sengine, sduck):
    sql = ("SELECT k, w, v, LAG(v) OVER w AS lv FROM c "
           "WINDOW w AS (PARTITION BY k ORDER BY w)")
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_named_window_broadcast_aggregate(sengine, sduck):
    # A named window with no ORDER BY -> whole-partition broadcast.
    sql = ("SELECT k, v, SUM(v) OVER w AS s FROM c "
           "WINDOW w AS (PARTITION BY k)")
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_named_window_running_aggregate_default_frame(sengine, sduck):
    # A named window with an ORDER BY and no explicit frame -> the parser
    # synthesizes the default RANGE UNBOUNDED PRECEDING TO CURRENT ROW frame
    # (peer-group cumulative), exactly as an inline window would.
    sql = ("SELECT k, w, v, SUM(v) OVER w AS s FROM c "
           "WINDOW w AS (PARTITION BY k ORDER BY w)")
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_named_window_explicit_rows_frame(sengine, sduck):
    # An explicit ROWS frame in the named def is inlined into the reference.
    sql = ("SELECT k, w, v, SUM(v) OVER w AS s FROM c "
           "WINDOW w AS (PARTITION BY k ORDER BY w "
           "ROWS BETWEEN 1 PRECEDING AND CURRENT ROW)")
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_named_window_no_partition(sengine, sduck):
    # A named window with no PARTITION BY (single partition).
    sql = ("SELECT k, w, v, ROW_NUMBER() OVER w AS rn FROM c "
           "WINDOW w AS (ORDER BY w)")
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_named_window_two_defs(sengine, sduck):
    # Two named windows, each referenced.
    sql = ("SELECT k, w, v, ROW_NUMBER() OVER w1 AS rn, SUM(v) OVER w2 AS s "
           "FROM c WINDOW w1 AS (PARTITION BY k ORDER BY w), "
           "w2 AS (PARTITION BY k)")
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_named_window_in_arithmetic(sengine, sduck):
    # A named-window reference inside an arithmetic expression.
    sql = ("SELECT k, w, v - LAG(v) OVER w AS diff FROM c "
           "WINDOW w AS (PARTITION BY k ORDER BY w)")
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_named_window_with_order_by(sengine, sduck):
    sql = ("SELECT k, w, v, ROW_NUMBER() OVER w AS rn FROM c "
           "WINDOW w AS (PARTITION BY k ORDER BY w) ORDER BY k DESC, w DESC")
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_named_window_typed_decimal_date(typed_engine):
    import duckdb

    d = typed_engine.catalog.data_dir
    con = duckdb.connect()
    con.execute(f"CREATE VIEW lineitem AS SELECT * FROM read_parquet('{d}/lineitem/*.parquet')")
    sql = ("SELECT l_orderkey, l_quantity, l_shipdate, "
           "ROW_NUMBER() OVER w AS rn FROM lineitem "
           "WINDOW w AS (PARTITION BY l_discount ORDER BY l_orderkey) "
           "QUALIFY rn = 1 ORDER BY l_orderkey")
    assert _ryu(typed_engine, sql) == _duck(con, sql)


def test_named_window_inlines_to_one_window_node():
    # After inlining, the plan has a single Window node with one func whose
    # OVER is the full inline spec (the named ref is gone).
    plan = parse("SELECT k, w, ROW_NUMBER() OVER w AS rn FROM a "
                 "WINDOW w AS (PARTITION BY k ORDER BY w)")
    win = _window_of(plan)
    assert len(win.funcs) == 1
    wf, name = win.funcs[0]
    assert name == "_wf1"
    assert wf.func == "ROW_NUMBER"
    assert [c.name for c in wf.partition_keys] == ["k"]
    assert [c.name for (c, _asc) in wf.order_keys] == ["w"]


def test_named_window_two_refs_two_funcs():
    # Two references to the same named window -> two WindowFuncs on the Window
    # node (the executor computes both; output matches DuckDB which also
    # assigns both the same per-row values).
    plan = parse("SELECT k, ROW_NUMBER() OVER w AS r1, RANK() OVER w AS r2 FROM a "
                 "WINDOW w AS (PARTITION BY k ORDER BY w)")
    win = _window_of(plan)
    funcs = {name: wf.func for wf, name in win.funcs}
    assert funcs == {"_wf1": "ROW_NUMBER", "_wf2": "RANK"}


# --- Named window rejections --- #


def test_named_window_chaining_rejected(sengine):
    # WINDOW w2 AS (w1 ...) -- a def that references another def.
    _rej(sengine, "SELECT ROW_NUMBER() OVER w2 AS rn FROM a "
                  "WINDOW w1 AS (PARTITION BY k), w2 AS (w1 ORDER BY w)")


def test_named_window_partial_override_rejected(sengine):
    # OVER (w ORDER BY ...) -- a reference with its own inline additions.
    _rej(sengine, "SELECT ROW_NUMBER() OVER (w ORDER BY w) AS rn FROM a "
                  "WINDOW w AS (PARTITION BY k)")


def test_named_window_unknown_ref_rejected(sengine):
    _rej(sengine, "SELECT ROW_NUMBER() OVER missing AS rn FROM a")


def test_named_window_duplicate_def_rejected(sengine):
    _rej(sengine, "SELECT ROW_NUMBER() OVER w AS rn FROM a "
                  "WINDOW w AS (PARTITION BY k), w AS (PARTITION BY k ORDER BY w)")


# --------------------------------------------------------------------------- #
# Expression PARTITION BY / ORDER BY keys (Phase G-6)
# --------------------------------------------------------------------------- #
#
# A PARTITION BY / ORDER BY key may be any expression, not just a bare column
# (G-6). The parser accepts any ``_expr(p)``; the executor materializes a
# non-column key into a synthetic sort column (``_wpN`` / ``_woN``) on a copy of
# the frame, sorts/groups by it, and the synthetic column never reaches the
# output. Bare columns keep their own name (no synthetic column). Composes with
# every window family (ranking / LAG-LEAD / broadcast / running + frames),
# QUALIFY, and named windows. Determinism: an expression ORDER BY key with ties
# (e.g. ``w + 1`` on _A's tied ``w``) is only deterministic when the output
# selects no payload column that distinguishes the tied rows -- the tests on _A
# select (k, w, rn) so tied rows are indistinguishable in the multiset; tests
# that need a payload use _C (unique w per partition).


def test_expr_partition_row_number(sengine, sduck):
    sql = "SELECT k, w, v, ROW_NUMBER() OVER (PARTITION BY k % 2 ORDER BY w) AS rn FROM c"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_expr_order_row_number(sengine, sduck):
    sql = "SELECT k, w, v, ROW_NUMBER() OVER (PARTITION BY k ORDER BY w * 2) AS rn FROM c"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_expr_partition_and_order(sengine, sduck):
    sql = ("SELECT k, w, v, ROW_NUMBER() OVER (PARTITION BY k + 1 ORDER BY w - 1) AS rn "
           "FROM c")
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_expr_order_desc(sengine, sduck):
    sql = ("SELECT k, w, v, ROW_NUMBER() OVER (PARTITION BY k ORDER BY w + 10 DESC) AS rn "
           "FROM c")
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_expr_partition_rank_ties(sengine, sduck):
    # RANK is deterministic on ties; k%2 puts both k=1 and k=... rows together.
    sql = "SELECT k, w, RANK() OVER (PARTITION BY k % 2 ORDER BY w) AS rk FROM a"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_expr_partition_dense_rank(sengine, sduck):
    sql = "SELECT k, w, DENSE_RANK() OVER (PARTITION BY k % 2 ORDER BY w) AS dr FROM a"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_expr_order_lag(sengine, sduck):
    sql = "SELECT k, w, v, LAG(v) OVER (PARTITION BY k ORDER BY w + 1) AS lv FROM c"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_expr_partition_broadcast(sengine, sduck):
    # Broadcast aggregate with an expression partition key (no ORDER BY).
    sql = "SELECT k, v, SUM(v) OVER (PARTITION BY k % 2) AS s FROM c ORDER BY k, v"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_expr_order_running_default_frame(sengine, sduck):
    # Running aggregate with an expression ORDER BY key -> default RANGE frame.
    sql = "SELECT k, w, SUM(v) OVER (PARTITION BY k ORDER BY w * 2) AS s FROM c"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_expr_order_running_rows_frame(sengine, sduck):
    # Explicit ROWS frame with an expression ORDER BY key.
    sql = ("SELECT k, w, SUM(v) OVER (PARTITION BY k ORDER BY w + 1 "
           "ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS s FROM c")
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_expr_order_count_star_running(sengine, sduck):
    sql = ("SELECT k, w, COUNT(*) OVER (PARTITION BY k ORDER BY w - 1) AS n FROM c")
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_expr_order_avg_running(sengine, sduck):
    sql = "SELECT k, w, AVG(v) OVER (PARTITION BY k ORDER BY w * 2) AS a FROM c"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_expr_order_min_cumulative(sengine, sduck):
    # Cumulative MIN with an expression ORDER BY key.
    sql = "SELECT k, w, MIN(v) OVER (PARTITION BY k ORDER BY w + 1) AS mn FROM c"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_expr_key_no_partition(sengine, sduck):
    # An expression ORDER BY key with no PARTITION BY (single partition).
    sql = "SELECT k, w, v, ROW_NUMBER() OVER (ORDER BY w * 2) AS rn FROM c"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_expr_key_with_qualify(sengine, sduck):
    sql = ("SELECT k, w, v FROM c "
           "QUALIFY ROW_NUMBER() OVER (PARTITION BY k % 2 ORDER BY w) = 1")
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_expr_key_with_named_window(sengine, sduck):
    sql = ("SELECT k, w, v, ROW_NUMBER() OVER w AS rn FROM c "
           "WINDOW w AS (PARTITION BY k % 2 ORDER BY w)")
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_expr_key_typed_decimal_date(typed_engine):
    import duckdb

    d = typed_engine.catalog.data_dir
    con = duckdb.connect()
    con.execute(f"CREATE VIEW lineitem AS SELECT * FROM read_parquet('{d}/lineitem/*.parquet')")
    # l_orderkey % 7 has only 7 distinct values (massive ties) -> add l_orderkey
    # as a deterministic tiebreaker so ROW_NUMBER is well-defined.
    sql = ("SELECT l_orderkey, l_quantity, "
           "ROW_NUMBER() OVER (PARTITION BY l_discount ORDER BY l_orderkey % 7, l_orderkey) AS rn "
           "FROM lineitem QUALIFY rn = 1 ORDER BY l_orderkey")
    assert _ryu(typed_engine, sql) == _duck(con, sql)


def test_expr_key_synthetic_columns_dropped():
    # The synthetic sort columns (_wpN / _woN) must not appear in the output.
    from ryudb.sql.plan import Col

    plan = parse("SELECT k, w, ROW_NUMBER() OVER (PARTITION BY k + 1 ORDER BY w * 2) AS rn "
                 "FROM a")
    win = _window_of(plan)
    wf, _name = win.funcs[0]
    # partition_keys / order_keys hold the expression Exprs (not bare Cols).
    assert not all(isinstance(p, Col) for p in wf.partition_keys)
    assert not all(isinstance(e, Col) for e, _ in wf.order_keys)
