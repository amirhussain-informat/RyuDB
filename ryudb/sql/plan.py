"""Logical/physical plan nodes and an expression AST for RyuDB.

The expression AST is a small, explicit tree that the executor lowers to cuDF
operations. Keeping our own representation (rather than passing sqlglot nodes
through to execution) makes the optimizer straightforward: it can rebuild and
rewrite expression trees without touching the parser's internals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# --------------------------------------------------------------------------- #
# Expressions
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Expr:
    """Base expression node."""

    def columns(self) -> set[str]:
        return set()


@dataclass(frozen=True)
class Col(Expr):
    name: str
    # Optional table/alias qualifier (e.g. ``a`` in ``a.v``). Preserved at parse
    # so a self-join / cross-table same-named-column join can disambiguate
    # ``a.v`` from ``b.v`` after the join executor renames colliding columns to
    # ``{alias}__{name}``. ``None`` for unqualified columns (the common case).
    # ``columns()`` deliberately returns the BARE name so the optimizer's scan
    # pruning / referenced-set logic (which keys on real table schemas) is
    # unchanged -- the qualifier is resolved only at eval time.
    table: str | None = None

    def columns(self) -> set[str]:
        return {self.name}


@dataclass(frozen=True)
class Lit(Expr):
    value: Any
    dtype: str | None = None  # e.g. "int", "float", "str", "date"

    def columns(self) -> set[str]:
        return set()


@dataclass(frozen=True)
class Star(Expr):
    """Used for COUNT(*)."""


@dataclass(frozen=True)
class BinOp(Expr):
    op: str  # + - * / = != < <= > >=
    left: Expr
    right: Expr

    def columns(self) -> set[str]:
        return self.left.columns() | self.right.columns()


@dataclass(frozen=True)
class And(Expr):
    left: Expr
    right: Expr

    def columns(self) -> set[str]:
        return self.left.columns() | self.right.columns()


@dataclass(frozen=True)
class Or(Expr):
    left: Expr
    right: Expr

    def columns(self) -> set[str]:
        return self.left.columns() | self.right.columns()


@dataclass(frozen=True)
class Not(Expr):
    expr: Expr

    def columns(self) -> set[str]:
        return self.expr.columns()


@dataclass(frozen=True)
class AggFunc(Expr):
    func: str  # COUNT SUM AVG MIN MAX
    arg: Expr  # Star() for COUNT(*)
    filter: "Expr | None" = None  # SQL ``FILTER (WHERE ...)`` predicate; None = no filter
    distinct: bool = False  # ``F(DISTINCT x)``: dedupe the arg within each group before reducing

    def columns(self) -> set[str]:
        cols = self.arg.columns() if not isinstance(self.arg, Star) else set()
        if self.filter is not None:
            cols |= self.filter.columns()
        return cols


@dataclass(frozen=True)
class FrameBound:
    """One bound of a window frame. ``kind`` is one of ``UNBOUNDED_PRECEDING`` /
    ``PRECEDING`` / ``CURRENT_ROW`` / ``FOLLOWING`` / ``UNBOUNDED_FOLLOWING``.
    ``n`` is the integer row offset for PRECEDING/FOLLOWING (None otherwise)."""
    kind: str
    n: int | None = None


@dataclass(frozen=True)
class Frame:
    """A window frame ``<mode> BETWEEN <start> AND <end>``. ``mode`` is ``ROWS``
    or ``RANGE``. RANGE peer-group semantics (rows with equal order keys share
    a cumulative value) apply only to the supported bounds (see parse); value
    offsets are deferred. ``None`` on a WindowFunc means no frame (whole-partition
    broadcast, i.e. no ORDER BY on an aggregate window)."""
    mode: str
    start: FrameBound
    end: FrameBound


@dataclass(frozen=True)
class WindowFunc(Expr):
    """A window function call ``func(arg) OVER (PARTITION BY .. ORDER BY .. [frame])``.

    Each window function carries its OWN OVER clause (partition + order can
    differ per function). ``func`` is one of ROW_NUMBER / RANK / DENSE_RANK /
    LAG / LEAD / SUM / COUNT / AVG / MIN / MAX, plus the statistical / logical /
    population aggregates STDDEV / STDDEV_SAMP / VARIANCE / MEDIAN / STDDEV_POP /
    VAR_POP / BOOL_AND / BOOL_OR. ``arg`` is the function argument
    (``Star()`` for COUNT(*), ``None`` for the no-argument rank funcs
    ROW_NUMBER / RANK / DENSE_RANK). ``offset`` / ``default`` are LAG/LEAD's
    optional integer offset (default 1) and default value (default NULL).
    ``partition_keys`` / ``order_keys`` are the OVER clause's PARTITION BY and
    ORDER BY (``order_keys`` is ``((Expr, ascending), ...)``); either may be
    empty. ``frame`` is the window frame (ROWS/RANGE BETWEEN ...); it is set for
    aggregate funcs (SUM/COUNT/AVG/MIN/MAX and the statistical/logical/population
    aggregates) when an ORDER BY is present -- the SQL default (RANGE UNBOUNDED
    PRECEDING TO CURRENT ROW, peer-group cumulative) is synthesized when no
    explicit frame is given -- and ``None`` for the whole-partition broadcast
    (aggregate, no ORDER BY) and for ranking/offset funcs (which ignore frames).
    ``columns()`` does not include the frame: its bound offsets are literal ints,
    not per-row expressions. WindowFunc does not carry FILTER / DISTINCT (only
    the non-window ``AggFunc`` does); window-form FILTER / DISTINCT aggs are not
    supported.
    """
    func: str
    arg: Expr | None
    partition_keys: tuple[Expr, ...] = ()
    order_keys: tuple[tuple[Expr, bool], ...] = ()
    offset: Expr | None = None
    default: Expr | None = None
    frame: Frame | None = None

    def columns(self) -> set[str]:
        cols: set[str] = set()
        for p in self.partition_keys:
            cols |= p.columns()
        for e, _ in self.order_keys:
            cols |= e.columns()
        if self.arg is not None and not isinstance(self.arg, Star):
            cols |= self.arg.columns()
        if self.offset is not None:
            cols |= self.offset.columns()
        if self.default is not None:
            cols |= self.default.columns()
        return cols


@dataclass(frozen=True)
class IsNull(Expr):
    """``expr IS [NOT] NULL`` -- ``negated`` for IS NOT NULL."""
    expr: Expr
    negated: bool = False

    def columns(self) -> set[str]:
        return self.expr.columns()


@dataclass(frozen=True)
class In(Expr):
    """``expr [NOT] IN (v1, v2, ...)`` -- list form only (subquery IN defers)."""
    expr: Expr
    values: tuple[Expr, ...]
    negated: bool = False

    def columns(self) -> set[str]:
        cols = self.expr.columns()
        for v in self.values:
            cols |= v.columns()
        return cols


@dataclass(frozen=True)
class Like(Expr):
    """``expr [NOT] LIKE pattern`` (or ``ILIKE`` when ``case_sensitive`` is False)."""
    expr: Expr
    pattern: Expr
    negated: bool = False
    case_sensitive: bool = True

    def columns(self) -> set[str]:
        return self.expr.columns() | self.pattern.columns()


@dataclass(frozen=True)
class Case(Expr):
    """``CASE [operand] WHEN cond THEN val ... [ELSE default]``.

    ``operand`` is non-None for a simple CASE (each branch condition is
    ``operand = when_value``); None for a searched CASE (each branch condition is
    a predicate). ``branches`` is ``[(cond, value), ...]``; ``default`` is the
    ELSE expression or ``None`` (-> NULL)."""
    operand: Expr | None
    branches: tuple[tuple[Expr, Expr], ...]
    default: Expr | None = None

    def columns(self) -> set[str]:
        cols: set[str] = set()
        if self.operand is not None:
            cols |= self.operand.columns()
        for cond, val in self.branches:
            cols |= cond.columns() | val.columns()
        if self.default is not None:
            cols |= self.default.columns()
        return cols


@dataclass(frozen=True)
class Coalesce(Expr):
    """``COALESCE(a, b, ...)`` -- first non-NULL argument per row."""
    args: tuple[Expr, ...]

    def columns(self) -> set[str]:
        cols: set[str] = set()
        for a in self.args:
            cols |= a.columns()
        return cols


@dataclass(frozen=True)
class Cast(Expr):
    """``CAST(expr AS type)`` for a non-literal ``expr`` (literal casts stay
    ``Lit`` with a dtype). ``dtype`` is a RyuDB type tag: int/float/str/bool/
    date/timestamp."""
    expr: Expr
    dtype: str

    def columns(self) -> set[str]:
        return self.expr.columns()


@dataclass(frozen=True)
class Func(Expr):
    """A scalar function call ``name(arg, arg, ...)``. ``name`` is a RyuDB
    function tag (upper/lower/length/substr/trim/concat/concat_pipe/replace/
    strpos/left/right/initcap/reverse/abs/round/ceil/floor); the tag selects the
    cuDF op in ``ops._func``. Using one generic node (rather than a dataclass per
    function) keeps the AST small -- the parse/ops tables carry the per-function
    specifics, and ``columns()``/``_estr`` work for free."""
    name: str
    args: tuple[Expr, ...]

    def columns(self) -> set[str]:
        cols: set[str] = set()
        for a in self.args:
            cols |= a.columns()
        return cols


# --------------------------------------------------------------------------- #
# Plan nodes
# --------------------------------------------------------------------------- #


@dataclass
class Scan:
    table: str
    # None => all columns; otherwise the projected set (pushed down by optimizer)
    columns: set[str] | None = None
    # The FROM/JOIN alias (``a`` in ``FROM t a``), or ``None`` when unaliased.
    # The join executor reads it to alias-prefix colliding columns on a self-
    # join / cross-table same-named-column join (``v`` -> ``a__v``).
    alias: str | None = None


@dataclass
class Filter:
    input: "PlanNode"
    predicate: Expr


@dataclass
class Project:
    input: "PlanNode"
    # list of (expression, output_name)
    items: list[tuple[Expr, str]] = field(default_factory=list)


@dataclass
class Join:
    left: "PlanNode"
    right: "PlanNode"
    on_left: list[str]   # columns from the left input
    on_right: list[str]  # columns from the right input (same length)
    how: str = "inner"   # inner | left | right | full | cross | semi | anti
    # ``semi``/``anti`` are the IN/NOT IN subquery lowering: the right side is the
    # subquery plan (a normal subtree the optimizer recurses into), the left side
    # is preserved (semi/anti keep left rows only; the right side's columns are
    # NOT in the output). on_left/on_right are the single IN key pair; on_predicate
    # is None. Semi = keep left rows whose key is in the subquery set; anti = keep
    # left rows whose key is not. See executor._join (cuDF ``isin``).
    # Residual ON predicate that is NOT a pure equi-key (e.g. ``ON a=b AND r.x>10``
    # folds the equi keys into on_left/on_right and the non-equi ``r.x>10`` here).
    # Kept *separate* from the WHERE Filter so outer-join semantics survive: an ON
    # residual filters only matched rows, never the null-padded unmatched rows.
    # ``None`` for USING/NATURAL (no residual) and CROSS. The executor applies it
    # inside the join (see ``Engine._apply_on_predicate``); the optimizer never
    # pushes it down (it is not a Filter).
    on_predicate: Expr | None = None
    # ``True`` for USING / NATURAL joins: their equi keys COALESCE into one column
    # (cuDF merges same-named keys to a single column), so the executor's
    # alias-rename collision path must NOT fire for them (it would un-coalesce the
    # key into ``a__k``/``b__k``). Non-key column collisions on a USING/NATURAL
    # join are rejected at parse time, so the only shared columns are the keys.
    using: bool = False


@dataclass
class Aggregate:
    input: "PlanNode"
    group_keys: list[tuple[Expr, str]]  # (expr, output_name)
    aggs: list[tuple[AggFunc, str]]     # (agg, output_name)


@dataclass
class Window:
    """Window-function computation (Phase F). A row-preserving "compute" node:
    it evaluates each window function over ``input`` and outputs the input
    columns PLUS one column per window function (``funcs`` is
    ``[(WindowFunc, output_name), ...]``). Unlike ``Aggregate`` it does NOT
    collapse rows -- every input row produces one output row with the window
    value attached, so the outer ``Project``/``Sort`` can reference both input
    columns and window outputs by name. ``input`` columns are passed through
    verbatim (the optimizer prunes scans below by what's referenced above).

    F-1: ranking (ROW_NUMBER/RANK/DENSE_RANK) and offset (LAG/LEAD) require an
    ORDER BY in the window; aggregate funcs (SUM/COUNT/AVG/MIN/MAX) broadcast
    over the whole partition (no ORDER BY). Running/cumulative aggregates (an
    ORDER BY on an aggregate window) and explicit frames are deferred."""
    input: "PlanNode"
    funcs: list[tuple[WindowFunc, str]]  # (window func, output_name)


@dataclass
class Sort:
    input: "PlanNode"
    keys: list[tuple[Expr, bool]]  # (expr, ascending)


@dataclass
class Limit:
    input: "PlanNode"
    n: int
    offset: int = 0


@dataclass
class Distinct:
    """``SELECT DISTINCT`` (Phase F-2a): a row-preserving node that drops
    duplicate rows from its ``input``. Like ``Sort``/``Limit`` it carries no
    expressions (it dedups on all of ``input``'s columns) and is placed between
    the projection and ``Sort``/``Limit`` in ``_build_select``. cuDF
    ``drop_duplicates`` treats NaN as equal, so DISTINCT is NULL-correct and
    matches DuckDB. ``DISTINCT ON (...)`` is a different feature (keep-first per
    key) and is rejected at parse time, not lowered to this node."""
    input: "PlanNode"


@dataclass
class Derived:
    """A derived table / FROM-subquery (Phase F-2b): ``FROM (SELECT ...) [AS] t``.
    A row-preserving source node whose execution returns the subplan's output
    frame verbatim -- the subplan's top ``Project``/``Aggregate`` names its output
    columns, and the outer query references them as ordinary flat column names
    (the flat-column model: ``Col`` carries only a name, no table qualifier).
    Like ``Window`` it is a *scope barrier*: outer predicates/projections do not
    push across it (the optimizer never pushes a conjunct into the subplan), but
    the optimizer recurses into ``input`` so the subplan is optimized -- predicate
    pushdown, projection pruning, join-side selection -- within its own scope.
    ``alias`` is the FROM alias and is required (anonymous derived tables are
    rejected at parse time); it doubles as the derived table's routing identity
    (``schema.get(alias)`` is empty, so join-key routing falls back to table
    qualifiers)."""
    input: "PlanNode"
    alias: str


@dataclass
class SetOp:
    """UNION / INTERSECT / EXCEPT over two child relations. ``op`` is
    "union" | "intersect" | "except"; ``distinct`` is False for UNION ALL (the
    only ALL variant supported -- INTERSECT ALL / EXCEPT ALL raise). Output
    column names come from the left child's projection; the executor renames the
    right child's columns positionally. A predicate-pushdown *barrier*: nothing
    crosses a set op, but the optimizer still recurses into each branch so
    joins/projections *inside* a branch get optimized (see optimize.py)."""
    left: "PlanNode"
    right: "PlanNode"
    op: str
    distinct: bool = True


@dataclass
class Insert:
    """Write node: append rows to a table's delta (step 3 + INSERT...SELECT).

    Two mutually exclusive forms: ``INSERT ... VALUES`` carries literal rows in
    ``rows`` (one ``list[Expr]`` per value row, each cell a ``Lit``);
    ``INSERT ... SELECT`` carries a relational subplan in ``source``. Exactly one
    is set. For VALUES ``rows`` is a leaf (no ``input``); for SELECT the subplan
    is the single child. ``columns`` is the user-supplied target column list
    (``None`` => catalog order); row cells / SELECT output columns map positionally
    to it (SELECT output names are ignored). The executor resolves the full
    schema, fills DEFAULTs, enforces NOT NULL, builds a typed cuDF batch, and
    appends it to the delta. PK/UNIQUE is enforced before any durable write.
    """

    table: str
    columns: list[str] | None = None
    rows: list[list[Expr]] = field(default_factory=list)
    source: "PlanNode | None" = None  # INSERT ... SELECT subplan (else rows)


@dataclass
class Delete:
    """Write node: tombstone rows of ``DELETE FROM t [WHERE pred]`` (step 9).

    A non-relational leaf (no ``input``), like ``Insert``. ``predicate`` is the
    optional WHERE row-selector (``None`` => delete every visible row). The
    executor evaluates it against the visible snapshot, collects the PK values
    of the matched rows, and stores them as a tombstone batch (see
    ``Engine._delete``). Requires a declared PRIMARY KEY on ``t`` (row identity
    is by PK value, not position)."""

    table: str
    predicate: Expr | None = None


@dataclass
class Update:
    """Write node: ``UPDATE t SET col = expr [, ...] [WHERE pred]`` (step 10).

    A non-relational leaf (no ``input``), like ``Insert``/``Delete``.
    ``assignments`` is an ordered list of ``(column, Expr)`` pairs from
    ``SET a = e, b = e``. ``predicate`` is the optional WHERE row-selector
    (``None`` => update every visible row). The executor evaluates the predicate
    against the visible snapshot, builds the post-SET rows, tombstones the matched
    rows' old PKs and re-inserts the new rows in one atomic two-ts commit (tombstone
    at ``T``, re-insert at ``T+1``), so the new row's ``ins_ts`` strictly exceeds
    the tombstone's ``tomb_ts`` and survives ``_merge_delta``'s
    ``keep = tomb_ts < ins_ts`` rule (see ``Engine._update``). Requires a declared
    PRIMARY KEY on ``t`` (row identity is by PK value, not position), and is
    supported in autocommit only (explicit-txn UPDATE raises
    ``NotImplementedError`` in v1)."""

    table: str
    assignments: list[tuple[str, Expr]] = field(default_factory=list)
    predicate: Expr | None = None


@dataclass
class TxnControl:
    """Transaction-control leaf (Phase 2 step 5): BEGIN / COMMIT / ROLLBACK.

    A non-relational leaf (no ``input``), like ``Insert``. The executor dispatches
    on ``kind`` to the Engine's ``_begin``/``_commit``/``_rollback`` and returns
    ``None`` (no result frame). Snapshot/restore are NOT plan nodes -- they are
    non-standard SQL and bypass ``parse`` entirely via a regex pre-sniff in
    ``Engine.sql``."""
    kind: str  # "begin" | "commit" | "rollback"


PlanNode = Scan | Filter | Project | Join | Aggregate | Window | Sort | Limit | Distinct | Derived | SetOp | Insert | Delete | Update | TxnControl


def walk(node: PlanNode):
    """Yield every plan node in the tree (pre-order)."""
    yield node
    for child in children(node):
        yield from walk(child)


def children(node: PlanNode) -> list[PlanNode]:
    if isinstance(node, (Filter, Project, Sort, Distinct, Derived)):
        return [node.input]
    if isinstance(node, (Join,)):
        return [node.left, node.right]
    if isinstance(node, SetOp):
        return [node.left, node.right]
    if isinstance(node, (Aggregate, Window, Limit)):
        return [node.input]
    if isinstance(node, Insert):
        # INSERT ... SELECT: the subplan is the single child. INSERT ... VALUES
        # (rows set, source None) is a leaf.
        return [node.source] if node.source is not None else []
    return []


def exprs_in(node: PlanNode) -> list[Expr]:
    """All expressions referenced by a node (for column analysis)."""
    if isinstance(node, Filter):
        return [node.predicate]
    if isinstance(node, Project):
        return [e for e, _ in node.items]
    if isinstance(node, Aggregate):
        return [e for e, _ in node.group_keys] + [a for a, _ in node.aggs]
    if isinstance(node, Sort):
        return [e for e, _ in node.keys]
    if isinstance(node, Window):
        return [wf for wf, _ in node.funcs]
    if isinstance(node, Join):
        return [node.on_predicate] if node.on_predicate is not None else []
    if isinstance(node, Delete):
        return [node.predicate] if node.predicate is not None else []
    if isinstance(node, Update):
        return [e for _, e in node.assignments] + (
            [node.predicate] if node.predicate is not None else []
        )
    # Insert's payload (VALUES Lits / SELECT subplan) is not a relational
    # expression to surface here; the optimizer does not rewrite a write node.
    return []


def pretty(node: PlanNode, indent: int = 0) -> str:
    pad = "  " * indent
    if isinstance(node, Scan):
        cols = "*" if node.columns is None else ",".join(sorted(node.columns))
        return f"{pad}Scan({node.table} cols={cols})"
    if isinstance(node, Filter):
        return f"{pad}Filter({_estr(node.predicate)})\n" + pretty(node.input, indent + 1)
    if isinstance(node, Project):
        items = ", ".join(f"{_estr(e)} AS {n}" for e, n in node.items)
        return f"{pad}Project({items})\n" + pretty(node.input, indent + 1)
    if isinstance(node, Join):
        on = " AND ".join(f"{lk}={rk}" for lk, rk in zip(node.on_left, node.on_right))
        on_pred = "" if node.on_predicate is None else f" [{_estr(node.on_predicate)}]"
        return (
            f"{pad}Join({node.how} on {on}{on_pred})\n"
            + pretty(node.left, indent + 1) + "\n"
            + pretty(node.right, indent + 1)
        )
    if isinstance(node, Aggregate):
        g = ", ".join(n for _, n in node.group_keys)
        a = ", ".join(f"{_estr(af)} AS {n}" for af, n in node.aggs)
        return f"{pad}Aggregate(group=[{g}] aggs=[{a}])\n" + pretty(node.input, indent + 1)
    if isinstance(node, Window):
        fs = ", ".join(f"{_estr(wf)} AS {n}" for wf, n in node.funcs)
        return f"{pad}Window({fs})\n" + pretty(node.input, indent + 1)
    if isinstance(node, Sort):
        k = ", ".join(f"{_estr(e)} {'ASC' if a else 'DESC'}" for e, a in node.keys)
        return f"{pad}Sort({k})\n" + pretty(node.input, indent + 1)
    if isinstance(node, Limit):
        return f"{pad}Limit({node.n} offset={node.offset})\n" + pretty(node.input, indent + 1)
    if isinstance(node, Distinct):
        return f"{pad}Distinct()\n" + pretty(node.input, indent + 1)
    if isinstance(node, Derived):
        return f"{pad}Derived({node.alias})\n" + pretty(node.input, indent + 1)
    if isinstance(node, SetOp):
        kw = "DISTINCT" if node.distinct else "ALL"
        return (
            f"{pad}SetOp({node.op} {kw})\n"
            + pretty(node.left, indent + 1) + "\n"
            + pretty(node.right, indent + 1)
        )
    if isinstance(node, Insert):
        cols = ",".join(node.columns) if node.columns else "*"
        return f"{pad}Insert({node.table} cols={cols} rows={len(node.rows)})"
    if isinstance(node, Delete):
        pred = "" if node.predicate is None else f" WHERE {_estr(node.predicate)}"
        return f"{pad}Delete({node.table}{pred})"
    if isinstance(node, Update):
        sets = ", ".join(f"{c}={_estr(e)}" for c, e in node.assignments)
        pred = "" if node.predicate is None else f" WHERE {_estr(node.predicate)}"
        return f"{pad}Update({node.table} SET {sets}{pred})"
    if isinstance(node, TxnControl):
        return f"{pad}TxnControl({node.kind})"
    return f"{pad}<{type(node).__name__}>"


def _estr_bound(b: FrameBound) -> str:
    if b.kind == "UNBOUNDED_PRECEDING":
        return "UNBOUNDED PRECEDING"
    if b.kind == "UNBOUNDED_FOLLOWING":
        return "UNBOUNDED FOLLOWING"
    if b.kind == "CURRENT_ROW":
        return "CURRENT ROW"
    side = "PRECEDING" if b.kind == "PRECEDING" else "FOLLOWING"
    return f"{b.n} {side}"


def _estr_frame(f: Frame) -> str:
    return f"{f.mode} BETWEEN {_estr_bound(f.start)} AND {_estr_bound(f.end)}"


def _estr(e: Expr) -> str:
    if isinstance(e, Col):
        return e.name
    if isinstance(e, Lit):
        return repr(e.value)
    if isinstance(e, Star):
        return "*"
    if isinstance(e, BinOp):
        return f"({_estr(e.left)} {e.op} {_estr(e.right)})"
    if isinstance(e, And):
        return f"({_estr(e.left)} AND {_estr(e.right)})"
    if isinstance(e, Or):
        return f"({_estr(e.left)} OR {_estr(e.right)})"
    if isinstance(e, Not):
        return f"(NOT {_estr(e.expr)})"
    if isinstance(e, AggFunc):
        s = f"{e.func}({_estr(e.arg)})"
        if e.filter is not None:
            s += f" FILTER (WHERE {_estr(e.filter)})"
        return s
    if isinstance(e, WindowFunc):
        arg = "" if e.arg is None else _estr(e.arg)
        part = ", ".join(_estr(p) for p in e.partition_keys)
        order = ", ".join(f"{_estr(o)} {'ASC' if a else 'DESC'}" for o, a in e.order_keys)
        over = []
        if part:
            over.append(f"PARTITION BY {part}")
        if order:
            over.append(f"ORDER BY {order}")
        if e.frame is not None:
            over.append(_estr_frame(e.frame))
        return f"{e.func}({arg}) OVER ({' '.join(over)})"
    if isinstance(e, IsNull):
        return f"{_estr(e.expr)} IS {'NOT ' if e.negated else ''}NULL"
    if isinstance(e, In):
        vals = ", ".join(_estr(v) for v in e.values)
        return f"{_estr(e.expr)} {'NOT ' if e.negated else ''}IN ({vals})"
    if isinstance(e, Like):
        op = "NOT ILIKE" if (e.negated and not e.case_sensitive) else (
            "NOT LIKE" if e.negated else ("ILIKE" if not e.case_sensitive else "LIKE"))
        return f"{_estr(e.expr)} {op} {_estr(e.pattern)}"
    if isinstance(e, Case):
        whens = " ".join(f"WHEN {_estr(c)} THEN {_estr(v)}" for c, v in e.branches)
        dflt = f" ELSE {_estr(e.default)}" if e.default is not None else ""
        op = f"{_estr(e.operand)} " if e.operand is not None else ""
        return f"CASE {op}{whens}{dflt} END"
    if isinstance(e, Coalesce):
        return f"COALESCE({', '.join(_estr(a) for a in e.args)})"
    if isinstance(e, Cast):
        return f"CAST({_estr(e.expr)} AS {e.dtype})"
    if isinstance(e, Func):
        return f"{e.name}({', '.join(_estr(a) for a in e.args)})"
    return repr(e)