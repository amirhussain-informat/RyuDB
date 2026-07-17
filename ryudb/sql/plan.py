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

    def columns(self) -> set[str]:
        return self.arg.columns() if not isinstance(self.arg, Star) else set()


# --------------------------------------------------------------------------- #
# Plan nodes
# --------------------------------------------------------------------------- #


@dataclass
class Scan:
    table: str
    # None => all columns; otherwise the projected set (pushed down by optimizer)
    columns: set[str] | None = None


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
    how: str = "inner"   # inner (Phase 1)


@dataclass
class Aggregate:
    input: "PlanNode"
    group_keys: list[tuple[Expr, str]]  # (expr, output_name)
    aggs: list[tuple[AggFunc, str]]     # (agg, output_name)


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
class Insert:
    """Write node: append literal value rows to a table's delta (Phase 2 step 3).

    A leaf (no ``input``), like ``Scan``. ``columns`` is the user-supplied column
    list (``None`` => INSERT without a column list => use the table's catalog
    column order); ``rows`` is one ``list[Expr]`` per value row, each cell a
    ``Lit`` lowered by the parser. The executor resolves the full schema, fills
    DEFAULTs, enforces NOT NULL, builds a typed cuDF batch, and appends it to the
    delta. PK/UNIQUE enforcement is deferred (step 4+).
    """

    table: str
    columns: list[str] | None = None
    rows: list[list[Expr]] = field(default_factory=list)


PlanNode = Scan | Filter | Project | Join | Aggregate | Sort | Limit | Insert


def walk(node: PlanNode):
    """Yield every plan node in the tree (pre-order)."""
    yield node
    for child in children(node):
        yield from walk(child)


def children(node: PlanNode) -> list[PlanNode]:
    if isinstance(node, (Filter, Project, Sort)):
        return [node.input]
    if isinstance(node, (Join,)):
        return [node.left, node.right]
    if isinstance(node, Aggregate):
        return [node.input]
    if isinstance(node, Limit):
        return [node.input]
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
    if isinstance(node, Join):
        return []
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
        return (
            f"{pad}Join({node.how} on {on})\n"
            + pretty(node.left, indent + 1) + "\n"
            + pretty(node.right, indent + 1)
        )
    if isinstance(node, Aggregate):
        g = ", ".join(n for _, n in node.group_keys)
        a = ", ".join(f"{af.func}({_estr(af.arg)}) AS {n}" for af, n in node.aggs)
        return f"{pad}Aggregate(group=[{g}] aggs=[{a}])\n" + pretty(node.input, indent + 1)
    if isinstance(node, Sort):
        k = ", ".join(f"{_estr(e)} {'ASC' if a else 'DESC'}" for e, a in node.keys)
        return f"{pad}Sort({k})\n" + pretty(node.input, indent + 1)
    if isinstance(node, Limit):
        return f"{pad}Limit({node.n} offset={node.offset})\n" + pretty(node.input, indent + 1)
    if isinstance(node, Insert):
        cols = ",".join(node.columns) if node.columns else "*"
        return f"{pad}Insert({node.table} cols={cols} rows={len(node.rows)})"
    return f"{pad}<{type(node).__name__}>"


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
        return f"{e.func}({_estr(e.arg)})"
    return repr(e)