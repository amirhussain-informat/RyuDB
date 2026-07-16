"""SQL -> logical plan translation.

Uses sqlglot to parse SQL, then lowers the AST into RyuDB's relational algebra
(plan.py). Phase 1 supports a deliberate subset:

  SELECT [cols | *] [, expr AS alias]*
  FROM table [AS alias]
  [JOIN table [AS alias] ON t1.k = t2.k [AND ...]]*
  [WHERE predicate]
  [GROUP BY expr [, ...]]
  [ORDER BY col [ASC|DESC] [, ...]]
  [LIMIT n [OFFSET m]]

Not yet supported: subqueries, CTEs, window functions, HAVING, UNION, non-equi
joins, correlated predicates, cross joins. Unsupported constructs raise
NotImplementedError with a clear message.
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp

from .plan import (
    AggFunc,
    And,
    BinOp,
    Col,
    Expr,
    Filter,
    Join,
    Limit,
    Lit,
    Not,
    Or,
    Project,
    Scan,
    Sort,
    Star,
)
from .plan import Aggregate  # noqa: E402 (kept separate for readability)

AGG_FUNCS = {
    exp.Count: "COUNT",
    exp.Sum: "SUM",
    exp.Avg: "AVG",
    exp.Min: "MIN",
    exp.Max: "MAX",
}


class ParseError(ValueError):
    pass


def parse(sql: str, schema: dict[str, list[str]] | None = None) -> object:
    """Parse a single SELECT statement into a logical plan node.

    `schema` (table -> columns) is used to route unqualified join columns to the
    correct side of a join. When omitted, only table-qualified columns route.
    """
    statements = sqlglot.parse(sql)
    if len(statements) != 1:
        raise ParseError(f"expected exactly one statement, got {len(statements)}")
    stmt = statements[0]
    if not isinstance(stmt, exp.Select):
        raise ParseError(f"only SELECT is supported (got {type(stmt).__name__})")
    return _build_select(stmt, schema)


def _build_select(sel: exp.Select, schema: dict[str, list[str]] | None = None):
    if sel.args.get("distinct"):
        raise NotImplementedError("DISTINCT is not supported yet")
    if sel.args.get("having"):
        raise NotImplementedError("HAVING is not supported yet")
    if sel.args.get("qualify") or sel.args.get("windows"):
        raise NotImplementedError("window functions are not supported yet")
    if sel.args.get("connect") or sel.args.get("start"):
        raise NotImplementedError("recursive/hierarchical queries are not supported")

    # --- FROM + JOINs -> base relation ----------------------------------- #
    from_ = sel.args.get("from") or sel.args.get("from_")
    if from_ is None:
        raise NotImplementedError("SELECT without FROM is not supported")
    base_table, base_alias = _table_ref(from_.this)
    plan: object = Scan(base_table)
    aliases: dict[str, str] = {base_alias: base_table}

    for j in sel.args.get("joins", []) or []:
        if j.side and j.side != "" and j.side.upper() not in ("", "INNER"):
            raise NotImplementedError(f"{j.side} joins are not supported (inner only)")
        if j.method and j.method.upper() in ("CROSS", "NATURAL"):
            raise NotImplementedError(f"{j.method} joins are not supported")
        rtbl, ralias = _table_ref(j.this)
        left_tables = {n.table for n in _walk_scans(plan)}
        on_left, on_right, leftover = _join_keys(j, aliases, ralias, rtbl, schema, left_tables)
        plan = Join(
            left=plan,
            right=Scan(rtbl),
            on_left=on_left,
            on_right=on_right,
            how="inner",
        )
        aliases[ralias] = rtbl
        if leftover is not None:
            plan = Filter(plan, leftover)

    # --- WHERE ----------------------------------------------------------- #
    where = sel.args.get("where")
    if where is not None:
        plan = Filter(plan, _expr(where.this))

    # --- projection / aggregate ------------------------------------------ #
    proj_items = list(sel.expressions)
    has_agg = any(_contains_agg(it) for it in proj_items)
    group = sel.args.get("group")
    group_exprs = list(group.expressions) if group else []

    out_names = []
    if group_exprs or has_agg:
        if any(_is_star(it) for it in proj_items):
            raise NotImplementedError("SELECT * with GROUP BY/aggregates is not supported")
        # Build group keys and aggregates from the projection list so that
        # output aliases (e.g. SELECT l_returnflag AS flag ... GROUP BY ...) are
        # preserved for ORDER BY resolution. Phase 1 assumes every GROUP BY
        # expression appears in the SELECT list.
        group_keys: list[tuple[Expr, str]] = []
        aggs: list[tuple[AggFunc, str]] = []
        for it in proj_items:
            e, alias = _proj_item(it)
            if isinstance(e, AggFunc):
                aggs.append((e, alias))
            else:
                group_keys.append((e, alias))
            out_names.append(alias)
        plan = Aggregate(plan, group_keys=group_keys, aggs=aggs)
    else:
        if len(proj_items) == 1 and _is_star(proj_items[0]):
            plan = _maybe_project_star(plan)  # no-op pass-through
        else:
            items = []
            for it in proj_items:
                e, alias = _proj_item(it)
                items.append((e, alias))
                out_names.append(alias)
            plan = Project(plan, items=items)

    # --- ORDER BY -------------------------------------------------------- #
    order = sel.args.get("order")
    if order is not None:
        keys = []
        for o in order.expressions:
            if not isinstance(o, exp.Ordered):
                raise NotImplementedError(f"unsupported ORDER BY term: {o}")
            e = _expr(o.this)
            if not isinstance(e, Col):
                raise NotImplementedError("ORDER BY only supports column references")
            # resolve to an output column name (alias) when possible
            name = e.name
            if name not in out_names and len(out_names) == len(set(out_names)):
                # fall back to the raw column name
                pass
            keys.append((Col(name), not o.args.get("desc", False)))
        plan = Sort(plan, keys=keys)

    # --- LIMIT / OFFSET -------------------------------------------------- #
    limit = sel.args.get("limit")
    if limit is not None:
        n = int(_literal_value(_limit_value(limit)))
        off = 0
        offset = sel.args.get("offset")
        if offset is not None:
            off = int(_literal_value(_limit_value(offset)))
        plan = Limit(plan, n=n, offset=off)

    return plan


# --------------------------------------------------------------------------- #
# Expression translation
# --------------------------------------------------------------------------- #


def _expr(node) -> Expr:
    if isinstance(node, exp.Column):
        if isinstance(node.this, exp.Star):
            return Star()
        return Col(node.name)
    if isinstance(node, exp.Identifier):
        return Col(node.name)
    if isinstance(node, exp.Star):
        return Star()
    if isinstance(node, exp.Literal):
        return Lit(_literal_value(node), "str" if node.is_string else _infer_num(node))
    if isinstance(node, exp.Boolean):
        return Lit(bool(node.this), "bool")
    if isinstance(node, exp.Null):
        return Lit(None, "null")
    if isinstance(node, exp.Paren):
        return _expr(node.this)
    if isinstance(node, exp.Cast):
        target = node.to.name.upper() if node.to else None
        inner = node.this
        if isinstance(inner, exp.Literal):
            return Lit(_literal_value(inner), target)
        # general cast of an expression: represent as BinOp-like? keep as inner
        # with a dtype hint is not enough; unsupported for Phase 1.
        raise NotImplementedError(f"cast of {type(inner).__name__} not supported")
    if isinstance(node, exp.And):
        return And(_expr(node.this), _expr(node.expression))
    if isinstance(node, exp.Or):
        return Or(_expr(node.this), _expr(node.expression))
    if isinstance(node, exp.Not):
        return Not(_expr(node.this))
    if isinstance(node, exp.Binary):
        op = _binop_symbol(node)
        return BinOp(op, _expr(node.this), _expr(node.expression))
    if isinstance(node, exp.AggFunc):
        if isinstance(node, (exp.Count, exp.Sum, exp.Avg, exp.Min, exp.Max)):
            arg = _expr(node.this) if node.this is not None else Star()
            return AggFunc(AGG_FUNCS[type(node)], arg)
    raise NotImplementedError(f"unsupported expression: {type(node).__name__}")


_BINOP = {
    exp.EQ: "=", exp.NEQ: "!=", exp.GT: ">", exp.LT: "<",
    exp.GTE: ">=", exp.LTE: "<=", exp.Add: "+", exp.Sub: "-",
    exp.Mul: "*", exp.Div: "/", exp.Mod: "%",
}


def _binop_symbol(node: exp.Binary) -> str:
    sym = _BINOP.get(type(node))
    if sym is None:
        raise NotImplementedError(f"unsupported operator: {type(node).__name__}")
    return sym


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _table_ref(node) -> tuple[str, str]:
    if isinstance(node, exp.Subquery):
        raise NotImplementedError("subqueries in FROM are not supported")
    if not isinstance(node, exp.Table):
        raise ParseError(f"expected a table reference, got {type(node).__name__}")
    name = node.name
    if not name:
        raise ParseError("table reference has no name")
    alias = node.alias or name
    return name, alias


def _join_keys(j, left_aliases, ralias, rtable, schema, left_tables):
    """Extract equi-join keys. Returns (on_left, on_right, leftover_predicate).

    `on_left` are columns from the already-built left relation, `on_right` from
    the table being joined. When `schema` is provided, unqualified columns are
    routed by membership; otherwise we fall back to table qualifiers.
    """
    on = j.args.get("on")
    using = j.args.get("using")
    on_left: list[str] = []
    on_right: list[str] = []
    leftover_parts: list[Expr] = []

    if using:
        for u in using:
            col = u.name if isinstance(u, exp.Column) else u.alias_or_name
            on_left.append(col)
            on_right.append(col)
        return on_left, on_right, None

    if on is None:
        raise NotImplementedError("JOIN requires an ON or USING clause")

    for conj in _flatten_and(on):
        if isinstance(conj, exp.EQ) and isinstance(conj.this, exp.Column) \
                and isinstance(conj.expression, exp.Column):
            lcol, rcol = conj.this, conj.expression
            left_name, right_name = _route_join_cols(lcol, rcol, rtable, schema, left_tables)
            on_left.append(left_name)
            on_right.append(right_name)
        else:
            leftover_parts.append(_expr(conj))

    if not on_left:
        raise NotImplementedError("only equi-joins are supported")
    leftover = None
    if leftover_parts:
        leftover = _conjoin(leftover_parts)
    return on_left, on_right, leftover


def _route_join_cols(lcol, rcol, rtable, schema, left_tables) -> tuple[str, str]:
    """Return (left_column_name, right_column_name) for an equi-join predicate."""
    if schema:
        right_cols = set(schema.get(rtable, []))
        left_cols = set()
        for t in left_tables:
            left_cols |= set(schema.get(t, []))
        l_in_right = lcol.name in right_cols
        r_in_right = rcol.name in right_cols
        l_in_left = lcol.name in left_cols
        r_in_left = rcol.name in left_cols
        if r_in_right and not r_in_left:
            return lcol.name, rcol.name
        if l_in_right and not l_in_left:
            return rcol.name, lcol.name
        if r_in_right:
            return lcol.name, rcol.name
        if l_in_right:
            return rcol.name, lcol.name
    # Fallback: use table qualifiers if present.
    if lcol.table == rtable:
        return rcol.name, lcol.name
    return lcol.name, rcol.name


def _walk_scans(plan):
    from .plan import Scan, walk
    return [n for n in walk(plan) if isinstance(n, Scan)]


def _flatten_and(node):
    if isinstance(node, exp.And):
        yield from _flatten_and(node.this)
        yield from _flatten_and(node.expression)
    else:
        yield node


def _conjoin(parts: list[Expr]) -> Expr:
    acc = parts[0]
    for p in parts[1:]:
        acc = And(acc, p)
    return acc


def _contains_agg(node) -> bool:
    return node.find(exp.AggFunc) is not None


def _is_star(node) -> bool:
    return isinstance(node, exp.Star) or (
        isinstance(node, exp.Column) and isinstance(node.this, exp.Star)
    )


def _maybe_project_star(plan):
    """SELECT * : pass through with no explicit projection."""
    return plan


def _proj_item(node) -> tuple[Expr, str]:
    if isinstance(node, exp.Alias):
        return _expr(node.this), node.alias
    if _is_star(node):
        raise NotImplementedError("'*' mixed with other expressions is not supported")
    e = _expr(node)
    return e, _output_name(node)


def _output_name(node) -> str:
    if isinstance(node, exp.Alias):
        return node.alias
    if isinstance(node, exp.Column):
        return node.name
    if isinstance(node, exp.AggFunc):
        return f"{AGG_FUNCS.get(type(node), type(node).__name__).lower()}_{_describe(node.this)}"
    return node.alias_or_name


def _describe(node) -> str:
    if node is None or isinstance(node, exp.Star):
        return "star"
    if isinstance(node, exp.Column):
        return node.name
    return "expr"


def _limit_value(node):
    """LIMIT/OFFSET store their value in `.expression` in sqlglot 28+."""
    expr = node.expression if node.expression is not None else node.this
    if expr is None:
        raise ParseError("LIMIT/OFFSET value is missing")
    return expr


def _literal_value(node):
    if isinstance(node, exp.Literal):
        return node.this
    if isinstance(node, exp.Boolean):
        return bool(node.this)
    if isinstance(node, exp.Null):
        return None
    if isinstance(node, exp.Paren):
        return _literal_value(node.this)
    # some literals arrive wrapped (e.g. in Limit)
    if hasattr(node, "this") and node.this is not None:
        return _literal_value(node.this)
    raise ParseError(f"expected a literal, got {type(node).__name__}")


def _infer_num(node: exp.Literal) -> str | None:
    v = node.this
    try:
        int(v)
        return "int"
    except (ValueError, TypeError):
        try:
            float(v)
            return "float"
        except (ValueError, TypeError):
            return None