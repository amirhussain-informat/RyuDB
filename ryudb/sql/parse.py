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

Supported joins: INNER, LEFT/RIGHT/FULL OUTER, CROSS, and NATURAL (lowered to
an equi-join on the intersection of common column names). A non-equi ON residual
alongside equi keys (e.g. ``ON a.k=b.k AND b.x>10``) is kept on the Join as an
``on_predicate`` and applied *inside* the join so outer-join semantics survive
(it filters only matched rows, never the null-padded unmatched rows).

Not yet supported: subqueries, CTEs, window functions, HAVING, UNION, pure
non-equi/theta joins (an ON with no equi key), correlated predicates,
table-qualified column output (bare ``Col`` only). Unsupported constructs raise
NotImplementedError with a clear message.
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp

from .plan import (
    AggFunc,
    And,
    BinOp,
    Case,
    Cast,
    Coalesce,
    Col,
    Delete,
    Expr,
    Filter,
    In,
    Insert,
    IsNull,
    Join,
    Like,
    Limit,
    Lit,
    Not,
    Or,
    Project,
    Scan,
    Sort,
    Star,
    TxnControl,
    Update,
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
    """Parse a single SELECT / INSERT / transaction-control statement into a
    logical plan node.

    `schema` (table -> columns) is used to route unqualified join columns to the
    correct side of a join. When omitted, only table-qualified columns route.

    Transaction control (BEGIN/COMMIT/ROLLBACK) lowers to a ``TxnControl`` leaf.
    Snapshot/restore (``CREATE SNAPSHOT``/``RESTORE TO SNAPSHOT``) are non-standard
    SQL and are handled by a regex pre-sniff in ``Engine.sql`` before this is
    called -- they never reach the parser. ``START TRANSACTION`` does NOT parse in
    sqlglot's default dialect; use ``BEGIN`` (optionally ``BEGIN TRANSACTION`` or
    ``BEGIN WORK``).
    """
    statements = sqlglot.parse(sql)
    if len(statements) != 1:
        raise ParseError(f"expected exactly one statement, got {len(statements)}")
    stmt = statements[0]
    if isinstance(stmt, exp.Select):
        return _build_select(stmt, schema)
    if isinstance(stmt, exp.Insert):
        return _build_insert(stmt)
    if isinstance(stmt, exp.Delete):
        return _build_delete(stmt)
    if isinstance(stmt, exp.Update):
        return _build_update(stmt)
    if isinstance(stmt, exp.Transaction):
        # sqlglot lowers BEGIN[/TRANSACTION/WORK/DEFERRED] to exp.Transaction;
        # this/modes/mark are ignored (no nested txns, no SAVEPOINTs yet).
        return TxnControl("begin")
    if isinstance(stmt, exp.Commit):
        if stmt.args.get("chain"):
            raise NotImplementedError("COMMIT AND CHAIN is not supported")
        return TxnControl("commit")
    if isinstance(stmt, exp.Rollback):
        if stmt.args.get("savepoint"):
            raise NotImplementedError("ROLLBACK TO SAVEPOINT is not supported")
        return TxnControl("rollback")
    raise ParseError(f"only SELECT/INSERT/UPDATE/DELETE/BEGIN/COMMIT/ROLLBACK is supported "
                     f"(got {type(stmt).__name__})")


def _build_insert(stmt: exp.Insert) -> Insert:
    """Lower ``INSERT INTO t [(cols)] VALUES (...),(...)`` into an Insert node.

    sqlglot wraps the target in ``exp.Schema`` when a column list is present and
    leaves it as a bare ``exp.Table`` otherwise; the value rows live in
    ``stmt.expression`` as an ``exp.Values`` (one ``exp.Tuple`` per row). Each cell
    is lowered with ``_expr`` to a ``Lit`` (so ``date '...'`` casts and NULLs are
    handled by the existing expression machinery). Unsupported INSERT variants
    (ON CONFLICT, RETURNING, WHERE, INSERT ... SELECT, ...) raise.
    """
    for arg in ("conflict", "returning", "where", "partition", "ignore",
                "overwrite", "alternative", "source"):
        if stmt.args.get(arg) is not None:
            raise NotImplementedError(f"INSERT with {arg.upper()} is not supported yet")
    target = stmt.this.this if isinstance(stmt.this, exp.Schema) else stmt.this
    if not isinstance(target, exp.Table):
        raise ParseError("INSERT target is not a table")
    table = target.name
    if not table:
        raise ParseError("INSERT target has no table name")
    cols = None
    if isinstance(stmt.this, exp.Schema):
        cols = [c.name for c in stmt.this.expressions]
        if not cols or any(not c for c in cols):
            raise ParseError("INSERT column list is empty or malformed")
        if len(set(cols)) != len(cols):
            raise ParseError("INSERT column list has duplicates")
    values = stmt.expression
    if not isinstance(values, exp.Values):
        raise NotImplementedError("only INSERT ... VALUES is supported")
    rows = [[_expr(cell) for cell in row.expressions] for row in values.expressions]
    if cols is not None:
        for i, row in enumerate(rows):
            if len(row) != len(cols):
                raise ParseError(
                    f"INSERT row {i} has {len(row)} values for {len(cols)} columns"
                )
    if not rows:
        raise ParseError("INSERT has no value rows")
    return Insert(table=table, columns=cols, rows=rows)


def _build_delete(stmt: exp.Delete) -> Delete:
    """Lower ``DELETE FROM t [WHERE pred]`` into a Delete node (step 9).

    sqlglot puts the target table in ``stmt.this`` (an ``exp.Table``; ``.name``
    is the table name) and the optional predicate in ``stmt.args["where"]`` (an
    ``exp.Where`` whose ``.this`` is the predicate expression, or ``None`` when
    no WHERE was given). Only the bare form is supported: USING/RETURNING/ORDER
    BY/LIMIT and multi-table DELETEs raise (sqlglot ``exp.Delete`` args keys:
    ``tables, this, using, cluster, where, returning, order, limit``).
    """
    for arg in ("using", "cluster", "returning", "order", "limit", "tables"):
        if stmt.args.get(arg):
            raise NotImplementedError(f"DELETE with {arg.upper()} is not supported yet")
    target = stmt.this
    if not isinstance(target, exp.Table):
        raise ParseError("DELETE target is not a table")
    table = target.name
    if not table:
        raise ParseError("DELETE target has no table name")
    where = stmt.args.get("where")
    predicate = _expr(where.this) if where is not None else None
    return Delete(table=table, predicate=predicate)


def _build_update(stmt: exp.Update) -> Update:
    """Lower ``UPDATE t SET col = expr [, ...] [WHERE pred]`` into an Update node.

    sqlglot puts the target table in ``stmt.this`` (an ``exp.Table``; ``.name`` is
    the table name), the SET list in ``stmt.expressions`` (a list of bare ``exp.EQ``
    -- one per ``col = expr``; ``eq.this`` is an ``exp.Column`` whose ``.name`` is
    the target column, ``eq.expression`` is the value expression), and the optional
    predicate in ``stmt.args["where"]`` (an ``exp.Where`` whose ``.this`` is the
    predicate, or ``None``). Only the bare single-table form is supported:
    FROM/JOIN/RETURNING/ORDER BY/LIMIT/WITH/OPTIONS raise (sqlglot ``exp.Update``
    ``arg_types``: ``with_, this, expressions, from_, where, returning, order,
    limit, options`` -- there is no ``joins`` key, but JOIN is rejected
    defensively).
    """
    for arg in ("from_", "returning", "order", "limit", "with_", "options"):
        if stmt.args.get(arg):
            raise NotImplementedError(f"UPDATE with {arg.upper()} is not supported yet")
    if stmt.args.get("joins"):
        raise NotImplementedError("UPDATE with JOIN is not supported yet")
    target = stmt.this
    if not isinstance(target, exp.Table):
        raise ParseError("UPDATE target is not a table")
    table = target.name
    if not table:
        raise ParseError("UPDATE target has no table name")
    sets = stmt.expressions or []
    assignments: list[tuple[str, Expr]] = []
    for eq in sets:
        if not isinstance(eq, exp.EQ) or not isinstance(eq.this, exp.Column):
            raise ParseError(
                f"UPDATE SET clause must be `col = expr` (got {type(eq).__name__})"
            )
        assignments.append((eq.this.name, _expr(eq.expression)))
    if not assignments:
        raise ParseError("UPDATE has no SET assignments")
    cols = [c for c, _ in assignments]
    if len(set(cols)) != len(cols):
        raise ParseError(f"UPDATE SET has duplicate columns: {cols}")
    where = stmt.args.get("where")
    predicate = _expr(where.this) if where is not None else None
    return Update(table=table, assignments=assignments, predicate=predicate)


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
        rtbl, ralias = _table_ref(j.this)
        left_tables = {n.table for n in _walk_scans(plan)}
        how, on_left, on_right, on_predicate = _join_spec(
            j, aliases, ralias, rtbl, schema, left_tables
        )
        plan = Join(
            left=plan,
            right=Scan(rtbl),
            on_left=on_left,
            on_right=on_right,
            how=how,
            on_predicate=on_predicate,
        )
        aliases[ralias] = rtbl

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
        return _cast(node)
    if isinstance(node, exp.And):
        return And(_expr(node.this), _expr(node.expression))
    if isinstance(node, exp.Or):
        return Or(_expr(node.this), _expr(node.expression))
    if isinstance(node, exp.Not):
        return _not(node)
    # These are all exp.Binary subclasses in sqlglot, so they must be matched
    # before the generic exp.Binary branch below.
    if isinstance(node, exp.Is):
        if isinstance(node.expression, exp.Null):
            return IsNull(_expr(node.this), negated=False)
        raise NotImplementedError("only IS NULL / IS NOT NULL is supported")
    if isinstance(node, exp.In):
        return _in(node, negated=False)
    if isinstance(node, exp.Between):
        return _between(node, negated=False)
    if isinstance(node, (exp.Like, exp.ILike)):
        return Like(_expr(node.this), _expr(node.expression),
                    negated=False, case_sensitive=isinstance(node, exp.Like))
    if isinstance(node, exp.Coalesce):
        args = [_expr(node.this)] + [_expr(v) for v in node.expressions]
        return Coalesce(tuple(args))
    if isinstance(node, exp.Case):
        return _case(node)
    if isinstance(node, exp.Binary):
        op = _binop_symbol(node)
        return BinOp(op, _expr(node.this), _expr(node.expression))
    if isinstance(node, exp.AggFunc):
        if isinstance(node, (exp.Count, exp.Sum, exp.Avg, exp.Min, exp.Max)):
            arg = _expr(node.this) if node.this is not None else Star()
            return AggFunc(AGG_FUNCS[type(node)], arg)
    raise NotImplementedError(f"unsupported expression: {type(node).__name__}")


# --------------------------------------------------------------------------- #
# Predicate / expression helpers (IS NULL, IN, BETWEEN, LIKE, CASE, COALESCE,
# CAST). BETWEEN lowers to AND/OR of comparisons (no plan node); the NOT
# variants set a `negated` flag on the inner Expr so three-valued logic on a
# NULL operand stays correct (a Not wrapper would invert NA wrongly).
# --------------------------------------------------------------------------- #


def _not(node) -> Expr:
    """Lower ``NOT x``. NOT IN / NOT BETWEEN / NOT LIKE / IS NOT NULL fold the
    negation into the inner Expr; anything else stays a ``Not`` wrapper."""
    inner = node.this
    if isinstance(inner, exp.In) and inner.args.get("query") is None:
        return _in(inner, negated=True)
    if isinstance(inner, exp.Between):
        return _between(inner, negated=True)
    if isinstance(inner, (exp.Like, exp.ILike)):
        return Like(_expr(inner.this), _expr(inner.expression),
                    negated=True, case_sensitive=isinstance(inner, exp.Like))
    if isinstance(inner, exp.Is) and isinstance(inner.expression, exp.Null):
        return IsNull(_expr(inner.this), negated=True)
    return Not(_expr(inner))


def _in(node, negated: bool) -> Expr:
    if node.args.get("query") is not None:
        raise NotImplementedError("IN with a subquery is not supported yet")
    values = tuple(_expr(v) for v in node.expressions)
    if not values:
        raise ParseError("IN list is empty")
    return In(_expr(node.this), values, negated=negated)


def _between(node, negated: bool) -> Expr:
    e = _expr(node.this)
    lo = _expr(node.args["low"])
    hi = _expr(node.args["high"])
    if not negated:
        return And(BinOp(">=", e, lo), BinOp("<=", e, hi))
    return Or(BinOp("<", e, lo), BinOp(">", e, hi))


def _cast(node) -> Expr:
    target = node.to.name.upper() if node.to else None
    inner = node.this
    if isinstance(inner, exp.Literal):
        # Literal cast stays a typed Lit (preserves the date-literal path).
        return Lit(_literal_value(inner), target)
    return Cast(_expr(inner), _sqlglot_type_tag(node.to))


def _case(node) -> Expr:
    operand = _expr(node.this) if node.this is not None else None
    ifs = node.args.get("ifs") or []
    branches: list[tuple[Expr, Expr]] = []
    for iff in ifs:
        cond = _expr(iff.this)
        if operand is not None:
            # Simple CASE: WHEN v THEN ... -> (operand = v).
            cond = BinOp("=", operand, cond)
        then = _expr(iff.args.get("true"))
        branches.append((cond, then))
    default = _expr(node.args.get("default")) if node.args.get("default") is not None else None
    return Case(operand, tuple(branches), default)


_SQLGLOT_TYPE_TAG = {
    "TINYINT": "int", "SMALLINT": "int", "INT": "int", "INTEGER": "int",
    "BIGINT": "int",
    "FLOAT": "float", "DOUBLE": "float", "REAL": "float",
    "DECIMAL": "float", "NUMERIC": "float",
    "VARCHAR": "str", "CHAR": "str", "TEXT": "str", "STRING": "str",
    "BOOLEAN": "bool", "BOOL": "bool",
    "DATE": "date",
    "TIMESTAMP": "timestamp", "DATETIME": "timestamp",
}


def _sqlglot_type_tag(dt) -> str:
    name = dt.this.name.upper() if dt is not None and dt.this is not None else ""
    tag = _SQLGLOT_TYPE_TAG.get(name)
    if tag is None:
        raise NotImplementedError(f"CAST to {name or 'unknown type'} is not supported")
    return tag


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


def _join_spec(j, left_aliases, ralias, rtable, schema, left_tables):
    """Lower a sqlglot ``Join`` into (how, on_left, on_right, on_predicate).

    ``how`` is the cuDF merge kind (inner/left/right/full/cross). ``on_left`` /
    ``on_right`` are the equi-join key column names (empty for a pure CROSS).
    ``on_predicate`` is the non-equi ON residual the executor applies *inside* the
    join with outer-correct semantics (``None`` when there is no residual).

    Keeping the residual on the Join -- not wrapped in a top Filter -- is what
    preserves ON-vs-WHERE for outer joins: a WHERE Filter would wrongly drop the
    null-padded unmatched rows that an outer join must retain. ``Filter`` is still
    used for the query's actual WHERE clause (see ``_build_select``).

    sqlglot encodes the join shape across three attrs: ``method`` (NATURAL),
    ``side`` (LEFT/RIGHT/FULL), ``kind`` (INNER/OUTER/CROSS). CROSS lives in
    ``kind``, not ``method``; NATURAL lives in ``method`` and may combine with a
    ``side`` (NATURAL LEFT/RIGHT/FULL).
    """
    method = (j.method or "").upper()
    side = (j.side or "").upper()
    kind = (j.kind or "").upper()

    if method == "NATURAL":
        if schema is None:
            raise NotImplementedError("NATURAL joins require a table schema")
        how = _side_how(side, kind)
        on_left, on_right = _common_columns(left_tables, rtable, schema)
        if not on_left:
            # SQL standard: a NATURAL join over no common columns is a cross join.
            return "cross", [], [], None
        return how, on_left, on_right, None

    if kind == "CROSS":
        on = j.args.get("on")
        if on is None:
            return "cross", [], [], None
        # `CROSS JOIN ... ON ...` is semantically an inner join on that ON; lower
        # it through _join_keys so equi columns route correctly (a post-cross
        # filter would collide on bare column names that appear on both sides).
        on_left, on_right, leftover = _join_keys(
            j, left_aliases, ralias, rtable, schema, left_tables
        )
        return "inner", on_left, on_right, leftover

    how = _side_how(side, kind)
    on_left, on_right, leftover = _join_keys(
        j, left_aliases, ralias, rtable, schema, left_tables
    )
    return how, on_left, on_right, leftover


def _side_how(side: str, kind: str) -> str:
    """Map sqlglot ``side``/``kind`` to a cuDF merge ``how``."""
    if side == "LEFT":
        return "left"
    if side == "RIGHT":
        return "right"
    if side == "FULL":
        return "full"
    if side == "" and kind == "OUTER":
        return "full"  # bare `OUTER JOIN` -> FULL OUTER (standard reading)
    return "inner"


def _common_columns(left_tables, rtable, schema) -> tuple[list[str], list[str]]:
    """Sorted intersection of the right table's columns with the union of the
    left relation's tables' columns -- the NATURAL-join key set (same name on
    both sides). Returns (keys, keys) since NATURAL keys share a name."""
    right_cols = set(schema.get(rtable, []))
    left_cols: set[str] = set()
    for t in left_tables:
        left_cols |= set(schema.get(t, []))
    common = sorted(left_cols & right_cols)
    return common, common


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