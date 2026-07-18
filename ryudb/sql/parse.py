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

Not yet supported: CTEs, window functions, HAVING,
INTERSECT ALL / EXCEPT ALL (multiset variants), pure non-equi/theta joins (an
ON with no equi key), correlated predicates, table-qualified column output
(bare ``Col`` only). Unsupported constructs raise NotImplementedError with a
clear message.

Uncorrelated ``x IN (SELECT ...)`` / ``x NOT IN (SELECT ...)`` in WHERE lower to
semi/anti joins (see ``_apply_where_subqueries``): the subquery becomes the
Join's right child, a normal subtree the optimizer recurses into. IN under OR is
deferred. Uncorrelated scalar subqueries (single-row aggregates such as
``SELECT COUNT(*)/MAX(...) FROM ...``) and ``EXISTS (SELECT ...)`` -- anywhere a
boolean/value is legal in WHERE or projection -- are flattened into cross-joins
of a 1-row relation onto the outer plan (see ``_flatten_outer_subqueries``);
NOT EXISTS falls out via ``Not``.

Correlated subqueries with a single equi-correlation ``inner.col = outer.col`` are
decorrelated at parse time into uncorrelated joins (Phase E-3, no outer-scope
binding needed): correlated ``EXISTS``/``NOT EXISTS`` -> semi/anti join on the
correlation key; correlated scalar (single-row aggregate) -> LEFT join onto a
grouped aggregate. Multi-equi / non-equi correlation, correlated IN, and outer
refs outside the WHERE are deferred.

Set operators (UNION [ALL] / INTERSECT / EXCEPT) compose two SELECTs into a
``SetOp`` node (see ``_build_query`` / ``_build_setop``); DISTINCT set ops use
cuDF ``drop_duplicates``/``merge`` which are NULL-safe.

Date/time functions (EXTRACT, YEAR/MONTH/DAY/HOUR/MINUTE/SECOND/DAYOFWEEK/
DAYOFYEAR, DATE_TRUNC, DATEDIFF, date +/- INTERVAL, DAYNAME/MONTHNAME, LAST_DAY,
STRFTIME, CURRENT_DATE/CURRENT_TIMESTAMP/NOW) lower to the generic ``Func`` node
(see ``_SCALAR_FUNC_BUILDERS`` and ``_interval_arith``); the cuDF lowering lives
in ``ops._func``. ``SELECT`` without FROM is still unsupported, so CURRENT_DATE
etc. require a FROM table (the scalar broadcasts per row).
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
    Derived,
    Distinct,
    Expr,
    Filter,
    Func,
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
    SetOp,
    Sort,
    Star,
    TxnControl,
    Update,
    Window,
    WindowFunc,
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
    if isinstance(stmt, (exp.Select, exp.Union, exp.Intersect, exp.Except, exp.Subquery)):
        return _build_query(stmt, schema)
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


def _build_select(sel: exp.Select, schema: dict[str, list[str]] | None = None, ctes=None):
    if sel.args.get("qualify") or sel.args.get("windows"):
        raise NotImplementedError("window functions are not supported yet")
    if sel.args.get("connect") or sel.args.get("start"):
        raise NotImplementedError("recursive/hierarchical queries are not supported")
    having = sel.args.get("having")

    # --- WITH / CTEs (Phase F-2c) ---------------------------------------- #
    # Build each CTE's subplan eagerly, in order, into a *copy* of the incoming
    # ``ctes`` map (so this scope's CTEs don't leak to an outer scope). A later CTE
    # sees the earlier ones (forward / self references are not in the map and fall
    # through to a ``Scan`` -> execution error -- an acceptable limitation). Each
    # body is built once via ``_build_query`` and shared by all references; the
    # optimizer rebuilds rather than mutates, so multiple ``Derived`` references
    # each get their own optimized copy during the single ``optimize()`` pass.
    ctes = dict(ctes or {})
    with_ = sel.args.get("with_") or sel.args.get("with")
    if with_ is not None:
        if with_.args.get("recursive"):
            raise NotImplementedError("recursive CTEs (WITH RECURSIVE) are not supported")
        for cte in with_.expressions:
            # A column-list CTE (``name(col,...) AS (...)``) renames/restricts the
            # CTE's output columns; sqlglot stores the list on the CTE's
            # TableAlias (``alias.columns``), not ``key_expressions``. Not supported
            # in the flat-column model (the CTE's output names come from its body's
            # projection; a column list would shadow them), so reject explicitly.
            if cte.args["alias"].args.get("columns"):
                raise NotImplementedError(
                    "column-list CTEs (name(col,...) AS ...) are not supported"
                )
            name = cte.args["alias"].name
            ctes[name] = _build_query(cte.args["this"], schema, ctes)

    # --- FROM + JOINs -> base relation ----------------------------------- #
    from_ = sel.args.get("from") or sel.args.get("from_")
    if from_ is None:
        raise NotImplementedError("SELECT without FROM is not supported")
    base_source, base_alias, base_name = _table_ref(from_.this, schema, ctes)
    plan: object = base_source
    aliases: dict[str, str] = {base_alias: base_name}

    for j in sel.args.get("joins", []) or []:
        r_source, ralias, rname = _table_ref(j.this, schema, ctes)
        left_tables = {n.table for n in _walk_scans(plan)}
        how, on_left, on_right, on_predicate = _join_spec(
            j, aliases, ralias, rname, schema, left_tables
        )
        plan = Join(
            left=plan,
            right=r_source,
            on_left=on_left,
            on_right=on_right,
            how=how,
            on_predicate=on_predicate,
        )
        aliases[ralias] = rname

    # --- uncorrelated scalar / EXISTS subqueries -> cross-join broadcast -- #
    # Flatten each uncorrelated scalar subquery (a single-row aggregate, e.g.
    # ``SELECT COUNT(*)/MAX(...) FROM ...``) and each ``EXISTS (SELECT ...)`` that
    # appear in THIS select's WHERE or projection into a cross-join of a 1-row
    # relation onto ``plan``, replacing the subquery node with a bare Column
    # reference to the broadcast column. Done before the WHERE block so the
    # replaced Columns flow into the residual Filter / IN-subquery handling
    # normally. Correlated subqueries are rejected here (outer aliases are in
    # scope); IN/NOT IN subqueries are left untouched (handled below). See
    # ``_flatten_outer_subqueries``. Non-aggregate and GROUP BY scalar subqueries
    # are deferred (no 1-row guarantee).
    plan = _flatten_outer_subqueries(plan, sel, schema, aliases, ctes)

    # --- WHERE ----------------------------------------------------------- #
    where = sel.args.get("where")
    if where is not None:
        # Uncorrelated ``x IN (SELECT ...)`` / ``x NOT IN (SELECT ...)`` conjuncts
        # fold into semi/anti joins on `plan`; the remaining conjuncts stay as a
        # Filter. See _apply_where_subqueries.
        plan, residual = _apply_where_subqueries(plan, where.this, schema, aliases, ctes)
        if residual is not None:
            plan = Filter(plan, residual)

    # --- projection / aggregate ------------------------------------------ #
    proj_items = list(sel.expressions)
    # Window functions are detected and lowered BEFORE the aggregate check:
    # sqlglot's aggregate-window funcs (SUM/LAG/RANK/.. OVER) are ``exp.AggFunc``
    # subclasses, so ``_contains_agg`` would otherwise misroute them into the
    # Aggregate branch (and ``ROW_NUMBER`` is not an AggFunc at all). Each
    # ``exp.Window`` is rewritten into a ``Window`` plan node and replaced in the
    # projection item by a bare Column ref to its output; the rewritten items then
    # flow into the normal Project/Aggregate building.
    if any(_contains_window(it) for it in proj_items):
        if sel.args.get("group") or sel.args.get("having"):
            raise NotImplementedError(
                "window functions with GROUP BY / HAVING are not supported yet"
            )
        plan = _build_window(plan, proj_items, schema)
    else:
        # A window in GROUP BY / HAVING (not the projection) is malformed for F-1
        # and would otherwise be silently dropped (the GROUP BY builder reads keys
        # from the projection list). Catch it explicitly.
        _reject_window_outside_projection(sel)
    has_agg = any(_contains_agg(it) for it in proj_items)
    group = sel.args.get("group")
    group_exprs = list(group.expressions) if group else []

    if group_exprs or has_agg or having is not None:
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
        if having is not None and not group_exprs and group_keys:
            # HAVING with no explicit GROUP BY and a non-aggregate SELECT column
            # is invalid SQL (the column must be grouped or aggregated); DuckDB
            # rejects it too. Don't implicitly group by the stray columns.
            raise NotImplementedError(
                "HAVING without GROUP BY requires a pure-aggregate SELECT list "
                "(add a GROUP BY for non-aggregate columns)"
            )
        if having is not None:
            pred, extra_aggs = _having_predicate(having, proj_items, group_keys, aggs)
            user_aggs = list(aggs)
            plan = Aggregate(plan, group_keys=group_keys, aggs=aggs + extra_aggs)
            plan = Filter(plan, pred)
            if extra_aggs:
                # The HAVING-only aggregates (_hvN) were added to compute the
                # predicate but must not leak into the output; re-project only the
                # user's columns (group keys + selected aggregates).
                items = (
                    [(Col(a), a) for _, a in group_keys]
                    + [(Col(a), a) for _, a in user_aggs]
                )
                plan = Project(plan, items=items)
        else:
            plan = Aggregate(plan, group_keys=group_keys, aggs=aggs)
    else:
        if len(proj_items) == 1 and _is_star(proj_items[0]):
            plan = _maybe_project_star(plan)  # no-op pass-through
        else:
            items = []
            for it in proj_items:
                e, alias = _proj_item(it)
                items.append((e, alias))
            plan = Project(plan, items=items)

    if sel.args.get("distinct"):
        d = sel.args["distinct"]
        if d.args.get("on") is not None:
            raise NotImplementedError("DISTINCT ON is not supported")
        plan = Distinct(plan)

    return _build_tail(sel, plan)


def _having_predicate(having, proj_items, group_keys, aggs):
    """Lower a HAVING clause to a predicate ``Expr`` over the ``Aggregate``'s
    output frame, returning ``(predicate, extra_aggs)``.

    The executor's ``Aggregate`` emits one column per group key and per selected
    aggregate, named by their output aliases (the same aliases ``_proj_item`` /
    ``_output_name`` produce). HAVING is evaluated by a ``Filter`` over that
    frame, so every column HAVING references must be one of those aliases (or a
    synthetic aggregate added below).

    Each ``exp.AggFunc`` in the HAVING expression is rewritten to a bare ``Col``
    of its matching SELECT-list aggregate's output alias (matched by the
    sqlglot ``.sql()`` of the aggregate node, which is deterministic for the
    default dialect -- an unaliased SELECT aggregate and a bare HAVING aggregate
    of the same shape render identically and so match automatically). A HAVING
    aggregate that is NOT in the SELECT list is added as a synthetic aggregate
    ``_hvN`` (computed during grouping, then pruned from the output by a wrapping
    ``Project``) and the HAVING reference rewrites to ``Col("_hvN")``.

    Every remaining ``Col`` in the lowered predicate must name a group-key alias
    or an aggregate alias; a reference to any other column is rejected (it is
    neither grouped nor aggregated -- standard SQL requires one or the other).
    Subqueries in HAVING are deferred.
    """
    if having.find(exp.Subquery) is not None or having.find(exp.Select) is not None:
        raise NotImplementedError("subqueries in HAVING are not supported")
    # Map each SELECT-list aggregate's sqlglot .sql() -> its output alias.
    agg_map: dict[str, str] = {}
    for it in proj_items:
        if isinstance(it, exp.Alias):
            inner = it.this
            alias = it.alias
        else:
            inner = it
            alias = _output_name(it)
        if isinstance(inner, exp.AggFunc):
            agg_map.setdefault(inner.sql(), alias)

    extra_aggs: list[tuple[AggFunc, str]] = []
    n_hv = 0
    for a in list(having.this.find_all(exp.AggFunc)):
        key = a.sql()
        if key in agg_map:
            a.replace(exp.column(agg_map[key]))
        else:
            n_hv += 1
            name = f"_hv{n_hv}"
            extra_aggs.append((_expr(a), name))
            a.replace(exp.column(name))

    pred = _expr(having.this)
    allowed = {a for _, a in group_keys} | {a for _, a in aggs} | {a for _, a in extra_aggs}
    for c in pred.columns():
        if c not in allowed:
            raise NotImplementedError(
                f"HAVING references a non-aggregated, non-grouped column {c!r}"
            )
    return pred, extra_aggs


# --------------------------------------------------------------------------- #
# Set operators (UNION [ALL] / INTERSECT / EXCEPT) and the shared ORDER BY /
# LIMIT tail. A top-level query may be a SELECT, a set-op of two queries, or a
# parenthesized subquery wrapping either; ``_build_query`` dispatches and
# unwraps. ORDER BY / LIMIT on a set-op attach to the set-op node itself
# (sqlglot puts them in the Union/Intersect/Except ``args``, not the right
# SELECT), so ``_build_tail`` reads them off whichever node it is given.
# --------------------------------------------------------------------------- #


_SETOP_NAME = {"Union": "union", "Intersect": "intersect", "Except": "except"}


def _build_query(node, schema: dict[str, list[str]] | None = None, ctes=None):
    """Dispatch a top-level query node (SELECT / set-op / parenthesized
    subquery) to the right builder. A ``Subquery`` wrapping a set-op or SELECT
    is unwrapped (``.this``) so ``SELECT ... UNION (SELECT ... UNION ...)``
    parses -- the parenthesized right side arrives as an ``exp.Subquery``.

    ``ctes`` is the in-scope CTE name -> built-subplan map (Phase F-2c), threaded
    so a CTE reference resolves to a ``Derived`` wherever it appears -- the main
    FROM, a join partner, a derived-table body, or an IN/EXISTS/scalar subquery
    nested anywhere under this query."""
    if isinstance(node, exp.Select):
        return _build_select(node, schema, ctes)
    if isinstance(node, (exp.Union, exp.Intersect, exp.Except)):
        return _build_setop(node, schema, ctes)
    if isinstance(node, exp.Subquery):
        return _build_query(node.this, schema, ctes)
    raise ParseError(f"unsupported query: {type(node).__name__}")


def _build_setop(node, schema: dict[str, list[str]] | None = None, ctes=None):
    """Lower ``left {UNION|INTERSECT|EXCEPT} [ALL] right`` into a SetOp node.

    sqlglot gives every set-op node ``this`` (left), ``expression`` (right), and
    ``distinct`` (True unless ``ALL``). INTERSECT ALL / EXCEPT ALL (the multiset
    variants) are deferred -- the executor raises ``NotImplementedError`` for
    them rather than failing at parse time, so ``_build_setop`` records
    ``distinct`` faithfully and lets the executor decide. ORDER BY / LIMIT on the
    set-op wrap the SetOp via the shared ``_build_tail``."""
    op = _SETOP_NAME[type(node).__name__]
    distinct = bool(node.args.get("distinct"))
    left = _build_query(node.this, schema, ctes)
    right = _build_query(node.expression, schema, ctes)
    plan = SetOp(left, right, op, distinct)
    return _build_tail(node, plan)


def _build_tail(stmt, plan):
    """Apply ORDER BY then LIMIT/OFFSET from ``stmt.args`` (shared by SELECT and
    set-op nodes). ORDER BY only supports column references that name an output
    column; the executor's Sort resolves them against the produced frame. No-op
    when neither clause is present."""
    order = stmt.args.get("order")
    if order is not None:
        keys = []
        for o in order.expressions:
            if not isinstance(o, exp.Ordered):
                raise NotImplementedError(f"unsupported ORDER BY term: {o}")
            e = _expr(o.this)
            if not isinstance(e, Col):
                raise NotImplementedError("ORDER BY only supports column references")
            keys.append((Col(e.name), not o.args.get("desc", False)))
        plan = Sort(plan, keys=keys)

    limit = stmt.args.get("limit")
    if limit is not None:
        n = int(_literal_value(_limit_value(limit)))
        off = 0
        offset = stmt.args.get("offset")
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
    if isinstance(node, exp.Neg):
        # Unary minus (e.g. a LAG/LEAD default of -1, or -x). Lower to 0 - inner
        # so NULL/numeric three-valued semantics match DuckDB.
        return BinOp("-", Lit(0, "int"), _expr(node.this))
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
    # sqlglot's *default* dialect (which RyuDB uses) parses several date funcs
    # oddly: STRFTIME is an exp.Anonymous, DATEDIFF('u', a, b) is a typed but
    # MANGLED exp.DateDiff (see the exp.DateDiff branch below), and DATE_TRUNC is
    # a typed exp.DateTrunc (handled in _SCALAR_FUNC_BUILDERS). The anonymous
    # forms are intercepted here by name so any of these funcs also works when a
    # dialect emits Anonymous. Other anonymous funcs (AGE, DATE_FORMAT, ...)
    # still fall through to NotImplementedError below.
    if isinstance(node, exp.Anonymous):
        up = node.name.upper()
        xs = node.expressions
        if up == "NOW":
            return Func("current_timestamp", ())
        if up == "STRFTIME" and len(xs) == 2:
            return Func("strftime", (_expr(xs[0]), _expr(xs[1])))
        if up == "DATE_TRUNC" and len(xs) == 2:
            return Func("date_trunc", (_expr(xs[1]), Lit(xs[0].name.lower(), "str")))
        if up == "DATEDIFF" and len(xs) == 3:
            return Func("datediff", (_expr(xs[1]), _expr(xs[2]),
                                    Lit(xs[0].name.lower(), "str")))
    # DATEDIFF('unit', start, end): the default dialect mis-parses this as
    # exp.DateDiff(this=Literal('unit'), expression=start, unit=Var(end)) -- the
    # end column lands in the `unit` slot, uppercased as if it were a unit
    # keyword. Reconstruct positionally (end coerced back to a lowercase Col).
    if isinstance(node, exp.DateDiff):
        return _datediff(node)
    # Date +/- INTERVAL: ``d + INTERVAL '1' DAY`` is an exp.Add/Sub whose operand
    # is an exp.Interval. Intercept before the generic exp.Binary branch (Add/Sub
    # are Binary subclasses) and lower to a date_add/date_sub Func.
    if isinstance(node, (exp.Add, exp.Sub)) and (
        isinstance(node.expression, exp.Interval) or isinstance(node.this, exp.Interval)
    ):
        return _interval_arith(node)
    # Scalar functions. DPipe (``||``) and Mod are exp.Binary subclasses, so they
    # must be matched before the generic exp.Binary branch; Trim and the
    # _SCALAR_FUNC_BUILDERS table are exp.Func (not Binary). exp.Mod is NOT in the
    # table -- it falls through to exp.Binary -> BinOp("%", ...) (the _BINOP map
    # already has exp.Mod: "%").
    if isinstance(node, exp.DPipe):
        return Func("concat_pipe", (_expr(node.this), _expr(node.expression)))
    if isinstance(node, exp.Trim):
        return _trim(node)
    if type(node) in _SCALAR_FUNC_BUILDERS:
        return _SCALAR_FUNC_BUILDERS[type(node)](node)
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
        # The WHERE top-conjunct path intercepts IN/NOT IN subqueries before
        # _expr (see _apply_where_subqueries). Reaching here means the subquery
        # is nested under something we don't fold (OR, CASE, projection, ...) --
        # deferred.
        raise NotImplementedError(
            "IN with a subquery is only supported as a top-level WHERE conjunct"
        )
    values = tuple(_expr(v) for v in node.expressions)
    if not values:
        raise ParseError("IN list is empty")
    return In(_expr(node.this), values, negated=negated)


# --------------------------------------------------------------------------- #
# WHERE subquery conjuncts -> semi/anti join. ``x IN (SELECT ...)`` / ``x NOT IN
# (SELECT ...)`` (uncorrelated, E-1) and correlated ``EXISTS (SELECT ...)`` /
# ``NOT EXISTS (SELECT ...)`` (E-3) fold into semi/anti joins on ``plan``: the
# subquery is the Join's right child (a normal PlanNode subtree the optimizer
# recurses into), so eval_expr stays engine-free. Semi/anti preserve the left
# side; the right (subquery) side's columns are not in the output.
#
# A correlated EXISTS/NOT EXISTS with a single equi-correlation ``inner.k =
# outer.k`` decorrelates to a semi/anti join on those columns -- the correlation
# predicate *is* the join key (the subquery's WHERE is reduced to its local
# conjuncts, its projection replaced by the inner key column since the SELECT
# list is irrelevant to EXISTS). NOT EXISTS -> anti-join is NULL-correct (a NULL
# outer key matches nothing -> NOT EXISTS true -> anti-join keeps it; matches
# DuckDB). IN under OR, correlated IN, non-equi/multi correlation, and
# non-bare-column IN keys are deferred.
# --------------------------------------------------------------------------- #


def _apply_where_subqueries(plan, where_node, schema, outer_aliases, ctes=None):
    """Fold IN / NOT IN (uncorrelated) and correlated EXISTS / NOT EXISTS WHERE
    conjuncts into semi/anti joins on ``plan``; return ``(plan, residual)`` where
    ``residual`` is the conjoined non-subquery conjuncts (or None).

    Only AND-combined conjuncts fold (semi/anti are not distributive over OR); a
    subquery conjunct under OR stays in the residual and raises when ``_expr``
    lowers it. Uncorrelated EXISTS conjuncts never reach here --
    ``_flatten_outer_subqueries`` already replaced them with ``col > 0`` (E-2) --
    so an ``exp.Exists`` conjunct here is correlated by construction."""
    residual: list[Expr] = []
    for c in _split_and_exp(where_node):
        sub = _where_subquery_conjunct(c, outer_aliases)
        if sub is None:
            residual.append(_expr(c))
            continue
        how, outer_key, inner_select = sub
        subplan = _build_query(inner_select, schema, ctes)
        on_right = _subquery_output_col(subplan)
        plan = Join(plan, subplan, [outer_key], [on_right], how, None)
    return plan, (_conjoin_exprs(residual) if residual else None)


def _where_subquery_conjunct(c, outer_aliases):
    """If conjunct ``c`` folds to a semi/anti join, return
    ``(how, outer_key, inner_select)``; else None (it stays in the residual).

    ``outer_key`` is the outer-side join key column name; ``inner_select`` is the
    sqlglot select for the Join's right side (already decorrelated for EXISTS:
    correlation stripped from its WHERE, projection set to the inner key)."""
    # Correlated EXISTS / NOT EXISTS.
    if isinstance(c, exp.Exists):
        return _exists_conjunct(c, negated=False, outer_aliases=outer_aliases)
    if isinstance(c, exp.Not) and isinstance(c.this, exp.Exists):
        return _exists_conjunct(c.this, negated=True, outer_aliases=outer_aliases)
    # Uncorrelated IN / NOT IN (correlated IN is deferred).
    in_sub = _in_subquery_conjunct(c)
    if in_sub is not None:
        how, left_node, subq_node = in_sub
        subq_select = subq_node.this if isinstance(subq_node, exp.Subquery) else subq_node
        if _classify_correlation(subq_select, outer_aliases) is not None:
            raise NotImplementedError("correlated IN subqueries are not supported yet")
        on_left = _in_subquery_key(left_node)  # raises if not a bare column
        return how, on_left[0], subq_select
    return None


def _exists_conjunct(exists_node, negated, outer_aliases):
    """Decorrelate a correlated ``EXISTS``/``NOT EXISTS`` to a semi/anti join.
    Returns ``(how, outer_key, inner_select)`` or None (uncorrelated -> residual)."""
    inner = exists_node.this
    corr = _classify_correlation(inner, outer_aliases)
    if corr is None:
        return None  # uncorrelated (already handled by _flatten_outer_subqueries)
    outer_key, inner_key, local_conjuncts = corr
    _set_local_where(inner, local_conjuncts)
    # The SELECT list is irrelevant to EXISTS -- project the inner key so the
    # right side exposes it as the join key column.
    inner.set("expressions", [exp.column(inner_key)])
    return ("anti" if negated else "semi"), outer_key, inner


def _in_subquery_conjunct(c):
    """If conjunct ``c`` is an IN/NOT IN subquery, return ``(how, left_node, subq)``."""
    if isinstance(c, exp.In) and c.args.get("query") is not None:
        return "semi", c.this, c.args["query"]
    if (isinstance(c, exp.Not) and isinstance(c.this, exp.In)
            and c.this.args.get("query") is not None):
        return "anti", c.this.this, c.this.args["query"]
    return None


def _in_subquery_key(left_node):
    if not isinstance(left_node, exp.Column):
        raise NotImplementedError(
            "IN-subquery key must be a bare column (expression keys are deferred)"
        )
    return [left_node.name]


def _split_and_exp(node):
    """Split a sqlglot predicate on top-level ``exp.And`` into conjuncts."""
    if isinstance(node, exp.And):
        return _split_and_exp(node.this) + _split_and_exp(node.expression)
    return [node]


def _conjoin_exprs(parts: list[Expr]) -> Expr:
    acc = parts[0]
    for p in parts[1:]:
        acc = And(acc, p)
    return acc


def _classify_correlation(subq_select, outer_aliases):
    """Classify a subquery's correlation with the outer scope.

    Returns ``(outer_key, inner_key, local_conjuncts)`` for a subquery whose only
    outer reference is a single equi-correlation ``inner.col = outer.col`` conjunct
    in its WHERE (both sides bare columns, one referencing an outer alias, the
    other local); ``local_conjuncts`` are the remaining (non-correlated) WHERE
    conjuncts. Returns ``None`` for an uncorrelated subquery. Raises
    ``NotImplementedError`` for the deferred forms: more than one equi-correlation,
    a non-equi outer reference in the WHERE, or any outer reference outside the
    WHERE (SELECT list / aggregate args / GROUP BY / HAVING / ORDER BY).

    Bare (unqualified) columns are assumed local (the flat-column model); a bare
    column that actually belongs outside raises a KeyError at execution rather
    than silently producing wrong results."""
    local = _local_aliases(subq_select)
    outer = set(outer_aliases)
    where = subq_select.args.get("where")

    equi: tuple[str, str] | None = None
    local_conjuncts: list = []
    if where is not None:
        for conj in _split_and_exp(where.this):
            if _references_outer(conj, outer, local):
                pair = _equi_correlation_pair(conj, outer, local)
                if pair is None:
                    raise NotImplementedError(
                        "correlated subqueries are only supported with a single "
                        "equality correlation (inner.col = outer.col); non-equi "
                        "correlations are not supported yet"
                    )
                if equi is not None:
                    raise NotImplementedError(
                        "correlated subqueries with multiple correlation "
                        "predicates are not supported yet"
                    )
                equi = pair
            else:
                local_conjuncts.append(conj)

    # Outer references outside the WHERE can't be turned into an equi-join key.
    for part in _non_where_subquery_parts(subq_select):
        if _references_outer(part, outer, local):
            raise NotImplementedError(
                "correlated subqueries are only supported when the outer reference "
                "is a WHERE equality (outer refs in the SELECT list, aggregate "
                "arguments, or HAVING are not supported yet)"
            )

    if equi is None:
        return None  # uncorrelated
    return (*equi, local_conjuncts)


def _references_outer(node, outer, local) -> bool:
    """True if ``node`` contains a qualified column whose table is an outer alias
    (and not one of the subquery's own FROM/JOIN aliases)."""
    for col in node.find_all(exp.Column):
        tbl = col.table
        if tbl and tbl not in local and tbl in outer:
            return True
    return False


def _equi_correlation_pair(conj, outer, local):
    """If ``conj`` is ``inner.col = outer.col`` (bare columns, exactly one side
    referencing an outer alias), return ``(outer_key, inner_key)``; else None."""
    if not isinstance(conj, exp.EQ):
        return None
    lhs, rhs = conj.this, conj.expression
    if not (isinstance(lhs, exp.Column) and isinstance(rhs, exp.Column)):
        return None
    lhs_outer = _col_table_is_outer(lhs, outer, local)
    rhs_outer = _col_table_is_outer(rhs, outer, local)
    if lhs_outer and not rhs_outer:
        return rhs.name, lhs.name  # rhs is outer, lhs is inner
    if rhs_outer and not lhs_outer:
        return lhs.name, rhs.name  # lhs is outer, rhs is inner
    return None  # both outer or both inner -> not a single correlation pair


def _col_table_is_outer(col, outer, local) -> bool:
    tbl = col.table
    return bool(tbl and tbl not in local and tbl in outer)


def _non_where_subquery_parts(subq_select):
    """The subquery's expression-bearing parts OUTSIDE its WHERE (SELECT list,
    GROUP BY, HAVING, ORDER BY) -- checked for outer references that can't be
    turned into an equi-join key."""
    parts: list = list(subq_select.expressions)  # SELECT list
    group = subq_select.args.get("group")
    if group is not None:
        parts.extend(group.expressions)
    having = subq_select.args.get("having")
    if having is not None:
        parts.append(having.this)
    order = subq_select.args.get("order")
    if order is not None:
        parts.extend(order.expressions)
    return parts


def _set_local_where(inner_select, local_conjuncts):
    """Rewrite the inner select's WHERE to only the local (non-correlated)
    conjuncts -- the equi-correlation has been lifted out as the join key. Drops
    the WHERE when no local conjuncts remain."""
    if not local_conjuncts:
        inner_select.args.pop("where", None)
    else:
        inner_select.set("where", exp.Where(this=_conjoin_exp_nodes(local_conjuncts)))


def _conjoin_exp_nodes(nodes):
    """Rebuild an ``exp.And`` tree from a list of sqlglot predicate nodes."""
    acc = nodes[0]
    for n in nodes[1:]:
        acc = exp.And(this=acc, expression=n)
    return acc


def _local_aliases(subq_select) -> set[str]:
    """Aliases / table names declared in the subquery's own FROM and JOINs."""
    names: set[str] = set()
    f = subq_select.args.get("from")
    if f is not None and f.this is not None:
        names.add(_table_alias_name(f.this))
    for j in subq_select.args.get("joins", []) or []:
        if j.this is not None:
            names.add(_table_alias_name(j.this))
    return names


def _table_alias_name(node) -> str:
    if isinstance(node, exp.Table):
        return node.alias or node.name
    return getattr(node, "alias", "") or getattr(node, "name", "") or ""


def _subquery_output_col(subplan) -> str:
    """The single output column name of a one-column subquery plan.

    Peeks through Sort/Limit (ORDER BY/LIMIT in an IN-subquery are meaningless;
    DuckDB ignores them). Requires exactly one output column and an explicit
    projection (``SELECT *`` is ambiguous and rejected)."""
    node = subplan
    while isinstance(node, (Sort, Limit)):
        node = node.input
    if isinstance(node, Project):
        if len(node.items) != 1:
            raise NotImplementedError("IN-subquery must project exactly one column")
        return node.items[0][1]
    if isinstance(node, Aggregate):
        n = len(node.group_keys) + len(node.aggs)
        if n != 1:
            raise NotImplementedError("IN-subquery must project exactly one column")
        return (node.aggs[0][1] if node.aggs else node.group_keys[0][1])
    if isinstance(node, SetOp):
        return _subquery_output_col(node.left)
    raise NotImplementedError(
        "IN-subquery must project a single named column (SELECT * is ambiguous)"
    )


# --------------------------------------------------------------------------- #
# Scalar / EXISTS subqueries -> cross-join broadcast (Phase E-2) or left-join
# decorrelation (Phase E-3). A subquery that appears in THIS select's WHERE or
# projection is flattened into a join on the running ``plan``; the subquery
# sqlglot node is replaced in place by a bare Column reference to the broadcast
# column. Keeping it relational (a Join.right subtree the optimizer recurses
# into) -- not an Expr-embedded subplan -- leaves ``eval_expr`` engine-free,
# consistent with the IN/NOT IN lowering (E-1).
#
# Uncorrelated forms (E-2):
#   * scalar (single-row aggregate) -> cross-join broadcast of the 1-row
#     aggregate; node -> ``Col(_sqN)``.
#   * EXISTS -> ``(SELECT COUNT(*) FROM (subq) LIMIT 1) > 0``; node -> ``col > 0``
#     so EXISTS works under AND/OR/NOT and in projection (not just as a top-level
#     WHERE conjunct). NOT EXISTS falls out via ``Not`` (count is non-NULL int).
#
# Correlated forms (E-3): a single equi-correlation ``inner.k = outer.k`` is
# decorrelated at parse time into an uncorrelated join -- no outer-scope binding:
#   * scalar -> LEFT join onto a grouped aggregate (the subquery's aggregate, now
#     grouped by the correlation key); node -> ``Col(_sqN)`` (NULL for unmatched
#     outer rows, matching DuckDB).
#   * EXISTS -> left for ``_apply_where_subqueries`` to fold as a semi/anti join
#     on the correlation key (the node is NOT replaced here).
# Non-aggregate / GROUP BY scalars are deferred; multi-equi / non-equi correlation
# and outer refs outside the WHERE are deferred (see ``_classify_correlation``).
# --------------------------------------------------------------------------- #


def _flatten_outer_subqueries(plan, sel, schema, outer_aliases, ctes=None):
    """Flatten scalar / EXISTS subqueries of THIS select into joins on ``plan``,
    replacing each scalar subquery node with a Column ref. Correlated EXISTS
    nodes are left in place (``_apply_where_subqueries`` folds them as semi/anti
    joins); uncorrelated EXISTS are replaced with ``col > 0``.

    Only subqueries that belong to ``sel`` itself are collected -- ones nested
    inside an inner SELECT (a subquery's own body, or an IN-subquery's body) are
    NOT descended into; they are handled recursively by that inner select's own
    ``_build_select`` (or by ``_apply_where_subqueries`` for IN). IN/NOT IN
    subquery nodes (``exp.In`` with a ``query``) are left untouched here."""
    acc: list[tuple[str, exp.Expression]] = []
    where = sel.args.get("where")
    if where is not None:
        _collect_outer_subqueries(where.this, acc)
    for it in sel.expressions:
        _collect_outer_subqueries(it, acc)

    n = 0
    for kind, node in acc:
        # exp.Exists.this is the inner Select directly; a scalar exp.Subquery.this
        # is the inner query (Select / set-op / nested Subquery) -> unwrap once.
        inner = node.this if kind == "exists" else (
            node.this if isinstance(node, exp.Subquery) else node
        )
        corr = _classify_correlation(inner, outer_aliases)
        if kind == "exists":
            if corr is not None:
                # Correlated EXISTS: leave for _apply_where_subqueries (semi/anti).
                continue
            n += 1
            name = f"_sq{n}"
            subplan = _build_query(inner, schema, ctes)
            # COUNT(*) of <=1 row > 0: 1 iff the subquery has any row, else 0.
            # COUNT must be uppercase (the executor's _scalar_global_agg special-
            # cases ``func == "COUNT"`` with a Star arg); ``AGG_FUNCS`` (used by
            # _build_select) already yields the uppercase tag.
            count_plan = Aggregate(
                Limit(subplan, 1, 0), [], [(AggFunc("COUNT", Star()), name)]
            )
            plan = Join(plan, count_plan, [], [], "cross", None)
            node.replace(exp.GT(this=exp.column(name), expression=exp.Literal.number(0)))
        else:  # scalar
            if corr is None:
                n += 1
                name = f"_sq{n}"
                subplan = _build_query(inner, schema, ctes)
                _require_global_aggregate(subplan)
                existing = _subquery_output_col(subplan)
                # Rename the aggregate's single output column to the broadcast
                # name via a 1-item Project so the cross-joined column is clean.
                renamed = Project(subplan, [(Col(existing), name)])
                plan = Join(plan, renamed, [], [], "cross", None)
                node.replace(exp.column(name))
            else:
                n += 1
                name = f"_sq{n}"
                plan, replacement = _decorrelate_scalar(plan, inner, corr, name, schema, ctes)
                node.replace(replacement)
    return plan


def _decorrelate_scalar(plan, inner_select, corr, name, schema, ctes=None):
    """Rewrite a correlated scalar subquery (single equi-correlation, global
    aggregate, inner-only agg arg) into a LEFT join of ``plan`` onto a grouped
    aggregate. The correlation key becomes the GROUP BY key and the left-join key;
    the aggregate output and the group key are renamed to unique ``_sqN`` names so
    they never collide with the outer frame's columns in the merge.

    Returns ``(plan, replacement)`` where ``replacement`` is the sqlglot node to
    substitute for the subquery. ``COUNT`` yields ``COALESCE(_sqN, 0)`` because
    COUNT over an empty correlation is 0, not NULL; any other aggregate yields
    ``_sqN`` (NULL over an empty correlation, matching DuckDB). The grouped
    aggregate's NULL-key group is dropped before the join so the LEFT join is
    NULL-safe: SQL ``inner.k = outer.k`` matches nothing when either side is NULL,
    but cuDF ``merge`` would match NULL==NULL, so the NULL-key group is removed and
    a NULL outer key finds no match -> null-pad (MAX/MIN/SUM -> NULL; COUNT -> the
    COALESCE default 0)."""
    outer_key, inner_key, local_conjuncts = corr
    _set_local_where(inner_select, local_conjuncts)
    subplan = _build_query(inner_select, schema, ctes)
    _require_global_aggregate(subplan)
    # subplan is Aggregate(input, [], [(agg, orig_name)]) -- inject the correlation
    # key as the GROUP BY key. Peek through Sort/Limit to the global aggregate.
    core = subplan
    while isinstance(core, (Sort, Limit)):
        core = core.input
    agg_expr, orig_name = core.aggs[0]
    grouped = Aggregate(
        core.input,
        [(Col(inner_key), name + "_k")],
        [(agg_expr, orig_name)],
    )
    # Rename both outputs to unique names (avoid the optimizer's global column-
    # name-uniqueness assumption and merge suffix collisions).
    renamed = Project(
        grouped,
        [(Col(orig_name), name), (Col(name + "_k"), name + "_k")],
    )
    # Drop the NULL-key group so the left join is NULL-safe (SQL ``=`` matches
    # nothing on NULL; cuDF merge would match NULL==NULL).
    renamed = Filter(renamed, IsNull(Col(name + "_k"), negated=True))
    plan = Join(plan, renamed, [outer_key], [name + "_k"], "left", None)
    # COUNT over an empty correlation is 0, not the null-pad NULL; COALESCE the
    # broadcast column to 0 for COUNT only (MAX/MIN/SUM stay NULL over empty).
    if agg_expr.func == "COUNT":
        replacement = exp.Coalesce(
            this=exp.column(name), expressions=[exp.Literal.number(0)]
        )
    else:
        replacement = exp.column(name)
    return plan, replacement


def _collect_outer_subqueries(node, acc):
    """Collect scalar (``exp.Subquery``) and ``exp.Exists`` subqueries that belong
    to the outer select, WITHOUT descending into any inner select's body (those
    are owned by the inner query and handled recursively). IN-subquery nodes
    (``exp.In`` carrying a ``query``) are skipped: their subquery is owned by
    ``_apply_where_subqueries``."""
    if isinstance(node, exp.Exists):
        acc.append(("exists", node))
        return  # do not descend into .this (the inner SELECT)
    if isinstance(node, exp.Subquery):
        parent = node.parent
        if isinstance(parent, exp.In) and parent.args.get("query") is node:
            return  # IN-subquery: owned by _apply_where_subqueries
        acc.append(("scalar", node))
        return  # do not descend into .this (the inner SELECT)
    if isinstance(node, exp.Select):
        return  # an inner select reached without an owning Subquery/Exists wrapper
    for child in node.args.values():
        if isinstance(child, exp.Expression):
            _collect_outer_subqueries(child, acc)
        elif isinstance(child, list):
            for c in child:
                if isinstance(c, exp.Expression):
                    _collect_outer_subqueries(c, acc)


def _require_global_aggregate(subplan):
    """A scalar subquery must be a single-row aggregate (a global aggregate with
    no GROUP BY keys) so the cross-join broadcasts exactly one value per outer
    row. Non-aggregate and GROUP BY scalar subqueries are deferred."""
    core = subplan
    while isinstance(core, (Sort, Limit)):
        core = core.input
    if not isinstance(core, Aggregate) or core.group_keys:
        raise NotImplementedError(
            "scalar subquery must be a single-row aggregate (e.g. "
            "SELECT COUNT(*)/MAX(...) FROM ...); non-aggregate and GROUP BY "
            "scalar subqueries are not supported yet"
        )


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


# --------------------------------------------------------------------------- #
# Scalar functions (UPPER/LOWER/LENGTH/SUBSTR/TRIM/CONCAT/||/REPLACE/POSITION/
# LEFT/RIGHT/INITCAP/REVERSE/ABS/ROUND/CEIL/FLOOR). Each maps a sqlglot node to
# a generic ``Func(tag, args)``; the per-tag cuDF op lives in ``ops._func``.
# sqlglot node shapes were confirmed by introspection (v28.10.1).
# --------------------------------------------------------------------------- #


def _trim(node) -> Expr:
    """Lower ``TRIM`` / ``LTRIM`` / ``RTRIM`` / ``TRIM(LEADING x FROM s)``.

    sqlglot unifies these as ``exp.Trim``: ``position`` is the side
    (``"LEADING"``/``"TRAILING"``/``"BOTH"``/``None`` for default both),
    ``expression`` is the trim-chars expr (``None`` -> whitespace). Encoded as
    ``Func("trim", (this, chars_or_NoneLit, sideLit))`` so ops has both without a
    sub-dataclass; ``args[1]`` is the chars expr, ``args[2]`` the side.
    """
    pos = node.args.get("position") or "BOTH"
    chars = node.args.get("expression")
    chars_expr = _expr(chars) if chars is not None else Lit(None, "str")
    return Func("trim", (_expr(node.this), chars_expr, Lit(pos, "str")))


def _scalar_unary(tag: str):
    """Builder for a 1-arg scalar func: ``F(x)`` -> ``Func(tag, (_expr(x),))``."""
    def build(node) -> Expr:
        return Func(tag, (_expr(node.this),))
    return build


def _substr(node) -> Expr:
    start = node.args.get("start")
    length = node.args.get("length")
    args = [_expr(node.this), _expr(start)]
    if length is not None:
        args.append(_expr(length))
    return Func("substr", tuple(args))


def _round(node) -> Expr:
    decimals = node.args.get("decimals")
    if decimals is not None:
        return Func("round", (_expr(node.this), _expr(decimals)))
    return Func("round", (_expr(node.this),))


def _interval_arith(node) -> Expr:
    """Lower ``date +/- INTERVAL n UNIT`` into a ``date_add``/``date_sub`` Func.

    sqlglot gives ``exp.Add``/``exp.Sub`` with an ``exp.Interval`` operand
    (``Interval(this=Literal(n), unit=Var(UNIT))``). The interval may sit on
    either side for ``+`` (commutative); for ``-`` only interval-on-the-right is
    sensible (``INTERVAL - date`` is not), so that raises. Encoded as
    ``Func(tag, (date_expr, Lit(n,"int"), Lit(unit,"str")))``; the executor picks
    a GPU timedelta for fixed units (day/week/hour/minute/second) and a pandas
    ``DateOffset`` fallback for variable units (month/year).
    """
    left_iv = isinstance(node.this, exp.Interval)
    right_iv = isinstance(node.expression, exp.Interval)
    if right_iv:
        date_side, iv = node.this, node.expression
    else:
        date_side, iv = node.expression, node.this
    n = int(_literal_value(iv.this))
    unit = iv.args["unit"].name.lower()
    if isinstance(node, exp.Add):
        tag = "date_add"
    else:
        if left_iv:
            raise NotImplementedError("INTERVAL - date is not supported")
        tag = "date_sub"
    return Func(tag, (_expr(date_side), Lit(n, "int"), Lit(unit, "str")))


def _datediff(node) -> Expr:
    """Lower ``DATEDIFF(unit, start, end)`` into ``Func("datediff", (start, end, unit))``.

    sqlglot's *default* dialect mis-parses the 3-arg form as
    ``exp.DateDiff(this=Literal('unit'), expression=start, unit=Var(end))`` -- the
    end column lands in the ``unit`` slot, uppercased as if it were a unit keyword
    (DAY/MONTH). Reconstruct positionally: ``start`` from ``expression``, ``end``
    from the ``unit`` slot coerced back to a lowercase ``Col`` (Var uppercases it;
    a clean Column is left alone), and ``unit`` from the ``this`` literal. The
    executor computes ``end - start`` in the unit, matching DuckDB's
    ``DATEDIFF(unit, a, b) = b - a``.
    """
    this = node.this
    if not isinstance(this, exp.Literal):
        raise NotImplementedError(
            "only DATEDIFF(unit, start, end) is supported (got an unrecognized form)"
        )
    unit = this.name.lower()
    start = _expr(node.expression)
    end_node = node.args.get("unit")
    if isinstance(end_node, exp.Var):
        end = Col(end_node.name.lower())
    elif end_node is not None:
        end = _expr(end_node)
    else:
        raise NotImplementedError("DATEDIFF is missing its end operand")
    return Func("datediff", (start, end, Lit(unit, "str")))


# exp.Type -> (node) -> Func. Looked up by exact type in ``_expr``.
_SCALAR_FUNC_BUILDERS = {
    exp.Upper: _scalar_unary("upper"),
    exp.Lower: _scalar_unary("lower"),
    exp.Length: _scalar_unary("length"),
    exp.Abs: _scalar_unary("abs"),
    exp.Ceil: _scalar_unary("ceil"),
    exp.Floor: _scalar_unary("floor"),
    exp.Initcap: _scalar_unary("initcap"),
    exp.Reverse: _scalar_unary("reverse"),
    exp.Substring: _substr,
    exp.Round: _round,
    exp.Concat: lambda n: Func("concat", tuple(_expr(x) for x in n.expressions)),
    exp.Replace: lambda n: Func(
        "replace", (_expr(n.this), _expr(n.expression), _expr(n.args["replacement"]))
    ),
    exp.StrPosition: lambda n: Func("strpos", (_expr(n.this), _expr(n.args["substr"]))),
    exp.Left: lambda n: Func("left", (_expr(n.this), _expr(n.expression))),
    exp.Right: lambda n: Func("right", (_expr(n.this), _expr(n.expression))),
    # --- date/time functions ------------------------------------------- #
    # Date-part functions reuse the "extract" tag with a FIELD literal arg; the
    # DAYOFWEEK/DAYOFYEAR function forms become EXTRACT(DOW/DOY FROM x).
    exp.Year:      lambda n: Func("extract", (_expr(n.this), Lit("YEAR", "str"))),
    exp.Month:     lambda n: Func("extract", (_expr(n.this), Lit("MONTH", "str"))),
    exp.Day:       lambda n: Func("extract", (_expr(n.this), Lit("DAY", "str"))),
    exp.Hour:      lambda n: Func("extract", (_expr(n.this), Lit("HOUR", "str"))),
    exp.Minute:    lambda n: Func("extract", (_expr(n.this), Lit("MINUTE", "str"))),
    exp.Second:    lambda n: Func("extract", (_expr(n.this), Lit("SECOND", "str"))),
    exp.DayOfWeek: lambda n: Func("extract", (_expr(n.this), Lit("DOW", "str"))),
    exp.DayOfYear: lambda n: Func("extract", (_expr(n.this), Lit("DOY", "str"))),
    # EXTRACT(field FROM x): this=Var(field), expression=x.
    exp.Extract: lambda n: Func("extract", (_expr(n.expression), Lit(n.this.name.upper(), "str"))),
    # DATE_TRUNC(unit, x): the default dialect emits exp.DateTrunc(this=x,
    # unit=Literal('UNIT')); duckdb-style emits exp.TimestampTrunc(this=x,
    # unit=Var). Both have .this + args['unit']; .name lowercases either.
    exp.DateTrunc: lambda n: Func("date_trunc", (_expr(n.this), Lit(n.args["unit"].name.lower(), "str"))),
    exp.TimestampTrunc: lambda n: Func("date_trunc", (_expr(n.this), Lit(n.args["unit"].name.lower(), "str"))),
    # STRFTIME(x, fmt): duckdb-style emits exp.TimeToStr(this=x, format=fmt); the
    # default dialect emits exp.Anonymous (handled in _expr).
    exp.TimeToStr: lambda n: Func("strftime", (_expr(n.this), _expr(n.args["format"]))),
    exp.LastDay: lambda n: Func("last_day", (_expr(n.this),)),
    exp.Dayname: lambda n: Func("dayname", (_expr(n.this),)),
    exp.Monthname: lambda n: Func("monthname", (_expr(n.this),)),
    # No-arg current-date/timestamp: scalars broadcast per row by _project.
    exp.CurrentDate: lambda n: Func("current_date", ()),
    exp.CurrentTimestamp: lambda n: Func("current_timestamp", ()),
}


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


def _table_ref(node, schema, ctes=None) -> tuple[object, str, str]:
    """Lower a FROM/JOIN table reference into ``(source, alias, name)``.

    ``source`` is a ``Scan(name)`` for a real catalog table, a
    ``Derived(subplan, alias)`` for a FROM-subquery, or a ``Derived`` wrapping a
    CTE's built subplan for a CTE reference (Phase F-2c). ``alias`` is the FROM
    alias. ``name`` is the routing identity used by ``_join_spec`` /
    ``_route_join_cols``: the real table name for a base table, or the alias for a
    derived table / CTE reference (``schema.get(alias)`` is empty, so equi-key
    routing falls back to table qualifiers -- correct for a relation whose output
    columns carry no schema entry). Derived tables require an alias (DuckDB does
    too); a CTE reference may use the CTE name as its alias or an explicit ``AS x``.
    A CTE reference is a plain ``exp.Table`` whose ``name`` is in the in-scope CTE
    map -- indistinguishable by type from a base table, so the map lookup happens
    here. The built CTE subplan is shared across references (the optimizer rebuilds
    rather than mutates, so each wrapping ``Derived`` gets its own optimized copy)."""
    if isinstance(node, exp.Subquery):
        subplan = _build_query(node.this, schema, ctes)
        alias = node.alias
        if not alias:
            raise ParseError("subquery in FROM must have an alias (e.g. FROM (...) AS t)")
        return Derived(subplan, alias), alias, alias
    if isinstance(node, exp.Table) and ctes and node.name in ctes:
        # A CTE reference: ``FROM cte1`` / ``FROM cte1 AS x`` -> Derived wrapping
        # the CTE body's built subplan. Same lowering as a FROM-subquery, so all
        # scope-barrier / pruning / routing semantics apply unchanged.
        alias = node.alias or node.name
        return Derived(ctes[node.name], alias), alias, alias
    if not isinstance(node, exp.Table):
        raise ParseError(f"expected a table reference, got {type(node).__name__}")
    name = node.name
    if not name:
        raise ParseError("table reference has no name")
    alias = node.alias or name
    return Scan(name), alias, name


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
    # Scans in the OUTER scope only. A ``Derived`` (FROM-subquery) is an opaque
    # relation: its inner scans belong to the subquery's own scope, not this
    # select's table set, so do NOT descend into it (otherwise ``left_tables``
    # in _build_select's join routing would be polluted with the subquery's
    # inner tables and mis-route the outer join keys).
    from .plan import Scan, Derived, children

    out: list = []

    def go(n):
        if isinstance(n, Scan):
            out.append(n)
        elif isinstance(n, Derived):
            return
        else:
            for c in children(n):
                go(c)

    go(plan)
    return out


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


def _contains_window(node) -> bool:
    """True if ``node`` (a projection item) has a window function at THIS select's
    level. By the time projection is built, scalar/EXISTS subqueries have been
    flattened to Column refs, so ``find(exp.Window)`` cannot descend into a
    subquery's own window -- any ``exp.Window`` found belongs to this select."""
    return node.find(exp.Window) is not None


def _reject_window_outside_projection(sel):
    """Raise if a window function appears in GROUP BY / HAVING / QUALIFY (a
    window in the projection list is supported; one elsewhere is malformed for
    F-1 and would otherwise be silently dropped)."""
    for key in ("group", "having", "qualify"):
        part = sel.args.get(key)
        if part is not None and part.find(exp.Window) is not None:
            raise NotImplementedError(
                "window functions in GROUP BY / HAVING / QUALIFY are not supported"
            )


# Window functions supported in F-1: ranking (ROW_NUMBER/RANK/DENSE_RANK) and
# offset (LAG/LEAD) require an ORDER BY; aggregate funcs (SUM/COUNT/AVG/MIN/MAX)
# broadcast over the whole partition (no ORDER BY). Running/cumulative aggregates
# (an ORDER BY on an aggregate window), explicit frames (ROWS/RANGE BETWEEN),
# QUALIFY, expression PARTITION BY / ORDER BY keys, and window + GROUP BY are
# deferred.
_WINDOW_RANK_FUNCS = {exp.RowNumber: "ROW_NUMBER", exp.Rank: "RANK",
                      exp.DenseRank: "DENSE_RANK"}


def _build_window(plan, proj_items, schema):
    """Extract every ``exp.Window`` from the projection items, lower each to a
    ``WindowFunc`` on a shared ``Window`` plan node, and replace the node in its
    item with a bare Column ref (``_wfN``). An item may contain multiple windows
    (e.g. ``LAG(w) - LEAD(w) OVER (...)``); each is extracted separately. Mutates
    ``proj_items`` in place (the sqlglot tree) so the rewritten items flow into
    the normal Project/Aggregate building. Returns the new plan (Window over the
    prior plan)."""
    funcs: list[tuple[WindowFunc, str]] = []
    n = 0
    for it in proj_items:
        for win in list(it.find_all(exp.Window)):
            n += 1
            name = f"_wf{n}"
            wf = _build_one_window(win, schema)
            funcs.append((wf, name))
            win.replace(exp.column(name))
    return Window(plan, funcs=funcs)


def _build_one_window(win, schema):
    """Lower an ``exp.Window`` to a ``WindowFunc`` (or raise for deferred forms)."""
    fn = win.this
    part = win.args.get("partition_by") or []
    partition_keys = []
    for p in part:
        e = _expr(p)
        if not isinstance(e, Col):
            raise NotImplementedError(
                "PARTITION BY expressions are not supported yet (only bare columns)"
            )
        partition_keys.append(e)
    order_node = win.args.get("order")
    order_keys: list[tuple[Expr, bool]] = []
    if order_node is not None:
        for o in order_node.expressions:
            if not isinstance(o, exp.Ordered):
                raise NotImplementedError(f"unsupported ORDER BY term in window: {o}")
            e = _expr(o.this)
            if not isinstance(e, Col):
                raise NotImplementedError(
                    "ORDER BY expressions in a window are not supported yet "
                    "(only bare columns)"
                )
            order_keys.append((e, not o.args.get("desc", False)))
    if win.args.get("spec") is not None:
        raise NotImplementedError(
            "window frames (ROWS/RANGE BETWEEN) are not supported yet"
        )

    offset = None
    default = None
    if isinstance(fn, exp.Lag) or isinstance(fn, exp.Lead):
        func = "LAG" if isinstance(fn, exp.Lag) else "LEAD"
        arg = _expr(fn.this)
        off = fn.args.get("offset")
        dflt = fn.args.get("default")
        offset = _expr(off) if off is not None else None
        default = _expr(dflt) if dflt is not None else None
        if not order_keys:
            raise NotImplementedError(f"{func} requires ORDER BY in the window")
    elif type(fn) in _WINDOW_RANK_FUNCS:
        func = _WINDOW_RANK_FUNCS[type(fn)]
        arg = None
        if not order_keys:
            raise NotImplementedError(f"{func} requires ORDER BY in the window")
    elif isinstance(fn, (exp.Count, exp.Sum, exp.Avg, exp.Min, exp.Max)):
        func = AGG_FUNCS[type(fn)]
        arg = _expr(fn.this) if fn.this is not None else Star()
        if order_keys:
            raise NotImplementedError(
                "aggregate window functions with ORDER BY (running/cumulative "
                "aggregates) are not supported yet"
            )
    else:
        raise NotImplementedError(
            f"window function {type(fn).__name__} is not supported yet"
        )
    return WindowFunc(
        func=func,
        arg=arg,
        partition_keys=tuple(partition_keys),
        order_keys=tuple(order_keys),
        offset=offset,
        default=default,
    )


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