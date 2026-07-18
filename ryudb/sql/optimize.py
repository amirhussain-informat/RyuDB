"""Rule-based logical optimizer.

Phase 1 rules:
  1. Projection pruning  — each Scan reads only the columns referenced above it.
  2. Predicate pushdown  — conjuncts in a Filter above a Join that reference a
     single table are pushed down to a Filter directly above that table's Scan.
     For OUTER joins a conjunct is pushed only into the *preserved* side (LEFT ->
     left, RIGHT -> right); a conjunct on the null-supplying side would drop the
     null-padded rows and silently turn the outer join into an inner join, so it
     stays above. FULL/CROSS have no preserved side, so nothing is pushed.
  3. Join-side selection  — when row-count statistics are available, the smaller
     subtree is placed on the left (build) side of each Join. Swapping a
     LEFT/RIGHT join rewrites ``how`` (left<->right) so the preserved side stays
     correct; inner/cross/full are symmetric.

The optimizer assumes unqualified column names are unique across the tables in
a query (true for TPC-H). If a column name is ambiguous (present in >1 table),
pushdown for that conjunct is skipped (it stays above the join).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from .plan import (
    Aggregate,
    And,
    Derived,
    Distinct,
    Expr,
    Filter,
    Join,
    Limit,
    PlanNode,
    Project,
    Scan,
    SetOp,
    Sort,
    Window,
    walk,
)

# Schema: table -> list of column names. Stats: table -> row count.
Schema = dict[str, list[str]]
Stats = dict[str, int]


def optimize(plan: PlanNode, schema: Schema | None = None, stats: Stats | None = None) -> PlanNode:
    schema = schema or {}
    stats = stats or {}
    plan = push_predicates(plan, schema)
    plan = prune_projections(plan, schema)
    plan = select_join_sides(plan, stats)
    return plan


# --------------------------------------------------------------------------- #
# Projection pruning
# --------------------------------------------------------------------------- #


def prune_projections(plan: PlanNode, schema: Schema) -> PlanNode:
    # SELECT * (no Project, no Aggregate) => keep all columns on every scan.
    has_project = any(isinstance(n, Project) for n in walk(plan))
    has_aggregate = any(isinstance(n, Aggregate) for n in walk(plan))
    star_mode = not has_project and not has_aggregate

    if star_mode or not schema:
        return plan

    referenced = _all_referenced_columns(plan)
    if not referenced:
        return plan

    def rewrite(node: PlanNode) -> PlanNode:
        if isinstance(node, Scan):
            table_cols = set(schema.get(node.table, []))
            cols = referenced & table_cols if table_cols else None
            return replace(node, columns=cols)
        if isinstance(node, Join):
            return Join(rewrite(node.left), rewrite(node.right), node.on_left,
                        node.on_right, node.how, node.on_predicate)
        if isinstance(node, SetOp):
            # A set op is a projection barrier: the columns each branch needs are
            # exactly what that branch projects, so recurse into both children
            # independently (a column referenced only in the left branch must not
            # be pruned from the right branch's scans, and vice versa).
            return SetOp(rewrite(node.left), rewrite(node.right),
                         node.op, node.distinct)
        if isinstance(node, Filter):
            return Filter(rewrite(node.input), node.predicate)
        if isinstance(node, Project):
            return Project(rewrite(node.input), node.items)
        if isinstance(node, Aggregate):
            return Aggregate(rewrite(node.input), node.group_keys, node.aggs)
        if isinstance(node, Window):
            # A projection barrier: it passes input columns through, so recurse
            # into its input's scans (which get pruned by what the window's
            # funcs AND the nodes above reference).
            return Window(rewrite(node.input), node.funcs)
        if isinstance(node, Sort):
            return Sort(rewrite(node.input), node.keys)
        if isinstance(node, Limit):
            return Limit(rewrite(node.input), node.n, node.offset)
        if isinstance(node, Distinct):
            return Distinct(rewrite(node.input))
        if isinstance(node, Derived):
            # A projection barrier from the outer scope: outer columns do not
            # push into the subplan's scans. But recurse so the subplan's own
            # scans get pruned -- the inner scans' pruned set is
            # ``referenced ∩ schema[inner_table]`` over the GLOBAL referenced set
            # (which ``_all_referenced_columns`` collects by walking into the
            # Derived, picking up the inner Project's real-column refs), so e.g.
            # ``SELECT t.x FROM (SELECT a AS x FROM A) t`` keeps column ``a`` on
            # A's scan (x is not a real column of A and is correctly excluded).
            return Derived(rewrite(node.input), node.alias)
        return node

    return rewrite(plan)


def _all_referenced_columns(plan: PlanNode) -> set[str]:
    cols: set[str] = set()
    for node in walk(plan):
        if isinstance(node, Filter):
            cols |= node.predicate.columns()
        elif isinstance(node, Project):
            for e, _ in node.items:
                cols |= e.columns()
        elif isinstance(node, Aggregate):
            for e, _ in node.group_keys:
                cols |= e.columns()
            for a, _ in node.aggs:
                cols |= a.columns()
        elif isinstance(node, Window):
            for wf, _ in node.funcs:
                cols |= wf.columns()
        elif isinstance(node, Sort):
            for e, _ in node.keys:
                cols |= e.columns()
        elif isinstance(node, Join):
            cols.update(node.on_left)
            cols.update(node.on_right)
            if node.on_predicate is not None:
                cols |= node.on_predicate.columns()
    return cols


# --------------------------------------------------------------------------- #
# Predicate pushdown
# --------------------------------------------------------------------------- #


def push_predicates(plan: PlanNode, schema: Schema) -> PlanNode:
    col_to_tables: dict[str, list[str]] = {}
    for table, cols in schema.items():
        for c in cols:
            col_to_tables.setdefault(c, []).append(table)

    def tables_of(conjunct: Expr) -> set[str]:
        tables: set[str] = set()
        for c in conjunct.columns():
            for t in col_to_tables.get(c, []):
                tables.add(t)
        return tables

    def subtree_tables(node: PlanNode) -> set[str]:
        # The real tables under ``node`` in THIS scope. A ``Derived`` (FROM-subquery)
        # is opaque from the outer scope: its inner scans belong to the subquery's
        # own scope, so do not descend into it (a derived table contributes one
        # opaque relation, not its inner table set, to an enclosing join).
        from .plan import children as _children

        tables: set[str] = set()
        stack: list[PlanNode] = [node]
        while stack:
            n = stack.pop()
            if isinstance(n, Scan):
                tables.add(n.table)
            elif isinstance(n, Derived):
                continue  # boundary: inner scans are a separate scope
            else:
                stack.extend(_children(n))
        return tables

    def insert(plan: PlanNode, per_table: dict[str, list[Expr]]) -> PlanNode:
        if isinstance(plan, Scan):
            conjuncts = per_table.get(plan.table)
            if conjuncts:
                return Filter(plan, _conjoin(conjuncts))
            return plan
        if isinstance(plan, Derived):
            # A FROM-subquery is a pushdown BARRIER: an outer conjunct must never
            # be pushed into the subplan (it would alter the derived table's
            # contents -- even if the subplan happens to scan a same-named real
            # table). The subplan's own WHERE is pushed by ``go`` recursing into
            # ``Derived.input``; ``insert`` only distributes OUTER conjuncts, so
            # stop here and leave the subplan untouched.
            return plan
        if isinstance(plan, Join):
            left_tables = subtree_tables(plan.left)
            right_tables = subtree_tables(plan.right)
            left_map = {t: cs for t, cs in per_table.items() if t in left_tables}
            right_map = {t: cs for t, cs in per_table.items() if t in right_tables}
            return Join(
                insert(plan.left, left_map),
                insert(plan.right, right_map),
                plan.on_left,
                plan.on_right,
                plan.how,
                plan.on_predicate,
            )
        if isinstance(plan, SetOp):
            # A set op is a pushdown barrier: a predicate above it cannot be
            # pushed across (it applies to the combined rows). But predicates
            # already routed to a table that lives entirely in one branch are
            # forwarded into that branch only, so a WHERE inside a UNION arm
            # still reaches its own scan.
            left_tables = subtree_tables(plan.left)
            right_tables = subtree_tables(plan.right)
            left_map = {t: cs for t, cs in per_table.items() if t in left_tables}
            right_map = {t: cs for t, cs in per_table.items() if t in right_tables}
            return SetOp(
                insert(plan.left, left_map),
                insert(plan.right, right_map),
                plan.op,
                plan.distinct,
            )
        if isinstance(plan, Filter):
            # Push past existing filters; they re-wrap below.
            return Filter(insert(plan.input, per_table), plan.predicate)
        if isinstance(plan, Project):
            return Project(insert(plan.input, per_table), plan.items)
        if isinstance(plan, Aggregate):
            return Aggregate(insert(plan.input, per_table), plan.group_keys, plan.aggs)
        if isinstance(plan, Window):
            return Window(insert(plan.input, per_table), plan.funcs)
        if isinstance(plan, Sort):
            return Sort(insert(plan.input, per_table), plan.keys)
        if isinstance(plan, Limit):
            return Limit(insert(plan.input, per_table), plan.n, plan.offset)
        if isinstance(plan, Distinct):
            return Distinct(insert(plan.input, per_table))
        return plan

    def go(node: PlanNode) -> PlanNode:
        # recurse children first
        if isinstance(node, Filter):
            node = Filter(go(node.input), node.predicate)
        elif isinstance(node, Project):
            node = Project(go(node.input), node.items)
        elif isinstance(node, Aggregate):
            node = Aggregate(go(node.input), node.group_keys, node.aggs)
        elif isinstance(node, Window):
            # A predicate barrier: a WHERE above the Window is built below it
            # (Filter under Window) and stays there; conjuncts never push across a
            # Window in normal shapes (changing window partition contents would
            # be wrong), so just recurse into the input.
            node = Window(go(node.input), node.funcs)
        elif isinstance(node, Sort):
            node = Sort(go(node.input), node.keys)
        elif isinstance(node, Limit):
            node = Limit(go(node.input), node.n, node.offset)
        elif isinstance(node, Distinct):
            node = Distinct(go(node.input))
        elif isinstance(node, Derived):
            # A scope barrier for OUTER conjuncts (see ``insert``), but recurse so
            # the subplan's own WHERE (a Filter over its inner join/scan) gets
            # pushed within its own scope.
            node = Derived(go(node.input), node.alias)
        elif isinstance(node, Join):
            node = Join(go(node.left), go(node.right), node.on_left,
                        node.on_right, node.how, node.on_predicate)
        elif isinstance(node, SetOp):
            node = SetOp(go(node.left), go(node.right), node.op, node.distinct)

        if isinstance(node, Filter) and isinstance(node.input, Join):
            join = node.input
            how = join.how
            left_tables = subtree_tables(join.left)
            right_tables = subtree_tables(join.right)
            join_tables = left_tables | right_tables
            # A conjunct is safe to push below an OUTER join only into the
            # *preserved* side. Pushing into the null-supplying side would filter
            # out the null-padded unmatched rows an outer join must keep, silently
            # turning it into an inner join. FULL/CROSS have no preserved side, so
            # nothing is pushed (every conjunct stays above as a true WHERE).
            if how == "left":
                pushable_sides = left_tables
            elif how == "right":
                pushable_sides = right_tables
            elif how == "inner":
                pushable_sides = left_tables | right_tables
            elif how in ("semi", "anti"):
                # Semi/anti preserve the left side only; the right side is the
                # subquery (a separate scope). Push left conjuncts below; never
                # push into the subquery.
                pushable_sides = left_tables
            else:  # full / cross
                pushable_sides = set()

            conjuncts = _split_and(node.predicate)
            pushable: dict[str, list[Expr]] = {}
            remaining: list[Expr] = []
            for c in conjuncts:
                # A conjunct referencing a non-base column -- e.g. a cross-join
                # broadcast column (``_sq1``) produced by a flattened scalar /
                # EXISTS subquery -- depends on a column produced *above* the
                # join, so it cannot be pushed below any scan (``tables_of``
                # ignores unknown columns, which would otherwise let the
                # conjunct be mis-routed to a scan that lacks the column).
                if any(col not in col_to_tables for col in c.columns()):
                    remaining.append(c)
                    continue
                ts = tables_of(c) & join_tables
                if len(ts) == 1:
                    t = next(iter(ts))
                    if t in pushable_sides:
                        pushable.setdefault(t, []).append(c)
                    else:
                        remaining.append(c)
                else:
                    remaining.append(c)
            if pushable:
                new_input = insert(join, pushable)
                if remaining:
                    return Filter(new_input, _conjoin(remaining))
                return new_input
        return node

    return go(plan)


# --------------------------------------------------------------------------- #
# Join-side selection
# --------------------------------------------------------------------------- #


def select_join_sides(plan: PlanNode, stats: Stats) -> PlanNode:
    if not stats:
        return plan

    def est_rows(node: PlanNode) -> float:
        if isinstance(node, Scan):
            return float(stats.get(node.table, 10**9))
        if isinstance(node, Join):
            left_rows = est_rows(node.left)
            right_rows = est_rows(node.right)
            return min(left_rows * right_rows, 10**15)
        if isinstance(node, SetOp):
            # UNION is at most the sum; INTERSECT/EXCEPT at most the left. Use the
            # sum (capped) as a conservative upper bound -- this only feeds
            # join-side selection of an *enclosing* join, and a SetOp is never
            # side-swapped itself.
            return min(est_rows(node.left) + est_rows(node.right), 10**15)
        if hasattr(node, "input"):
            return est_rows(node.input)
        return 10**9

    def go(node: PlanNode) -> PlanNode:
        if isinstance(node, Join):
            left = go(node.left)
            right = go(node.right)
            if node.how in ("semi", "anti"):
                # Semi/anti preserve the left side; swapping would flip which side
                # is preserved (and the right side is a subquery, not a probe
                # input). Keep the order, just recurse into children.
                return Join(left, right, node.on_left, node.on_right,
                            node.how, node.on_predicate)
            if est_rows(right) < est_rows(left):
                # swap sides so the smaller subtree is the build (left) side.
                # Swapping a LEFT/RIGHT join flips the preserved side, so rewrite
                # how to match (left<->right); inner/cross/full are symmetric.
                how = node.how
                if how == "left":
                    how = "right"
                elif how == "right":
                    how = "left"
                return Join(right, left, node.on_right, node.on_left,
                            how, node.on_predicate)
            return Join(left, right, node.on_left, node.on_right,
                        node.how, node.on_predicate)
        if isinstance(node, SetOp):
            # Symmetric: no build/probe side to choose, just recurse so joins
            # inside each branch still get side-selected.
            return SetOp(go(node.left), go(node.right), node.op, node.distinct)
        if isinstance(node, Filter):
            return Filter(go(node.input), node.predicate)
        if isinstance(node, Project):
            return Project(go(node.input), node.items)
        if isinstance(node, Aggregate):
            return Aggregate(go(node.input), node.group_keys, node.aggs)
        if isinstance(node, Window):
            return Window(go(node.input), node.funcs)
        if isinstance(node, Sort):
            return Sort(go(node.input), node.keys)
        if isinstance(node, Limit):
            return Limit(go(node.input), node.n, node.offset)
        if isinstance(node, Distinct):
            return Distinct(go(node.input))
        if isinstance(node, Derived):
            # Recurse so joins inside the subplan get side-selected; the derived
            # table itself is one opaque relation (est_rows recurses via the
            # ``hasattr(node, "input")`` fallback for an enclosing join).
            return Derived(go(node.input), node.alias)
        return node

    return go(plan)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _split_and(e: Expr) -> list[Expr]:
    if isinstance(e, And):
        return _split_and(e.left) + _split_and(e.right)
    return [e]


def _conjoin(parts: Iterable[Expr]) -> Expr:
    parts = list(parts)
    acc = parts[0]
    for p in parts[1:]:
        acc = And(acc, p)
    return acc