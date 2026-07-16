"""Rule-based logical optimizer.

Phase 1 rules:
  1. Projection pruning  — each Scan reads only the columns referenced above it.
  2. Predicate pushdown  — conjuncts in a Filter above a Join that reference a
     single table are pushed down to a Filter directly above that table's Scan.
  3. Join-side selection  — when row-count statistics are available, the smaller
     subtree is placed on the left (build) side of each Join.

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
    Expr,
    Filter,
    Join,
    Limit,
    PlanNode,
    Project,
    Scan,
    Sort,
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
            return Join(rewrite(node.left), rewrite(node.right), node.on_left, node.on_right, node.how)
        if isinstance(node, Filter):
            return Filter(rewrite(node.input), node.predicate)
        if isinstance(node, Project):
            return Project(rewrite(node.input), node.items)
        if isinstance(node, Aggregate):
            return Aggregate(rewrite(node.input), node.group_keys, node.aggs)
        if isinstance(node, Sort):
            return Sort(rewrite(node.input), node.keys)
        if isinstance(node, Limit):
            return Limit(rewrite(node.input), node.n, node.offset)
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
        elif isinstance(node, Sort):
            for e, _ in node.keys:
                cols |= e.columns()
        elif isinstance(node, Join):
            cols.update(node.on_left)
            cols.update(node.on_right)
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
        return {n.table for n in walk(node) if isinstance(n, Scan)}

    def insert(plan: PlanNode, per_table: dict[str, list[Expr]]) -> PlanNode:
        if isinstance(plan, Scan):
            conjuncts = per_table.get(plan.table)
            if conjuncts:
                return Filter(plan, _conjoin(conjuncts))
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
            )
        if isinstance(plan, Filter):
            # Push past existing filters; they re-wrap below.
            return Filter(insert(plan.input, per_table), plan.predicate)
        if isinstance(plan, Project):
            return Project(insert(plan.input, per_table), plan.items)
        if isinstance(plan, Aggregate):
            return Aggregate(insert(plan.input, per_table), plan.group_keys, plan.aggs)
        if isinstance(plan, Sort):
            return Sort(insert(plan.input, per_table), plan.keys)
        if isinstance(plan, Limit):
            return Limit(insert(plan.input, per_table), plan.n, plan.offset)
        return plan

    def go(node: PlanNode) -> PlanNode:
        # recurse children first
        if isinstance(node, Filter):
            node = Filter(go(node.input), node.predicate)
        elif isinstance(node, Project):
            node = Project(go(node.input), node.items)
        elif isinstance(node, Aggregate):
            node = Aggregate(go(node.input), node.group_keys, node.aggs)
        elif isinstance(node, Sort):
            node = Sort(go(node.input), node.keys)
        elif isinstance(node, Limit):
            node = Limit(go(node.input), node.n, node.offset)
        elif isinstance(node, Join):
            node = Join(go(node.left), go(node.right), node.on_left, node.on_right, node.how)

        if isinstance(node, Filter) and isinstance(node.input, Join):
            conjuncts = _split_and(node.predicate)
            pushable: dict[str, list[Expr]] = {}
            remaining: list[Expr] = []
            join_tables = subtree_tables(node.input)
            for c in conjuncts:
                ts = tables_of(c) & join_tables
                if len(ts) == 1:
                    pushable.setdefault(next(iter(ts)), []).append(c)
                else:
                    remaining.append(c)
            if pushable:
                new_input = insert(node.input, pushable)
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
            l = est_rows(node.left)
            r = est_rows(node.right)
            return min(l * r, 10**15)
        if hasattr(node, "input"):
            return est_rows(node.input)
        return 10**9

    def go(node: PlanNode) -> PlanNode:
        if isinstance(node, Join):
            left = go(node.left)
            right = go(node.right)
            if est_rows(right) < est_rows(left):
                # swap sides so the smaller subtree is the build (left) side
                return Join(right, left, node.on_right, node.on_left, node.how)
            return Join(left, right, node.on_left, node.on_right, node.how)
        if isinstance(node, Filter):
            return Filter(go(node.input), node.predicate)
        if isinstance(node, Project):
            return Project(go(node.input), node.items)
        if isinstance(node, Aggregate):
            return Aggregate(go(node.input), node.group_keys, node.aggs)
        if isinstance(node, Sort):
            return Sort(go(node.input), node.keys)
        if isinstance(node, Limit):
            return Limit(go(node.input), node.n, node.offset)
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