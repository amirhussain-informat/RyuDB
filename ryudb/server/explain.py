"""Structured EXPLAIN — a JSON plan tree, not the flat ``engine.explain()`` string.

``engine.explain()`` returns ``pretty(plan)`` text. For a visual explain tree
the frontend needs structure, so this module rebuilds the *optimized* plan
exactly as ``engine.sql``/``engine.explain`` do (parse -> optimize the
INSERT/MERGE source in place -> optimize the relational root) and walks the
``PlanNode`` tree into a JSON object:

    {op, est_rows, fused, detail: {...}, children: [...]}

- ``op``: the node type label (``Scan``/``Filter``/``Join``/``Aggregate``/...).
- ``est_rows``: an estimated row count. Only ``Scan`` carries a real value
  (the catalog's row count from ``stats_dict``); other nodes are ``null`` in
  Phase 1 (a future phase can propagate cardinality estimates through the tree).
- ``fused``: ``true`` when the node is an ``Aggregate`` directly over a ``Join``
  — the fused star-join+aggregate shape eligible for the C++ ``probe_*_kernel``.
  This is *eligibility*, not a guarantee the fused path fired (that depends on
  runtime cardinality, which the static plan cannot see); the field is the seed
  of the visual-explain ``fused`` badge.
- ``detail``: a few scalar fields per node (table name, join how/keys, group
  keys, ...). Rich expression rendering is a later phase; Phase 1 keeps it lean.
"""

from __future__ import annotations

from typing import Any

from ..sql.optimize import optimize
from ..sql.parse import parse
from ..sql.plan import (
    Aggregate,
    Delete,
    Derived,
    Distinct,
    Filter,
    Insert,
    Join,
    Limit,
    Merge,
    Project,
    Scan,
    SetOp,
    Sort,
    TxnControl,
    Update,
    Window,
    children,
)

# Write nodes bypass the optimizer (mirrors engine.sql); their plan tree is a
# single labelled leaf.
_WRITE_LEAVES = (Insert, Delete, Update, Merge, TxnControl)


def build_plan(sql: str, catalog: Any):
    """Parse + optimize a statement, replicating engine.sql/explain's plan
    construction exactly (so EXPLAIN shows the same plan that would run)."""
    plan = parse(sql, catalog.schema_dict())
    if isinstance(plan, Insert) and plan.source is not None:
        plan.source = optimize(
            plan.source, catalog.schema_dict(), catalog.stats_dict()
        )
    if isinstance(plan, Merge) and plan.source is not None:
        plan.source = optimize(
            plan.source, catalog.schema_dict(), catalog.stats_dict()
        )
    if not isinstance(plan, _WRITE_LEAVES):
        plan = optimize(plan, catalog.schema_dict(), catalog.stats_dict())
    return plan


def plan_tree(plan: Any, stats: dict[str, int]) -> dict[str, Any]:
    """Walk a PlanNode tree into a JSON-friendly dict."""
    if isinstance(plan, Scan):
        node: dict[str, Any] = {
            "op": "Scan",
            "est_rows": stats.get(plan.table, 0),
            "fused": False,
            "detail": {"table": plan.table, "alias": plan.alias},
        }
    elif isinstance(plan, Filter):
        node = {"op": "Filter", "est_rows": None, "fused": False, "detail": {}}
    elif isinstance(plan, Project):
        node = {
            "op": "Project",
            "est_rows": None,
            "fused": False,
            "detail": {"items": [name for _, name in plan.items]},
        }
    elif isinstance(plan, Join):
        node = {
            "op": "Join",
            "est_rows": None,
            "fused": False,
            "detail": {
                "how": plan.how,
                "on_left": list(plan.on_left),
                "on_right": list(plan.on_right),
            },
        }
    elif isinstance(plan, Aggregate):
        node = {
            "op": "Aggregate",
            "est_rows": None,
            "fused": isinstance(plan.input, Join),
            "detail": {
                "group_keys": [name for _, name in plan.group_keys],
                "aggs": [name for _, name in plan.aggs],
            },
        }
    elif isinstance(plan, Window):
        node = {
            "op": "Window",
            "est_rows": None,
            "fused": False,
            "detail": {"funcs": [name for _, name in plan.funcs]},
        }
    elif isinstance(plan, Sort):
        node = {"op": "Sort", "est_rows": None, "fused": False, "detail": {}}
    elif isinstance(plan, Limit):
        node = {
            "op": "Limit",
            "est_rows": None,
            "fused": False,
            "detail": {"n": plan.n, "offset": plan.offset},
        }
    elif isinstance(plan, Distinct):
        node = {"op": "Distinct", "est_rows": None, "fused": False, "detail": {}}
    elif isinstance(plan, Derived):
        node = {
            "op": "Derived",
            "est_rows": None,
            "fused": False,
            "detail": {"alias": plan.alias},
        }
    elif isinstance(plan, SetOp):
        node = {
            "op": "SetOp",
            "est_rows": None,
            "fused": False,
            "detail": {"op": plan.op, "distinct": plan.distinct},
        }
    elif isinstance(plan, Insert):
        node = {
            "op": "Insert",
            "est_rows": None,
            "fused": False,
            "detail": {"table": plan.table, "columns": plan.columns},
        }
    elif isinstance(plan, Delete):
        node = {
            "op": "Delete",
            "est_rows": None,
            "fused": False,
            "detail": {"table": plan.table},
        }
    elif isinstance(plan, Update):
        node = {
            "op": "Update",
            "est_rows": None,
            "fused": False,
            "detail": {
                "table": plan.table,
                "assignments": [c for c, _ in plan.assignments],
            },
        }
    elif isinstance(plan, Merge):
        node = {
            "op": "Merge",
            "est_rows": None,
            "fused": False,
            "detail": {"table": plan.table},
        }
    elif isinstance(plan, TxnControl):
        node = {
            "op": "TxnControl",
            "est_rows": None,
            "fused": False,
            "detail": {"kind": plan.kind},
        }
    else:  # unknown node — surface its type so the UI degrades gracefully
        node = {
            "op": type(plan).__name__,
            "est_rows": None,
            "fused": False,
            "detail": {},
        }
    node["children"] = [plan_tree(c, stats) for c in children(plan)]
    return node