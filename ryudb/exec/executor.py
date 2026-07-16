"""Plan executor: lowers physical plan nodes to cuDF operations on the GPU.

The executor walks the plan bottom-up, producing a cuDF DataFrame at each node.
Index hygiene is deliberate: scans and every reshaping op reset to a clean
RangeIndex so that Series and scalar broadcasts line up in Project/Aggregate.
"""

from __future__ import annotations

import cudf

from ..catalog import Catalog
from ..sql.optimize import optimize
from ..sql.parse import parse
from ..sql.plan import (
    Aggregate,
    Filter,
    Join,
    Limit,
    PlanNode,
    Project,
    Scan,
    Sort,
    Star,
)
from ..storage import scan
from .ops import eval_expr

_AGG_METHOD = {"SUM": "sum", "AVG": "mean", "MIN": "min", "MAX": "max", "COUNT": "count"}


class Engine:
    """Front door: parse -> optimize -> execute on GPU, returning a cuDF frame."""

    def __init__(self, catalog: Catalog):
        self.catalog = catalog

    def sql(self, sql: str) -> cudf.DataFrame:
        plan = parse(sql, self.catalog.schema_dict())
        plan = optimize(
            plan,
            self.catalog.schema_dict(),
            self.catalog.stats_dict(),
        )
        return self.execute(plan)

    def explain(self, sql: str) -> str:
        from ..sql.plan import pretty

        plan = parse(sql, self.catalog.schema_dict())
        plan = optimize(plan, self.catalog.schema_dict(), self.catalog.stats_dict())
        return pretty(plan)

    def execute(self, plan: PlanNode) -> cudf.DataFrame:
        return self._exec(plan)

    def _exec(self, node: PlanNode) -> cudf.DataFrame:
        if isinstance(node, Scan):
            return scan(self.catalog.get(node.table), node.columns)
        if isinstance(node, Filter):
            df = self._exec(node.input)
            mask = eval_expr(node.predicate, df)
            if isinstance(mask, cudf.Series):
                return df[mask]
            return df if mask else df.iloc[0:0]
        if isinstance(node, Join):
            left = self._exec(node.left)
            right = self._exec(node.right)
            return left.merge(
                right,
                left_on=node.on_left,
                right_on=node.on_right,
                how=node.how,
                suffixes=("_x", "_y"),
            )
        if isinstance(node, Aggregate):
            return self._aggregate(node)
        if isinstance(node, Project):
            return self._project(node)
        if isinstance(node, Sort):
            return self._sort(node)
        if isinstance(node, Limit):
            return self._limit(node)
        raise NotImplementedError(f"no executor for {type(node).__name__}")

    # ------------------------------------------------------------------ #
    def _aggregate(self, node: Aggregate) -> cudf.DataFrame:
        df = self._exec(node.input)
        group_keys = node.group_keys
        aggs = node.aggs

        if not group_keys:
            row: dict[str, list] = {}
            for af, n in aggs:
                if af.func == "COUNT" and isinstance(af.arg, Star):
                    row[n] = [int(len(df))]
                else:
                    col = eval_expr(af.arg, df)
                    row[n] = [_scalar_agg(af.func, col)]
            return cudf.DataFrame(row)

        work = df  # mutate in place; df is a fresh frame not reused upstream
        by_names = [gn for _, gn in group_keys]
        for ge, gn in group_keys:
            if gn not in work.columns:
                work[gn] = eval_expr(ge, df)

        arg_cols: dict[str, str] = {}
        for af, n in aggs:
            if af.func == "COUNT" and isinstance(af.arg, Star):
                continue
            tmp = f"__arg_{n}"
            work[tmp] = eval_expr(af.arg, df)
            arg_cols[n] = tmp

        grouped = work.groupby(by_names)
        out = None
        for af, n in aggs:
            if af.func == "COUNT" and isinstance(af.arg, Star):
                s = grouped.size().rename(n)
            elif af.func == "COUNT":
                s = grouped[arg_cols[n]].count().rename(n)
            else:
                s = grouped[arg_cols[n]].agg(_AGG_METHOD[af.func]).rename(n)
            out = s if out is None else cudf.concat([out, s], axis=1)
        out = out.reset_index()
        return out

    def _project(self, node: Project) -> cudf.DataFrame:
        df = self._exec(node.input)
        # Build on the input's index so Series columns align and scalar columns
        # broadcast without materializing a Python list of len(df) elements.
        out = cudf.DataFrame(index=df.index)
        for e, name in node.items:
            v = eval_expr(e, df)
            out[name] = v  # Series aligns by index; scalar broadcasts to all rows
        return out

    def _sort(self, node: Sort) -> cudf.DataFrame:
        df = self._exec(node.input)
        if not node.keys:
            return df
        by = [k.name for k, _ in node.keys]
        ascending = [a for _, a in node.keys]
        return df.sort_values(by=by, ascending=ascending)

    def _limit(self, node: Limit) -> cudf.DataFrame:
        df = self._exec(node.input)
        end = node.offset + node.n
        return df.iloc[node.offset:end]


def _scalar_agg(func: str, series) -> object:
    if func == "COUNT":
        return int(series.count())
    method = _AGG_METHOD[func]
    return getattr(series, method)()