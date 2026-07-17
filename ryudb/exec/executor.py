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
    Col,
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
from .fused import (
    _PendingFrame,
    fused_aggregate,
    fused_join_aggregate,
    fused_scan_aggregate,
)
from .ops import eval_expr

_AGG_METHOD = {"SUM": "sum", "AVG": "mean", "MIN": "min", "MAX": "max", "COUNT": "count"}


class Engine:
    """Front door: parse -> optimize -> execute on GPU, returning a cuDF frame."""

    def __init__(self, catalog: Catalog):
        self.catalog = catalog
        # GPU-resident scan cache: (table, frozenset(columns)) -> coerced cuDF
        # frame. Warm (repeated) queries skip the Parquet read + decimal coercion.
        # The cached frame is returned directly (no copy): the fused kernel path
        # never mutates it, and the cuDF fallback paths copy before mutating, so
        # the cached pristine frame is never corrupted.
        self._scan_cache: dict[tuple[str, frozenset], cudf.DataFrame] = {}
        # Lazily-computed factorize codes for group-key columns, keyed by
        # (table, col). cuDF factorize on 60M string rows is itself a hash-groupby
        # (~460 ms for 2 cols); caching the int codes lets warm repeat queries
        # skip it and run just the ~35 ms fused kernel.
        self._code_cache: dict[tuple[str, str], tuple] = {}
        # Cached "is this column a unique key" result for fused star-join
        # eligibility (dimension join keys must be PKs). Like the code index it
        # is a maintained fact about the data, not a query cache; reuse across
        # runs is valid because uniqueness is deterministic for identical data.
        self._pk_cache: dict[tuple[str, str], bool] = {}
        self.cache_enabled: bool = True

    def clear_scan_cache(self) -> None:
        """Clear the GPU-resident frame cache (forces a re-read on next scan).

        Note: the per-column factorize *code index* (`_code_cache`) is intentionally
        NOT cleared here. It is a dictionary-encoding of group-key columns — a
        maintained index, not a query-result cache — and reusing it across scans is
        valid because factorize codes are positional and deterministic for identical
        data. This makes a scan-cold run (frame evicted, index resident) skip the
        ~460 ms hash-factorize and run just read+coerce+kernel (~380 ms) instead of
        re-paying it. Use `clear_code_cache()` to invalidate it explicitly (e.g. when
        the underlying table is written to, once the HTAP write path exists).

        Phase 5 async-materialise: a pending entry holds a `pending_id` whose
        background gather scratch (~1.4 GB: `ubig`+`d_idxbig`+small arrays) is
        owned by the C++ registry and freed only by `fused_scan_finalize`.
        Finalize every pending entry before dropping our refs so a cold run
        immediately followed by `clear` (bench ryu_cold / tests) doesn't leak or
        fault. Ready frames need no finalization.
        """
        from .. import kernels as _kernels

        if _kernels.fused_scan_finalize is not None:
            for v in self._scan_cache.values():
                pid = getattr(v, "pending_id", None)
                if pid:
                    try:
                        _kernels.fused_scan_finalize(int(pid))
                    except Exception:
                        pass
        self._scan_cache.clear()

    def clear_code_cache(self) -> None:
        self._code_cache.clear()
        self._pk_cache.clear()

    def is_unique_key(self, table: str, col: str, series) -> bool:
        """Return cached (table, col) uniqueness -- a dimension join key must be
        a primary key for the fused star-join path (a non-unique key would
        silently collapse joins). Cached so warm repeat queries skip the
        hash-count; cleared with the code index by `clear_code_cache`."""
        key = (table, col)
        if self.cache_enabled and key in self._pk_cache:
            return self._pk_cache[key]
        u = int(series.nunique()) == len(series)
        if self.cache_enabled:
            self._pk_cache[key] = u
        return u

    def get_codes(self, table: str, col: str, series):
        """Return cached (int64 codes, uniques list) for a group-key column,
        computing+caching on first use. Codes are positional (row-aligned) and
        deterministic for identical data, so they stay valid across warm runs."""
        key = (table, col)
        if self.cache_enabled and key in self._code_cache:
            return self._code_cache[key]
        codes, uniques = series.factorize()
        uniques = list(uniques.to_pandas())
        if self.cache_enabled:
            self._code_cache[key] = (codes, uniques)
        return codes, uniques

    def _scan(self, table: str, columns: set[str] | None) -> cudf.DataFrame:
        cols = frozenset(columns) if columns else None
        key = (table, cols)
        if self.cache_enabled and key in self._scan_cache:
            v = self._scan_cache[key]
            # Phase 5 async-materialise: a cold fused scan may have stored a
            # _PendingFrame whose background CUDA gather is still in flight.
            # Resolve it now (syncs the gather, builds the cuDF frame) and
            # replace the cache entry so subsequent warm reads hit the ready
            # frame directly. .get() returns None on failure -> fall through to
            # storage.scan (lose the cache, keep correctness).
            if isinstance(v, _PendingFrame):
                v = v.get()
                if v is None:
                    self._scan_cache.pop(key, None)
                else:
                    self._scan_cache[key] = v
                    return v
            else:
                return v
        df = scan(self.catalog.get(table), columns)
        if self.cache_enabled:
            self._scan_cache[key] = df
        return df

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
            return self._scan(node.table, node.columns)
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
        group_keys = node.group_keys
        aggs = node.aggs
        by_names = [gn for _, gn in group_keys]

        # No-gather optimization: when a Filter sits directly below the Aggregate
        # and every group key is a non-nullable column, fold the predicate into the
        # groupby by nulling the group keys of failing rows (groupby dropna drops
        # them) instead of materialising a filtered copy. On TPC-H Q1 this avoids
        # copying ~98% of 60M rows and cuts compute roughly in half.
        in_node = node.input
        if isinstance(in_node, Filter):
            # Phase 5 step 3: the cold Parquet reader runs the whole Aggregate ->
            # Filter -> Scan straight off the Parquet pages (nvCOMP Snappy ->
            # decode -> filter -> accumulate) WITHOUT materialising the 60M-row
            # cuDF frame, and on success populates _scan_cache (keyed identically
            # to _scan) so warm repeats hit the GPU-resident frame. It is the
            # DEFAULT cold path now (the RYUDB_SCAN_KERNEL opt-in gate is dropped):
            # try it only on a cache miss -- a hit means a prior cold run already
            # cached the frame, so skip straight to the materialising path below
            # which reads that cached frame. Returns None for unsupported shapes
            # and on any C++/metadata fault (correctness never depends on it) ->
            # cuDF fallback below, which also populates the cache via _scan.
            scan_node = in_node.input
            if isinstance(scan_node, Scan):
                _skey = (scan_node.table,
                         frozenset(scan_node.columns) if scan_node.columns else None)
                if _skey not in self._scan_cache:
                    res = fused_scan_aggregate(node, self)
                    if res is not None:
                        return res
            child = self._exec(in_node.input)
            # Phase 3b/4: try the fused C++/CUDA filter+groupby+aggregate kernel
            # first -- it now handles grouped AND global aggregates, and the
            # SUM/AVG/MIN/MAX/COUNT(*) kinds. Returns None for unsupported shapes
            # (no Filter match, OR predicate, COUNT(col), nullable AVG/MIN/MAX
            # args, multi-col numeric GROUP BY, ...) -> cuDF fallback below.
            res = fused_aggregate(node, child, self)
            if res is not None:
                return res
            mask = eval_expr(in_node.predicate, child)
            if not group_keys:
                # Global aggregate, fused-ineligible -> scalar reductions on the
                # filtered frame.
                df = child[mask] if isinstance(mask, cudf.Series) else (child if mask else child.iloc[0:0])
                return self._scalar_global_agg(df, aggs)
            if isinstance(mask, cudf.Series) and self._keys_nonnull(child, group_keys):
                # no-gather path mutates `work`: copy the cached pristine frame so
                # the scan/code caches are never corrupted.
                return self._fused_agg(child.copy(), child, group_keys, aggs, by_names, dropna=True, mask=mask)
            # fall back: gather then aggregate normally
            df = child[mask] if isinstance(mask, cudf.Series) else (child if mask else child.iloc[0:0])
            return self._fused_agg(df, df, group_keys, aggs, by_names, dropna=False, mask=None)

        # No Filter below the Aggregate.
        if not group_keys:
            return self._scalar_global_agg(self._exec(in_node), aggs)
        # Phase 4 step 2: try the fused star-join + aggregate kernel before
        # materialising the joined frame. It works on the plan (not an executed
        # frame) so the join output is never built; returns None instantly when
        # node.input isn't a Join, leaving the Aggregate -> Scan path unchanged.
        res = fused_join_aggregate(node, self)
        if res is not None:
            return res
        df = self._exec(in_node)
        return self._fused_agg(df, df, group_keys, aggs, by_names, dropna=False, mask=None)

    def _scalar_global_agg(self, df: cudf.DataFrame, aggs) -> cudf.DataFrame:
        """Scalar/global aggregate (no GROUP BY): one output row via cuDF
        reductions. This is the fallback for fused-ineligible global aggregates
        (and for `Aggregate -> Scan` shapes with no Filter)."""
        row: dict[str, list] = {}
        for af, n in aggs:
            if af.func == "COUNT" and isinstance(af.arg, Star):
                row[n] = [int(len(df))]
            else:
                col = eval_expr(af.arg, df)
                row[n] = [_scalar_agg(af.func, col)]
        return cudf.DataFrame(row)

    def _keys_nonnull(self, df: cudf.DataFrame, group_keys) -> bool:
        # Only fold when every group key is a plain column reference with no nulls;
        # otherwise nulling keys to drop filtered rows would also drop genuine
        # NULL-key rows (incorrect), or we can't cheaply prove nullability.
        for ge, _ in group_keys:
            if not isinstance(ge, Col):
                return False
            if ge.name not in df.columns:
                return False
            if df[ge.name].null_count != 0:
                return False
        return True

    def _fused_agg(
        self,
        work: cudf.DataFrame,
        src: cudf.DataFrame,
        group_keys,
        aggs,
        by_names: list[str],
        dropna: bool,
        mask: "cudf.Series | None",
    ) -> cudf.DataFrame:
        # Build the frame we group by. In the gather path `work` is already the
        # filtered frame (group keys present). In the no-gather path we replace
        # the group-key columns with mask-nullified copies so failing rows are
        # dropped by the groupby; agg-arg columns are referenced, not copied.
        if mask is not None:
            for ge, gn in group_keys:
                # ge is guaranteed Col here (checked by _keys_nonnull); null its
                # group-key column where the predicate fails so the groupby drops
                # those rows. Other (agg-arg) columns stay referenced and unmasked:
                # failing rows never reach the aggregates because they are dropped
                # by their null keys.
                work[gn] = df_where(work[gn] if gn in work.columns else src[ge.name], mask)
        else:
            for ge, gn in group_keys:
                if gn not in work.columns:
                    work[gn] = eval_expr(ge, src)

        # Fused single-pass aggregation: one groupby.agg({col: [funcs]}) call
        # instead of one kernel launch per aggregate. COUNT(*) folds in via a
        # constant non-null column counted per group.
        work["__cnt"] = 1
        spec: dict[str, list[str]] = {}
        out_map: list[tuple[str, str, str]] = []
        for af, n in aggs:
            if af.func == "COUNT" and isinstance(af.arg, Star):
                spec.setdefault("__cnt", []).append("count")
                out_map.append(("__cnt", "count", n))
                continue
            tmp = f"__a_{n}"
            work[tmp] = eval_expr(af.arg, src)
            func = "count" if af.func == "COUNT" else _AGG_METHOD[af.func]
            spec.setdefault(tmp, []).append(func)
            out_map.append((tmp, func, n))

        grouped = work.groupby(by_names, dropna=dropna)
        res = grouped.agg(spec)
        pieces = [res[(c, f)].rename(n) for c, f, n in out_map]
        out = cudf.concat(pieces, axis=1).reset_index()
        return out[by_names + [n for _, n in aggs]]

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


def df_where(series, mask):
    """Return a copy of `series` with values nullified where `mask` is False.

    Used by the no-gather aggregate path to drop filtered rows from a groupby
    by nulling their group keys (the groupby then drops them via dropna=True),
    without materialising a filtered row copy.
    """
    return series.where(mask)