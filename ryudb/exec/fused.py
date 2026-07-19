"""Fused filter+groupby+aggregate CUDA kernel (Numba), with a cuDF fallback.

The cuDF executor path for `Aggregate -> Filter -> Scan` issues many synchronous
GPU ops (mask, gather/noon-gather, groupby, one kernel per aggregate, concat),
each with a Python round-trip and a kernel launch. Profiling showed that this
*orchestration* — not the Parquet reader — is the dominant cost at SF<=10 (Q1
compute-only, data resident on the GPU, is ~534 ms vs DuckDB's 98 ms total).

This module replaces that orchestration with **one fused CUDA kernel** authored
via Numba `@cuda.jit`: a single pass over the device data evaluates the predicate,
computes every aggregate's argument expression, and atomically accumulates into
per-group slots. A small per-query code generator specialises the kernel to the
query's predicate and aggregate expressions (Numba cannot dispatch on Python AST
types inside a kernel, so we emit a source string and JIT it).

`fused_aggregate(node, child)` returns a cuDF DataFrame when the plan matches the
supported shape, or `None` when it does not — the caller then falls back to the
existing cuDF path, so correctness is never compromised by an unsupported edge.

Supported shape (v1, targets TPC-H Q1):
  - `Aggregate` whose input is a `Filter` (predicate folded into the kernel).
  - Group keys are `Col`s, factorisable to int codes, with
    product-of-distinct-counts * number-of-aggregates <= MAX_ACC_CELLS (a dense
    per-group accumulator is used; high-cardinality GROUP BY falls back).
  - Aggregates are `COUNT(*)` or `SUM(expr)` where `expr` is `+ - * /` arithmetic
    over numeric `Col`s and numeric literals.
  - Predicate is a conjunction (`AND`) of `Col OP literal` comparisons over
    numeric or datetime columns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import cuda

from .. import kernels as _kernels
from ..sql.plan import Aggregate, And, BinOp, Col, Filter, Join, Lit, Or, Scan, Star, walk

if TYPE_CHECKING:
    import cudf

# Per-block shared accumulator cells (float64). Sized to a safe constant so the
# kernel can declare a static shared array; the gate ensures
# n_groups * n_aggs <= MAX_ACC_CELLS. 8192 float64 = 64 KB, within RTX 3090
# per-block shared memory without opt-in.
MAX_ACC_CELLS = 4096
THREADS = 256

_KERNEL_CACHE: dict[str, object] = {}

# --- C++ kernel descriptor codes (must mirror kernels/fused.cu) -------------- #
_DT_INT64, _DT_FLOAT64 = 0, 1
_OP = {"=": 0, "!=": 1, "<": 2, "<=": 3, ">": 4, ">=": 5}
_TOP = {"+": 1, "-": 2, "*": 3, "/": 4}
_TK_COL, _TK_LIT, _TK_OP = 0, 1, 2
_AGG_COUNT, _AGG_SUM, _AGG_MIN, _AGG_MAX, _AGG_AVG = 0, 1, 2, 3, 4
_STRAT_DENSE, _STRAT_HASH = 0, 1
# Hash-table accumulator memory budget (bytes) and slot cap. Sized from row count
# (catalog has no NDV): capacity = next_pow2(min(n, 2**25, BUDGET//(nagg*8))).
_HASH_ACC_BUDGET = 2 * 10**9
_HASH_CAP_MAX = 1 << 25  # 33.5M slots


class _PendingFrame:
    """A lazily-materialised scan frame awaiting a background CUDA gather.

    Phase 5 async-materialise: when the C++ fused scan runs the cache-populate
    `materialise_kernel` on a side stream, it returns a `pending_id` (>0) and
    hands ownership of the gather scratch to a registry. The cold query stores
    this object in `engine._scan_cache` instead of a ready `cudf.DataFrame`; the
    aggregate result itself is already computed and returned synchronously.

    The first warm read calls `.get()`, which runs `fused_scan_finalize` (syncs
    the gather's completion event `E_mat` so the per-column numba buffers are
    fully written and visible), then builds the cuDF Series from those buffers.
    Building the Series *after* the `E_mat` sync preserves the step-3 race fix
    (`as_column`'s async copy must read completed data). `frame_bufs` are kept
    alive here until the Series are built.

    If `finalize`/build fails, `.get()` returns `None` so `Engine._scan` falls
    through to `storage.scan` -- we lose the cache but keep correctness (the
    async side effect must never break a query).
    """

    __slots__ = ("_bufs", "_meta", "_pid", "_proj", "_frame")

    def __init__(self, frame_bufs, frame_meta, pending_id, proj):
        self._bufs = frame_bufs
        self._meta = frame_meta  # list[(name, out_kind)]
        self._pid = int(pending_id)
        self._proj = proj
        self._frame = None  # cached cudf.DataFrame once built

    def get(self):
        if self._frame is not None:
            return self._frame
        import cudf
        import cudf.core.column as _cc
        try:
            # Sync the background gather (no-op if already done) and free its
            # scratch. Sole freer of the materialise device buffers.
            if _kernels.fused_scan_finalize is not None:
                _kernels.fused_scan_finalize(self._pid)
            cols_out: dict[str, "cudf.Series"] = {}
            for c, (name, ok) in enumerate(self._meta):
                series = cudf.Series._from_column(_cc.as_column(self._bufs[c]))
                if ok == 3:
                    series = series.astype("datetime64[s]")
                cols_out[name] = series
            self._frame = cudf.DataFrame(
                {n: cols_out[n] for n in sorted(self._proj)}
            )
            # Buffers are now owned by the cuDF columns; release our refs.
            self._bufs = None
            self._pid = 0
            return self._frame
        except Exception:
            # Never let the async side effect break a query: drop the cache so
            # the warm path re-scans via storage.scan.
            self._frame = None
            self._bufs = None
            self._pid = 0
            return None

    @property
    def pending_id(self) -> int:
        return self._pid


def fused_aggregate(node: Aggregate, child, engine=None) -> "cudf.DataFrame | None":
    """Try to run `node` as a fused CUDA kernel over `child`.

    `child` is the already-executed frame below the Aggregate's Filter (i.e. the
    Scan result, decimals coerced to float64). `engine`, when given, supplies a
    per-(table,col) factorize-code cache so warm repeat queries skip the ~460 ms
    string-factorize. Returns a cuDF DataFrame, or `None` if the plan shape is
    not supported (caller falls back to cuDF).
    """
    spec = _match(node, child)
    if spec is None:
        return None

    n = len(child)
    if n == 0:
        # A global aggregate (no GROUP BY) still returns one row with COUNT=0 and
        # NULL for the other aggs; a grouped aggregate returns zero rows.
        if not spec["group_keys"]:
            return _global_null_result(spec)
        return _empty_result(spec)

    # Phase 3b: prefer the C++/CUDA backend (nvcc+pybind11) when built. It handles
    # both the dense low-cardinality path AND the hash-table high-cardinality path
    # (numeric group keys read directly -- no factorize). Returns None if the shape
    # isn't handled here (e.g. OR predicate, string high-card) -> fall through to
    # the Numba dense path, then cuDF.
    table = _source_table(node)
    if _kernels.is_available:
        try:
            res = _run_cpp(spec, child, engine, table)
        except Exception:  # noqa: BLE001 -- never let a C++ fault break correctness
            res = None
        if res is not None:
            return res

    # Resolve the source table name for the group-key code cache (best-effort:
    # walk the subtree for a Scan). Falls back to None -> no caching. (`table` was
    # already computed above for the C++ path; reuse it here.)
    # ---- data prep ------------------------------------------------------- #
    # The Numba dense kernel only supports grouped COUNT(*) and SUM; AVG/MIN/MAX
    # and the global-aggregate shape are handled by the C++ path above. If we get
    # here with an unsupported shape, bail so the executor falls back to cuDF.
    if not spec["group_keys"] or any(af.func in ("AVG", "MIN", "MAX") for af, _ in spec["aggs"]):
        return None

    # Factorise each group-key column to int codes; compute row-major strides
    # (last key varies fastest, stride 1). Use the engine's code cache when
    # available so warm repeats skip cuDF's 460 ms hash-factorize.
    code_arrays = []
    uniques = []
    strides = []
    sizes = []
    for ge, _gn in spec["group_keys"]:
        if engine is not None and table is not None:
            codes, uniq = engine.get_codes(table, ge.name, child[ge.name])
            code_arrays.append(codes)
            sizes.append(len(uniq))
            uniques.append(uniq)
        else:
            codes, uniq = child[ge.name].factorize()
            code_arrays.append(codes)
            sizes.append(len(uniq))
            uniques.append(list(uniq.to_pandas()))
    # strides: stride_j = product(sizes[k] for k>j)
    for j in range(len(sizes)):
        strides.append(int(np.prod(sizes[j + 1:], dtype=np.int64)))
    n_groups = int(np.prod(sizes, dtype=np.int64)) if sizes else 1

    nagg = len(spec["aggs"])
    if n_groups * nagg > MAX_ACC_CELLS:
        return None  # too many accumulator cells for shared memory -> fall back

    # Bind device arrays for every column referenced by predicate/agg-args.
    # Datetime columns are normalised to int64 seconds so date literals compare
    # correctly regardless of the stored time unit.
    arrays: dict[str, object] = {}
    date_cols: set[str] = set()
    for name in spec["cols_used"]:
        col = child[name]
        if np.issubdtype(col.dtype, np.datetime64):
            col = _to_int64_seconds(col)
            date_cols.add(name)
        arrays[name] = cuda.as_cuda_array(col)

    # Bind code arrays for group keys.
    code_dev = [cuda.as_cuda_array(c) for c in code_arrays]

    # ---- codegen --------------------------------------------------------- #
    src, call_args, arg_names = _codegen(spec, n_groups, nagg, strides, date_cols)
    kernel = _compile(src)

    # ---- launch ---------------------------------------------------------- #
    gacc = cuda.to_device(np.zeros(n_groups * nagg, dtype=np.float64))
    blocks = (n + THREADS - 1) // THREADS
    # call_args maps positional kernel args; build the actual arg list in the
    # order the generated signature declares them.
    kargs = _bind_args(arg_names, arrays, code_dev, spec, n)
    kernel[blocks, THREADS](*kargs, gacc, n_groups, nagg, n)
    cuda.synchronize()

    # ---- read-out -------------------------------------------------------- #
    acc = gacc.copy_to_host().reshape(n_groups, nagg)
    return _build_result(acc, spec, uniques, sizes, strides, n_groups)


# --------------------------------------------------------------------------- #
# Shape matching
# --------------------------------------------------------------------------- #


def _source_table(node: Aggregate) -> "str | None":
    """Best-effort: find the Scan feeding this Aggregate and return its table
    name (used as the code-cache key). Returns None if no Scan is present."""
    try:
        scan = next(n for n in walk(node) if isinstance(n, Scan))
    except StopIteration:
        return None
    return getattr(scan, "table", None)


def _match(node: Aggregate, child) -> "dict | None":
    if not isinstance(node.input, Filter):
        return None
    pred = node.input.predicate

    group_keys = node.group_keys
    aggs = node.aggs
    # NOTE: empty group_keys is allowed -- a global aggregate (single group).

    # Group keys must be plain columns present in the frame, and non-null: the
    # kernel factorises group keys (NA -> -1 code), which would drop genuine
    # NULL-key groups. Defer to cuDF, whose groupby(dropna=False) keeps them
    # (matching DuckDB, which retains NULL groups). See test_null_group_keys.
    for ge, _gn in group_keys:
        if not isinstance(ge, Col) or ge.name not in child.columns:
            return None
        if child[ge.name].null_count != 0:
            return None

    # Aggregates: COUNT(*), or SUM/AVG/MIN/MAX over a numeric arithmetic expr.
    # COUNT(col) is deferred (the kernel counts passing rows, not non-nulls).
    # AVG/MIN/MAX require their arg columns to be non-null: the kernel reads raw
    # device values and does not skip nulls, and AVG = sum / passing-row-count is
    # only correct when the arg is non-null on every passing row. Nullable args
    # defer to cuDF (which skips nulls correctly).
    for af, _n in aggs:
        if af.func == "COUNT" and isinstance(af.arg, Star):
            continue
        if af.func in ("SUM", "AVG", "MIN", "MAX"):
            if not _is_numeric_expr(af.arg, child):
                return None
            if af.func in ("AVG", "MIN", "MAX") and not _arg_cols_nonnull(af.arg, child):
                return None
            continue
        return None  # COUNT(col) and anything else -> fall back

    # Predicate: conjunction of Col OP literal comparisons (numeric/datetime).
    if not _is_supported_predicate(pred, child):
        return None

    cols_used: set[str] = set()
    # Only predicate/agg-arg columns need raw device arrays in the kernel;
    # group-key columns are accessed via their factorised int codes instead.
    for af, _ in aggs:
        if not (af.func == "COUNT" and isinstance(af.arg, Star)):
            cols_used |= af.arg.columns()
    cols_used |= pred.columns()

    # Ensure every referenced column exists.
    for c in cols_used:
        if c not in child.columns:
            return None

    return {
        "group_keys": group_keys,
        "aggs": aggs,
        "predicate": pred,
        "cols_used": sorted(cols_used),
    }


def _arg_cols_nonnull(e, child) -> bool:
    """True if every column referenced by `e` has no nulls (kernel skips nothing)."""
    for name in e.columns():
        if name not in child.columns or child[name].null_count != 0:
            return False
    return True


def _is_numeric_expr(e, child) -> bool:
    """True if `e` is arithmetic over numeric columns and numeric literals."""
    if isinstance(e, Col):
        return e.name in child.columns and _is_numeric_dtype(child[e.name].dtype)
    if isinstance(e, Lit):
        return e.dtype in ("int", "float", "bool") or isinstance(e.value, (int, float))
    if isinstance(e, BinOp) and e.op in ("+", "-", "*", "/"):
        return _is_numeric_expr(e.left, child) and _is_numeric_expr(e.right, child)
    return False


def _is_supported_predicate(e, child) -> bool:
    if isinstance(e, And):
        return _is_supported_predicate(e.left, child) and _is_supported_predicate(e.right, child)
    if isinstance(e, Or):
        return _is_supported_predicate(e.left, child) and _is_supported_predicate(e.right, child)
    if isinstance(e, BinOp) and e.op in ("=", "!=", "<", "<=", ">", ">="):
        lcol = _col_ref(e.left)
        rcol = _col_ref(e.right)
        col = lcol or rcol
        if col is None or col not in child.columns:
            return False
        dt = child[col].dtype
        if not (_is_numeric_dtype(dt) or np.issubdtype(dt, np.datetime64)):
            return False
        # the other side must be a literal
        other = e.right if lcol else e.left
        return isinstance(other, Lit)
    return False


def _col_ref(e):
    if isinstance(e, Col):
        return e.name
    return None


def _is_numeric_dtype(dt) -> bool:
    return np.issubdtype(dt, np.number) or "float" in str(dt) or "int" in str(dt)


def _to_int64_seconds(series):
    """Normalise a datetime column to int64 seconds since epoch."""
    try:
        return series.astype("datetime64[s]").astype("int64")
    except Exception:  # noqa: BLE001
        return series.astype("int64")


# --------------------------------------------------------------------------- #
# Expression emission
# --------------------------------------------------------------------------- #

_CMP = {"=": "==", "!=": "!=", "<": "<", "<=": "<=", ">": ">", ">=": ">="}


def _emit_expr(e) -> str:
    """Emit a numeric agg-arg expression referencing per-column arrays."""
    if isinstance(e, Col):
        return f"c_{e.name}[i]"
    if isinstance(e, Lit):
        if e.dtype == "int" or (isinstance(e.value, int) and not isinstance(e.value, bool)):
            return str(int(e.value))
        if e.dtype == "float" or isinstance(e.value, float):
            return repr(float(e.value))
        if e.dtype == "bool":
            return repr(bool(e.value))
        raise ValueError(f"unsupported literal in fused kernel: {e.value!r}")
    if isinstance(e, BinOp) and e.op in ("+", "-", "*", "/"):
        return f"({_emit_expr(e.left)} {e.op} {_emit_expr(e.right)})"
    raise ValueError(f"unsupported expression in fused kernel: {e!r}")


def _lit_seconds(lit: Lit) -> str:
    """A date literal compared to a datetime column -> int64 seconds since epoch.

    sqlglot leaves the literal's dtype as '' for `date '...'` casts, so we convert
    based on the column it is compared against (caller has already routed here).
    """
    return str(int(np.datetime64(lit.value, "s").astype("int64")))


def _emit_pred(e, date_cols: set[str]) -> str:
    if isinstance(e, And):
        return f"({_emit_pred(e.left, date_cols)} and {_emit_pred(e.right, date_cols)})"
    if isinstance(e, Or):
        return f"({_emit_pred(e.left, date_cols)} or {_emit_pred(e.right, date_cols)})"
    if isinstance(e, BinOp) and e.op in _CMP:
        # One side is a Col, the other a Lit. If the column is a datetime column,
        # the literal is a date -> emit int64 seconds to match the (already
        # converted) int64-seconds column array.
        if isinstance(e.left, Col) and isinstance(e.right, Lit):
            col, lit, swapped = e.left, e.right, False
        elif isinstance(e.left, Lit) and isinstance(e.right, Col):
            col, lit, swapped = e.right, e.left, True
        else:
            raise ValueError(f"unsupported comparison in fused kernel: {e!r}")
        col_str = f"c_{col.name}[i]"
        if col.name in date_cols:
            lit_str = _lit_seconds(lit)
        else:
            lit_str = _emit_expr(lit)
        if swapped:
            return f"({lit_str} {_CMP[e.op]} {col_str})"
        return f"({col_str} {_CMP[e.op]} {lit_str})"
    raise ValueError(f"unsupported predicate in fused kernel: {e!r}")


# --------------------------------------------------------------------------- #
# Codegen
# --------------------------------------------------------------------------- #


def _codegen(spec, n_groups, nagg, strides, date_cols):
    """Emit a specialised kernel source string.

    Returns (source, call_args_description, arg_names) where arg_names is the
    ordered list of array-argument names the kernel declares (each bound to a
    device array at launch), followed positionally by (gacc, n_groups, nagg, n).
    """
    # Argument names: one per referenced column (c_<col>) plus group-key code
    # arrays (k_<col>). Keep a stable order.
    col_args = [f"c_{c}" for c in spec["cols_used"]]
    key_args = [f"k_{ge.name}" for ge, _ in spec["group_keys"]]
    arg_names = col_args + key_args
    sig = ", ".join(arg_names + ["gacc", "n_groups", "nagg", "n"])

    # Group index expression: sum(code_j[i] * stride_j)
    gexpr = " + ".join(f"{k}[i] * {s}" for k, s in zip(key_args, strides)) or "0"

    # Predicate
    pred_src = _emit_pred(spec["predicate"], date_cols)

    # Aggregates: build per-agg accumulation lines. SUM(expr) -> atomic add of
    # the emitted expression; COUNT(*) -> atomic add of 1.0.
    agg_lines = []
    for slot, (af, _n) in enumerate(spec["aggs"]):
        if af.func == "COUNT" and isinstance(af.arg, Star):
            val = "1.0"
        else:  # SUM
            val = _emit_expr(af.arg)
        agg_lines.append(f"        cuda.atomic.add(sh, g * nagg + {slot}, {val})")
    agg_block = "\n".join(agg_lines)

    src = f"""
from numba import cuda, float64

def _kernel({sig}):
    sh = cuda.shared.array({MAX_ACC_CELLS}, dtype=float64)
    t = cuda.threadIdx.x
    nga = n_groups * nagg
    for k in range(t, nga, cuda.blockDim.x):
        sh[k] = 0.0
    cuda.syncthreads()
    i = cuda.grid(1)
    if i < n and {pred_src}:
        g = {gexpr}
{agg_block}
    cuda.syncthreads()
    for k in range(t, nga, cuda.blockDim.x):
        cuda.atomic.add(gacc, k, sh[k])
"""
    return src, None, arg_names


def _compile(src: str):
    if src in _KERNEL_CACHE:
        return _KERNEL_CACHE[src]
    ns: dict = {}
    exec(compile(src, "<fused_kernel>", "exec"), ns)
    kernel = cuda.jit(ns["_kernel"])
    _KERNEL_CACHE[src] = kernel
    return kernel


def _bind_args(arg_names, arrays, code_dev, spec, n):
    """Build the positional arg list matching the generated signature order."""
    args = []
    for name in arg_names:
        if name.startswith("c_"):
            args.append(arrays[name[2:]])
        else:  # k_<col>
            args.append(code_dev.pop(0))
    return args


# --------------------------------------------------------------------------- #
# Result assembly
# --------------------------------------------------------------------------- #


def _build_result(acc, spec, uniques, sizes, strides, n_groups):
    import cudf

    # Enumerate group cells in row-major order; emit those with count(*) > 0.
    # The COUNT(*) slot (if any) determines non-empty groups; otherwise use any
    # agg slot being nonzero is unsafe (a real sum can be 0), so require a
    # COUNT(*) or fall back to "any nonzero across aggs".
    cnt_slot = None
    for slot, (af, _n) in enumerate(spec["aggs"]):
        if af.func == "COUNT" and isinstance(af.arg, Star):
            cnt_slot = slot
            break

    key_cols = [gn for _, gn in spec["group_keys"]]
    out_cols: dict = {gn: [] for gn in key_cols}
    for af, n in spec["aggs"]:
        out_cols[n] = []

    for g in range(n_groups):
        if cnt_slot is not None:
            if acc[g, cnt_slot] <= 0:
                continue
        elif not np.any(acc[g] != 0):
            continue
        # decode group index back to per-key codes
        rem = g
        for j, size in enumerate(sizes):
            code = rem // strides[j]
            rem = rem % strides[j]
            out_cols[key_cols[j]].append(uniques[j][code])
        for slot, (af, n) in enumerate(spec["aggs"]):
            out_cols[n].append(float(acc[g, slot]))

    # Preserve dtypes: group-key label columns keep the original column dtype
    # (string here); agg columns are float64. COUNT(*) output cast to int64.
    data = {}
    for gn in key_cols:
        data[gn] = out_cols[gn]
    for af, n in spec["aggs"]:
        col = out_cols[n]
        if af.func == "COUNT" and isinstance(af.arg, Star):
            data[n] = [int(x) for x in col]
        else:
            data[n] = col
    return cudf.DataFrame(data)


def _empty_result(spec):
    import cudf

    data = {gn: [] for _, gn in spec["group_keys"]}
    for af, n in spec["aggs"]:
        data[n] = []
    return cudf.DataFrame(data)


def _global_null_result(spec):
    """One-row result for a global aggregate over zero matching rows: COUNT(*)=0
    and NULL (NaN) for every other aggregate (SQL semantics)."""
    import cudf

    data: dict = {}
    for af, n in spec["aggs"]:
        if af.func == "COUNT" and isinstance(af.arg, Star):
            data[n] = cudf.Series([0], dtype=np.int64)
        else:
            data[n] = cudf.Series([np.nan], dtype=np.float64)
    return cudf.DataFrame(data)


# --------------------------------------------------------------------------- #
# C++/CUDA backend (nvcc + pybind11): descriptor lowering + frame assembly
# --------------------------------------------------------------------------- #


def _dev_ptr(arr) -> int:
    """Raw device pointer (int) from a cuDF/numpy array exposing CAI."""
    return int(arr.__cuda_array_interface__["data"][0])


def _next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p <<= 1
    return p


def _lit_to_double(lit: Lit) -> float:
    """Numeric literal -> double (mirrors _emit_expr; date lits handled by caller)."""
    v = lit.value
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    return float(v)


def _to_postfix(e, bind_named) -> list[tuple[int, int, float]]:
    """Lower an agg-arg Expr to postfix tokens: (kind, col_or_op, lit).

    kind: _TK_COL (col idx in 2nd), _TK_LIT (lit in 3rd), _TK_OP (op code in 2nd).
    """
    if isinstance(e, Col):
        return [(_TK_COL, bind_named(e.name), 0.0)]
    if isinstance(e, Lit):
        return [(_TK_LIT, 0, _lit_to_double(e))]
    if isinstance(e, BinOp) and e.op in _TOP:
        return _to_postfix(e.left, bind_named) + _to_postfix(e.right, bind_named) + [
            (_TK_OP, _TOP[e.op], 0.0)
        ]
    raise ValueError(f"unsupported agg expr in C++ lowering: {e!r}")


def _flatten_and_pred(e, bind_named, child) -> list[tuple[int, int, float]] | None:
    """Flatten an AND-of-comparisons predicate to (col_idx, op, lit) tuples.

    Returns None if the predicate contains OR or an unsupported shape (the C++
    kernel only evaluates conjunctions; OR falls back to Numba/cuDF).
    """
    if isinstance(e, And):
        left = _flatten_and_pred(e.left, bind_named, child)
        right = _flatten_and_pred(e.right, bind_named, child)
        if left is None or right is None:
            return None
        return left + right
    if isinstance(e, BinOp) and e.op in _OP:
        lcol = e.left if isinstance(e.left, Col) else None
        rcol = e.right if isinstance(e.right, Col) else None
        col = lcol or rcol
        lit = e.right if lcol else e.left
        if col is None or not isinstance(lit, Lit):
            return None
        idx = bind_named(col.name)
        # Date column -> literal is a date; convert to int64 seconds (the column
        # was bound as int64 seconds). Else numeric literal.
        if np.issubdtype(child[col.name].dtype, np.datetime64):
            lv = float(int(np.datetime64(lit.value, "s").astype("int64")))
        else:
            lv = _lit_to_double(lit)
        op = _OP[e.op]
        # Preserve direction: if the literal was on the left, mirror the operator.
        if lcol is None and rcol is not None:
            op = {0: 0, 1: 1, 2: 4, 3: 5, 4: 2, 5: 3}[op]  # swap < <-> > , <= <-> >=
        return [(idx, op, lv)]
    return None


def _run_cpp(spec, child, engine, table) -> "cudf.DataFrame | None":
    """Lower `spec` to descriptors, run the C++ fused kernel, build a cuDF frame.

    Returns None when the shape isn't handled by the C++ path (OR predicate,
    string high-cardinality GROUP BY -- deferred) or the hash table overflowed;
    the caller then falls back to Numba/cuDF. `table` is the source table name for
    the engine's factorize-code cache (None -> uncached factorize).
    """
    n = len(child)
    nagg = len(spec["aggs"])
    ngkey = len(spec["group_keys"])
    # AVG stores a running sum + needs a hidden per-group passing-row count slot
    # (the denominator). nagg_eff is the true internal slot count, used for the
    # dense-accumulator shared-memory gate.
    has_avg = any(af.func == "AVG" for af, _ in spec["aggs"])
    nagg_eff = nagg + (1 if has_avg else 0)

    # Column binding: name -> index for predicate/agg-arg cols; group-key arrays
    # get their own indices appended (they may bind codes, not the raw column).
    col_ptrs: list[int] = []
    col_dtypes: list[int] = []
    name_idx: dict[str, int] = {}
    date_cols: set[str] = set()
    # Hold references to any temporary arrays (datetime->int64 casts, factorize
    # codes) so cuDF does not free their device buffers before the kernel reads
    # them via the raw pointers we stashed in col_ptrs.
    _kept: list[object] = []

    def bind_named(name: str) -> int:
        if name in name_idx:
            return name_idx[name]
        col = child[name]
        if np.issubdtype(col.dtype, np.datetime64):
            arr = _to_int64_seconds(col)
            _kept.append(arr)
            ptr, dt = _dev_ptr(arr), _DT_INT64
            date_cols.add(name)
        elif np.issubdtype(col.dtype, np.integer) or "int" in str(col.dtype):
            ptr, dt = _dev_ptr(col), _DT_INT64
        elif np.issubdtype(col.dtype, np.floating) or "float" in str(col.dtype):
            ptr, dt = _dev_ptr(col), _DT_FLOAT64
        else:
            raise ValueError(f"C++ kernel cannot bind non-numeric column {name}")
        idx = len(col_ptrs)
        col_ptrs.append(ptr)
        col_dtypes.append(dt)
        name_idx[name] = idx
        return idx

    # --- group-key encoding: numeric -> HASH direct; else factorize -> DENSE ---
    gkey_decoders: list[tuple] = []  # ('codes', uniques) | ('int',) | ('datetime',)
    gkey_idx: list[int] = []
    gkey_stride: list[int] = []
    strategy = _STRAT_DENSE
    n_groups = 1
    capacity = 0

    def _numeric_key(ge) -> bool:
        dt = child[ge.name].dtype
        return np.issubdtype(dt, np.integer) or "int" in str(dt) or np.issubdtype(dt, np.datetime64)

    all_numeric = all(_numeric_key(ge) for ge, _ in spec["group_keys"])

    if ngkey == 0:
        # Global aggregate (no GROUP BY): a single group, no keys. Always DENSE
        # with n_groups=1. (all([]) is True, so without this guard the empty-keys
        # case would wrongly enter the HASH branch and hit the ngkey!=1 check.)
        strategy = _STRAT_DENSE
        n_groups = 1
        capacity = 0
    elif all_numeric and ngkey == 1:
        # HASH direct: bind the SINGLE int64 group key raw (datetime -> seconds).
        # The C++ hash kernel reads p.gkey_idx[0] and uses atomicCAS-on-key, so
        # only a single int64 column can bind here. Multi-column numeric GROUP BY
        # falls through to the factorize->stride-combine branch below (collapse the
        # key tuple to one int64 perfect-hash code -> same single-int64 hash_kernel).
        strategy = _STRAT_HASH
        capacity = _next_pow2(min(n, _HASH_CAP_MAX, _HASH_ACC_BUDGET // (nagg * 8)))
        if capacity < 4:
            return None
        for ge, _gn in spec["group_keys"]:
            col = child[ge.name]
            if np.issubdtype(col.dtype, np.datetime64):
                arr = _to_int64_seconds(col)
                gkey_decoders.append(("datetime",))
            else:
                arr = col
                gkey_decoders.append(("int",))
            _kept.append(arr)
            idx = len(col_ptrs)
            col_ptrs.append(_dev_ptr(arr))
            col_dtypes.append(_DT_INT64)
            gkey_idx.append(idx)
    else:
        # Non-int (string/float) keys, or multi-column numeric keys. Factorize each
        # key col (cached) -> per-col int64 codes; compute row-major strides. If the
        # full product of per-cardinalities fits the DENSE accumulator (<= MAX_ACC_CELLS)
        # bind the per-col codes + strides for dense_kernel (unchanged low-card path).
        # Otherwise collapse the key TUPLE to a single int64 code (sum code_j*stride_j
        # -- a lossless perfect hash of the tuple) and feed it to the existing
        # single-int64 hash_kernel (global HT, high-card). The kernel reads gkey_idx[0]
        # unchanged; keys are reconstructed at read-out by decomposing the code back to
        # per-col codes (// stride % size) and mapping via the cached uniques.
        sizes: list[int] = []
        uniques: list = []
        per_col_codes: list = []
        for ge, _gn in spec["group_keys"]:
            if engine is not None and table is not None:
                codes, uniq = engine.get_codes(table, ge.name, child[ge.name])
            else:
                codes, uniq = child[ge.name].factorize()
                uniq = list(uniq.to_pandas())
            per_col_codes.append(codes)
            sizes.append(len(uniq))
            uniques.append(uniq)
        for j in range(len(sizes)):
            gkey_stride.append(int(np.prod(sizes[j + 1:], dtype=np.int64)))
        n_groups = int(np.prod(sizes, dtype=np.int64)) if sizes else 1
        if n_groups * nagg_eff <= MAX_ACC_CELLS:
            # DENSE: bind each per-col code col; dense_kernel combines them in-kernel.
            for codes, uniq in zip(per_col_codes, uniques):
                idx = len(col_ptrs)
                _kept.append(codes)
                col_ptrs.append(_dev_ptr(codes))
                col_dtypes.append(_DT_INT64)
                gkey_idx.append(idx)
                gkey_decoders.append(("codes", uniq))
            strategy = _STRAT_DENSE
        else:
            # HASH: combine per-col codes into one int64 code = the DENSE group index
            # (perfect hash). Vectorized cuDF, no host loop; reuses the cached factorize.
            combined = per_col_codes[0].astype("int64") * gkey_stride[0]
            for j in range(1, len(per_col_codes)):
                combined = combined + per_col_codes[j].astype("int64") * gkey_stride[j]
            capacity = _next_pow2(min(n, _HASH_CAP_MAX, _HASH_ACC_BUDGET // (nagg * 8)))
            if capacity < 4:
                return None
            idx = len(col_ptrs)
            _kept.append(combined)
            col_ptrs.append(_dev_ptr(combined))
            col_dtypes.append(_DT_INT64)
            gkey_idx.append(idx)
            gkey_decoders.append(
                ("codes_strided", uniques, list(gkey_stride), sizes)
            )
            strategy = _STRAT_HASH

    # --- predicate (conjunction only) ---
    pred = _flatten_and_pred(spec["predicate"], bind_named, child)
    if pred is None:
        return None
    pred_col = np.array([p[0] for p in pred], dtype=np.int32)
    pred_op = np.array([p[1] for p in pred], dtype=np.int32)
    pred_lit = np.array([p[2] for p in pred], dtype=np.float64)

    # --- aggregates ---
    # Kind per visible agg. AVG stores a running SUM in its slot and is divided by
    # a hidden per-group passing-row-count slot at read-out; in-kernel AVG behaves
    # exactly like SUM. The hidden count slot (one, when any AVG is present) is
    # appended after the visible aggs as an AGG_COUNT with empty tokens.
    _AGG_KIND = {"COUNT": _AGG_COUNT, "SUM": _AGG_SUM, "AVG": _AGG_AVG,
                 "MIN": _AGG_MIN, "MAX": _AGG_MAX}
    agg_kind: list[int] = []
    agg_tok_start: list[int] = []
    agg_tok_len: list[int] = []
    tok_kind: list[int] = []
    tok_col: list[int] = []
    tok_lit: list[float] = []
    tok_op: list[int] = []
    for af, _n in spec["aggs"]:
        start = len(tok_kind)
        if af.func == "COUNT" and isinstance(af.arg, Star):
            agg_kind.append(_AGG_COUNT)
            agg_tok_start.append(start)
            agg_tok_len.append(0)
            continue
        toks = _to_postfix(af.arg, bind_named)
        for k, a, b in toks:
            tok_kind.append(k)
            tok_lit.append(b)
            if k == _TK_OP:
                # _to_postfix packs the op code in the 2nd tuple slot; route it to
                # tok_op (the array the kernel's eval_agg reads for TK_OP tokens).
                tok_col.append(0)
                tok_op.append(a)
            else:
                tok_col.append(a)
                tok_op.append(0)
        agg_kind.append(_AGG_KIND[af.func])
        agg_tok_start.append(start)
        agg_tok_len.append(len(toks))

    # The HASH path now supports MIN/MAX/AVG too (hash_kernel mirrors dense_kernel's
    # per-slot dispatch; AVG stores a running SUM + hidden COUNT, divided at read-out;
    # MIN/MAX use +/-inf-inited slots + atomic_min/max_d). No deferral here.

    # Hidden per-group passing-row count slot: the AVG denominator. One slot,
    # appended after the visible aggs; not emitted as an output column.
    hidden_count_idx = None
    if has_avg:
        hidden_count_idx = len(agg_kind)
        agg_kind.append(_AGG_COUNT)
        agg_tok_start.append(len(tok_kind))
        agg_tok_len.append(0)

    col_ptrs_np = np.array(col_ptrs, dtype=np.int64)
    col_dtypes_np = np.array(col_dtypes, dtype=np.int32)
    gkey_idx_np = np.array(gkey_idx, dtype=np.int32)
    gkey_stride_np = np.array(gkey_stride, dtype=np.int64)
    tok_kind_np = np.array(tok_kind, dtype=np.int32)
    tok_col_np = np.array(tok_col, dtype=np.int32)
    tok_lit_np = np.array(tok_lit, dtype=np.float64)
    tok_op_np = np.array(tok_op, dtype=np.int32)
    agg_kind_np = np.array(agg_kind, dtype=np.int32)
    agg_tok_start_np = np.array(agg_tok_start, dtype=np.int32)
    agg_tok_len_np = np.array(agg_tok_len, dtype=np.int32)

    # Per-slot accumulator init: +inf for MIN, -inf for MAX, 0 otherwise. DENSE tiles
    # per_slot across n_groups (shared-mem acc); HASH tiles it across `capacity` (the
    # global HT accumulator) when any MIN/MAX is present -- the host copies acc_init
    # into the HASH acc and hash_kernel's atomic_min/max_d lower from +/-inf. SUM/COUNT
    # and AVG (running sum + hidden count) all start at 0, so a HASH plan with no
    # MIN/MAX passes an empty acc_init and the host memsets the acc to 0.
    per_slot = np.array(
        [np.inf if k == _AGG_MIN else -np.inf if k == _AGG_MAX else 0.0
         for k in agg_kind], dtype=np.float64,
    )
    if strategy == _STRAT_DENSE:
        acc_init = np.tile(per_slot, n_groups)
    elif strategy == _STRAT_HASH and any(k in (_AGG_MIN, _AGG_MAX) for k in agg_kind):
        acc_init = np.tile(per_slot, capacity)
    else:
        acc_init = np.empty(0, dtype=np.float64)

    overflow, n_out, keys_list, aggs_list = _kernels.fused_agg(
        col_ptrs_np, col_dtypes_np, gkey_idx_np, gkey_stride_np,
        pred_col, pred_op, pred_lit, agg_kind_np, agg_tok_start_np, agg_tok_len_np,
        tok_kind_np, tok_col_np, tok_lit_np, tok_op_np, acc_init,
        int(strategy), int(n_groups), int(capacity), int(n),
    )
    if overflow != 0:
        return None  # hash table filled -> cuDF fallback

    # Global aggregate (no GROUP BY): always exactly one output row. If the filter
    # matched zero rows the kernel returns n_out=0 -> synthesize one row with
    # COUNT=0 and NULL (NaN) for the other aggs.
    is_global = ngkey == 0
    if is_global and n_out == 0:
        return _global_null_result(spec)

    return _assemble_agg_frame(spec, keys_list, aggs_list, gkey_decoders, hidden_count_idx)


def _assemble_agg_frame(spec, keys_list, aggs_list, gkey_decoders, hidden_count_idx):
    """Build the cuDF result frame from the C++ kernel's output arrays. Shared by
    the warm `_run_cpp` path and the cold `fused_scan_aggregate` path.

    `gkey_decoders[j]` is ('codes', uniques) | ('datetime',) | ('int',) for the
    per-key DENSE/HASH-int paths, OR a single ('codes_strided', uniques, strides,
    sizes) entry spanning ALL key columns for the multi-col stride-combined HASH
    path (one bound int64 code = sum code_j*stride_j; decomposed at read-out).
    `hidden_count_idx` is the internal AVG-denominator slot index, or None."""
    import cudf

    hidden_count = (
        np.asarray(aggs_list[hidden_count_idx], dtype=np.float64)
        if hidden_count_idx is not None else None
    )
    data: dict = {}
    # Multi-col stride-combined HASH: one bound key column holds the perfect-hash
    # code for the whole key tuple; decompose it back into per-col codes
    # (sub = (code // strides[j]) % sizes[j]) and map each via its cached uniques.
    if len(gkey_decoders) == 1 and gkey_decoders[0][0] == "codes_strided":
        _uniq, strides, sizes = gkey_decoders[0][1], gkey_decoders[0][2], gkey_decoders[0][3]
        codes = np.asarray(keys_list[0], dtype=np.int64)
        for j, (_ge, gn) in enumerate(spec["group_keys"]):
            sub = (codes // strides[j]) % sizes[j]
            uniq = _uniq[j]
            data[gn] = cudf.Series([uniq[int(c)] for c in sub])
    else:
        for j, (_ge, gn) in enumerate(spec["group_keys"]):
            codes = np.asarray(keys_list[j], dtype=np.int64)
            dec = gkey_decoders[j]
            if dec[0] == "codes":
                uniq = dec[1]
                data[gn] = cudf.Series([uniq[int(c)] for c in codes])
            elif dec[0] == "datetime":
                data[gn] = cudf.Series(codes.astype("datetime64[s]"))
            else:  # 'int'
                data[gn] = cudf.Series(codes)
    for a, (af, n) in enumerate(spec["aggs"]):
        vals = np.asarray(aggs_list[a], dtype=np.float64)
        if af.func == "AVG":
            # Running sum / per-group passing-row count (non-null arg guaranteed).
            with np.errstate(invalid="ignore", divide="ignore"):
                vals = vals / hidden_count
        if af.func == "COUNT" and isinstance(af.arg, Star):
            data[n] = cudf.Series(vals.astype(np.int64))
        else:
            data[n] = cudf.Series(vals)

    by_names = [gn for _, gn in spec["group_keys"]]
    out = cudf.DataFrame(data)
    return out[by_names + [n for _, n in spec["aggs"]]]


# --------------------------------------------------------------------------- #
# Fused star-join + aggregate (C++/CUDA backend)
# --------------------------------------------------------------------------- #


def fused_join_aggregate(node: Aggregate, engine) -> "cudf.DataFrame | None":
    """Try to run an `Aggregate -> Join*` (snowflake star-join + grouped
    aggregate) as one fused CUDA kernel.

    Streams the fact table once, probes a chain of dimension hash tables
    in-kernel, and accumulates straight into a dense per-group accumulator --
    the joined frame is never materialised (the dominant cost of the cuDF
    `merge` + `groupby` path). Returns a cuDF DataFrame, or None if the shape is
    unsupported (the caller falls back to `self._exec(join)` + cuDF groupby).

    Scope (deferred shapes return None -> cuDF): a single group key that is a
    `Col` living in a dimension reached by the join chain; SUM / COUNT(*) over
    fact-table columns; every join key int64/int32; a linear snowflake chain
    from the fact table to the group-key dimension that covers every joined
    table (no off-path branching dims, no pure-star legs); INNER and LEFT/RIGHT
    outer joins (RIGHT arrives via the optimizer's side-swap; the fact side is
    streamed, the dim hashed, and a miss at an outer stage null-pads to a NULL
    group slot instead of dropping the fact row); and single-stage FULL outer
    (a FULL join reuses the LEFT null-pad for fact misses, then the host readout
    emits the dim-only groups -- COUNT=1, SUM=NULL -- that FULL adds over LEFT);
    high-cardinality group-from-join (SUM/COUNT over a high-NDV dimension group
    key reaches the chain -- the carried code is hashed into a global open-addressed
    HT instead of the dense shared-mem accumulator, the direct analogue of the
    non-join HASH path); and no Filter/Project/Sort under the Aggregate. Q3
    (multi-key group + cross-table filter), non-int join keys, dimension agg args,
    AVG/MIN/MAX/COUNT(col) over joins, CROSS joins, multi-stage FULL outer (the
    dim-only semantics across a chain are subtle and rare), outer joins whose
    preserved side is the dimension (the fact would be null-supplying -> not
    streamable), and global-over-join all defer.
    """
    import cudf

    if not _kernels.is_available:
        return None
    if not isinstance(node.input, Join):
        return None

    group_keys = node.group_keys
    aggs = node.aggs
    if len(group_keys) != 1:
        return None
    ge, gname = group_keys[0]
    if not isinstance(ge, Col):
        return None

    # Aggs: COUNT(*) or SUM (numeric + fact-table-only checked later, once the
    # fact frame is available). AVG/MIN/MAX/COUNT(col) over joins defer.
    for af, _n in aggs:
        if af.func == "COUNT" and isinstance(af.arg, Star):
            continue
        if af.func == "SUM":
            continue
        return None

    # Only Scan and Join may appear under the Aggregate (no Filter/Project/...).
    # Each join must be a single-pair equi-join; INNER / LEFT / RIGHT / FULL outer
    # (CROSS defers to cuDF -- it has no key). The how is carried per edge so the
    # path walk can tell an INNER stage (a probe miss drops the fact row) from an
    # outer stage (a miss null-pads).
    scans: list[Scan] = []
    edges: list[tuple[str, str, str]] = []
    for n in walk(node.input):
        if isinstance(n, Scan):
            scans.append(n)
        elif isinstance(n, Join):
            if n.how not in ("inner", "left", "right", "full") or len(n.on_left) != 1:
                return None  # cross/non-equi -> cuDF (FULL single-stage handled below)
            edges.append((n.on_left[0], n.on_right[0], n.how))
        else:
            return None
    if not scans or len(edges) != len(scans) - 1:
        return None  # not a tree -> defer

    # Attribute each bare column name to a table among this query's scans. TPC-H
    # prefixes keep names globally unique; an ambiguous name -> defer.
    schema = engine.catalog.schema_dict()
    scan_tables = [s.table for s in scans]
    col_to_table: dict[str, str] = {}
    for t in scan_tables:
        for c in schema.get(t, []):
            if c in col_to_table:
                return None
            col_to_table[c] = t

    # Fact = the largest scan (streamed). Dimensions = the rest. The optimizer's
    # join-side swap is handled by walking the join graph undirected from the
    # fact, so the fact's position in the tree does not matter.
    stats = engine.catalog.stats_dict()
    fact_scan = max(scans, key=lambda s: stats.get(s.table, 0))
    fact_table = fact_scan.table
    target_dim = col_to_table.get(ge.name)
    if target_dim is None or target_dim == fact_table:
        return None  # group key must live in a joined dimension, not the fact

    # Undirected adjacency over the join tree. Each entry is (neigh, col_here,
    # col_neigh, how, here_is_left): here_is_left marks whether the table we're
    # expanding FROM is the LEFT child of that join (on_left_col is its col) --
    # needed to decide which side an outer join preserves.
    adj: dict[str, list[tuple[str, str, str, str, int]]] = {t: [] for t in scan_tables}
    for a, b, how in edges:
        ta, tb = col_to_table.get(a), col_to_table.get(b)
        if ta is None or tb is None or ta == tb:
            return None
        adj[ta].append((tb, a, b, how, 1))  # ta is the LEFT child (on_left col a)
        adj[tb].append((ta, b, a, how, 0))  # tb is the RIGHT child

    # BFS from fact to the group-key dimension; record each step's parent edge.
    parent: dict[str, tuple[str, str, str, str, int] | None] = {fact_table: None}
    order = [fact_table]
    i = 0
    while i < len(order) and target_dim not in parent:
        cur = order[i]
        i += 1
        for nb, ca, cb, how, here_is_left in adj[cur]:
            if nb not in parent:
                parent[nb] = (cur, ca, cb, how, here_is_left)
                order.append(nb)
    if target_dim not in parent:
        return None  # group-key dim not reachable from fact -> defer

    # Walk back target_dim -> fact: path_dims[j] = (dim_table, dim_key_col,
    # probe_key_col) where probe_key_col is on the already-connected side (fact
    # for j=0, dim j-1 for j>0). The kernel reads only the FIRST probe key from a
    # fact column; later probe keys are carried as the previous dim's payload.
    # path_left[j] = 1 when stage j preserves the fact side (a LEFT/RIGHT join
    # whose null-supplying side is this dimension -> a probe miss null-pads to
    # the NULL group instead of dropping the fact row); 0 for an inner stage. An
    # outer join that preserves the DIMENSION (fact side null-supplying) is not
    # streamable as fact-streamed probing -> defer to cuDF.
    path_dims: list[tuple[str, str, str]] = []
    path_left: list[int] = []
    full_outer = False
    cur = target_dim
    while cur != fact_table:
        par, ca, cb, how, fact_is_left = parent[cur]
        if how == "inner":
            lps = 0
        elif how == "full":
            # FULL preserves both sides; the fact miss -> NULL group behaves as a
            # LEFT stage (lps=1). The dim-only groups FULL adds are emitted by the
            # host readout; single-stage only (multi-stage FULL defers below).
            lps = 1
            full_outer = True
        else:
            # LEFT preserves the LEFT child; RIGHT preserves the RIGHT child.
            preserved = fact_is_left if how == "left" else (not fact_is_left)
            if not preserved:
                return None  # dim preserved, fact null-supplying -> not streamable
            lps = 1
        path_dims.append((cur, cb, ca))
        path_left.append(lps)
        cur = par
    path_dims.reverse()
    path_left.reverse()
    has_left = any(path_left)
    if full_outer and len(path_dims) != 1:
        return None  # multi-stage FULL-outer-agg -> cuDF (dim-only across a chain)
    if path_dims[0][2] not in schema.get(fact_table, []):
        return None
    # Every scan must lie on the path (no off-path branching dims).
    if {fact_table} | {d[0] for d in path_dims} != set(scan_tables):
        return None

    # Payloads: dim j bridges to dim j+1 via payload = next step's probe key (a
    # col of dim j). The last dim's payload = the group-key col (factorised to a
    # dense code on the host). A pure-star leg (next probe is a fact col, not a
    # col of dim j) defers.
    n_joins = len(path_dims)
    payloads: list[str] = []
    for j in range(n_joins):
        if j < n_joins - 1:
            pl = path_dims[j + 1][2]
            if pl not in schema.get(path_dims[j][0], []):
                return None
            payloads.append(pl)
        else:
            payloads.append(ge.name)
    if ge.name not in schema.get(path_dims[-1][0], []):
        return None

    # --- execute the scan leaves (cached on warm runs; same nodes the cuDF
    # merge path executes, so the scan cache is shared) ---
    scan_by_table = {s.table: s for s in scans}
    fact_frame = engine._exec(scan_by_table[fact_table])
    dim_frames = [engine._exec(scan_by_table[path_dims[j][0]]) for j in range(n_joins)]
    n = len(fact_frame)

    # Verify needed columns survived projection pushdown; bail if not.
    fact_cols_needed = {path_dims[0][2]}
    for af, _ in aggs:
        if not (af.func == "COUNT" and isinstance(af.arg, Star)):
            fact_cols_needed |= af.arg.columns()
    for c in fact_cols_needed:
        if c not in fact_frame.columns:
            return None
    for j in range(n_joins):
        if path_dims[j][1] not in dim_frames[j].columns or payloads[j] not in dim_frames[j].columns:
            return None

    # Agg args must be numeric and live in the fact table.
    for af, _n in aggs:
        if af.func == "COUNT" and isinstance(af.arg, Star):
            continue
        if not _is_numeric_expr(af.arg, fact_frame):
            return None
        for c in af.arg.columns():
            if col_to_table.get(c) != fact_table:
                return None

    # n==0 -> empty grouped result.
    if n == 0:
        return _empty_result({"group_keys": group_keys, "aggs": aggs})

    # --- bind fact columns (first probe key + agg-arg cols) ---
    col_ptrs: list[int] = []
    col_dtypes: list[int] = []
    name_idx: dict[str, int] = {}
    _kept: list[object] = []

    def bind_fact(name: str) -> int:
        if name in name_idx:
            return name_idx[name]
        col = fact_frame[name]
        dt = col.dtype
        if np.issubdtype(dt, np.datetime64):
            arr = _to_int64_seconds(col)
            _kept.append(arr)
            ptr, dcode = _dev_ptr(arr), _DT_INT64
        elif np.issubdtype(dt, np.integer) or "int" in str(dt):
            if np.issubdtype(dt, np.int64):
                arr = col
            else:
                arr = col.astype(np.int64)
                _kept.append(arr)
            ptr, dcode = _dev_ptr(arr), _DT_INT64
        elif np.issubdtype(dt, np.floating) or "float" in str(dt):
            ptr, dcode = _dev_ptr(col), _DT_FLOAT64
        else:
            raise ValueError(f"fused join cannot bind non-numeric column {name}")
        idx = len(col_ptrs)
        col_ptrs.append(ptr)
        col_dtypes.append(dcode)
        name_idx[name] = idx
        return idx

    # The first probe key is a join key -> must be integer (read raw int64).
    first_probe_name = path_dims[0][2]
    fp_dt = fact_frame[first_probe_name].dtype
    if not (np.issubdtype(fp_dt, np.integer) or "int" in str(fp_dt)):
        return None  # non-int join key deferred
    try:
        for c in fact_cols_needed:
            bind_fact(c)
    except ValueError:
        return None
    first_probe_col = name_idx[first_probe_name]

    # --- build dimension HT inputs: (key, payload) int64 device ptrs ---
    dim_key_ptrs: list[int] = []
    dim_payload_ptrs: list[int] = []
    dim_n: list[int] = []
    ht_cap: list[int] = []
    n_groups = 0
    gkey_uniques: list = []

    def _as_int64(series):
        if np.issubdtype(series.dtype, np.int64):
            return series
        arr = series.astype(np.int64)
        _kept.append(arr)
        return arr

    for j in range(n_joins):
        dframe = dim_frames[j]
        key_col = path_dims[j][1]
        kseries = dframe[key_col]
        if not (np.issubdtype(kseries.dtype, np.integer) or "int" in str(kseries.dtype)):
            return None  # non-int join key deferred
        # PK guard (cached): a non-unique dim key would silently collapse joins.
        if not engine.is_unique_key(path_dims[j][0], key_col, kseries):
            return None
        dim_key_ptrs.append(_dev_ptr(_as_int64(kseries)))

        if j < n_joins - 1:
            pseries = dframe[payloads[j]]
            if not (np.issubdtype(pseries.dtype, np.integer) or "int" in str(pseries.dtype)):
                return None  # bridging payload is the next join key -> must be int
            dim_payload_ptrs.append(_dev_ptr(_as_int64(pseries)))
        else:
            # Last dim: payload = factorised group-key codes (dense 0..n-1). For a
            # LEFT-outer plan (has_left), offset the codes by +1 so dim rows occupy
            # slots 1..n and slot 0 is reserved for the NULL group (fact rows that
            # miss at an outer stage); n_groups grows by 1 to include that slot. A
            # pure-inner plan skips the offset -- slot 0 is an ordinary group and
            # the output is byte-identical to the inner-only kernel.
            codes, uniq = engine.get_codes(path_dims[j][0], payloads[j], dframe[payloads[j]])
            if has_left:
                codes = codes + 1  # int64; code 0 -> NULL group slot
            _kept.append(codes)
            dim_payload_ptrs.append(_dev_ptr(codes))
            n_groups = len(uniq) + (1 if has_left else 0)
            gkey_uniques = uniq

        dn = len(dframe)
        if 2 * dn > _HASH_CAP_MAX:
            return None  # HT too large -> cuDF fallback
        dim_n.append(dn)
        ht_cap.append(_next_pow2(2 * dn))

    nagg = len(aggs)
    strategy = _STRAT_DENSE
    capacity = 0
    if n_groups * nagg <= MAX_ACC_CELLS:
        strategy = _STRAT_DENSE                       # unchanged low-card path
    else:
        # High-card group-from-join: the carried group code is hashed into a global
        # open-addressed HT instead of indexing a shared-mem dense accumulator (the
        # direct analogue of the non-join HASH path, PR #50).
        strategy = _STRAT_HASH
        capacity = _next_pow2(min(n, _HASH_CAP_MAX, _HASH_ACC_BUDGET // (nagg * 8)))
        if capacity < 4:
            return None  # too few rows for an HT -> cuDF

    # --- aggregate descriptors (SUM/COUNT only this step; no hidden AVG slot) ---
    _AGG_KIND = {"COUNT": _AGG_COUNT, "SUM": _AGG_SUM}
    agg_kind: list[int] = []
    agg_tok_start: list[int] = []
    agg_tok_len: list[int] = []
    tok_kind: list[int] = []
    tok_col: list[int] = []
    tok_lit: list[float] = []
    tok_op: list[int] = []
    for af, _n in aggs:
        start = len(tok_kind)
        if af.func == "COUNT" and isinstance(af.arg, Star):
            agg_kind.append(_AGG_COUNT)
            agg_tok_start.append(start)
            agg_tok_len.append(0)
            continue
        toks = _to_postfix(af.arg, bind_fact)
        for k, a, b in toks:
            tok_kind.append(k)
            tok_lit.append(b)
            if k == _TK_OP:
                tok_col.append(0)
                tok_op.append(a)
            else:
                tok_col.append(a)
                tok_op.append(0)
        agg_kind.append(_AGG_KIND[af.func])
        agg_tok_start.append(start)
        agg_tok_len.append(len(toks))

    # No predicate (this path requires no Filter under the Aggregate).
    pred_col = np.empty(0, dtype=np.int32)
    pred_op = np.empty(0, dtype=np.int32)
    pred_lit = np.empty(0, dtype=np.float64)

    # Per-slot acc init: +inf/-inf for MIN/MAX, 0 otherwise (all 0 for SUM/COUNT).
    per_slot = np.array(
        [np.inf if k == _AGG_MIN else -np.inf if k == _AGG_MAX else 0.0 for k in agg_kind],
        dtype=np.float64,
    )
    if strategy == _STRAT_DENSE:
        acc_init = np.tile(per_slot, n_groups)
    elif strategy == _STRAT_HASH and any(k in (_AGG_MIN, _AGG_MAX) for k in agg_kind):
        acc_init = np.tile(per_slot, capacity)  # MIN/MAX over-join (future) -> inf/-inf init
    else:
        acc_init = np.empty(0, dtype=np.float64)  # SUM/COUNT -> host memsets acc to 0

    try:
        overflow, n_out, keys_list, aggs_list = _kernels.fused_join_agg(
            np.array(col_ptrs, dtype=np.int64), np.array(col_dtypes, dtype=np.int32),
            int(first_probe_col),
            np.array(dim_key_ptrs, dtype=np.int64),
            np.array(dim_payload_ptrs, dtype=np.int64),
            np.array(dim_n, dtype=np.int32), np.array(ht_cap, dtype=np.int32),
            np.array(path_left, dtype=np.int32),
            pred_col, pred_op, pred_lit,
            np.array(agg_kind, dtype=np.int32),
            np.array(agg_tok_start, dtype=np.int32),
            np.array(agg_tok_len, dtype=np.int32),
            np.array(tok_kind, dtype=np.int32), np.array(tok_col, dtype=np.int32),
            np.array(tok_lit, dtype=np.float64), np.array(tok_op, dtype=np.int32),
            acc_init, int(strategy), int(capacity), int(n_groups), int(n),
        )
    except Exception:  # noqa: BLE001 -- never let a C++ fault break correctness
        return None
    if overflow != 0:
        return None  # a dim HT filled -> cuDF fallback
    if n_out == 0:
        return _empty_result({"group_keys": group_keys, "aggs": aggs})

    # --- assemble cuDF frame (single group key: code -> unique value) ---
    # For an outer plan (has_left), code 0 is the NULL group (unmatched fact rows);
    # its group-key columns are NULL. A real dim group at code c (>=1) decodes to
    # gkey_uniques[c-1] (the host offsets the last dim's codes by +1).
    codes = np.asarray(keys_list[0], dtype=np.int64)
    if has_left:
        key_vals = [None if int(c) == 0 else gkey_uniques[int(c) - 1] for c in codes]
    else:
        key_vals = [gkey_uniques[int(c)] for c in codes]
    agg_vals: list[list] = []
    for a, (af, _n) in enumerate(aggs):
        vals = np.asarray(aggs_list[a], dtype=np.float64)
        if af.func == "COUNT" and isinstance(af.arg, Star):
            agg_vals.append([int(x) for x in vals.astype(np.int64)])
        else:  # SUM
            agg_vals.append([float(x) for x in vals])
    # FULL-outer: dim rows no fact row hit. COUNT(*)=1 (the null-padded dim-only
    # row), SUM=NULL (no fact value). `seen` already covers code 0 (NULL group) +
    # every matched dim; emit the rest (1..n_dim) as dim-only rows.
    if full_outer:
        n_dim = len(gkey_uniques)  # == n_groups - 1
        seen = {int(c) for c in codes}
        for g in range(1, n_dim + 1):
            if g in seen:
                continue
            key_vals.append(gkey_uniques[g - 1])
            for a, (af, _n) in enumerate(aggs):
                if af.func == "COUNT" and isinstance(af.arg, Star):
                    agg_vals[a].append(1)
                else:  # SUM
                    agg_vals[a].append(None)  # -> NaN in the float64 column
    data: dict = {gname: cudf.Series(key_vals)}
    for a, (af, n_) in enumerate(aggs):
        data[n_] = cudf.Series(agg_vals[a])
    out = cudf.DataFrame(data)
    return out[[gname] + [n_ for _, n_ in aggs]]


# --------------------------------------------------------------------------- #
# Fused scan + aggregate (C++/CUDA cold reader: nvCOMP Snappy -> decode -> agg)
# --------------------------------------------------------------------------- #
#
# Phase 5 cold path. `Aggregate -> Filter -> Scan` is run straight off the
# Parquet pages -- nvCOMP batched Snappy-decompress on the GPU, an RLE/bit-packed
# dict-index decode, decimal/date folds, predicate + per-group accumulate -- so
# the 60M-row cuDF frame is never materialised. Works on the PLAN (no executed
# frame), so the executor calls it BEFORE `self._exec(in_node.input)`.
#
# v1 scope (the certain cold wins): global aggregate (no GROUP BY -> Q6 /
# scan_agg_full) and HASH with a single PLAIN int64 group key (high-card
# l_orderkey). Columns are PLAIN or PLAIN_DICTIONARY numeric, Snappy, non-null.
# Dict-string DENSE group keys (Q1) are deferred: the dictionary is per-row-group
# so dict indices are local codes that don't map to a global DENSE accumulator
# without a per-RG local->global remap (Phase 5 step 2).

# Column-plan codes (mirror kernels/fused.cu).
_PK_PLAIN_RAW, _PK_DICT_NUMERIC_ARG = 0, 1
_PHYS_I32, _PHYS_I64 = 0, 1
_PQ_META_CACHE: dict[str, object] = {}


def _pq_file(path: str):
    """Cached pyarrow ParquetFile (footer metadata only; read-only Phase 1 -> the
    file is immutable so a module cache is safe)."""
    pf = _PQ_META_CACHE.get(path)
    if pf is None:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        _PQ_META_CACHE[path] = pf
    return pf


class _Defer(Exception):
    """Raised internally to abandon the scan path -> caller falls back to cuDF."""


class _ColProxy:
    __slots__ = ("dtype", "null_count")

    def __init__(self, dtype, null_count):
        self.dtype = dtype
        self.null_count = null_count


class _SchemaProxy:
    """Frame-like view over the arrow schema + null-count stats so the existing
    `_match` / `_is_numeric_expr` / `_is_supported_predicate` / `_flatten_and_pred`
    helpers work WITHOUT materialising the cuDF frame."""

    def __init__(self, columns, dtypes, null_counts):
        self.columns = columns
        self._dt = dtypes
        self._nc = null_counts

    def __getitem__(self, name):
        return _ColProxy(self._dt[name], self._nc[name])


def _arrow_match_dtype(t) -> object:
    """arrow type -> numpy dtype for shape matching (decimals -> float64 so
    _is_numeric_expr accepts them; the kernel folds the decimal scale itself)."""
    import pyarrow as pa

    if pa.types.is_decimal(t) or pa.types.is_floating(t):
        return np.dtype("float64")
    if pa.types.is_signed_integer(t) or pa.types.is_unsigned_integer(t):
        return np.dtype("int64")
    if pa.types.is_date(t):
        return np.dtype("datetime64[s]")
    return np.dtype("object")  # string/bool/other -> non-numeric -> defer


def _col_type_meta(t):
    """arrow numeric type -> (phys, scale, is_date) the reader supports, or None
    (strings/doubles/bools/etc. defer). The encoding (PLAIN vs PLAIN_DICTIONARY)
    is resolved per-chunk in bind_named."""
    import pyarrow as pa

    if pa.types.is_decimal(t):
        return (_PHYS_I64, int(t.scale), 0)
    if pa.types.is_date(t):
        return (_PHYS_I32, 0, 1)
    if pa.types.is_int64(t):
        return (_PHYS_I64, 0, 0)
    if pa.types.is_int32(t):
        return (_PHYS_I32, 0, 0)
    return None


def _frame_out_kind(phys: int, scale: int, is_date: int) -> int:
    """Store-width code consumed by ``materialise_kernel`` (mirrors fused.cu):
    3 = datetime64[s] (int64 seconds = days*86400), 2 = float64 (decimal scale
    folded on the device), 0 = int32, 1 = int64. Single-sources the store width
    so the host frame dtype and the kernel store cannot diverge."""
    if is_date:
        return 3
    if scale > 0:
        return 2
    if phys == _PHYS_I32:
        return 0
    return 1


def _frame_alloc_dtype(out_kind: int) -> np.dtype:
    """numpy dtype of the device buffer the kernel writes into. A date column
    (out_kind 3) is written as int64 seconds and ``astype``'d to
    ``datetime64[s]`` after the call, so its alloc dtype is int64."""
    if out_kind == 0:
        return np.dtype("int32")
    if out_kind == 2:
        return np.dtype("float64")
    return np.dtype("int64")  # 1 (int64) or 3 (date -> int64 seconds)


def fused_scan_aggregate(node: Aggregate, engine) -> "cudf.DataFrame | None":
    """Try to run `Aggregate -> Filter -> Scan` straight off the Parquet pages
    (cold reader) without materialising the cuDF frame. Returns a cuDF DataFrame,
    or None when the shape is unsupported (caller falls back to the existing
    materialising path). Called BEFORE `self._exec(in_node.input)` in the
    executor's `_aggregate`, so a None leaves the scan cache untouched."""
    import pyarrow as pa

    if not _kernels.is_available:
        return None
    if not isinstance(node.input, Filter):
        return None
    filt = node.input
    if not isinstance(filt.input, Scan):
        return None
    info = engine.catalog.get(filt.input.table)
    # Phase 2 delta guard: the cold Parquet reader bypasses _scan and would miss
    # unflushed delta rows (committed OR an in-txn buffer). Defer to the
    # materialising _scan + merge path when a table has pending writes. None
    # pending (step 2) -> never fires -> no regression.
    if engine.has_pending(filt.input.table):
        return None
    if len(info.paths) != 1:
        return None  # v1: single file per table
    path = info.paths[0]

    try:
        pf = _pq_file(path)
        md = pf.metadata
        schema_arrow = pf.schema_arrow
        nrg = md.num_row_groups
        if nrg == 0:
            return None
        names = schema_arrow.names
        name_to_j = {n: i for i, n in enumerate(names)}

        # Schema proxy for shape matching (decimals numeric, dates datetime,
        # null_count from per-chunk statistics -- 0 for TPC-H).
        dtypes = {n: _arrow_match_dtype(schema_arrow.field(n).type) for n in names}
        null_counts: dict[str, int] = {}
        for n in names:
            j = name_to_j[n]
            nc = 0
            for rg in range(nrg):
                st = md.row_group(rg).column(j).statistics
                if st is None or st.null_count != 0:
                    nc = 1
                    break
            null_counts[n] = nc
        proxy = _SchemaProxy(names, dtypes, null_counts)

        spec = _match(node, proxy)
        if spec is None:
            return None

        nagg = len(spec["aggs"])
        ngkey = len(spec["group_keys"])
        has_avg = any(af.func == "AVG" for af, _ in spec["aggs"])
        nagg_eff = nagg + (1 if has_avg else 0)

        # --- column plan: bind predicate/agg-arg/group-key cols, record per-chunk
        # descriptors. bind_named raises _Defer on any unsupported column/encoding. ---
        col_kind: list[int] = []
        col_phys: list[int] = []
        col_scale: list[int] = []
        col_is_date: list[int] = []
        chunk_off: list[int] = []
        chunk_total: list[int] = []
        chunk_nvals: list[int] = []
        name_idx: dict[str, int] = {}

        def bind_named(name: str) -> int:
            if name in name_idx:
                return name_idx[name]
            if name not in name_to_j:
                raise _Defer()
            t = schema_arrow.field(name).type
            tmeta = _col_type_meta(t)
            if tmeta is None:
                raise _Defer()
            phys, scale, is_date = tmeta
            j = name_to_j[name]
            kind = None
            for rg in range(nrg):
                cc = md.row_group(rg).column(j)
                if cc.compression != "SNAPPY":
                    raise _Defer()
                st = cc.statistics
                if st is None or st.null_count != 0:
                    raise _Defer()  # kernel reads raw values, does not skip nulls
                enc = set(cc.encodings)
                is_dict = "PLAIN_DICTIONARY" in enc
                if is_dict:
                    ck = _PK_DICT_NUMERIC_ARG
                elif enc == {"PLAIN"}:
                    ck = _PK_PLAIN_RAW
                else:
                    raise _Defer()
                if kind is None:
                    kind = ck
                elif ck != kind:
                    raise _Defer()  # encoding varies across row groups -> defer
                pt = cc.physical_type
                if phys == _PHYS_I64 and pt != "INT64":
                    raise _Defer()
                if phys == _PHYS_I32 and pt != "INT32":
                    raise _Defer()
                off = cc.dictionary_page_offset
                if off is None:
                    off = cc.data_page_offset
                chunk_off.append(int(off))
                chunk_total.append(int(cc.total_compressed_size))
                chunk_nvals.append(int(cc.num_values))
            idx = len(col_kind)
            col_kind.append(kind)
            col_phys.append(phys)
            col_scale.append(scale)
            col_is_date.append(is_date)
            name_idx[name] = idx
            return idx

        # --- group key / strategy ---
        gkey_idx: list[int] = []
        gkey_stride: list[int] = []
        gkey_decoders: list = []
        if ngkey == 0:
            strategy = _STRAT_DENSE
            n_groups = 1
            capacity = 0
        elif ngkey == 1:
            ge, _gname = spec["group_keys"][0]
            if not isinstance(ge, Col):
                return None
            t = schema_arrow.field(ge.name).type
            # v1 HASH: a single PLAIN int64 key read raw at values_off. int32/date
            # would misread the int64 slot; decimal would emit unscaled ints; dict
            # would read the index array; string is the deferred DENSE case.
            if not pa.types.is_int64(t):
                return None
            idx = bind_named(ge.name)
            if col_kind[idx] != _PK_PLAIN_RAW or col_phys[idx] != _PHYS_I64:
                return None
            strategy = _STRAT_HASH
            n_groups = 0  # HASH uses `capacity`, not n_groups (kernel ignores it)
            total_rows = info.row_count
            capacity = _next_pow2(min(total_rows, _HASH_CAP_MAX, _HASH_ACC_BUDGET // (nagg * 8)))
            if capacity < 4:
                return None
            gkey_idx = [idx]
            gkey_decoders = [("int",)]
        else:
            return None  # multi-key GROUP BY -> defer (v1)

        # --- predicate (conjunction of Col OP lit; date lit -> int64 seconds) ---
        pred = _flatten_and_pred(spec["predicate"], bind_named, proxy)
        if pred is None:
            return None
        pred_col = np.array([p[0] for p in pred], dtype=np.int32)
        pred_op = np.array([p[1] for p in pred], dtype=np.int32)
        pred_lit = np.array([p[2] for p in pred], dtype=np.float64)

        # --- aggregates (COUNT(*)/SUM/AVG/MIN/MAX over numeric arithmetic) ---
        _AGG_KIND = {"COUNT": _AGG_COUNT, "SUM": _AGG_SUM, "AVG": _AGG_AVG,
                     "MIN": _AGG_MIN, "MAX": _AGG_MAX}
        agg_kind: list[int] = []
        agg_tok_start: list[int] = []
        agg_tok_len: list[int] = []
        tok_kind: list[int] = []
        tok_col: list[int] = []
        tok_lit: list[float] = []
        tok_op: list[int] = []
        for af, _n in spec["aggs"]:
            start = len(tok_kind)
            if af.func == "COUNT" and isinstance(af.arg, Star):
                agg_kind.append(_AGG_COUNT)
                agg_tok_start.append(start)
                agg_tok_len.append(0)
                continue
            toks = _to_postfix(af.arg, bind_named)
            for k, a, b in toks:
                tok_kind.append(k)
                tok_lit.append(b)
                if k == _TK_OP:
                    tok_col.append(0)
                    tok_op.append(a)
                else:
                    tok_col.append(a)
                    tok_op.append(0)
            agg_kind.append(_AGG_KIND[af.func])
            agg_tok_start.append(start)
            agg_tok_len.append(len(toks))

        # Cold HASH now supports MIN/MAX/AVG over the single PLAIN int64 key
        # (page_hash_kernel mirrors page_dense_kernel's per-slot dispatch; the host
        # copies acc_init into the cold HASH acc when non-empty). Multi-col/string
        # HASH stays cold-deferred (the key is read raw from the Parquet page, so a
        # Python-combined code does not exist on disk) -> warm `fused_aggregate`.

        # Hidden per-group passing-row count slot: the AVG denominator.
        hidden_count_idx = None
        if has_avg:
            hidden_count_idx = len(agg_kind)
            agg_kind.append(_AGG_COUNT)
            agg_tok_start.append(len(tok_kind))
            agg_tok_len.append(0)

        # DENSE global: shared-mem gate on the internal slot count.
        if strategy == _STRAT_DENSE and n_groups * nagg_eff > MAX_ACC_CELLS:
            return None

        # Per-slot accumulator init: +inf for MIN, -inf for MAX, 0 otherwise. DENSE
        # tiles across n_groups (shared-mem acc); HASH tiles across `capacity` (the
        # global HT acc) when any MIN/MAX is present -- the host copies acc_init into
        # the HASH acc and page_hash_kernel's atomic_min/max_d lower from +/-inf.
        # SUM/COUNT and AVG (running sum + hidden count) start at 0, so a HASH plan
        # with no MIN/MAX passes an empty acc_init and the host memsets the acc to 0.
        per_slot = np.array(
            [np.inf if k == _AGG_MIN else -np.inf if k == _AGG_MAX else 0.0
             for k in agg_kind], dtype=np.float64,
        )
        if strategy == _STRAT_DENSE:
            acc_init = np.tile(per_slot, n_groups)
        elif strategy == _STRAT_HASH and any(k in (_AGG_MIN, _AGG_MAX) for k in agg_kind):
            acc_init = np.tile(per_slot, capacity)
        else:
            acc_init = np.empty(0, dtype=np.float64)

        ncol = len(col_kind)

        # --- Phase 5 step 3: populate Engine._scan_cache from the scan path. When
        # the bound column set equals the Scan projection, allocate one typed
        # non-nullable device buffer per bound column and hand each buffer's data
        # ptr to the kernel. materialise_kernel writes every decoded value at the
        # RG's global row offset; on success we cache the frame under the SAME key
        # _scan uses so warm repeats hit the GPU-resident frame. proj a strict
        # superset of the bound set -> skip (never cache a short frame); proj None
        # -> skip.
        #
        # Order matters: the kernel writes the numba buffers on the scan path's
        # custom stream, and the C++ entry syncs before returning. We build the
        # cuDF Series from each buffer ONLY AFTER the call, so any copy as_column
        # performs reads already-correct, sync-visible buffer data -- there is no
        # async copy racing the kernel (which previously clobbered writes with
        # stale recycled-buffer data, seen as scattered NaN). column_empty is
        # all-null and would mask the kernel's writes, so the numba->as_column
        # path (non-nullable) is used. frame_bufs keeps the buffers alive through
        # the call (the kernel's write targets).
        import cudf
        import cudf.core.column as _cc
        frame_ptrs: list[int] = []
        frame_out_kind: list[int] = []
        frame_bufs: list = []
        frame_meta: list[tuple[str, int]] = []
        idx_name: dict[int, str] = {}
        populate = False
        proj = filt.input.columns
        if engine.cache_enabled and proj is not None and set(name_idx) == set(proj):
            assert len(chunk_nvals) == nrg * ncol, (len(chunk_nvals), nrg, ncol)
            total_rows = int(sum(chunk_nvals[:nrg]))  # col-major: col 0's per-RG counts
            idx_name = {v: k for k, v in name_idx.items()}
            populate = True
            for c in range(ncol):
                ok = _frame_out_kind(col_phys[c], col_scale[c], col_is_date[c])
                buf = cuda.device_array(total_rows, dtype=_frame_alloc_dtype(ok))
                frame_bufs.append(buf)
                frame_ptrs.append(int(buf.__cuda_array_interface__["data"][0]))
                frame_out_kind.append(ok)
                frame_meta.append((idx_name[c], ok))

        overflow, n_out, keys_list, aggs_list, pending_id = _kernels.fused_scan_agg(
            path, int(ncol), int(nrg),
            np.array(chunk_off, dtype=np.int64),
            np.array(chunk_total, dtype=np.int32),
            np.array(chunk_nvals, dtype=np.int32),
            np.array(col_kind, dtype=np.int32),
            np.array(col_phys, dtype=np.int32),
            np.array(col_scale, dtype=np.int32),
            np.array(col_is_date, dtype=np.int32),
            np.array(gkey_idx, dtype=np.int32),
            np.array(gkey_stride, dtype=np.int64),
            pred_col, pred_op, pred_lit,
            np.array(agg_kind, dtype=np.int32),
            np.array(agg_tok_start, dtype=np.int32),
            np.array(agg_tok_len, dtype=np.int32),
            np.array(tok_kind, dtype=np.int32),
            np.array(tok_col, dtype=np.int32),
            np.array(tok_lit, dtype=np.float64),
            np.array(tok_op, dtype=np.int32),
            acc_init,
            np.array(frame_ptrs, dtype=np.int64),
            np.array(frame_out_kind, dtype=np.int32),
            int(strategy), int(n_groups), int(capacity),
        )
    except _Defer as _e:
        import os as _os
        if _os.environ.get("RYUDB_SCAN_DEBUG"):
            import traceback as _tb
            print("[scan_agg] DEFER:", _e)
            _tb.print_exc()
        return None
    except Exception as _e:  # noqa: BLE001 -- never let a C++/metadata fault break correctness
        import os as _os
        if _os.environ.get("RYUDB_SCAN_DEBUG"):
            import traceback as _tb
            print("[scan_agg] EXC:", _e)
            _tb.print_exc()
        return None

    if overflow != 0:
        import os as _os
        if _os.environ.get("RYUDB_SCAN_DEBUG"):
            print("[scan_agg] kernel overflow =", overflow)
        return None  # decompress/decode/hash-table error -> cuDF fallback

    # Cache the materialised frame so warm repeats hit the GPU-resident frame
    # instead of re-reading Parquet. overflow == 0 here, so every RG wrote its
    # full row range into the per-column numba buffers (the kernel's write
    # targets). Two paths:
    #
    #  * pending_id != 0 (async): the materialise gather is still running on the
    #    C++ side-stream (stream2). Store a _PendingFrame that *defers* building
    #    the cuDF Series until the first warm read calls .get() -> fused_scan_
    #    finalize, which syncs E_mat first. Building the Series *after* that sync
    #    preserves the step-3 race fix (as_column's async copy must read gather
    #    writes that are already visible). Keep frame_bufs alive on the pending
    #    object until then.
    #  * pending_id == 0 (sync fallback / kill switch): the gather already
    #    completed under the C++ entry's final sync, so build the cuDF Series NOW
    #    exactly as before. Date columns (out_kind 3) were written as int64
    #    seconds -> astype datetime64[s] to match storage.scan; the warm path is
    #    unit-agnostic via _to_int64_seconds.
    # Never reached when populate is False (no buffers).
    if populate:
        cache_key = (filt.input.table, frozenset(proj))
        if pending_id:
            engine._scan_cache[cache_key] = _PendingFrame(
                frame_bufs, frame_meta, pending_id, proj
            )
        else:
            cols_out: dict[str, "cudf.Series"] = {}
            for c, (name, ok) in enumerate(frame_meta):
                series = cudf.Series._from_column(_cc.as_column(frame_bufs[c]))
                if ok == 3:
                    series = series.astype("datetime64[s]")
                cols_out[name] = series
            engine._scan_cache[cache_key] = cudf.DataFrame(
                {n: cols_out[n] for n in sorted(proj)}
            )

    is_global = ngkey == 0
    if is_global and n_out == 0:
        return _global_null_result(spec)
    if n_out == 0:
        return _empty_result(spec)

    return _assemble_agg_frame(spec, keys_list, aggs_list, gkey_decoders, hidden_count_idx)