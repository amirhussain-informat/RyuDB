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

from ..sql.plan import Aggregate, And, BinOp, Col, Filter, Lit, Or, Scan, Star, walk

if TYPE_CHECKING:
    import cudf

# Per-block shared accumulator cells (float64). Sized to a safe constant so the
# kernel can declare a static shared array; the gate ensures
# n_groups * n_aggs <= MAX_ACC_CELLS. 8192 float64 = 64 KB, within RTX 3090
# per-block shared memory without opt-in.
MAX_ACC_CELLS = 4096
THREADS = 256

_KERNEL_CACHE: dict[str, object] = {}


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
        return _empty_result(spec)

    # Resolve the source table name for the group-key code cache (best-effort:
    # walk the subtree for a Scan). Falls back to None -> no caching.
    table = _source_table(node)

    # ---- data prep ------------------------------------------------------- #
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
    if not group_keys:
        return None  # global aggregate handled elsewhere

    # Group keys must be plain columns present in the frame.
    for ge, _gn in group_keys:
        if not isinstance(ge, Col) or ge.name not in child.columns:
            return None

    # Aggregates: COUNT(*) or SUM(arithmetic expr over numeric cols/lits).
    for af, _n in aggs:
        if af.func == "COUNT" and isinstance(af.arg, Star):
            continue
        if af.func == "SUM":
            if not _is_numeric_expr(af.arg, child):
                return None
            continue
        return None  # COUNT(col), AVG, MIN, MAX -> fall back

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