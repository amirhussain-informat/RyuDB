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

# --- C++ kernel descriptor codes (must mirror kernels/fused.cu) -------------- #
_DT_INT64, _DT_FLOAT64 = 0, 1
_OP = {"=": 0, "!=": 1, "<": 2, "<=": 3, ">": 4, ">=": 5}
_TOP = {"+": 1, "-": 2, "*": 3, "/": 4}
_TK_COL, _TK_LIT, _TK_OP = 0, 1, 2
_AGG_COUNT, _AGG_SUM = 0, 1
_STRAT_DENSE, _STRAT_HASH = 0, 1
# Hash-table accumulator memory budget (bytes) and slot cap. Sized from row count
# (catalog has no NDV): capacity = next_pow2(min(n, 2**25, BUDGET//(nagg*8))).
_HASH_ACC_BUDGET = 2 * 10**9
_HASH_CAP_MAX = 1 << 25  # 33.5M slots


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
    import cudf

    n = len(child)
    nagg = len(spec["aggs"])
    ngkey = len(spec["group_keys"])

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

    if all_numeric:
        # HASH direct: bind the group key as an int64 array (datetime -> seconds).
        # The C++ hash kernel handles a SINGLE int64 group key only (it reads
        # p.gkey_idx[0] and uses atomicCAS-on-key). Multi-column numeric GROUP BY
        # is deferred to cuDF for now.
        if ngkey != 1:
            return None
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
        # Factorize all group keys (cached) -> codes. DENSE if low-card, else the
        # string high-card HASH-codes path is deferred (host decode of millions of
        # codes is slow) -> return None and let cuDF handle it.
        sizes: list[int] = []
        uniques: list = []
        for ge, _gn in spec["group_keys"]:
            if engine is not None and table is not None:
                codes, uniq = engine.get_codes(table, ge.name, child[ge.name])
            else:
                codes, uniq = child[ge.name].factorize()
                uniq = list(uniq.to_pandas())
            idx = len(col_ptrs)
            _kept.append(codes)
            col_ptrs.append(_dev_ptr(codes))
            col_dtypes.append(_DT_INT64)
            gkey_idx.append(idx)
            sizes.append(len(uniq))
            uniques.append(uniq)
            gkey_decoders.append(("codes", uniq))
        for j in range(len(sizes)):
            gkey_stride.append(int(np.prod(sizes[j + 1:], dtype=np.int64)))
        n_groups = int(np.prod(sizes, dtype=np.int64)) if sizes else 1
        if n_groups * nagg > MAX_ACC_CELLS:
            return None  # string high-card -> defer to cuDF
        strategy = _STRAT_DENSE

    # --- predicate (conjunction only) ---
    pred = _flatten_and_pred(spec["predicate"], bind_named, child)
    if pred is None:
        return None
    pred_col = np.array([p[0] for p in pred], dtype=np.int32)
    pred_op = np.array([p[1] for p in pred], dtype=np.int32)
    pred_lit = np.array([p[2] for p in pred], dtype=np.float64)

    # --- aggregates ---
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
        # SUM(expr)
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
        agg_kind.append(_AGG_SUM)
        agg_tok_start.append(start)
        agg_tok_len.append(len(toks))

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

    overflow, n_out, keys_list, aggs_list = _kernels.fused_agg(
        col_ptrs_np, col_dtypes_np, gkey_idx_np, gkey_stride_np,
        pred_col, pred_op, pred_lit, agg_kind_np, agg_tok_start_np, agg_tok_len_np,
        tok_kind_np, tok_col_np, tok_lit_np, tok_op_np,
        int(strategy), int(n_groups), int(capacity), int(n),
    )
    if overflow != 0:
        return None  # hash table filled -> cuDF fallback

    # --- assemble cuDF frame ---
    data: dict = {}
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
        if af.func == "COUNT" and isinstance(af.arg, Star):
            data[n] = cudf.Series(vals.astype(np.int64))
        else:
            data[n] = cudf.Series(vals)

    by_names = [gn for _, gn in spec["group_keys"]]
    out = cudf.DataFrame(data)
    return out[by_names + [n for _, n in spec["aggs"]]]