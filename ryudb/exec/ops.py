"""Expression evaluation on cuDF Series / scalars.

`eval_expr(expr, df)` returns either a cuDF Series (column-aligned) or a Python
scalar. Comparisons return boolean Series (used as Filter masks); arithmetic
returns numeric Series. Date literals are converted to pandas timestamps so
they compare against cuDF date32/datetime64 columns.
"""

from __future__ import annotations


import math
import re

import cudf
import numpy as np
import pandas as pd

from ..sql.plan import (
    AggFunc,
    And,
    BinOp,
    Case,
    Cast,
    Coalesce,
    Col,
    Expr,
    Func,
    In,
    IsNull,
    Like,
    Lit,
    Not,
    Or,
    Star,
)

_CMP_SWAP = {"<": ">", ">": "<", "<=": ">=", ">=": "<=", "=": "=", "!=": "!="}


def eval_expr(e: Expr, df: cudf.DataFrame):
    if isinstance(e, Col):
        if e.name not in df.columns:
            raise KeyError(f"unknown column {e.name!r}; available: {list(df.columns)}")
        return df[e.name]
    if isinstance(e, Star):
        raise ValueError("'*' is only valid inside COUNT(*)")
    if isinstance(e, Lit):
        return _literal(e)
    if isinstance(e, BinOp):
        return _binop(e, df)
    if isinstance(e, And):
        return _bool_combine(eval_expr(e.left, df), eval_expr(e.right, df), "&")
    if isinstance(e, Or):
        return _bool_combine(eval_expr(e.left, df), eval_expr(e.right, df), "|")
    if isinstance(e, Not):
        v = eval_expr(e.expr, df)
        return ~v if isinstance(v, cudf.Series) else not v
    if isinstance(e, IsNull):
        return _isnull(e, df)
    if isinstance(e, In):
        return _in(e, df)
    if isinstance(e, Like):
        return _like(e, df)
    if isinstance(e, Case):
        return _case(e, df)
    if isinstance(e, Coalesce):
        return _coalesce(e, df)
    if isinstance(e, Cast):
        return _cast(e, df)
    if isinstance(e, Func):
        return _func(e, df)
    if isinstance(e, AggFunc):
        raise ValueError("aggregate used outside of GROUP BY / aggregate context")
    raise NotImplementedError(f"cannot evaluate {type(e).__name__}")


def _literal(e: Lit):
    v = e.value
    if e.dtype == "DATE":
        return pd.Timestamp(v)
    if e.dtype == "int":
        return int(v)
    if e.dtype == "float":
        return float(v)
    if e.dtype == "bool":
        return bool(v)
    if e.dtype == "null":
        return None
    return v


def _binop(e: BinOp, df: cudf.DataFrame):
    left = eval_expr(e.left, df)
    right = eval_expr(e.right, df)
    op = e.op
    if op in ("=", "!=", "<", "<=", ">", ">="):
        return _compare(op, left, right)
    return _arith(op, left, right)


def _compare(op, left, right):
    l_series = isinstance(left, cudf.Series)
    r_series = isinstance(right, cudf.Series)
    if not l_series and not r_series:
        lv, rv = left, right
        if op == "=":
            return lv == rv
        if op == "!=":
            return lv != rv
        if op == "<":
            return lv < rv
        if op == "<=":
            return lv <= rv
        if op == ">":
            return lv > rv
        if op == ">=":
            return lv >= rv
    # Ensure the Series is on the left for a clean `series <op> scalar` form.
    if r_series and not l_series:
        return _compare(_CMP_SWAP[op], right, left)
    # left is a Series here
    if op == "=":
        return left == right
    if op == "!=":
        return left != right
    if op == "<":
        return left < right
    if op == "<=":
        return left <= right
    if op == ">":
        return left > right
    if op == ">=":
        return left >= right
    raise NotImplementedError(f"unsupported comparison {op}")


def _arith(op, left, right):
    l_series = isinstance(left, cudf.Series)
    r_series = isinstance(right, cudf.Series)
    if not l_series and not r_series:
        return _py_arith(op, left, right)
    # cuDF supports reflected scalar ops (1 - s, 10 / s) via __rsub__/__rtruediv__.
    return _series_arith(op, left, right)


def _series_arith(op, lv, rv):
    if op == "+":
        return lv + rv
    if op == "-":
        return lv - rv
    if op == "*":
        return lv * rv
    if op == "/":
        return lv / rv
    if op == "%":
        # DuckDB MOD takes the sign of the dividend (truncated mod); Python/cuDF
        # `%` is floored (sign of the divisor), so -2.5 % 3 = 0.5 here vs -2.5 in
        # DuckDB. Match DuckDB: x - y * trunc(x / y).
        return lv - rv * np.trunc(lv / rv)
    raise NotImplementedError(f"unsupported arithmetic operator {op}")


def _py_arith(op, lv, rv):
    if op == "+":
        return lv + rv
    if op == "-":
        return lv - rv
    if op == "*":
        return lv * rv
    if op == "/":
        return lv / rv
    if op == "%":
        # Truncated mod (DuckDB sign-of-dividend convention); see _series_arith.
        return lv - rv * math.trunc(lv / rv)
    raise NotImplementedError(f"unsupported arithmetic operator {op}")


def _bool_combine(a, b, op):
    a_s = isinstance(a, cudf.Series)
    b_s = isinstance(b, cudf.Series)
    if a_s and b_s:
        return (a & b) if op == "&" else (a | b)
    if not a_s and not b_s:
        return (a and b) if op == "&" else (a or b)
    # mixed scalar/series
    if op == "&":
        return b if a else _full_false(b)
    return _full_true(b) if a else b


def _full_false(series):
    return cudf.Series([False] * len(series))


def _full_true(series):
    return cudf.Series([True] * len(series))


# --------------------------------------------------------------------------- #
# Predicates / expressions: IS NULL, IN, LIKE, CASE, COALESCE, CAST.
# Comparisons return boolean Series (Filter masks); CASE/COALESCE/CAST return
# value Series. NULL operands follow SQL three-valued logic where it matters:
# a NULL in IS NULL is TRUE (handled by isna); a NULL in IN/LIKE yields NA
# (dropped at the Filter layer via fillna(False)).
# --------------------------------------------------------------------------- #


def _as_series(v, df: cudf.DataFrame) -> "cudf.Series | pd.Series":
    """Broadcast a Python scalar to a Series aligned to df's index; pass a
    Series through. Lets scalar then-values / coalesce args / cast operands
    combine with column Series via .where/.fillna."""
    if isinstance(v, cudf.Series):
        return v
    if isinstance(v, pd.Series):
        return v
    return cudf.Series([v] * len(df), index=df.index)


def _isnull(e: IsNull, df: cudf.DataFrame):
    v = eval_expr(e.expr, df)
    if isinstance(v, cudf.Series):
        m = v.isna()
        return (~m) if e.negated else m
    isn = pd.isna(v)
    return (not isn) if e.negated else isn


def _in(e: In, df: cudf.DataFrame):
    s = _as_series(eval_expr(e.expr, df), df)
    vals = [eval_expr(v, df) for v in e.values]
    # Scalars expected; if a value is itself a column Series, isin can't take it.
    if any(isinstance(v, cudf.Series) for v in vals):
        raise NotImplementedError("IN with a column-typed list element is not supported")
    m = s.isin(vals)
    # NULL operand -> NA (not False), so NOT IN of a NULL is also NA (dropped).
    m = m.where(s.notna())
    return (~m) if e.negated else m


def _like_to_regex(pattern: str) -> str:
    """Translate a SQL LIKE pattern to an anchored regex. ``%`` -> ``.*``,
    ``_`` -> ``.``, every other literal char is regex-escaped."""
    body = []
    for ch in pattern:
        if ch == "%":
            body.append(".*")
        elif ch == "_":
            body.append(".")
        else:
            body.append(re.escape(ch))
    return f"^{''.join(body)}$"


def _like(e: Like, df: cudf.DataFrame):
    s = _as_series(eval_expr(e.expr, df), df)
    pat = eval_expr(e.pattern, df)
    if isinstance(pat, cudf.Series):
        raise NotImplementedError("LIKE with a non-literal pattern is not supported")
    pat = str(pat)
    # ILIKE: libcudf's regex engine rejects the inline ``(?i)`` flag, so lower-
    # case both sides and match case-sensitively instead.
    if not e.case_sensitive:
        s = s.str.lower()
        pat = pat.lower()
    regex = _like_to_regex(pat)
    m = s.str.contains(regex, regex=True, na=False)
    # NULL operand -> NA (na=False forced False; restore NA so NOT LIKE of a
    # NULL is NA, not True -- SQL three-valued logic).
    m = m.where(s.notna())
    return (~m) if e.negated else m


def _case(e: Case, df: cudf.DataFrame):
    # Default: the ELSE expr, or a full-NA Series typed to the first THEN so the
    # .where chain upcasts consistently. Apply branches in reverse so the FIRST
    # true condition wins (first-branch applied last overrides the rest).
    first_then = _as_series(eval_expr(e.branches[0][1], df), df) if e.branches else None
    if e.default is not None:
        result = _as_series(eval_expr(e.default, df), df)
    elif first_then is not None:
        result = cudf.Series([None] * len(df), index=df.index).astype(first_then.dtype)
    else:
        return cudf.Series([None] * len(df), index=df.index)
    for cond, then in reversed(e.branches):
        mask = _as_series(eval_expr(cond, df), df).fillna(False)
        result = _as_series(eval_expr(then, df), df).where(mask, result)
    return result


def _coalesce(e: Coalesce, df: cudf.DataFrame):
    result = _as_series(eval_expr(e.args[0], df), df)
    for a in e.args[1:]:
        result = result.fillna(_as_series(eval_expr(a, df), df))
    return result


def _cast(e: Cast, df: cudf.DataFrame):
    v = eval_expr(e.expr, df)
    tag = e.dtype
    if tag in ("date", "timestamp"):
        if isinstance(v, cudf.Series):
            ts = pd.to_datetime(v.to_pandas()) if isinstance(v, cudf.Series) else pd.to_datetime(v)
            ts = cudf.Series(ts, index=df.index) if not isinstance(ts, pd.Series) else ts
            return ts if tag == "timestamp" else ts.dt.date
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        return pd.Timestamp(v) if tag == "timestamp" else pd.Timestamp(v).date()
    if isinstance(v, cudf.Series):
        if tag == "int":
            # DuckDB CAST(double AS int) rounds to nearest; bare astype truncates
            # and int64 cannot hold NA. Round, then go through nullable Int64.
            if "float" in str(v.dtype):
                return v.round().astype("Int64")
            return v.astype("Int64")
        if tag == "float":
            return v.astype("float64")
        if tag == "bool":
            return v.astype("bool")
        return v.astype("str")
    if isinstance(v, pd.Series):
        return v.astype({"int": "Int64", "float": "float64", "bool": "bool", "str": "str"}[tag])
    # scalar
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if tag == "int":
        return int(round(float(v)))
    if tag == "float":
        return float(v)
    if tag == "bool":
        return bool(v)
    return str(v)


# --------------------------------------------------------------------------- #
# Scalar functions (UPPER/LOWER/LENGTH/SUBSTR/TRIM/CONCAT/||/REPLACE/POSITION/
# LEFT/RIGHT/INITCAP/REVERSE/ABS/ROUND/CEIL/FLOOR). Dispatched by Func.name to a
# cuDF Series op (numpy ufuncs for ceil/floor/sign). NULL operands propagate NA
# for the str accessors and ||; CONCAT fills NA with "" (NULL-ignoring, matching
# DuckDB). SUBSTR/LEFT/RIGHT bounds are scalar-only (cuDF str.slice requires
# scalar bounds). ROUND uses half-away-from-zero (cuDF Series.round is banker's
# rounding, which mismatches DuckDB on .5).
# --------------------------------------------------------------------------- #


def _as_str_series(v, df: cudf.DataFrame) -> cudf.Series:
    """Coerce an evaluated arg to a string Series aligned to df (casts ints /
    floats / scalars to str; broadcasts scalars). Lets CONCAT / || mix string
    and numeric args the way DuckDB does."""
    s = _as_series(v, df)
    if not isinstance(s, cudf.Series):
        # pandas Series fallback -> wrap as cudf
        s = cudf.Series(s, index=df.index)
    if not pd.api.types.is_string_dtype(s.dtype):
        s = s.astype("str")
    return s


def _scalar_int(v):
    """An int bound for SUBSTR/LEFT/RIGHT. Must be a python scalar (cuDF
    str.slice takes scalar bounds); a column-typed bound is unsupported."""
    if isinstance(v, (cudf.Series, pd.Series)):
        raise NotImplementedError("SUBSTR/LEFT/RIGHT with a column-typed bound is not supported")
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    return int(v)


def _func(e: Func, df: cudf.DataFrame):
    name = e.name
    args = e.args

    # --- single-arg string transforms ----------------------------------- #
    if name == "upper":
        v = eval_expr(args[0], df)
        return v.str.upper() if isinstance(v, cudf.Series) else (None if v is None else str(v).upper())
    if name == "lower":
        v = eval_expr(args[0], df)
        return v.str.lower() if isinstance(v, cudf.Series) else (None if v is None else str(v).lower())
    if name == "initcap":
        v = eval_expr(args[0], df)
        return v.str.title() if isinstance(v, cudf.Series) else (None if v is None else str(v).title())
    if name == "reverse":
        v = eval_expr(args[0], df)
        if isinstance(v, cudf.Series):
            return v.str.slice(None, None, -1)
        return None if v is None else str(v)[::-1]
    if name == "length":
        v = eval_expr(args[0], df)
        return v.str.len() if isinstance(v, cudf.Series) else (None if v is None else len(str(v)))

    # --- multi-arg string funcs ----------------------------------------- #
    if name == "substr":
        s = _as_str_series(eval_expr(args[0], df), df)
        start = _scalar_int(eval_expr(args[1], df))
        if start is None:
            return cudf.Series([None] * len(df), index=df.index)
        st = start - 1  # SQL 1-based -> cuDF 0-based
        if st < 0:
            st = 0
        if len(args) > 2:
            ln = _scalar_int(eval_expr(args[2], df))
            if ln is None:
                return cudf.Series([None] * len(df), index=df.index)
            if ln < 0:
                ln = 0
            return s.str.slice(st, st + ln)
        return s.str.slice(st)
    if name == "left":
        s = _as_str_series(eval_expr(args[0], df), df)
        n = _scalar_int(eval_expr(args[1], df))
        if n is None:
            return cudf.Series([None] * len(df), index=df.index)
        if n < 0:
            n = 0
        return s.str.slice(0, n)
    if name == "right":
        s = _as_str_series(eval_expr(args[0], df), df)
        n = _scalar_int(eval_expr(args[1], df))
        if n is None:
            return cudf.Series([None] * len(df), index=df.index)
        if n <= 0:
            return s.str.slice(0, 0)
        # cuDF rejects a negative slice start; reverse, take the prefix, reverse.
        return s.str.slice(None, None, -1).str.slice(0, n).str.slice(None, None, -1)
    if name == "replace":
        s = _as_str_series(eval_expr(args[0], df), df)
        frm = str(eval_expr(args[1], df))
        to = str(eval_expr(args[2], df))
        return s.str.replace(frm, to)
    if name == "strpos":
        hay = _as_str_series(eval_expr(args[0], df), df)
        needle = str(eval_expr(args[1], df))
        return hay.str.find(needle) + 1  # 0-based -1 (not found) -> 0; NA stays NA
    if name == "trim":
        s = _as_str_series(eval_expr(args[0], df), df)
        chars = eval_expr(args[1], df)  # None for whitespace
        side = eval_expr(args[2], df)
        to_strip = None if chars is None else str(chars)
        if side == "LEADING":
            fn = s.str.lstrip
        elif side == "TRAILING":
            fn = s.str.rstrip
        else:
            fn = s.str.strip
        return fn(to_strip) if to_strip is not None else fn()
    if name in ("concat", "concat_pipe"):
        parts = [_as_str_series(eval_expr(a, df), df) for a in args]
        if name == "concat":
            # DuckDB CONCAT ignores NULLs (treat as empty).
            parts = [p.fillna("") for p in parts]
        acc = parts[0]
        for p in parts[1:]:
            acc = acc.str.cat(p)
        return acc

    # --- numeric funcs --------------------------------------------------- #
    if name == "abs":
        v = eval_expr(args[0], df)
        return v.abs() if isinstance(v, cudf.Series) else (None if v is None else abs(v))
    if name == "ceil":
        v = eval_expr(args[0], df)
        if isinstance(v, cudf.Series):
            return np.ceil(v)
        return None if v is None else math.ceil(v)
    if name == "floor":
        v = eval_expr(args[0], df)
        if isinstance(v, cudf.Series):
            return np.floor(v)
        return None if v is None else math.floor(v)
    if name == "round":
        v = eval_expr(args[0], df)
        d = int(eval_expr(args[1], df)) if len(args) > 1 else 0
        f = 10 ** d
        if isinstance(v, cudf.Series):
            # Half-away-from-zero (DuckDB); cuDF Series.round is banker's rounding.
            return np.floor(v.abs() * f + 0.5) / f * np.sign(v)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        sign = -1.0 if v < 0 else 1.0
        return math.floor(abs(v) * f + 0.5) / f * sign

    raise NotImplementedError(f"unsupported scalar function: {name}")