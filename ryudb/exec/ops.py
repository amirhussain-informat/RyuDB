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


def _as_dt_series(v, df: cudf.DataFrame) -> cudf.Series:
    """Coerce an evaluated arg to a nanosecond datetime Series aligned to df.

    Storage DATE/TIMESTAMP columns arrive from ``Engine._scan`` as cuDF
    ``datetime64[ms]`` (storage.scan) or ``datetime64[s]`` (cold-cache path) --
    never date32/object. Normalizing to ``datetime64[ns]`` makes the EPOCH and
    DATEDIFF int64 formulas (validated on [ns]) unit-correct: a [ms] column
    ``.astype('int64')`` gives ms, not ns, so without normalization EPOCH would
    be off by 1e6. Object/string dates parse via ``cudf.to_datetime``. Scalars
    broadcast to a Series; a NULL scalar (None/NaT) yields a NaT Series.
    """
    if isinstance(v, cudf.Series):
        s = v
        if not pd.api.types.is_datetime64_any_dtype(s.dtype):
            s = cudf.to_datetime(s)  # parse string/object dates
        if str(s.dtype) != "datetime64[ns]":
            s = s.astype("datetime64[ns]")
        return s
    ts = v if isinstance(v, pd.Timestamp) else pd.Timestamp(v)
    return cudf.Series([ts] * len(df), index=df.index).astype("datetime64[ns]")


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

    # --- date/time functions ------------------------------------------- #
    if name == "extract":
        return _extract(args, df)
    if name in ("date_trunc", "date_add", "date_sub", "datediff",
                "dayname", "monthname", "last_day", "strftime",
                "current_date", "current_timestamp"):
        return _datetime_func(name, args, df)

    raise NotImplementedError(f"unsupported scalar function: {name}")


# --------------------------------------------------------------------------- #
# Date/time functions (EXTRACT/YEAR/.../DATE_TRUNC/DATEDIFF/date +/- INTERVAL/
# DAYNAME/MONTHNAME/LAST_DAY/STRFTIME/CURRENT_DATE/CURRENT_TIMESTAMP). Storage
# DATE/TIMESTAMP columns arrive as cuDF datetime64 (normalized to [ns] by
# _as_dt_series); most parts use the GPU ``.dt`` accessor. cuDF ``.dt.floor``
# lacks 'Y'/'M', so year/month truncation and month/year interval addition use a
# pandas ``to_period``/``DateOffset`` fallback. NULL -> NaT -> NaN (normalized
# to None by conftest.as_sorted); the manual int64 paths (EPOCH, DATEDIFF
# hour/min/sec) mask NaT explicitly since ``astype('int64')`` turns NaT into a
# sentinel, not NaN.
# --------------------------------------------------------------------------- #

_EXTRACT_FIELD = {
    "YEAR": "year", "MONTH": "month", "DAY": "day",
    "HOUR": "hour", "MINUTE": "minute", "SECOND": "second",
}


def _extract(args, df: cudf.DataFrame):
    v = eval_expr(args[0], df)
    field = str(eval_expr(args[1], df)).upper()
    if not isinstance(v, cudf.Series):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        ts = pd.Timestamp(v)
        if field == "DOW":
            return int((ts.weekday() + 1) % 7)
        if field == "DOY":
            return int(ts.timetuple().tm_yday)
        if field == "EPOCH":
            return float(ts.timestamp())
        return int(getattr(ts, _EXTRACT_FIELD[field]))
    s = _as_dt_series(v, df)
    if field in _EXTRACT_FIELD:
        return getattr(s.dt, _EXTRACT_FIELD[field])
    if field == "DOY":
        return s.dt.dayofyear
    if field == "DOW":
        return (s.dt.dayofweek + 1) % 7
    if field == "EPOCH":
        # s is [ns]; int64 is nanoseconds. Mask NaT (-> sentinel) to NaN.
        r = s.astype("int64") / 1e9
        return r.where(s.notna())
    raise NotImplementedError(f"unsupported EXTRACT field: {field}")


# Fixed-length interval units -> nanoseconds per unit (for GPU timedelta). Month/
# year are variable-length and handled separately via a pandas DateOffset.
_FIXED_UNIT_NS = {
    "day": 86_400, "days": 86_400,
    "week": 7 * 86_400, "weeks": 7 * 86_400,
    "hour": 3_600, "hours": 3_600,
    "minute": 60, "minutes": 60,
    "second": 1, "seconds": 1,
}
# cuDF .dt.floor unit codes (no 'Y'/'M' -- those use the pandas to_period path).
_TRUNC_FLOOR = {"day": "D", "hour": "h", "minute": "min", "second": "s"}


def _datetime_func(name, args, df: cudf.DataFrame):
    if name == "current_date":
        return pd.Timestamp.now().normalize()
    if name == "current_timestamp":
        return pd.Timestamp.now()

    v = eval_expr(args[0], df)

    if name == "dayname":
        if isinstance(v, cudf.Series):
            return _as_dt_series(v, df).dt.day_name()
        return None if v is None or pd.isna(v) else pd.Timestamp(v).day_name()
    if name == "monthname":
        if isinstance(v, cudf.Series):
            return _as_dt_series(v, df).dt.month_name()
        return None if v is None or pd.isna(v) else pd.Timestamp(v).month_name()
    if name == "strftime":
        fmt = str(eval_expr(args[1], df))
        if isinstance(v, cudf.Series):
            return _as_dt_series(v, df).dt.strftime(fmt)
        return None if v is None or pd.isna(v) else pd.Timestamp(v).strftime(fmt)
    if name == "last_day":
        if isinstance(v, cudf.Series):
            s = _as_dt_series(v, df)
            # MonthEnd(0) rolls to the current month's last day but keeps the time
            # component; DuckDB LAST_DAY returns a DATE at midnight, so floor.
            last = (s.to_pandas() + pd.offsets.MonthEnd(0)).dt.floor("D")
            return cudf.Series(last, index=df.index).astype("datetime64[ns]")
        if v is None or pd.isna(v):
            return None
        return (pd.Timestamp(v) + pd.offsets.MonthEnd(0)).normalize()

    if name == "date_trunc":
        unit = str(eval_expr(args[1], df)).lower()
        if isinstance(v, cudf.Series):
            s = _as_dt_series(v, df)
            if unit in ("year", "years"):
                return cudf.Series(s.to_pandas().dt.to_period("Y").dt.to_timestamp(),
                                   index=df.index).astype("datetime64[ns]")
            if unit in ("month", "months"):
                return cudf.Series(s.to_pandas().dt.to_period("M").dt.to_timestamp(),
                                   index=df.index).astype("datetime64[ns]")
            if unit in _TRUNC_FLOOR:
                return s.dt.floor(_TRUNC_FLOOR[unit])
            raise NotImplementedError(f"unsupported DATE_TRUNC unit: {unit}")
        if v is None or pd.isna(v):
            return None
        ts = pd.Timestamp(v)
        if unit in ("year", "years"):
            return ts.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        if unit in ("month", "months"):
            return ts.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if unit in ("day", "days"):
            return ts.normalize()
        if unit in ("hour", "hours"):
            return ts.replace(minute=0, second=0, microsecond=0)
        if unit in ("minute", "minutes"):
            return ts.replace(second=0, microsecond=0)
        if unit in ("second", "seconds"):
            return ts.replace(microsecond=0)
        raise NotImplementedError(f"unsupported DATE_TRUNC unit: {unit}")

    if name in ("date_add", "date_sub"):
        n = int(eval_expr(args[1], df))
        unit = str(eval_expr(args[2], df)).lower()
        sign = 1 if name == "date_add" else -1
        if unit in _FIXED_UNIT_NS:
            nsec = n * _FIXED_UNIT_NS[unit] * 10**9
            off = np.timedelta64(nsec, "ns")
            if isinstance(v, cudf.Series):
                return _as_dt_series(v, df) + sign * off
            if v is None or pd.isna(v):
                return None
            return pd.Timestamp(v) + sign * pd.Timedelta(nanoseconds=nsec)
        if unit in ("month", "months"):
            off = pd.DateOffset(months=n)
        elif unit in ("year", "years"):
            off = pd.DateOffset(years=n)
        else:
            raise NotImplementedError(f"unsupported INTERVAL unit: {unit}")
        delta = off if sign == 1 else -off
        if isinstance(v, cudf.Series):
            return cudf.Series(_as_dt_series(v, df).to_pandas() + delta,
                               index=df.index).astype("datetime64[ns]")
        if v is None or pd.isna(v):
            return None
        return pd.Timestamp(v) + delta

    if name == "datediff":
        start = eval_expr(args[0], df)
        end = eval_expr(args[1], df)
        unit = str(eval_expr(args[2], df)).lower()
        factor = {"day": 86_400, "hour": 3_600, "minute": 60, "second": 1}.get(
            unit.rstrip("s") if unit.endswith("s") else unit)
        if factor is None:
            raise NotImplementedError(f"unsupported DATEDIFF unit: {unit}")
        if isinstance(end, cudf.Series) or isinstance(start, cudf.Series):
            delta = _as_dt_series(end, df) - _as_dt_series(start, df)  # timedelta64[ns]
            if unit in ("day", "days"):
                return delta.dt.days
            r = delta.astype("int64") // (factor * 10**9)
            return r.where(delta.notna())
        if end is None or pd.isna(end) or start is None or pd.isna(start):
            return None
        delta = pd.Timestamp(end) - pd.Timestamp(start)
        if unit in ("day", "days"):
            return int(delta.days)
        return int(delta.total_seconds() // factor)

    raise NotImplementedError(f"unsupported date function: {name}")