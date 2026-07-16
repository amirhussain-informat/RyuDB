"""Expression evaluation on cuDF Series / scalars.

`eval_expr(expr, df)` returns either a cuDF Series (column-aligned) or a Python
scalar. Comparisons return boolean Series (used as Filter masks); arithmetic
returns numeric Series. Date literals are converted to pandas timestamps so
they compare against cuDF date32/datetime64 columns.
"""

from __future__ import annotations

from datetime import date

import cudf
import pandas as pd

from ..sql.plan import (
    AggFunc,
    And,
    BinOp,
    Col,
    Expr,
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
        l, r = left, right
        if op == "=":
            return l == r
        if op == "!=":
            return l != r
        if op == "<":
            return l < r
        if op == "<=":
            return l <= r
        if op == ">":
            return l > r
        if op == ">=":
            return l >= r
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


def _series_arith(op, l, r):
    if op == "+":
        return l + r
    if op == "-":
        return l - r
    if op == "*":
        return l * r
    if op == "/":
        return l / r
    raise NotImplementedError(f"unsupported arithmetic operator {op}")


def _py_arith(op, l, r):
    if op == "+":
        return l + r
    if op == "-":
        return l - r
    if op == "*":
        return l * r
    if op == "/":
        return l / r
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