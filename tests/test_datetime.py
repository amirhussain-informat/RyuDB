"""SQL surface, Phase D: date/time functions & arithmetic -- RyuDB vs DuckDB.

Storage DATE/TIMESTAMP columns arrive in ``Engine._scan`` as cuDF
``datetime64`` (normalized to [ns] by ``ops._as_dt_series``), so most parts use
the GPU ``.dt`` accessor. cuDF ``.dt.floor`` lacks ``'Y'``/``'M'`` (``Invalid
resolution``), so year/month truncation and month/year interval addition take a
pandas ``to_period``/``DateOffset`` fallback; day/hour/minute/second stay on the
GPU. ``EXTRACT(EPOCH ...)`` and ``DATEDIFF``'s hour/min/sec paths mask NaT
explicitly since ``astype('int64')`` turns NaT into a sentinel rather than NaN.
Comparison is via ``conftest.as_sorted`` (NULL -> None, floats rounded,
order-independent). The fixture is written by DuckDB so the DATE column is a
real parquet DATE logical type.

Deferred (still raise): MAKE_DATE, AGE, bare ``d1 - d2`` (use DATEDIFF),
``d + integer``, DATE_FORMAT (use STRFTIME), TO_TIMESTAMP.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ryudb import Catalog, Engine
from ryudb.sql.parse import parse

from .conftest import as_sorted

# d  -- DATE; ts -- TIMESTAMP; d2 -- DATE (a second date so DATEDIFF gets two
# stored columns and both engines see identical inputs, no nested intervals).
# Includes a leap day (2024-02-29), a month-end (2023-12-31), and a NULL row.
_ROWS = [
    ("2023-01-15", "2023-01-15 12:30:45", "2023-01-20"),
    ("2023-02-20", "2023-02-20 03:04:05", "2023-03-25"),
    ("2023-12-31", "2023-12-31 23:59:59", "2024-01-02"),
    ("2024-02-29", "2024-02-29 00:00:00", "2024-03-05"),
    (None, None, None),
]


@pytest.fixture
def sdir(tmp_path):
    import duckdb

    con = duckdb.connect()
    df = pd.DataFrame(_ROWS, columns=["d", "ts", "d2"])
    # DuckDB writes DATE/TIMESTAMP as the parquet logical types RyuDB reads.
    con.register("src", df)
    (tmp_path / "t").mkdir()
    con.execute(
        f"COPY (SELECT CAST(d AS DATE) AS d, CAST(ts AS TIMESTAMP) AS ts, "
        f"CAST(d2 AS DATE) AS d2 FROM src) TO '{tmp_path}/t/0.parquet' (FORMAT PARQUET)"
    )
    return tmp_path


@pytest.fixture
def sengine(sdir) -> Engine:
    cat = Catalog(str(sdir))
    cat.register("t", str(sdir / "t"))
    return Engine(cat)


@pytest.fixture
def sduck(sdir):
    import duckdb

    con = duckdb.connect()
    con.execute(f"CREATE VIEW t AS SELECT * FROM read_parquet('{sdir}/t/*.parquet')")
    return con


def _ryu(engine: Engine, sql: str):
    return as_sorted(engine.sql(sql))


def _duck(con, sql: str):
    return as_sorted(con.execute(sql).fetchdf())


# --------------------------------------------------------------------------- #
# EXTRACT and the YEAR()/MONTH()/... sugar
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("field,expr", [
    ("YEAR", "EXTRACT(YEAR FROM d)"),
    ("MONTH", "EXTRACT(MONTH FROM d)"),
    ("DAY", "EXTRACT(DAY FROM d)"),
])
def test_extract_date_parts(sengine, sduck, field, expr):
    assert _ryu(sengine, f"SELECT {expr} AS p FROM t") == \
        _duck(sduck, f"SELECT {expr} AS p FROM t")


@pytest.mark.parametrize("field,expr", [
    ("HOUR", "EXTRACT(HOUR FROM ts)"),
    ("MINUTE", "EXTRACT(MINUTE FROM ts)"),
    ("SECOND", "EXTRACT(SECOND FROM ts)"),
])
def test_extract_time_parts(sengine, sduck, field, expr):
    assert _ryu(sengine, f"SELECT {expr} AS p FROM t") == \
        _duck(sduck, f"SELECT {expr} AS p FROM t")


def test_extract_dow(sengine, sduck):
    # DOW convention differs (cuDF dayofweek Mon=0..Sun=6 vs DuckDB Sun=0..Sat=6);
    # ops applies (dayofweek + 1) % 7.
    assert _ryu(sengine, "SELECT EXTRACT(DOW FROM d) AS p FROM t WHERE d IS NOT NULL") == \
        _duck(sduck, "SELECT EXTRACT(DOW FROM d) AS p FROM t WHERE d IS NOT NULL")


def test_extract_doy(sengine, sduck):
    assert _ryu(sengine, "SELECT EXTRACT(DOY FROM d) AS p FROM t") == \
        _duck(sduck, "SELECT EXTRACT(DOY FROM d) AS p FROM t")


def test_extract_epoch(sengine, sduck):
    # EPOCH returns float seconds (DuckDB); int64 nanoseconds / 1e9.
    assert _ryu(sengine, "SELECT EXTRACT(EPOCH FROM ts) AS p FROM t") == \
        _duck(sduck, "SELECT EXTRACT(EPOCH FROM ts) AS p FROM t")


@pytest.mark.parametrize("expr", [
    "YEAR(d)", "MONTH(d)", "DAY(d)", "DAYOFWEEK(d)", "DAYOFYEAR(d)",
])
def test_year_func_sugar(sengine, sduck, expr):
    assert _ryu(sengine, f"SELECT {expr} AS p FROM t") == \
        _duck(sduck, f"SELECT {expr} AS p FROM t")


# --------------------------------------------------------------------------- #
# DATE_TRUNC
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("unit", ["year", "month", "day", "hour"])
def test_date_trunc(sengine, sduck, unit):
    # Truncate the TIMESTAMP so hour/min/sec truncation is non-trivial; year/month
    # go through the pandas to_period fallback (cuDF .dt.floor lacks 'Y'/'M').
    col = "ts"
    assert _ryu(sengine, f"SELECT DATE_TRUNC('{unit}', {col}) AS p FROM t") == \
        _duck(sduck, f"SELECT DATE_TRUNC('{unit}', {col}) AS p FROM t")


# --------------------------------------------------------------------------- #
# date +/- INTERVAL
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("sql", [
    "SELECT d + INTERVAL '1' DAY AS p FROM t",
    "SELECT d + INTERVAL '1' MONTH AS p FROM t",
    "SELECT d + INTERVAL '1' YEAR AS p FROM t",
    "SELECT ts + INTERVAL '1' HOUR AS p FROM t",
])
def test_date_add(sengine, sduck, sql):
    assert _ryu(sengine, sql) == _duck(sduck, sql)


def test_date_sub(sengine, sduck):
    # INTERVAL on the right of '-'; day uses the GPU timedelta path.
    assert _ryu(sengine, "SELECT d - INTERVAL '1' DAY AS p FROM t") == \
        _duck(sduck, "SELECT d - INTERVAL '1' DAY AS p FROM t")


def test_date_add_week(sengine, sduck):
    assert _ryu(sengine, "SELECT d + INTERVAL '1' WEEK AS p FROM t") == \
        _duck(sduck, "SELECT d + INTERVAL '7' DAY AS p FROM t")


# --------------------------------------------------------------------------- #
# DATEDIFF (end - start in unit)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("unit", ["day", "hour", "second"])
def test_datediff(sengine, sduck, unit):
    sql = f"SELECT DATEDIFF('{unit}', d, d2) AS p FROM t"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


# --------------------------------------------------------------------------- #
# DAYNAME / MONTHNAME / LAST_DAY / STRFTIME
# --------------------------------------------------------------------------- #


def test_dayname(sengine, sduck):
    assert _ryu(sengine, "SELECT DAYNAME(d) AS p FROM t WHERE d IS NOT NULL") == \
        _duck(sduck, "SELECT DAYNAME(d) AS p FROM t WHERE d IS NOT NULL")


def test_monthname(sengine, sduck):
    assert _ryu(sengine, "SELECT MONTHNAME(d) AS p FROM t WHERE d IS NOT NULL") == \
        _duck(sduck, "SELECT MONTHNAME(d) AS p FROM t WHERE d IS NOT NULL")


def test_last_day(sengine, sduck):
    # Leap day and month-end rows exercise MonthEnd; the result is a DATE at
    # midnight (MonthEnd(0) then floor-to-day).
    assert _ryu(sengine, "SELECT LAST_DAY(d) AS p FROM t") == \
        _duck(sduck, "SELECT LAST_DAY(d) AS p FROM t")


def test_strftime(sengine, sduck):
    assert _ryu(sengine, "SELECT STRFTIME(d, '%Y-%m-%d') AS p FROM t WHERE d IS NOT NULL") == \
        _duck(sduck, "SELECT STRFTIME(d, '%Y-%m-%d') AS p FROM t WHERE d IS NOT NULL")


# --------------------------------------------------------------------------- #
# CURRENT_DATE / CURRENT_TIMESTAMP / NOW()
# --------------------------------------------------------------------------- #


def test_current_date(sengine, sduck):
    # Both engines run in-process on the same day; _project broadcasts the scalar.
    r = sengine.sql("SELECT CURRENT_DATE AS today FROM t")
    assert list(r.columns) == ["today"]
    vals = r["today"].to_pandas().dropna().unique().tolist()
    assert len(vals) == 1
    assert pd.Timestamp(vals[0]).normalize() == pd.Timestamp.now().normalize()


def test_current_timestamp_year(sengine, sduck):
    # Avoid sub-second fragility; the year is the same on both engines.
    r = sengine.sql("SELECT EXTRACT(YEAR FROM CURRENT_TIMESTAMP) AS y FROM t")
    vals = r["y"].to_pandas().dropna().unique().tolist()
    assert len(vals) == 1
    assert int(vals[0]) == pd.Timestamp.now().year


def test_now_runs(sengine):
    # NOW() is exp.Anonymous -> Func("current_timestamp"). Don't exact-compare vs
    # DuckDB (sub-second skew); just confirm it parses, runs, and yields one
    # distinct timestamp per row.
    r = sengine.sql("SELECT NOW() AS n FROM t")
    assert list(r.columns) == ["n"]
    assert len(r) == len(_ROWS)


# --------------------------------------------------------------------------- #
# NULL semantics + GROUP BY over a date Func (cuDF groupby path; fused defers)
# --------------------------------------------------------------------------- #


def test_extract_null(sengine, sduck):
    # EXTRACT over the NULL row -> NULL (cuDF .dt -> NaN -> as_sorted -> None).
    r = _ryu(sengine, "SELECT EXTRACT(YEAR FROM d) AS p FROM t")
    d = _duck(sduck, "SELECT EXTRACT(YEAR FROM d) AS p FROM t")
    assert r == d


def test_date_func_in_groupby(sengine, sduck):
    # The Func group key runs on the cuDF groupby path (fused CUDA defers on any
    # unknown Func). NULL year groups to NULL.
    sql = "SELECT YEAR(d) AS y, COUNT(*) AS c FROM t GROUP BY YEAR(d)"
    assert _ryu(sengine, sql) == _duck(sduck, sql)


# --------------------------------------------------------------------------- #
# Parsing: date funcs lower to Func tags
# --------------------------------------------------------------------------- #


def test_extract_parses_to_func():
    from ryudb.sql.plan import Func, Project

    plan = parse("SELECT EXTRACT(YEAR FROM d) AS y FROM t")
    assert isinstance(plan, Project)
    proj = plan.items[0][0]  # (Expr, output_name)
    assert isinstance(proj, Func)
    assert proj.name == "extract"


def test_current_date_parses_to_func():
    from ryudb.sql.plan import Func, Project

    plan = parse("SELECT CURRENT_DATE AS today FROM t")
    assert isinstance(plan, Project)
    proj = plan.items[0][0]
    assert isinstance(proj, Func)
    assert proj.name == "current_date"