"""Benchmark RyuDB (GPU/cuDF) vs DuckDB (CPU) on TPC-H.

Data is generated with DuckDB's `tpch` extension (dbgen) and written to Parquet,
so both engines read the exact same columnar files. Each query is checked for
correctness against DuckDB and then timed (min of N repeats after a warm-up).

Usage:
  python bench/run_bench.py --scale 1 --repeats 3
  python bench/run_bench.py --scale 0.1   # quick smoke on a tiny dataset
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ryudb import Catalog, Engine  # noqa: E402

TABLES = ["nation", "region", "part", "supplier", "partsupp", "customer", "orders", "lineitem"]

# TPC-H and TPC-H-like queries that fit RyuDB's Phase-1 SQL subset
# (no subqueries, no CASE, no IN, equi-joins only).
QUERIES: dict[str, str] = {
    "Q1_pricing_summary": """
        SELECT l_returnflag, l_linestatus,
               sum(l_quantity) AS sum_qty,
               sum(l_extendedprice) AS sum_base_price,
               sum(l_extendedprice * (1 - l_discount)) AS sum_disc_price,
               count(*) AS count_order
          FROM lineitem
         WHERE l_shipdate <= date '1998-09-02'
         GROUP BY l_returnflag, l_linestatus
         ORDER BY l_returnflag, l_linestatus
    """,
    # High-cardinality numeric GROUP BY -- the Phase-3b headline: runs fused via
    # the C++ hash-table path (single int64 group key, no factorize, no dense
    # accumulator gate) instead of falling back to cuDF.
    "Q_high_card_orderkey": """
        SELECT l_orderkey,
               sum(l_quantity) AS sum_qty,
               sum(l_extendedprice) AS sum_base_price,
               count(*) AS count_order
          FROM lineitem
         WHERE l_shipdate <= date '1998-09-02'
         GROUP BY l_orderkey
         ORDER BY l_orderkey
    """,
    # High-cardinality single int64 GROUP BY with MIN/MAX/AVG -- exercises the
    # extended C++ hash_kernel per-slot dispatch (atomic_min/max_d + AVG running
    # sum/hidden count) over the raw int64 key. Previously deferred to cuDF.
    "Q_high_card_minmax": """
        SELECT l_orderkey,
               min(l_quantity) AS min_qty,
               max(l_quantity) AS max_qty,
               avg(l_quantity) AS avg_qty
          FROM lineitem
         WHERE l_shipdate <= date '1998-09-02'
         GROUP BY l_orderkey
         ORDER BY l_orderkey
    """,
    "Q6_filter_agg": """
        SELECT sum(l_extendedprice * l_discount) AS revenue, count(*) AS n
          FROM lineitem
         WHERE l_shipdate >= date '1994-01-01'
           AND l_shipdate < date '1995-01-01'
           AND l_discount BETWEEN 0.05 AND 0.07
           AND l_quantity < 24
    """,
    "Q3_shipping_priority": """
        SELECT l_orderkey, sum(l_extendedprice * (1 - l_discount)) AS revenue,
               o_orderdate, o_shippriority
          FROM customer
          JOIN orders   ON o_custkey = c_custkey
          JOIN lineitem ON l_orderkey = o_orderkey
         WHERE c_mktsegment = 'BUILDING'
           AND o_orderdate < date '1995-03-15'
           AND l_shipdate > date '1995-03-15'
         GROUP BY l_orderkey, o_orderdate, o_shippriority
         ORDER BY revenue DESC, o_orderdate
         LIMIT 10
    """,
    # High-cardinality group-from-join: GROUP BY a high-NDV dimension key reached
    # by the chain (o_orderkey, ~1.5M distinct at SF=10) with SUM/COUNT. Previously
    # deferred to cuDF by the DENSE-only join accumulator (n_groups*nagg >
    # MAX_ACC_CELLS); now hits the fused C++ probe_hash_agg_kernel (the join-path
    # analogue of Q_high_card_orderkey's non-join HASH).
    "Q_join_high_card_orderkey": """
        SELECT o_orderkey,
               sum(l_quantity) AS sum_qty,
               sum(l_extendedprice) AS sum_base_price,
               count(*) AS count_order
          FROM lineitem
          JOIN orders ON l_orderkey = o_orderkey
         GROUP BY o_orderkey
         ORDER BY o_orderkey
    """,
    "scan_agg_full": """
        SELECT count(*) AS n, sum(l_extendedprice) AS s, avg(l_quantity) AS q,
               min(l_discount) AS md, max(l_tax) AS mt
          FROM lineitem WHERE l_quantity > 25
    """,
    "four_table_join_agg": """
        SELECT n_name, sum(l_extendedprice) AS revenue
          FROM lineitem
          JOIN orders   ON l_orderkey = o_orderkey
          JOIN customer ON o_custkey = c_custkey
          JOIN nation   ON c_nationkey = n_nationkey
         GROUP BY n_name
         ORDER BY revenue DESC
         LIMIT 10
    """,
    # Cross-table WHERE folded into the fused join path: a fact predicate
    # (l_quantity < 24 -> the kernel's pass_pred) plus two dim predicates
    # (o_orderdate < date '...' on orders, c_mktsegment = 'BUILDING' on customer ->
    # pre-HT-build dim frame filters). Previously deferred to cuDF by the "no
    # Filter under the Aggregate" restriction on the join path; now hits the
    # fused C++ probe kernel (DENSE, ~25 n_name groups).
    "Q_join_where_filter": """
        SELECT n_name, sum(l_extendedprice) AS revenue, count(*) AS n
          FROM lineitem
          JOIN orders   ON l_orderkey = o_orderkey
          JOIN customer ON o_custkey = c_custkey
          JOIN nation   ON c_nationkey = n_nationkey
         WHERE c_mktsegment = 'BUILDING'
           AND o_orderdate < date '1995-03-15'
           AND l_quantity < 24
         GROUP BY n_name
         ORDER BY revenue DESC
         LIMIT 10
    """,
    # Group-key-in-fact on the fused join path: GROUP BY a FACT-table column
    # (l_orderkey) reached by the chain, with the join kept by a DIM predicate
    # (o_orderdate < date '...'). The kernel reads the group code from a fact
    # column (group_key_col), not the chain tail; the last dim's payload is a
    # zero array (tail=0 -> g = fcode). Previously deferred (the group key was
    # not a reached dim); now hits the fused C++ probe kernel (HASH, ~1.5M
    # distinct orderkeys at SF=10).
    "Q_join_fact_key": """
        SELECT l_orderkey, sum(l_extendedprice) AS revenue, count(*) AS n
          FROM lineitem
          JOIN orders ON l_orderkey = o_orderkey
         WHERE o_orderdate < date '1995-03-15'
         GROUP BY l_orderkey
         ORDER BY revenue DESC
         LIMIT 10
    """,
}

# Q6 uses BETWEEN, which RyuDB doesn't parse yet -> expand it for the GPU run.
RYU_QUERIES: dict[str, str] = dict(QUERIES)
RYU_QUERIES["Q6_filter_agg"] = """
    SELECT sum(l_extendedprice * l_discount) AS revenue, count(*) AS n
      FROM lineitem
     WHERE l_shipdate >= date '1994-01-01'
       AND l_shipdate < date '1995-01-01'
       AND l_discount >= 0.05 AND l_discount <= 0.07
       AND l_quantity < 24
"""


def generate_tpch(scale: float, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "lineitem" / "0.parquet").exists():
        print(f"[gen] {out_dir} already exists, skipping generation")
        return
    con = duckdb.connect()
    try:
        con.execute("INSTALL tpch; LOAD tpch;")
    except Exception as exc:  # noqa: BLE001
        print(f"[gen] could not load DuckDB tpch extension: {exc}", file=sys.stderr)
        raise
    print(f"[gen] generating TPC-H SF={scale} with DuckDB dbgen ...")
    con.execute(f"CALL dbgen(sf => {scale});")
    for t in TABLES:
        tdir = out_dir / t
        tdir.mkdir(exist_ok=True)
        con.execute(f"COPY (SELECT * FROM {t}) TO '{tdir}/0.parquet' (FORMAT PARQUET);")
        print(f"[gen]   wrote {t}")
    con.close()


def make_engine(data_dir: Path) -> Engine:
    cat = Catalog(str(data_dir))
    for t in TABLES:
        cat.register(t, str(data_dir / t))
    return Engine(cat)


def _to_pdf(df):
    import pandas as pd
    if isinstance(df, pd.DataFrame):
        return df
    if hasattr(df, "to_pandas"):
        return df.to_pandas()
    return pd.DataFrame(df)


def frames_match(a, b) -> bool:
    """Tolerance comparison: exact for strings/dates, allclose for numerics.

    TPC-H values are money (2 decimals) summed over many rows; GPU float64 and
    CPU decimal sums differ in low-order bits, so an exact equality check would
    report false mismatches. We use rtol=1e-6 / atol=1e-2 (one cent).
    """
    import numpy as np

    pa, pb = _to_pdf(a), _to_pdf(b)
    if list(pa.columns) != list(pb.columns):
        pa = pa.sort_index(axis=1)
        pb = pb.sort_index(axis=1)
    if len(pa) != len(pb):
        return False
    if len(pa) == 0:
        return True
    cols = list(pa.columns)
    pa = pa.sort_values(cols).reset_index(drop=True)
    pb = pb.sort_values(cols).reset_index(drop=True)
    for c in cols:
        va, vb = pa[c], pb[c]
        # Decimal/object numeric columns -> float for tolerance compare
        try:
            fa = va.astype("float64").to_numpy()
            fb = vb.astype("float64").to_numpy()
            if not np.allclose(fa, fb, rtol=1e-6, atol=1e-2, equal_nan=True):
                return False
        except (ValueError, TypeError):
            if not va.reset_index(drop=True).eq(vb.reset_index(drop=True)).all():
                return False
    return True


def _time(fn, repeats: int) -> float:
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def main() -> int:
    ap = argparse.ArgumentParser(description="RyuDB vs DuckDB on TPC-H")
    ap.add_argument("--scale", type=float, default=1.0, help="TPC-H scale factor")
    ap.add_argument("--repeats", type=int, default=3, help="timed runs per query (min is reported)")
    ap.add_argument("--data", default=None, help="data dir (default: data/tpch-sf{scale})")
    ap.add_argument("--queries", nargs="*", default=None, help="subset of query names to run")
    args = ap.parse_args()

    data_dir = Path(args.data) if args.data else Path("data") / f"tpch-sf{args.scale}"
    generate_tpch(args.scale, data_dir)

    engine = make_engine(data_dir)
    duck = duckdb.connect()
    for t in TABLES:
        duck.execute(f"CREATE VIEW {t} AS SELECT * FROM read_parquet('{data_dir}/{t}/*.parquet')")

    print(f"\nRyuDB (GPU/cuDF) vs DuckDB (CPU) — TPC-H SF={args.scale}, repeats={args.repeats}\n")
    header = f"{'query':<26}{'ryu cold':>10}{'ryu warm':>10}{'duckdb':>10}{'warm x':>9}  check"
    print(header)
    print("-" * len(header))

    names = args.queries or list(QUERIES)
    for name in names:
        duck_sql = QUERIES[name]
        ryu_sql = RYU_QUERIES[name]
        # correctness
        try:
            ryu_df = engine.sql(ryu_sql)
            duck_df = duck.execute(duck_sql).fetchdf()
            ok = frames_match(ryu_df, duck_df)
        except Exception as exc:  # noqa: BLE001
            print(f"{name:<26}{'ERROR':>10}{'':>10}{'':>10}{'':>9}  {exc}")
            continue

        # Warm up both engines (also builds RyuDB's GPU-resident frame + code
        # index so the cold/warm split below is meaningful).
        engine.sql(ryu_sql)
        duck.execute(duck_sql).fetchdf()

        # RyuDB cold: scan cache cleared before each run (forces a Parquet
        # re-read). The per-column factorize *code index* persists by design
        # (it is a dictionary-encoded column, not a query cache), so "cold" here
        # is scan-cold / index-resident -- the realistic "frame evicted from GPU
        # memory but dictionary index kept" state. The very first run on a
        # freshly started engine pays a one-time ~460 ms index build instead.
        def ryu_cold():
            engine.clear_scan_cache()
            engine.sql(ryu_sql)

        # RyuDB warm: frame + code index resident, repeated runs hit cache.
        def ryu_warm():
            engine.sql(ryu_sql)

        ryu_cold_ms = _time(ryu_cold, args.repeats) * 1000
        ryu_warm_ms = _time(ryu_warm, args.repeats) * 1000
        duck_ms = _time(lambda: duck.execute(duck_sql).fetchdf(), args.repeats) * 1000
        speedup = duck_ms / ryu_warm_ms if ryu_warm_ms > 0 else float("inf")
        print(
            f"{name:<26}{ryu_cold_ms:>10.1f}{ryu_warm_ms:>10.1f}{duck_ms:>10.1f}"
            f"{speedup:>8.2f}x  {'OK' if ok else 'MISMATCH'}"
        )

    print(
        "\nNotes:"
        "\n  ryu cold = scan cache cleared (Parquet re-read), code index resident;"
        "\n             the first run on a fresh engine also pays a one-time index build."
        "\n  ryu warm = frame + code index GPU-resident (repeated query)."
        "\n  duckdb   = DuckDB warm (repeated query, parquet metadata cached)."
        "\n  warm x   = duckdb / ryu_warm (speedup > 1x means the GPU won warm)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())