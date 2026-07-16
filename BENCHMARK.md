# RyuDB Phase-1 Benchmark — Honest Results

RyuDB (GPU / RAPIDS cuDF) vs DuckDB (CPU) on TPC-H, plus a pandas CPU baseline.
All queries verified correct against DuckDB (`check = OK`).

Hardware: NVIDIA RTX 3090 (24 GB), driver 591.86. Runs inside WSL2 Ubuntu, cuDF
26.06, DuckDB 1.5.4. Times are min of 3 runs after a warm-up, in milliseconds,
and **include Parquet read + execution** (end-to-end, results materialized).

## RyuDB vs DuckDB (end-to-end)

| query | SF=0.1 ryu / duck | SF=1 ryu / duck | SF=10 ryu / duck |
|---|---|---|---|
| Q1 pricing summary | 64 / 5 (0.08x) | 117 / 14 (0.08x) | 969 / 98 (0.10x) |
| Q6 filter + agg | 28 / 3 (0.12x) | 60 / 8 (0.14x) | 354 / 58 (0.16x) |
| Q3 3-table join + agg | 58 / 10 (0.17x) | 110 / 23 (0.21x) | 515 / 150 (0.29x) |
| scan + 5 aggs | 33 / 3 (0.09x) | 63 / 8 (0.13x) | 370 / 56 (0.15x) |
| 4-table join + agg | 72 / 14 (0.20x) | 161 / 42 (0.26x) | 869 / 396 (0.46x) |

**RyuDB does not beat DuckDB at SF=1 or SF=10 on these queries.** Speedup < 1x
means the GPU lost. The gap narrows with scale and with join complexity (the
4-table join reaches 0.46x at SF=10), but DuckDB stays ahead end-to-end.

## RyuDB GPU vs pandas CPU (Q1, SF=1)

| engine | Q1 time | speedup |
|---|---|---|
| RyuDB (cuDF, GPU) | ~117 ms | 1x |
| pandas (CPU) | ~3100 ms | **~27x slower** |

The GPU execution layer delivers a large win over a naive CPU dataframe engine.
DuckDB is simply a much stronger CPU baseline than pandas.

## Where the time goes — Q1 at SF=10 (60M rows, ~7 GB)

| stage | time |
|---|---|
| `cudf.read_parquet` (6 cols, raw) | ~271 ms |
| decimal→float coercion | ~50 ms |
| filter (`l_shipdate <= date`, gather ~59M rows) | ~100 ms |
| groupby + 4 aggregates | ~314 ms |
| **total** | **~969 ms** |

For comparison, **DuckDB's entire Q1 at SF=10 is ~98 ms** — faster than RyuDB's
Parquet read alone (271 ms). That is the crux of the gap.

## Why RyuDB loses to DuckDB (and where the GPU *would* win)

1. **Parquet reader.** cuDF's GPU Parquet decoder (~271 ms for 6 cols / 60M rows)
   is slower than DuckDB's world-class vectorized CPU reader. DuckDB is the
   ClickBench champion largely because of this reader. The read dominates
   scan-heavy queries (Q1, Q6) and is not amortized.
2. **Synchronous Python orchestration.** The executor issues many small cuDF
   ops (read, mask, gather, groupby, per-aggregate, concat, sort) with a Python
   round-trip + kernel launch + sync between each. At SF≤10 the per-op GPU
   kernels are short, so launch/Python overhead is a meaningful fraction.
3. **DECIMAL coercion.** TPC-H numeric columns are DECIMAL; cuDF decimal
   reductions are unsupported, so we cast to float64 on scan (~50 ms here).

The GPU's advantage is **bandwidth and parallelism once data is on-device**, not
Parquet decode. It wins clearly over pandas (~27x) and would be expected to win
over DuckDB when:

- the dataset is **larger than RAM** and compute is amortized over streaming
  reads (cuDF/cudf-polars streaming, multi-GPU);
- workloads are **compute-heavy** (very large joins, wide aggregations, iterative
  pipelines) rather than scan-dominated;
- a **custom GPU storage/reader** (Phase 3) avoids the generic Parquet path, or
  data is already resident on the GPU (e.g., serving repeated queries from GPU
  memory).

## How to run

```bash
python bench/run_bench.py --scale 1     # ~1 GB, quick
python bench/run_bench.py --scale 10    # ~7 GB, the table above
```

## Honest summary

Phase 1 succeeded at its primary goal: a **correct, working GPU RDBMS** (SQL
subset, optimizer, cuDF execution) with a **benchmark harness** and **tests
passing against DuckDB**. It did **not** meet the stretch exit criterion of
beating DuckDB end-to-end at SF≤10 — an honest, anticipated finding (the plan
called this out: "small/point queries will likely lose to DuckDB… document, not
hide"). Closing the gap is Phase-3 work: a custom GPU storage engine/reader,
streaming for larger-than-memory data, and C++/CUDA hot paths to remove Python
orchestration overhead.