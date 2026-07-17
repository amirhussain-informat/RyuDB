# RyuDB Phase-1 Benchmark — Honest Results

RyuDB (GPU / RAPIDS cuDF) vs DuckDB (CPU) on TPC-H, plus a pandas CPU baseline.
All queries verified correct against DuckDB (`check = OK`).

Hardware: NVIDIA RTX 3090 (24 GB), driver 591.86. Runs inside WSL2 Ubuntu, cuDF
26.06, DuckDB 1.5.4. Times are min of 3 runs after a warm-up, in milliseconds,
and **include Parquet read + execution** (end-to-end, results materialized).

## RyuDB vs DuckDB (end-to-end)

| query | SF=0.1 ryu / duck | SF=1 ryu / duck | SF=10 ryu / duck |
|---|---|---|---|
| Q1 pricing summary | 44 / 5 (0.11x) | 105 / 14 (0.14x) | 720 / 99 (0.14x) |
| Q6 filter + agg | 31 / 3 (0.10x) | 60 / 8 (0.14x) | 355 / 60 (0.17x) |
| Q3 3-table join + agg | 57 / 11 (0.20x) | 106 / 23 (0.22x) | 509 / 148 (0.29x) |
| scan + 5 aggs | 31 / 3 (0.08x) | 64 / 8 (0.13x) | 364 / 56 (0.15x) |
| 4-table join + agg | 66 / 17 (0.26x) | 149 / 42 (0.29x) | 876 / 391 (0.45x) |

**RyuDB does not beat DuckDB at SF=1 or SF=10 on these queries.** Speedup < 1x
means the GPU lost. The gap narrows with scale and with join complexity (the
4-table join reaches 0.45x at SF=10), but DuckDB stays ahead end-to-end.

### Phase-3 compute optimizations (this round)

Two executor changes targeting the *compute* path (see "Reframe" below):

1. **Fused aggregation** — all aggregates of a GROUP BY are issued in a single
   `groupby.agg({col: [funcs]})` call instead of one kernel per aggregate plus a
   `concat`. One pass over the columns instead of N.
2. **No-gather filter folding** — when a Filter sits directly below an
   Aggregate and the group keys are non-nullable columns, the predicate is
   folded into the groupby by nulling the group keys of failing rows
   (`groupby dropna=True` drops them) instead of materialising a filtered row
   copy. On Q1 (~98% of rows pass) this avoids copying ~59M of 60M rows.

Result: **Q1 improves ~1.35x** at SF=10 (969→720 ms), and across all scales
(SF0.1 64→44, SF1 117→105). The other queries are essentially unchanged because
they are either *global* aggregates with no GROUP BY (Q6, scan_agg — a
no-gather variant there was tried and reverted: null-aware reductions on the
masked column cost more than the gather) or *join-then-agg* where the join
dominates (Q3, 4-table).

## RyuDB GPU vs pandas CPU (Q1, SF=1)

| engine | Q1 time | speedup |
|---|---|---|
| RyuDB (cuDF, GPU) | ~117 ms | 1x |
| pandas (CPU) | ~3100 ms | **~27x slower** |

The GPU execution layer delivers a large win over a naive CPU dataframe engine.
DuckDB is simply a much stronger CPU baseline than pandas.

## Where the time goes — Q1 at SF=10 (60M rows, ~7 GB)

After the Phase-3 compute optimizations:

| stage | time |
|---|---|
| `cudf.read_parquet` (6 cols, raw) | ~278 ms |
| decimal→float coercion | ~50 ms |
| no-gather fold (null group keys) + fused groupby + 4 aggs | ~340 ms |
| **total** | **~720 ms** |

Before this round the same query was ~969 ms: a ~100 ms filter gather plus a
~314 ms groupby executed as four separate aggregate kernels plus a concat. The
no-gather fold removed the gather; fusing the four aggregates into one
`groupby.agg` removed the per-aggregate passes.

### Reframe: the gap is compute orchestration, not (only) the reader

The original hypothesis was that cuDF's Parquet reader was the bottleneck. It is
slow (278 ms), but a direct measurement disproved "reader-only":

| measurement | time |
|---|---|
| `cudf.read_parquet` 6 cols, SNAPPY (current) | 278 ms |
| `cudf.read_parquet` 6 cols, UNCOMPRESSED | 197 ms |
| **Q1 compute-only, data already on the GPU (no read)** | **534 ms** |
| DuckDB entire Q1 (read + compute) | 98 ms |

With the data already resident on the GPU, Q1's compute alone is **534 ms** —
over 5x DuckDB's *entire* query. So even eliminating the read entirely does not
close the gap. The dominant cost is **GPU compute orchestration**: many
synchronous cuDF ops (filter, groupby, per-aggregate kernels, arithmetic
materialisation, concat) each with a kernel launch + Python round-trip + sync.
This round's fused-agg + no-gather cuts that 534 ms roughly in half, but
DuckDB's vectorised, fully-fused pipeline still wins by a wide margin at SF≤10.

Reader experiments that did *not* pan out:
- **Row-group stats pruning** is useless on unclustered TPC-H data: 0/489 row
  groups prunable for Q1 (every group's `l_shipdate` range overlaps the
  predicate), and passing filters into `read_parquet(filters=...)` made reads
  *slower* (extra metadata work, no skipped I/O).
- **Uncompressed storage** saves ~80 ms on read (197 vs 278 ms) — real but
  small relative to the ~340 ms compute, and it bloats disk ~4x. Not pursued.

## Why RyuDB loses to DuckDB (and where the GPU *would* win)

1. **GPU compute orchestration.** The executor issues many small cuDF ops with
   a Python round-trip + kernel launch + sync between each. At SF≤10 the per-op
   GPU kernels are short, so launch/Python/async-sync overhead dominates. This
   is the single biggest factor (534 ms compute-only vs 98 ms DuckDB total).
2. **Parquet reader.** cuDF's GPU Parquet decoder (~278 ms for 6 cols / 60M
   rows) is slower than DuckDB's world-class vectorized CPU reader. A real but
   secondary cost (~80 ms recoverable via uncompressed storage).
3. **DECIMAL coercion.** TPC-H numeric columns are DECIMAL; cuDF decimal
   reductions are unsupported, so we cast to float64 on scan (~50 ms here).

The GPU's advantage is **bandwidth and parallelism once data is on-device**, not
Parquet decode or fine-grained orchestration. It wins clearly over pandas
(~27x) and would be expected to win over DuckDB when:

- the dataset is **larger than RAM** and compute is amortized over streaming
  reads (cuDF/cudf-polars streaming, multi-GPU);
- workloads are **compute-heavy** (very large joins, wide aggregations, iterative
  pipelines) rather than scan-dominated;
- a **custom CUDA kernel** fuses filter+groupby+aggregate into one launch
  (Phase 3 C++/CUDA hot path — the real route to beating DuckDB at this scale);
- or data is already resident on the GPU (serving repeated queries from GPU
  memory — eliminates the read, leaving only the ~340 ms compute to optimize).

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
hide").

A Phase-3 compute pass (fused aggregation + no-gather filter folding) cut Q1
~1.35x (969→720 ms at SF=10) and confirmed the real bottleneck is **GPU compute
orchestration**, not the reader: Q1 compute-only with data resident on the GPU
is 534 ms vs DuckDB's 98 ms total. Closing the gap at this scale therefore needs
the Phase-3 C++/CUDA work — a single fused filter+groupby+aggregate kernel — not
further reader tweaks. Reader-side, uncompressed storage recovers ~80 ms but is
secondary. The GPU's win over DuckDB remains at larger-than-memory scale and on
compute-heavy workloads, as documented above.