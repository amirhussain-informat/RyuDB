# RyuDB Benchmark — Honest Results

RyuDB (GPU / RAPIDS cuDF) vs DuckDB (CPU) on TPC-H, plus a pandas CPU baseline.
All queries verified correct against DuckDB (`check = OK`).

Hardware: NVIDIA RTX 3090 (24 GB), driver 591.86. Runs inside WSL2 Ubuntu, cuDF
26.06, DuckDB 1.5.4, Numba 0.64.0. Times are min of 4 runs after a warm-up, in
milliseconds. Two regimes are reported (see "Warm vs cold" below): **cold** (scan
cache cleared — Parquet re-read) and **warm** (frame + code index GPU-resident).

## RyuDB vs DuckDB — warm path (Phase 3b, SF=10)

The fused kernel is now an **ahead-of-time C++/CUDA extension** (nvcc + pybind11),
with a **hash-table groupby** so high-cardinality numeric GROUP BY keys run in the
fused path instead of falling back to cuDF. Numba + cuDF remain as fallbacks.

| query | ryu cold | ryu warm | duckdb | warm speedup |
|---|---|---|---|---|
| Q1 pricing summary | 414 | **52** | 101 | **1.95x** ✓ |
| Q high-card orderkey (NEW) | 990 | **677** | 1192 | **1.76x** ✓ |
| Q6 filter + agg | 360 | 47 | 64 | 1.36x ✓ |
| Q3 3-table join + agg | 525 | 108 | 161 | 1.49x ✓ |
| scan + 5 aggs | 371 | 66 | 59 | 0.89x |
| 4-table join + agg | 882 | 593 | 411 | 0.69x |

**4 of 6 queries beat DuckDB on the warm path.** Two headlines this round:

- **The C++ port is faster than the Numba kernel it replaces.** Q1 warm went
  59 → 52 ms (1.76x → 1.95x vs DuckDB) — ahead-of-time C++ removes Numba's
  per-launch JIT/dispatch overhead, and the generic interpreter handles Q1's
  string-keyed dense rollup with no per-query codegen.
- **High-cardinality numeric GROUP BY now runs fused.** `GROUP BY l_orderkey`
  (~1.5M distinct keys at SF10) previously fell back to cuDF; the new in-kernel
  open-addressing hash table reads the int64 key directly (no `factorize`, no
  dense-accumulator gate) and beats DuckDB 1.76x warm (677 vs 1192 ms). This is
  the Phase-3b capability the user asked for.

The two queries still losing — `scan_agg_full` (global aggregate, no GROUP BY →
no fused path) and `4-table join + agg` (join-dominated, no fused join path) —
are unchanged from Phase 3a and remain the targets.

## Phase-3b: C++/nvcc fused kernel + hash-table groupby (this round)

Phase 3a's fused kernel was Numba `@cuda.jit` with a dense per-group accumulator
gated by `n_groups * nagg ≤ MAX_ACC_CELLS = 4096`, so high-cardinality GROUP BY
fell back to cuDF. Phase 3b:

1. **Ported the kernel to C++/CUDA** compiled ahead-of-time with **nvcc 13.3 +
   pybind11** (no CUDA-toolkit install — nvcc was already in the `ryudb` conda
   env; a conda-forge host compiler `gxx_linux-64` provides `-ccbin`). The kernel
   is a **generic interpreter**: the Python side (`ryudb/exec/fused.py`) lowers a
   matched plan to small descriptor arrays (column device pointers + dtypes,
   predicate/aggregate token streams) and the C++ interprets them per row — no
   per-query C++ codegen (which would need NVRTC, not nvcc). Build on demand:
   `python -m ryudb.kernels.build` → `ryudb/kernels/fused.so`; if absent, the
   executor falls back to Numba/cuDF, so the package stays importable without
   nvcc and correctness never depends on the extension.
2. **Added a hash-table groupby** (`STRAT_HASH`) for a **single int64 group key**
   — the headline. The hash table *is* the int64 key array, initialised to
   `EMPTY = -1` (via `cudaMemset 0xFF`). Insert/lookup uses `atomicCAS` directly
   on the key slot: the CAS is the publish, so it's lock-free and race-free (an
   earlier `atomicCAS(&occupied,0,1)`-then-publish design raced — probing threads
   read stale keys and created duplicate groups). Numeric group keys (int,
   datetime→int64 seconds) are read in-kernel directly — **no `factorize`, no
   code index, no cardinality gate** — so cold *and* warm high-card numeric
   GROUP BY both run fused. Capacity is sized from the row count (the catalog has
   no NDV): `next_pow2(min(n, 2^25, 2GB // (nagg*8)))`; on overflow the C++ call
   returns a sentinel and the caller falls back to cuDF.
3. **Preserved every fallback.** `_match` is unchanged; the C++ backend runs
   first, then Numba (dense), then the cuDF no-gather path. Multi-column numeric
   GROUP BY, OR predicates, and string high-card keys still defer to cuDF
   (deferred stretch goals). Datetime group keys are normalised to int64 seconds
   on the Python side and take the hash path.

Descriptor codes mirror between `ryudb/exec/fused.py` and `ryudb/kernels/fused.cu`
(dtype / op / token-kind / agg-kind / strategy). One real bug found and fixed
during the port: the postfix operator code was being lowered into the wrong
descriptor array (`tok_col` instead of `tok_op`), so every operator evaluated as
division — `Q1`'s `sum(l_extendedprice * (1 - l_discount))` came out as
`ep / (1/disc)` = wrong by ~10x. Caught by the Q1-vs-DuckDB test.

Build / run:
```bash
conda install -n ryudb -c conda-forge gxx_linux-64 gcc_linux-64 sysroot_linux-64
pip install pybind11
python -m ryudb.kernels.build          # -> ryudb/kernels/fused.so
python -m pytest -q                    # 41 tests, incl. tests/test_kernels.py
```

## RyuDB vs DuckDB — warm path (Phase 3a, SF=10)

| query | ryu cold | ryu warm | duckdb | warm speedup |
|---|---|---|---|---|
| Q1 pricing summary | 416 | **59** | 105 | **1.76x** ✓ |
| Q6 filter + agg | 359 | **45** | 60 | **1.32x** ✓ |
| Q3 3-table join + agg | 518 | **103** | 148 | **1.44x** ✓ |
| scan + 5 aggs | 368 | 65 | 56 | 0.85x |
| 4-table join + agg | 879 | 594 | 407 | 0.69x |

**3 of 5 queries now beat DuckDB on the warm path** (Q1 1.76x, Q6 1.32x, Q3
1.44x). The scan cache benefits every query by skipping the re-read on repeated
queries; the fused CUDA kernel specifically accelerates the Q1-shaped
`Aggregate → Filter → Scan` low-cardinality rollup. The two queries that still
lose — `scan_agg_full` (a *global* aggregate with no GROUP BY, so the fused
kernel doesn't apply) and `4-table join + agg` (join-dominated, no fused path) —
are the remaining targets for the C++ port and join work.

### Warm vs cold (and the one-time index build)

- **warm** — the frame and the per-column factorize *code index* are GPU-resident
  (the realistic "serving repeated queries from GPU memory" case). Q1 warm is
  ~59 ms: the fused kernel (~35 ms) + datetime→int64 prep (~15 ms) + read-out +
  sort, with **no Parquet read, no factorize, no copy**.
- **cold** — the scan cache is cleared before each run (forces a Parquet re-read),
  but the code index is *kept* (it is a dictionary-encoded column, not a query
  cache, so it is valid across re-reads). Q1 cold is ~416 ms: read (~278) +
  decimal coercion (~50) + fused kernel (~35) + read-out/sort. This is better than
  the pre-Phase-3a 720 ms no-gather baseline.
- **first-ever run on a fresh engine** also pays a one-time ~460 ms to build the
  code index (cuDF `factorize` on 60M string rows × 2 group-key columns). After
  that it is reused by every warm and cold run on the same table.

## Phase-3a: fused filter+groupby+aggregate CUDA kernel (this round)

The Phase-3 reframe (below) showed the dominant cost was **GPU compute
orchestration** — many synchronous cuDF ops (filter, groupby, per-aggregate
kernels, concat) each with a kernel launch + Python round-trip + sync — not the
reader. Phase 3a replaces that orchestration for the
`Aggregate → Filter → Scan` shape with **one fused Numba `@cuda.jit` kernel**
that evaluates the predicate, computes every aggregate's argument expression,
and atomically accumulates into per-group slots in a single pass over the device
data. A small per-query code generator specialises the kernel to the query's
predicate and aggregate expressions (Numba can't dispatch on Python AST types
inside a kernel, so a source string is emitted and JIT-compiled).

Component breakdown at SF=10 (60M rows), Q1:

| component | no code cache | with code cache |
|---|---|---|
| factorize 2 string group-key cols (60M rows) | ~480 ms | 0 (cached) |
| copy of child frame | ~85 ms | 0 (fused path is non-mutating) |
| datetime→int64 prep | ~15 ms | ~15 ms |
| fused kernel launch | ~35 ms | ~35 ms |
| **`fused_aggregate` total** | **~541 ms** | **~55 ms** |

The code index turns a 541 ms fused aggregate into 55 ms — a **9.8x speedup** —
and the no-mutation design removes the 85 ms copy. Together they take warm Q1
from ~616 ms (where Phase 3a first landed, before the code cache) to ~59 ms,
beating DuckDB's 105 ms warm.

### The factorize discovery

Profiling the first cut of the fused kernel showed warm Q1 was still ~616 ms and
*not* beating DuckDB, despite the kernel itself being only ~35 ms. The new
dominant cost was cuDF `factorize()` on the two string group-key columns
(~480 ms) — itself a hash-groupby, i.e. exactly the work the kernel was meant to
avoid. The fix was to cache the factorize codes per `(table, col)` as a
persistent dictionary index, so warm repeat queries skip it entirely. This is
the lesson: a fused kernel only wins if *all* of its prep is also fused or
cached; an unfused prep step (factorize) can dwarf the kernel.

### Eligibility and fallback (correctness never compromised)

The fused path is gated by a shape matcher (`ryudb/exec/fused.py`); an
ineligible plan returns `None` and the executor falls back to the existing cuDF
path. Supported (v1, targets Q1):
- `Aggregate` whose input is a `Filter`;
- group keys are `Col`s factorizable to int codes, with
  product-of-distinct-counts × number-of-aggregates ≤ `MAX_ACC_CELLS` (4096; a
  dense per-group accumulator in shared memory — high-cardinality GROUP BY falls
  back);
- aggregates are `COUNT(*)` or `SUM(arithmetic over numeric Col/lit)`;
- predicate is a conjunction of `Col OP literal` comparisons (numeric/datetime).

A high-cardinality GROUP BY (e.g. `GROUP BY l_orderkey`) correctly falls back to
cuDF and matches DuckDB row-for-row (tested).

## RyuDB vs DuckDB — cold end-to-end (Phase-1 / Phase-3 baseline)

For reference, the previous end-to-end numbers (single warm-up then min of 3,
**including the Parquet read**), before the scan cache and fused kernel:

| query | SF=0.1 ryu / duck | SF=1 ryu / duck | SF=10 ryu / duck |
|---|---|---|---|
| Q1 pricing summary | 44 / 5 (0.11x) | 105 / 14 (0.14x) | 720 / 99 (0.14x) |
| Q6 filter + agg | 31 / 3 (0.10x) | 60 / 8 (0.14x) | 355 / 60 (0.17x) |
| Q3 3-table join + agg | 57 / 11 (0.20x) | 106 / 23 (0.22x) | 509 / 148 (0.29x) |
| scan + 5 aggs | 31 / 3 (0.08x) | 64 / 8 (0.13x) | 364 / 56 (0.15x) |
| 4-table join + agg | 66 / 17 (0.26x) | 149 / 42 (0.29x) | 876 / 391 (0.45x) |

The cold path still loses to DuckDB end-to-end (the reader floor), as expected;
Phase 3a's win is on the warm path, where the GPU's bandwidth advantage once data
is resident is realized.

### Prior Phase-3 compute optimizations (still in the fallback path)

Two executor changes that remain the cuDF fallback for ineligible shapes:

1. **Fused aggregation** — all aggregates of a GROUP BY are issued in a single
   `groupby.agg({col: [funcs]})` call instead of one kernel per aggregate plus a
   `concat`. One pass over the columns instead of N.
2. **No-gather filter folding** — when a Filter sits directly below an
   Aggregate and the group keys are non-nullable columns, the predicate is
   folded into the groupby by nulling the group keys of failing rows
   (`groupby dropna=True` drops them) instead of materialising a filtered row
   copy. On Q1 (~98% of rows pass) this avoids copying ~59M of 60M rows.

Result at the time: Q1 improved ~1.35x at SF=10 (969→720 ms).

## RyuDB GPU vs pandas CPU (Q1, SF=1)

| engine | Q1 time | speedup |
|---|---|---|
| RyuDB (cuDF, GPU) | ~117 ms | 1x |
| pandas (CPU) | ~3100 ms | **~27x slower** |

The GPU execution layer delivers a large win over a naive CPU dataframe engine.
DuckDB is simply a much stronger CPU baseline than pandas.

## Where the time goes — Q1 at SF=10 (60M rows, ~7 GB)

| stage | cold | warm |
|---|---|---|
| `cudf.read_parquet` (6 cols, raw) | ~278 ms | 0 (cached frame) |
| decimal→float coercion | ~50 ms | 0 (cached frame) |
| factorize 2 string group keys | 0 (code index resident) | 0 (code index resident) |
| datetime→int64 prep | ~15 ms | ~15 ms |
| fused kernel (filter+groupby+4 aggs) | ~35 ms | ~35 ms |
| **total (excl. one-time index build)** | **~416 ms** | **~59 ms** |

(First-ever run on a fresh engine adds a one-time ~460 ms to build the code
index, which is then reused by every subsequent run on that table.)

### Reframe: the gap was compute orchestration, not (only) the reader

The original hypothesis was that cuDF's Parquet reader was the bottleneck. It is
slow (~278 ms), but a direct measurement disproved "reader-only":

| measurement | time |
|---|---|
| `cudf.read_parquet` 6 cols, SNAPPY (current) | 278 ms |
| `cudf.read_parquet` 6 cols, UNCOMPRESSED | 197 ms |
| **Q1 compute-only, data already on the GPU (pre-Phase-3a cuDF path)** | **534 ms** |
| DuckDB entire Q1 (read + compute) | 98 ms |

With the data already resident on the GPU, Q1's compute alone was **534 ms** —
over 5x DuckDB's *entire* query. The dominant cost was **GPU compute
orchestration**: many synchronous cuDF ops (filter, groupby, per-aggregate
kernels, arithmetic materialisation, concat) each with a kernel launch + Python
round-trip + sync. Phase 3a's fused kernel collapses that into a single ~35 ms
launch, and the code index + cache remove the read and factorize — which is why
warm Q1 now beats DuckDB.

Reader experiments that did *not* pan out:
- **Row-group stats pruning** is useless on unclustered TPC-H data: 0/489 row
  groups prunable for Q1 (every group's `l_shipdate` range overlaps the
  predicate), and passing filters into `read_parquet(filters=...)` made reads
  *slower* (extra metadata work, no skipped I/O).
- **Uncompressed storage** saves ~80 ms on read (197 vs 278 ms) — real but
  small, and it bloats disk ~4x. Not pursued.

## Why RyuDB still loses on some queries (and where it now wins)

1. **Global aggregates / non-fused shapes** (`scan_agg_full`, Q6-style with no
   GROUP BY) — the fused kernel requires a GROUP BY, so these take the cuDF
   fallback. The scan cache still removes the re-read (warm 65 ms vs 56 ms
   DuckDB — close, slightly losing to DuckDB's vectorized reductions).
2. **Join-dominated queries** (`4-table join + agg`) — no fused path for joins;
   cuDF's merge orchestration dominates (warm 594 ms vs 407 ms DuckDB). The C++
   port should add a fused join+aggregate path.
3. **Cold reads** — cuDF's Parquet decoder (~278 ms) is slower than DuckDB's
   vectorized CPU reader. Real but secondary now that the warm path wins.

The GPU's advantage is **bandwidth and parallelism once data is on-device**, now
realized on the warm path for Q1/Q6/Q3. The remaining wins over DuckDB come from
extending the fused-kernel approach to joins and global aggregates (C++ port),
and from larger-than-memory / compute-heavy workloads.

## How to run

```bash
python bench/run_bench.py --scale 1     # ~1 GB, quick
python bench/run_bench.py --scale 10    # ~7 GB, the table above
python bench/run_bench.py --scale 10 --queries Q1_pricing_summary --repeats 4
```

## Honest summary

Phase 1 delivered a **correct, working GPU RDBMS** (SQL subset, optimizer, cuDF
execution) with a benchmark harness and tests passing against DuckDB.

Phase 3 (fused aggregation + no-gather filter folding) cut Q1 ~1.35x and
identified **GPU compute orchestration** as the real bottleneck (534 ms
compute-only vs DuckDB's 98 ms total).

Phase 3a replaced that orchestration with a **single fused Numba CUDA kernel**
for the `Aggregate → Filter → Scan` shape, plus a GPU-resident frame cache and a
persistent factorize **code index**. The code index was the key finding: the
fused kernel alone (~35 ms) was dwarfed by cuDF's 480 ms `factorize` prep until
the codes were cached as a dictionary index, turning a 541 ms fused aggregate
into 55 ms. Warm Q1 beat DuckDB 1.76x (59 vs 105 ms), and Q6 (1.32x) and Q3
(1.44x) beat DuckDB warm too via the scan cache.

Phase 3b (this round) **ported the proven kernel to ahead-of-time C++/CUDA**
(nvcc + pybind11, no per-query codegen — a generic interpreter over descriptor
arrays) and added a **hash-table groupby** so high-cardinality numeric GROUP BY
(`GROUP BY l_orderkey`, ~1.5M keys at SF10) runs fused instead of falling back
to cuDF — beating DuckDB 1.76x warm (677 vs 1192 ms). The C++ port is also
faster than the Numba kernel on Q1 (59 → 52 ms, 1.76x → 1.95x). Numba + cuDF
remain as fallbacks so a missing/failed C++ build never regresses correctness.
**4 of 6 queries now beat DuckDB warm.** The cold path still loses end-to-end
(reader floor, as expected).

Remaining work: extend the fused-kernel approach to **joins** (the 4-table
query is still 0.69x) and **global aggregates** (scan_agg_full still 0.89x), and
lift the hash path to **multi-column / string** group keys (currently single
int64 only).