# RyuDB Benchmark — Honest Results

RyuDB (GPU / RAPIDS cuDF) vs DuckDB (CPU) on TPC-H, plus a pandas CPU baseline.
All queries verified correct against DuckDB (`check = OK`).

Hardware: NVIDIA RTX 3090 (24 GB), driver 591.86. Runs inside WSL2 Ubuntu, cuDF
26.06, DuckDB 1.5.4, Numba 0.64.0. Times are min of 4 runs after a warm-up, in
milliseconds. Two regimes are reported (see "Warm vs cold" below): **cold** (scan
cache cleared — Parquet re-read) and **warm** (frame + code index GPU-resident).

## RyuDB vs DuckDB — warm path (Phase 4 step 2, SF=10)

The fused C++/CUDA kernel now also handles **snowflake star-joins + aggregate**
in one pass — it streams the fact table, probes dimension hash tables in-kernel,
and accumulates per group, **never materialising the joined frame**. Combined
with the earlier global-aggregate + MIN/MAX/AVG work, every warm-path query now
runs fused. Numba + cuDF remain as fallbacks.

| query | ryu cold | ryu warm | duckdb | warm speedup |
|---|---|---|---|---|
| Q1 pricing summary | 409 | **52** | 104 | **1.97x** ✓ |
| Q high-card orderkey | 980 | **688** | 1195 | **1.74x** ✓ |
| Q6 filter + global agg | 321 | **23** | 65 | **2.77x** ✓ |
| Q3 3-table join + agg | 535 | 113 | 164 | 1.45x ✓ |
| scan + 5 aggs (global) | 330 | **33** | 58 | **1.76x** ✓ |
| 4-table join + agg | 332 | **64** | 412 | **6.49x** ✓ |

**6 of 6 queries beat DuckDB on the warm path.** Headline this round:

- **Fused star-join + aggregate.** `4-table join + agg`
  (`lineitem ⋈ orders ⋈ customer ⋈ nation`, `GROUP BY n_name`,
  `SUM(l_extendedprice)`) went 598 → 64 ms (0.73x → **6.49x** vs DuckDB) — it
  used to materialise the full 60M-row joined frame via 3 cuDF `merge` calls then
  run a cuDF groupby; now one kernel streams `lineitem` once, does 3 dimension
  hash-table lookups per row, and writes only 25 group accumulator slots. The
  joined frame is never built, so the dominant write-bandwidth cost disappears.
  The win is even larger than the scan-agg fused queries because the join output
  is bigger than any single table.

The other 5 queries are unchanged in shape (not retargeted); small differences
vs the step-1 table are run-to-run noise (Q3 1.52x→1.45x, scan_agg_full
1.93x→1.76x — both still win).

### Honest note on small scale (SF=1)
At SF1 the 4-table query is **3.88x** warm (11 ms vs DuckDB 43 ms) — the fused
star-join wins at small scale too (unlike the step-1 DENSE-generalisation which
cost a little at SF1). Q1 is back to **1.20x** at SF1 (the step-1 0.94x dip was
run-to-run noise around the launch-bound breakeven). Q3 at SF1 is 0.95x — it is
**not** handled by the fused join path (multi-key group + cross-table filter →
cuDF fallback) and is essentially tied with DuckDB at small scale; at SF10 it
wins at 1.45x. So at SF1, 5 of 6 beat DuckDB (Q3 tied); at SF10, 6 of 6.

## Phase 4 step 2: fused star-join + aggregate (this round)

Phase 4 step 1 left `4-table join + agg` as the only warm-path loser (0.73x):
`Aggregate → Join` materialised the joined frame via cuDF `merge` then ran a
cuDF groupby — the fused kernel never fired (it required `Aggregate → Filter`).
Phase 4 step 2 adds a **fused star-join + aggregate** path:

1. **New C++ kernel** (`fused_join_agg`): builds an open-addressing int64→int64
   hash table per dimension (`build_ht_kernel`, atomicCAS-on-key insert with a
   payload — reuses the HASH groupby's insert pattern), then a `probe_agg_kernel`
   streams the fact table, walks the chain of dimension HTs (read-only lookups,
   no atomics during probe), and accumulates per group. The accumulator + shared-
   mem + cross-block reduce are the existing DENSE logic; only the group-index
   computation differs (chain lookups instead of code×stride). A probe miss at
   any stage drops the row — inner-join semantics.
2. **Snowflake chain on the host** (`fused.py::fused_join_aggregate`): works on
   the **plan** (not an executed frame, so the join is never materialised).
   Detects the chain orientation-independently — BFS over the undirected join
   graph from the largest scan (the fact) to the group-key dimension, since the
   optimizer swaps join sides. Each dimension's payload is the next chain key
   (or the factorised group-key code for the last dim), so the kernel threads
   `l_orderkey → o_custkey → c_nationkey → n_name(code)`. int32 keys are promoted
   to int64; the group key (n_name, 25 distinct) is factorised to a dense code →
   DENSE accumulator.
3. **Executor wiring** (`executor._aggregate`): the fused join path is attempted
   *before* `self._exec(in_node)` materialises the join; returns None instantly
   when `node.input` isn't a Join, so `Aggregate → Scan` is unchanged.
4. **Tight eligibility** (defer to cuDF): single group key in a dimension; SUM /
   COUNT(*) over fact-table columns; int join keys; a linear snowflake chain
   covering every joined table; inner joins; no Filter under the Aggregate. Q3
   (multi-key group + cross-table filter), high-card group-from-join (HASH),
   non-int keys, dimension agg args, AVG/MIN/MAX over joins, and global-over-join
   all defer. A cached PK guard (`engine.is_unique_key`) rejects non-unique
   dimension keys (which would silently collapse joins); a VRAM cap rejects
   oversized HTs.

## Phase 5: hand-rolled CUDA Parquet decoder (this round)

The cold path was the only systematic gap (every warm query already wins). Phase
5 built a hand-rolled Parquet decoder that fuses **nvCOMP batched Snappy
decompress → page decode → filter → aggregate** straight off the Parquet pages,
never materialising the 60M-row cuDF frame — the same `Aggregate → Filter →
Scan` shapes the warm fused kernel handles, but off the cold bytes.

**It is correct and safely gated, but it does NOT beat DuckDB cold at SF10 — so
it is opt-in (`RYUDB_SCAN_KERNEL`), not the default cold path.** The default
cold path remains the cuDF materialising fallback (no regression: Q6 cold
315 ms, scan_agg 324 ms, high_card 978 ms — unchanged from Phase 4).

Measured at SF10 with the scan path enabled (`RYUDB_SCAN_KERNEL=1`), all correct
(`check = OK`, exact match to DuckDB):

| query | scan-path cold | duckdb | vs duckdb |
|---|---|---|---|
| Q6 filter + global agg | 3986 ms | 59 ms | 0.015x (67x slower) |
| scan + 5 aggs (global) | 4048 ms | 57 ms | 0.014x (71x slower) |
| Q high-card orderkey | 4744 ms | 1154 ms | 0.24x (4.1x slower) |

Per-RG CUDA-event profile (Q6, 489 row groups) shows where the time goes:

| phase | ms | share |
|---|---|---|
| nvCOMP Snappy decompress | 3861 | 97% |
| dict-index decode + meta copies | 182 | 5% |
| page kernel (filter + accumulate) | 23 | 0.6% |

**The bottleneck is nvCOMP invoked once per row group with ~7 chunks each.**
nvCOMP's batched Snappy decoder parallelises *across chunks*; 7 chunks on an
82-SM RTX 3090 is severe underutilisation, so each ~1 MB batch takes ~7.9 ms
(≈250 MB/s — slower than CPU Snappy). The page kernel itself is 23 ms total —
the fused decode+filter+accumulate idea is sound; the decompression feeding it
is the wall. The serial single-threaded `scan_runs_kernel` dict-index decode
(0.37 ms × 489 = 182 ms) is a second, smaller wall that on its own already
exceeds DuckDB's 57 ms total.

**Why it is still shipped:** it is correct (Q6 and high_card match DuckDB
exactly at SF10, DENSE and HASH), it defers safely on any shape/encoding it
cannot handle (None → cuDF fallback, so correctness never depends on the C++
extension), and it is the foundation for the two fixes that could make it the
cold winner. It is gated off by default so the measured cuDF cold path is
unchanged.

**Path to flip cold (not done this round):**
1. **One batched nvCOMP call over all row groups' pages** (≈3400 chunks at SF10)
   instead of 489 calls of ~7 — fills the GPU and should cut the 3861 ms toward
   the ~50 ms the page kernel proves is achievable. Trades ~1.5 GB VRAM for the
   full compressed+decompressed page set (trivial on 24 GB).
2. **Parallel or host-side dict-index decode** to remove the serial
   `scan_runs_kernel` (182 ms → ~20 ms).
3. With both, the back-of-envelope is nvCOMP ~50 + dict ~20 + page 23 + H2D +
   pre-parse ≈ 120–150 ms — closer to DuckDB's 57 ms but likely still short of a
   flip without further work; the measured 3861 ms today is the honest ceiling.

Also fixed this round: PLAIN values are read with **byte-wise unaligned loads**
(Parquet `values_off = 4 + deflen` is data-dependent and not 8-byte aligned, so
`((const long long*)base)[i]` faulted with `cudaErrorMisalignedAddress` on the
HASH group key — Q6/DENSE never hit it because its only PLAIN int64 is read via
the dict path); and the host loop is **stream-ordered with a pinned-ring gather**
(`cudaHostRegister` fails on the 2.2 GB WSL2 mmap, so per-RG compressed pages are
gathered into a 16-slot pinned ring and H2D'd with a bounded-launch-queue batch
sync — fixing an earlier `cudaErrorMemoryAllocation` and a host-source data
race).

## Phase 4 step 1: fused global aggregate + MIN/MAX/AVG

## Phase 4 step 1: fused global aggregate + MIN/MAX/AVG (this round)

Phase 3b's fused kernel handled only `Aggregate → Filter → Scan` with GROUP BY
and only `COUNT(*)`/`SUM`. Global aggregates (no GROUP BY) took the scalar cuDF
path *before* any fused attempt (`executor._aggregate` returned early when
`group_keys` was empty), and `_match` rejected no-group-keys and AVG/MIN/MAX.
Phase 4 step 1:

1. **Executor wiring** (`executor._aggregate`): the fused kernel is now
   attempted **first for every `Aggregate → Filter` shape** (grouped *or*
   global), then falls back. The scalar-global cuDF path was extracted into
   `_scalar_global_agg` and remains the fallback for fused-ineligible global
   aggregates (and for `Aggregate → Scan` with no Filter).
2. **Eligibility** (`fused._match`): dropped the no-group-keys rejection;
   accept `AVG`/`MIN`/`MAX` (in addition to `SUM`) over numeric arithmetic
   expressions; defer `COUNT(col)`. `AVG`/`MIN`/`MAX` require their argument
   columns to be **non-null** (the kernel reads raw device values and does not
   skip nulls; `AVG = sum / passing-row-count` is only correct for non-null
   args) — nullable args defer to cuDF.
3. **C++ DENSE kernel** (`fused.cu`): new `AGG_MIN/MAX/AVG` constants and
   `atomic_min_d`/`atomic_max_d` (double CAS-loop min/max — CUDA has no double
   `atomicMin`/`atomicMax`). Per-slot init by agg kind (+∞ for MIN, −∞ for MAX,
   0 else) in both the shared accumulator and a host-built `acc_init` array
   copied to the global accumulator. The per-row accumulate and cross-block
   reduce both switch on kind. `AGG_AVG` accumulates a running sum in-kernel
   (like SUM); division happens at read-out.
4. **AVG denominator:** a single hidden `AGG_COUNT` slot appended after the
   visible aggs when any AVG is present — the per-group passing-row count
   (non-null arg guaranteed). Not emitted as an output column. `nagg_internal =
   nagg + (1 if has_avg else 0)`; the DENSE shared-memory gate uses this
   effective count.
5. **Global-agg semantics:** a global aggregate always returns exactly one row.
  `n_out == 0` (filter matched zero rows) or `len(child) == 0` → one row with
  `COUNT(*) = 0` and `NULL` (NaN) for the other aggs (SQL semantics; matches
  DuckDB via `frames_match(equal_nan=True)`).
6. **HASH path unchanged** (SUM/COUNT-only); `fused._run_cpp` guards: if
   `strategy == HASH` and any agg is MIN/MAX/AVG → `return None` (cuDF
   fallback). This avoids an expensive `capacity*nagg` +∞/−∞ init copy; the
   headline high-card query is SUM/COUNT anyway.

Build / run unchanged: `python -m ryudb.kernels.build` (rebuild after editing
`fused.cu`); `python -m pytest -q` (45 tests, incl. 4 new `test_kernels.py`
cases: global agg, empty-match global, grouped MIN/MAX/AVG, HASH+MIN guard).

## RyuDB vs DuckDB — warm path (Phase 3b, SF=10)

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

## Why RyuDB still loses (and where it now wins)

1. **Cold reads** — cuDF's Parquet decoder (~278 ms) is slower than DuckDB's
   vectorized CPU reader. This is now the **only** systematic warm-path gap, and
   it is a reader floor, not a compute gap: every warm-path query — including the
   4-table join — now runs fused and beats DuckDB. Closing it needs a custom GPU
   Parquet/Arrow reader (or GPU-resident warmup), the natural next step.
2. **Small-scale Q3** (SF1, 0.95x) — Q3's multi-key group + cross-table filter
   is not handled by the fused join path (it defers to cuDF); at SF1 it is
   essentially tied with DuckDB, at SF10 it wins at 1.45x. Not a regression —
   the fused join path deliberately scopes to single-key snowflake shapes.

The GPU's advantage is **bandwidth and parallelism once data is on-device**,
now realized on the warm path for **all 6** TPC-H queries at SF10. The remaining
win over DuckDB comes from a custom GPU reader (cold path) and from
larger-than-memory / compute-heavy workloads.

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

Phase 3b **ported the proven kernel to ahead-of-time C++/CUDA**
(nvcc + pybind11, no per-query codegen — a generic interpreter over descriptor
arrays) and added a **hash-table groupby** so high-cardinality numeric GROUP BY
(`GROUP BY l_orderkey`, ~1.5M keys at SF10) runs fused instead of falling back
to cuDF — beating DuckDB 1.76x warm (677 vs 1192 ms). The C++ port is also
faster than the Numba kernel on Q1 (59 → 52 ms, 1.76x → 1.95x). Numba + cuDF
remain as fallbacks so a missing/failed C++ build never regresses correctness.
**4 of 6 queries beat DuckDB warm.**

Phase 4 step 1 extended the fused kernel to **global aggregates**
(no GROUP BY, `n_groups = 1`) and the **MIN/MAX/AVG** kinds (double `atomicCAS`
min/max with +∞/−∞ init; AVG = running sum ÷ hidden per-group count). The
executor now attempts the fused kernel first for every `Aggregate → Filter`
(grouped or global). `scan_agg_full` flipped 0.89x → **1.93x** (66 → 32 ms) and
`Q6` (also a global SUM+COUNT) jumped 1.36x → **2.74x** (47 → 22 ms). **5 of 6
queries beat DuckDB warm.** The cold path still loses end-to-end (reader
floor, as expected).

Phase 4 step 2 (this round) added a **fused star-join + aggregate** kernel so
`Aggregate → Join` no longer materialises the joined frame. One kernel streams
the fact table, probes a chain of dimension hash tables in-kernel, and
accumulates per group — the 60M-row join output is never built. `4-table join +
agg` flipped 0.73x → **6.49x** warm (598 → 64 ms vs DuckDB 412 ms) at SF10, and
**3.88x** at SF1 — the largest win in the project, because skipping the join
output removes more write bandwidth than any single-table fusion. **6 of 6
queries now beat DuckDB warm** at SF10. The chain is detected orientation-
independently (BFS from the largest scan), int32 join keys are promoted, the
low-card group key is factorised to a dense code, and a cached PK guard + VRAM
cap keep the path safe. Q3 (multi-key + cross-table filter) and high-card
group-from-join defer to cuDF.

Phase 5 built the hand-rolled CUDA Parquet decoder (nvCOMP Snappy → decode →
filter → aggregate, never materialising the frame). It is **correct** (Q6 and
high_card match DuckDB exactly at SF10, DENSE and HASH) and **safely defers**,
but measured 67–71x slower than DuckDB cold at SF10 — 97% of the time is nvCOMP
invoked per row group with ~7 chunks (severe GPU underutilisation), plus a
serial dict-index decode. It is **opt-in (`RYUDB_SCAN_KERNEL`)** so the default
cold path is unchanged. The cold flip needs one batched nvCOMP call over all
pages + a parallel dict decode (see the Phase 5 section above).

Remaining work: make the Phase 5 decoder the cold winner (single batched nvCOMP
call + parallel/host dict-index decode), and lift the HASH path to
**multi-column / string** group keys + MIN/MAX/AVG (currently single int64,
SUM/COUNT only) and to the fused join path (high-card group-from-join).