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

**It is correct, and after step 2 it BEATS the cuDF fallback cold at SF10 on
every fused-eligible query — but it is still opt-in (`RYUDB_SCAN_KERNEL`), not
yet the default cold path** (dropping the gate is a separate decision; see
below). The default cold path remains the cuDF materialising fallback (no
regression: Q6 cold 315 ms, scan_agg 331 ms, high_card 1004 ms).

Measured at SF10 with the scan path enabled (`RYUDB_SCAN_KERNEL=1`), all correct
(`check = OK`, exact match to DuckDB). **Step 1b collapsed the 489 per-RG
nvCOMP calls into one batched call over all ~3400 pages; step 2 collapsed the
~1467 serial dict-decode launches into two batched launches:**

| query | scan-path cold (Phase 5) | after step 1b | after step 2 | cuDF fallback | duckdb |
|---|---|---|---|---|---|
| Q6 filter + global agg | 3986 ms | 446 ms | **228 ms** | 315 ms | 58 ms |
| scan + 5 aggs (global) | 4048 ms | 489 ms | **266 ms** | 331 ms | 55 ms |
| Q high-card orderkey | 4744 ms | 1025 ms | **874 ms** | 1004 ms | 1226 ms |

Step 2 makes the scan path the cold winner over the cuDF fallback on all three
(~1.4×, ~1.25×, ~1.15×), and high-card still beats DuckDB cold (874 vs 1226 ms).
End-to-end from Phase 5: Q6 cold 3986 → 228 ms (~17.5×).

Profile (Q6, 489 row groups, scan-cache-cleared / OS-cache-warm — the bench's
"cold"):

| phase | Phase 5 (ms) | after step 1b | after step 2 |
|---|---|---|---|
| pre-parse page headers (host) | — | 5 | 5.8 |
| H2D all compressed + scratch alloc | (per-RG) | 118 | 155 |
| **nvCOMP Snappy decompress** | **3861** | 13 | 16 |
| **dict-index decode** | (in 205) | (in 293) | **5.2** |
| page kernel (per-RG loop) | 23 | (in 293) | 28.6 |
| read-out | <1 | <1 | 0.1 |

Step 2: the serial per-RG `scan_runs_kernel<<<1,1>>>` (~1467 single-thread
launches for Q6) + `apply_runs` is replaced by **two batched launches** —
`scan_runs_batched` (one block per dict page) + `apply_runs_batched` (2D grid,
`blockIdx.y`=page slot) — each dict page getting a permanent `d_idxbig` index
slot and `d_runs` run-table slot. dict decode ~293 ms → 5.2 ms (~56×); the
per-RG loop is now page-kernel-only (~29 ms). The remaining cold time is
dominated by **H2D + scratch alloc (~155 ms)** — the 490 MB compressed H2D plus
allocating the ~1.7 GB per-page dict scratch (`d_idxbig` ~721 MB, `d_runs`
~960 MB).

**Why it is still opt-in:** it is correct (Q6 and high_card match DuckDB exactly
at SF10, DENSE and HASH) and defers safely on any shape/encoding it cannot handle
(None → cuDF fallback, so correctness never depends on the C++ extension). It
now beats the cuDF fallback cold, so dropping `RYUDB_SCAN_KERNEL` to make it the
default cold path is the natural next step — held as a separate decision because
enabling a more complex C++ path by default carries the sticky-CUDA-context risk
(a fault could poison the cuDF fallback); the clean None-deferral on unsupported
shapes and the `d_overflow` flag on decode errors mitigate that, but it wants an
explicit sign-off and a full-suite cold run first.

**Path to flip cold (steps 1–2 done; remaining):**
1. ✅ **One batched nvCOMP call over all row groups' pages** — done (step 1b).
   nvCOMP 3861 ms → 13 ms.
2. ✅ **Batched dict-index decode** — done (step 2). dict decode ~293 ms → 5 ms
   via two batched launches with per-page slots. Trades ~1.7 GB VRAM (`d_idxbig`
   + `d_runs`) — fits 24 GB.
3. **Drop the `RYUDB_SCAN_KERNEL` gate** to make the scan path the default cold
   path (now that it beats cuDF cold), after a full-suite cold regression run.
4. **H2D/scratch-alloc ~155 ms** is then the wall — overlap the 490 MB
   compressed H2D with nvCOMP on a second stream, or a CUDA-direct (`cuFile`)
   read, and defer the 1.7 GB dict-scratch allocation to need. Target: Q6 cold
   ~100 ms, approaching DuckDB's 58 ms.

### Phase 5 step 3: drop the gate + populate the scan cache (this round)

Step 3 drops `RYUDB_SCAN_KERNEL` and makes the scan path the **default cold path**
(taken on a `_scan_cache` miss). The blocker: the scan path's cold win came from
*not* materialising the 60M-row frame, so it never populated `_scan_cache` and
warm repeats re-read Parquet — Q6 warm **22.9 → 224.9 ms** (the warm regression
that kept the gate on). Step 3 fixes it with a **`materialise_kernel`** that
gathers the scan path's already-decoded GPU buffers (PLAIN loads + DICT gathers
via `page_col_raw64`, no double-fold) into Python-owned cuDF frame columns, one
typed Series per bound column (int32/int64/float64/datetime64[s], store width
single-sourced through a passed `d_out_kind`). On success the frame is cached
under the *same* key `_scan` uses, so warm repeats hit the GPU-resident frame.

Measured at SF10 (min of 3, scan-cache-cleared = cold, frame-resident = warm):

| query | ryu cold | ryu warm | duckdb | warm x |
|---|---|---|---|---|
| Q6 filter + global agg | 318.7 | **23.8** | 62.3 | **2.62x** ✓ |
| scan + 5 aggs (global) | 361.6 | **34.6** | 60.1 | **1.74x** ✓ |
| Q high-card orderkey | 952.4 | **685.3** | 1375.0 | **2.01x** ✓ |

**Warm is preserved — the regression is gone** (23.8 / 34.6 / 685.3 ms, all three
still beat DuckDB warm; 6 of 6 overall). The gate is dropped; the scan path is
the default cold path; 62 tests pass including a new `test_scan_cache_populate.py`.

**Honest cold tradeoff:** populating the cache inherently requires reading every
bound column once, so the materialise-gather is a **full re-gather (~70–95 ms:
~50–80 ms in the kernel + ~15 ms cuDF Series build + ~8 ms alloc)**, not the
~30 ms estimated. Cold is now scan-path + one-full-gather ≈ the cuDF fallback:
Q6 318.7 (< 315–320 cuDF, roughly tied), high_card 952 (< 1004 cuDF, still wins),
but scan_agg 361.6 > 331 cuDF — the scan path's pre-step-3 cold win (228/266/874)
is **eroded by the synchronous materialise**. You cannot have scan-path cold speed
*and* cache population for free; the materialise costs ~a full scan. Recovering
the cold win needs **async/background materialise** (launch the gather on a side
stream, return the aggregate immediately, have the warm path wait on a recorded
event) — left as the follow-up. Correctness never depends on the C++ extension
(`None` deferral + `overflow` guard + cache populated only on `overflow==0`).

A subtle cuDF-26 idiom note for the frame allocation: `cudf.core.column.column_empty`
returns an **all-null** column (`null_count == N`) that masks the kernel's writes,
and `as_column(numba.device_array)` is **non-nullable** but `as_column` *copies*
(async, default stream) for some dtypes — that copy once raced the kernel's writes
on the custom stream and clobbered them with stale recycled-buffer data (scattered
NaN, only under prior GPU load). Fix: build the cuDF Series from the numba buffers
**after** the kernel call (post-sync), so any copy reads already-correct data.

### Phase 5 step 4: async/background materialise (this round)

Step 3's synchronous materialise blocks the cold return (the gather runs on the
scan stream before the host returns the aggregate), eroding the pre-step-3 cold
win. Step 4 moves the gather to a **non-blocking side CUDA stream** so the cold
query returns the aggregate *before* the gather finishes; the first warm read
waits on a recorded event (`E_mat`) only if the gather hasn't completed.

Design (see `ryudb/kernels/fused.cu` + `ryudb/exec/fused.py` + `executor.py`):
the per-RG scan loop is unchanged (page kernels on `stream`, periodic 16-RG
syncs). After the loop, stable per-RG pointer arrays (`d_buf_all`/`d_dict_all`,
one H2D each) are uploaded on `stream`; `E_page` captures all scan writes; the
`materialise_kernel` launches per-RG on `stream2` (non-blocking, so the
default-stream read-out doesn't wait for it), then `E_mat` is recorded. The cold
return syncs **only** `stream` (and the default stream for `compact_kernel`) —
not `stream2` — so the host returns while the gather runs. Every device
allocation the gather reads (`ubig`, `d_idxbig`, `d_buf_all`, `d_dict_all`,
`d_row_off`, `d_frame_ptrs`, `d_out_kind`, `d_kind/d_phys/d_scale`) is moved into
a `PendingMatCtx` registry keyed by a returned `pending_id`; `fused_scan_finalize
(id)` is the **sole freer** (called on the first warm read or on
`clear_scan_cache`). The Python side stores a `_PendingFrame` in `_scan_cache`
and lazily builds the cuDF Series in `.get()` **after** the `E_mat` sync —
preserving the step-3 `as_column` race fix. Kill switch
`RYUDB_ASYNC_MATERIALISE=0` restores the synchronous path (`pending_id==0`).

Measured at SF10 (min of 3, async path = default):

| query | ryu cold (async) | ryu cold (sync kill-switch) | ryu warm | duckdb | warm x |
|---|---|---|---|---|---|
| Q6 filter + global agg | 298.8 | 307.9 | **23.2** | 58.9 | **2.54x** ✓ |
| scan + 5 aggs (global) | 349.9 | 361.4 | **34.1** | 58.6 | **1.72x** ✓ |
| Q high-card orderkey | 924.8 | 960.9 | **687.6** | 1204.3 | **1.75x** ✓ |

**Warm is fully preserved** (23.2 / 34.1 / 687.6 vs step-3's 23.8 / 34.6 / 685.3
— noise; all three still beat DuckDB warm). Cold drops vs the sync kill-switch
on every query (-9 / -12 / -36 ms), confirming the gather is genuinely off the
scan stream and the cold return is no longer blocked by it. 64 tests pass (incl.
4 in `test_scan_cache_populate.py`: cold stores a `_PendingFrame`, warm `.get()`
matches `storage.scan` + DuckDB, `clear_scan_cache` finalizes pending, kill
switch → `pending_id==0` + ready DataFrame), 3× clean.

**Honest result — the cold recovery is partial, not the full pre-step-3 win.**
The plan expected cold back to ~230 / ~270 / ~880; measured 299 / 350 / 925.
Empirically the synchronous gather only costs ~9–36 ms of cold (the sync-vs-async
delta), far less than the ~70–95 ms estimated — most of the cold time is the
scan + nvCOMP decode + accumulate itself, not the gather. So hiding the gather
behind the inter-query gap recovers a modest 3–4 %. The win is larger in a real
interactive workload (the gather hides behind user think time, which the tight
bench loop doesn't model) and for the biggest gather (`high_card`, -36 ms). A
bigger cold win would need overlapping the gather with the scan itself (per-RG
side-stream launches with per-RG events during the loop) — a riskier change left
as the next follow-up. The `cudaMalloc(d_buf_all/d_dict_all)` on the post-loop
host critical path is a smaller residual; pre-allocating those before the loop is
a low-risk future tweak. Correctness never depends on the async path: a failed
`cudaStreamCreate`/`cudaEventCreate` falls back to the sync materialise; a
`_PendingFrame.get()` failure drops the cache and falls through to `storage.scan`.

Also fixed this round: PLAIN values are read with **byte-wise unaligned loads**
(Parquet `values_off = 4 + deflen` is data-dependent and not 8-byte aligned, so
`((const long long*)base)[i]` faulted with `cudaErrorMisalignedAddress` on the
HASH group key — Q6/DENSE never hit it because its only PLAIN int64 is read via
the dict path); and the host loop is **stream-ordered with a pinned-ring gather**
(`cudaHostRegister` fails on the 2.2 GB WSL2 mmap, so per-RG compressed pages are
gathered into a 16-slot pinned ring and H2D'd with a bounded-launch-queue batch
sync — fixing an earlier `cudaErrorMemoryAllocation` and a host-source data
race).

### Phase 5 step 5: attack the cold wall — meta-cache + async-pooled alloc (this round)

Step 4 left cold at 299 / 350 / 925 ms. A `RYUDB_SCAN_PROFILE` run (Q6 SF10,
steady-state, `min` over 3 cold runs in one process — the reported cold is run
2/3, after the OS page cache is warm) showed where cold time actually goes:
preparse+plan ~44 ms, scratch-alloc ~70 ms, h2d-comp ~86 ms (490 MB H2D at the
~6.6 GB/s WSL2 PCIe ceiling), nvcomp ~13, stream-loop+sync ~27. Two levers
attack the CPU/alloc portion (the H2D bytes are PCIe-capped — a microbench
confirmed pinned AND pageable H2D both top out at ~6.6 GB/s on WSL2's
paravirtualized GPU, so a chunked-`cudaHostRegister` direct-pin lever was
**skipped** as not worth the ~11 ms host-memcpy hop it would save).

**Lever 2 — `cudaMallocAsync` pooling (big scratch → ~0 on repeat cold).** Every
big scratch buffer (`dbig` ~490 MB, `ubig` ~635 MB, `d_idxbig` ~721 MB,
`d_runs` ~960 MB, the nvCOMP metadata arrays, `d_temp`, the materialise arrays)
is allocated with `cudaMallocAsync(ptr, sz, stream)` via a one-time-probed
`dev_alloc_async` helper (falls back to `cudaMalloc` if unsupported). Load-bearing
CUDA semantics: `cudaFree` on async memory is valid but **RELEASES** the block
from the pool (no reuse); only `cudaFreeAsync(ptr, stream)` **returns** it for
reuse. So every converted alloc's free became `cudaFreeAsync(ptr, 0)` (the
`stream` is destroyed before the cleanup/finalize frees, so the default stream
is used; the blocks are idle by then — the cleanup runs after the final stream
sync, and `fused_scan_finalize` runs after `cudaEventSynchronize(E_mat)`).
Second load-bearing detail: the default mempool's `ReleaseThreshold` is 0, so
freed blocks are handed back to the OS immediately → **no reuse**; the helper
raises it to ~256 GB once so freed blocks stay in the pool. Result: `scratch-
alloc` 70 → ~1 ms on run 3 (pool reuse).

**Lever 1 — page-metadata cache (preparse → ~0 on repeat cold) + async pre-
preparse setup.** The cold scan's host-side preparse walks every (col, row-group)
chunk's Thrift page headers (~3400 pages for Q6 SF10) to recover page offsets
Parquet file metadata does not store. The result depends only on the file bytes
+ the bound columns' chunk descriptors — never on the mmap address (offsets are
byte offsets) or the predicate/aggregate plan — so it is memoized in a
file-scope `g_meta_cache` (`unordered_map<MetaKey, ParsedMeta>`, LRU-capped at
16 paths, no lock — single Python thread under the GIL, same convention as the
`g_pending` registry) keyed by `{path, file_len, mtime_ns (ext4 nanosecond),
ncol, nrg, chunk_off[ncol*nrg], chunk_total[ncol*nrg], chunk_nvals[nrg],
col_kind[ncol]}`. The preparse body moved into a file-scope `run_preparse`; on a
hit the ~44 ms parse is skipped (`preparse+plan` 44 → 0.2 ms). Staleness (same-
size + same-mtime rewrite) would feed wrong offsets → nvCOMP/gather fault →
`overflow=1` → cuDF fallback (the standing "correctness never depends on the
extension" rule); nanosecond mtime makes it near-zero. A parse/layout error is
NOT cached.

Profiling then showed the `preparse+plan` mark's bulk was NOT the Thrift parse
(~7 ms) but ~16 **synchronous** `cudaMemcpy` round-trips in the plan-descriptor
uploads (`np_dev`, each a device sync ≈ 2 ms on WSL2/GPU-PV). So the pre-preparse
setup — the 16 plan descriptors, `d_overflow`, the `acc/seen` (DENSE) /
`key/acc/distinct` (HASH) accumulators, and `d_buf/d_dict` — was made **async on
the default stream**: `cudaMalloc`→`dev_alloc_async(..., 0)`,
`cudaMemcpy`/`cudaMemset`→`...Async(..., 0)`. They queue without per-call sync
round-trips and complete before any `stream` kernel via null-stream
synchronization (the scan `stream` is a legacy stream). The accumulators are
now pooled too (helps the HASH `high_card` cold). Every device free in
`fused_scan_agg` is now `cudaFreeAsync(..., 0)`; `fused_scan_finalize` returns
the registry scratch to the pool the same way.

Measured at SF10 (min of 3, async materialise + both levers = default):

| query | ryu cold (step 4) | ryu cold (step 5) | ryu warm | duckdb | warm x |
|---|---|---|---|---|---|
| Q6 filter + global agg | 298.8 | **215.3** | **23.6** | 60.8 | **2.57x** ✓ |
| scan + 5 aggs (global) | 349.9 | **269.6** | **36.2** | 60.8 | **1.68x** ✓ |
| Q high-card orderkey | 924.8 | **830.3** | **682.7** | 1245.0 | **1.82x** ✓ |

**Cold drops ~84 ms on every query** (-84 / -80 / -94 vs step 4); **warm is
preserved** (23.6 / 36.2 / 682.7 vs step-4 23.2 / 34.1 / 687.6 — noise; all
three still beat DuckDB warm). The bench's reported cold is the repeat-cold min,
so the meta-cache and pool reuse both fully manifest there (first-cold unchanged
— the parse and first allocs still pay). 64 tests pass 3× clean (the registry-
lifetime / cross-stream race surface is the guard).

**Honest result — the cold win is the CPU/alloc portion, not the H2D bytes.**
Cold is now ~215 / ~270 / ~830; DuckDB warm is still faster on the two global
aggs (60.8 ms) — beating DuckDB cold on the scan path at SF10 is out of reach
while H2D is PCIe-capped at 6.6 GB/s (the 490 MB compressed H2D alone is ~75 ms
of irreducible transfer). The remaining cold is dominated by `h2d-comp` (~80 ms,
the 490 MB H2D + batch sync) and the scan/nvCOMP/accumulate loop (~27 ms) —
bytes-on-the-wire, not CPU/alloc. A further cold win would need reducing H2D
bytes (Parquet page-index skipping / column pruning) or a faster transport, both
out of scope here. Correctness never depends on any of this: `cudaMallocAsync`
unsupported → `cudaMalloc` fallback; meta-cache staleness → `overflow=1` → cuDF
fallback.

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
high_card match DuckDB exactly at SF10, DENSE and HASH) and **safely defers**.
Step 1b collapsed the 489 per-row-group nvCOMP calls into **one batched call over
all ~3400 pages** (nvCOMP 3861 ms → 13 ms); step 2 collapsed the ~1467 serial
dict-decode launches into **two batched launches** with per-page index slots
(dict decode ~293 ms → 5 ms). Scan-path cold Q6 3986 → 228 ms (~17.5×); the scan
path now **beats the cuDF fallback cold on every fused-eligible query** (Q6 228
vs 315, scan_agg 266 vs 331, high-card 874 vs 1004 ms) and high-card beats DuckDB
cold. It is still **opt-in (`RYUDB_SCAN_KERNEL`)** — dropping the gate to make it
the default cold path is the next decision. The remaining cold wall is H2D +
scratch alloc (~155 ms).

Remaining work: drop `RYUDB_SCAN_KERNEL` to make the scan path the default cold
path (after a full-suite cold regression run), then attack the H2D/scratch-alloc
wall (overlap compressed H2D with nvCOMP on a 2nd stream, or `cuFile`). Also lift
the HASH path to **multi-column / string** group keys + MIN/MAX/AVG (currently
single int64, SUM/COUNT only) and to the fused join path (high-card
group-from-join).