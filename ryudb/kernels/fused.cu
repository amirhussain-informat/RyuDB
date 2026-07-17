// Fused filter + groupby + aggregate CUDA kernel (C++/nvcc + pybind11).
//
// One pass over device column data evaluates the predicate, computes every
// aggregate's argument expression, and accumulates per group. The Python side
// (ryudb/exec/fused.py) lowers a matched plan to small descriptor arrays and
// interprets them here (no per-query C++ codegen). Datetimes are normalised to
// int64 seconds on the Python side, so this kernel only ever sees int64 /
// float64 device columns. Two strategies:
//   DENSE  -- low-cardinality group keys: dense per-group accumulator (group
//             index from int codes + strides), per-block shared + global atomic.
//   HASH   -- high-cardinality group keys: in-kernel open-addressing hash table
//             (atomicCAS insert/lookup -> group id) + global atomic accumulators.
//             Numeric group keys are read directly (no factorize); string keys
//             arrive as cached int codes (see engine.get_codes).
//
// On hash-table overflow the host returns an overflow sentinel and the caller
// falls back to the cuDF path. If this extension is not built, the executor uses
// the Numba/cuDF paths -- correctness never depends on it.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cuda_runtime.h>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <nvcomp.h>
#include <nvcomp/snappy.h>

#include "pqpages.h"

#ifdef RYUDB_SCAN_PROFILE
#include <chrono>
#include <cstdio>
struct _ScanTimer {
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    const char *last = "start";
    void mark(const char *name) {
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        fprintf(stderr, "[profile] %-22s %8.1f ms\n", name, ms);
        t0 = t1; last = name;
    }
};
#endif

namespace py = pybind11;

// dtype codes
static constexpr int DT_INT64 = 0;
static constexpr int DT_FLOAT64 = 1;
// op codes (predicate)
static constexpr int OP_EQ = 0, OP_NE = 1, OP_LT = 2, OP_LE = 3, OP_GT = 4, OP_GE = 5;
// token kinds
static constexpr int TK_COL = 0, TK_LIT = 1, TK_OP = 2;
// token op codes
static constexpr int TOP_ADD = 1, TOP_SUB = 2, TOP_MUL = 3, TOP_DIV = 4;
// agg kinds
static constexpr int AGG_COUNT = 0, AGG_SUM = 1, AGG_MIN = 2, AGG_MAX = 3, AGG_AVG = 4;
// strategies
static constexpr int STRAT_DENSE = 0, STRAT_HASH = 1;

// Descriptor bundle (device pointers + counts), passed to kernels by value.
struct Plan {
    const void **cols;          // device array of column device pointers
    const int *dtypes;          // device array of DT_* per column
    const int *gkey_idx;        // device: column index per group key
    const long long *gkey_stride;  // device: row-major strides (DENSE)
    int ngkey;

    int n_pred;
    const int *pred_col;        // device
    const int *pred_op;         // device
    const double *pred_lit;     // device

    int nagg;
    const int *agg_kind;        // device
    const int *agg_tok_start;   // device
    const int *agg_tok_len;     // device

    int ntok;
    const int *tok_kind;        // device
    const int *tok_col;         // device (TK_COL -> column index)
    const double *tok_lit;      // device (TK_LIT -> literal value)
    const int *tok_op;          // device (TK_OP -> TOP_*)

    int n;                      // row count
};

__device__ inline double col_val(const Plan &p, int c, int i) {
    if (p.dtypes[c] == DT_INT64) return (double)((const long long *)p.cols[c])[i];
    return ((const double *)p.cols[c])[i];
}

__device__ inline bool pass_pred(const Plan &p, int i) {
    for (int j = 0; j < p.n_pred; j++) {
        double v = col_val(p, p.pred_col[j], i);
        double lit = p.pred_lit[j];
        int op = p.pred_op[j];
        bool ok = op == OP_EQ ? v == lit : op == OP_NE ? v != lit
                       : op == OP_LT  ? v < lit
                       : op == OP_LE  ? v <= lit
                       : op == OP_GT  ? v > lit
                                      : v >= lit;
        if (!ok) return false;
    }
    return true;
}

// Postfix expression evaluation with a tiny stack (expressions are small).
__device__ inline double eval_agg(const Plan &p, int i, int agg) {
    int s = p.agg_tok_start[agg], len = p.agg_tok_len[agg];
    double stack[8];
    int sp = 0;
    for (int t = 0; t < len; t++) {
        int idx = s + t;
        int kind = p.tok_kind[idx];
        if (kind == TK_COL) {
            stack[sp++] = col_val(p, p.tok_col[idx], i);
        } else if (kind == TK_LIT) {
            stack[sp++] = p.tok_lit[idx];
        } else {  // TK_OP
            double b = stack[--sp], a = stack[--sp];
            int op = p.tok_op[idx];
            double r = op == TOP_ADD ? a + b : op == TOP_SUB ? a - b
                            : op == TOP_MUL ? a * b
                                           : a / b;
            stack[sp++] = r;
        }
    }
    return sp > 0 ? stack[0] : 0.0;
}

// ---------------- DENSE ----------------
//
// Per-group accumulator slot init/semantics depend on the agg kind:
//   COUNT/SUM/AVG -> 0.0 (additive; AVG stores a running sum, divided at read-out)
//   MIN -> +inf   (reduced by atomic_min_d)
//   MAX -> -inf   (raised by atomic_max_d)
// `nagg` here is the INTERNAL slot count (visible aggs + one hidden per-group
// passing-row-count slot when any AVG is present, kind=AGG_COUNT). The hidden
// slot is the denominator for AVG and is not emitted as an output column.
static __device__ inline double init_for_kind(int kind) {
    if (kind == AGG_MIN) return __longlong_as_double(0x7ff0000000000000ULL);   // +inf
    if (kind == AGG_MAX) return __longlong_as_double(0xfff0000000000000ULL);   // -inf
    return 0.0;
}

// CUDA has no double atomicMin/atomicMax; use a compare-and-swap loop on the
// 64-bit bit pattern. Works on both global and shared memory.
static __device__ inline double atomic_min_d(double *addr, double val) {
    unsigned long long *a = (unsigned long long *)addr;
    unsigned long long old = *a, assumed;
    do {
        assumed = old;
        double cur = __longlong_as_double(assumed);
        double nv = (val < cur) ? val : cur;
        old = atomicCAS(a, assumed, __double_as_longlong(nv));
    } while (assumed != old);
    return __longlong_as_double(old);
}

static __device__ inline double atomic_max_d(double *addr, double val) {
    unsigned long long *a = (unsigned long long *)addr;
    unsigned long long old = *a, assumed;
    do {
        assumed = old;
        double cur = __longlong_as_double(assumed);
        double nv = (val > cur) ? val : cur;
        old = atomicCAS(a, assumed, __double_as_longlong(nv));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__global__ void dense_kernel(Plan p, double *acc, int *seen, int n_groups, int nagg) {
    extern __shared__ double sh[];
    int t = threadIdx.x;
    int nga = n_groups * nagg;
    // Per-slot init by agg kind (slot a = k % nagg).
    for (int k = t; k < nga; k += blockDim.x) sh[k] = init_for_kind(p.agg_kind[k % nagg]);
    __syncthreads();
    for (int i = blockIdx.x * blockDim.x + t; i < p.n; i += gridDim.x * blockDim.x) {
        if (!pass_pred(p, i)) continue;
        long long g = 0;
        for (int j = 0; j < p.ngkey; j++)
            g += ((const long long *)p.cols[p.gkey_idx[j]])[i] * p.gkey_stride[j];
        if (g < 0 || g >= n_groups) continue;
        for (int a = 0; a < p.nagg; a++) {
            int kind = p.agg_kind[a];
            double *slot = &sh[g * nagg + a];
            if (kind == AGG_MIN) {
                atomic_min_d(slot, eval_agg(p, i, a));
            } else if (kind == AGG_MAX) {
                atomic_max_d(slot, eval_agg(p, i, a));
            } else {
                double val = kind == AGG_COUNT ? 1.0 : eval_agg(p, i, a);
                atomicAdd(slot, val);  // SUM, AVG (running sum), hidden COUNT
            }
        }
        atomicMax(&seen[g], 1);
    }
    __syncthreads();
    // Cross-block reduce: MIN/MAX reduce by min/max, the rest by add.
    for (int k = t; k < nga; k += blockDim.x) {
        int kind = p.agg_kind[k % nagg];
        if (kind == AGG_MIN) atomic_min_d(&acc[k], sh[k]);
        else if (kind == AGG_MAX) atomic_max_d(&acc[k], sh[k]);
        else atomicAdd(&acc[k], sh[k]);
    }
}

// ---------------- HASH (single int64 group key, lock-free) ----------------
//
// The group key column is a single int64 array (a real numeric key like
// l_orderkey read directly, or cached factorize codes for a string key). The
// hash table is the key array itself: each slot is int64, initialised to EMPTY
// (-1, set via cudaMemset 0xFF). Insert/lookup uses atomicCAS on the slot:
//   atomicCAS(&key[slot], EMPTY, mykey) -> old value
//     old == EMPTY : we claimed the slot (new distinct group), gid = slot
//     old == mykey : key already present, gid = slot
//     else         : collision, linear-probe to the next slot
// The atomicCAS on the 64-bit key IS the publish, so this is race-free (no
// separate occupied/key publish ordering hazard). EMPTY = -1 is safe because
// real keys are >= 0 (codes) or >= 1 (l_orderkey) or >> 0 (datetime seconds).
static const long long HASH_EMPTY = -1;

__global__ void fill_i64(long long *a, long long val, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        a[i] = val;
}

__global__ void hash_kernel(Plan p, long long *key, double *acc, int capacity, int nagg,
                            int *distinct, int *overflow) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < p.n; i += gridDim.x * blockDim.x) {
        if (!pass_pred(p, i)) continue;
        long long mykey = ((const long long *)p.cols[p.gkey_idx[0]])[i];  // ngkey == 1
        if (mykey == HASH_EMPTY) { atomicExch(overflow, 1); continue; }
        unsigned long long h = (unsigned long long)mykey;
        h ^= h >> 33;
        h *= 0xff51afd7ed558ccdULL;
        h ^= h >> 33;
        h *= 0xc4ceb9fe1a85ec53ULL;
        h ^= h >> 33;  // MurmurHash3 finalizer (good avalanche for int keys)
        int slot = (int)(h & (unsigned long long)(capacity - 1));
        int gid = -1;
        for (int probe = 0; probe < 64; probe++) {
            long long old = (long long)atomicCAS((unsigned long long *)&key[slot],
                                                 (unsigned long long)HASH_EMPTY,
                                                 (unsigned long long)mykey);
            if (old == HASH_EMPTY) { atomicAdd(distinct, 1); gid = slot; break; }
            if (old == mykey) { gid = slot; break; }
            slot = (slot + 1) & (capacity - 1);
        }
        if (gid < 0) { atomicExch(overflow, 1); continue; }
        for (int a = 0; a < p.nagg; a++) {
            double val = p.agg_kind[a] == AGG_COUNT ? 1.0 : eval_agg(p, i, a);
            atomicAdd(&acc[(long long)gid * nagg + a], val);
        }
    }
}

__global__ void compact_kernel(const long long *key, const double *acc, int capacity, int nagg,
                               long long *out_keys, double *out_acc, int *counter) {
    for (int slot = blockIdx.x * blockDim.x + threadIdx.x; slot < capacity;
         slot += gridDim.x * blockDim.x) {
        if (key[slot] == HASH_EMPTY) continue;
        int idx = atomicAdd(counter, 1);
        out_keys[idx] = key[slot];
        for (int a = 0; a < nagg; a++)
            out_acc[(long long)idx * nagg + a] = acc[(long long)slot * nagg + a];
    }
}

// ---------------- FUSED STAR-JOIN + AGGREGATE ----------------
//
// Streams the fact table once, walks a chain of dimension hash tables in-kernel
// (probe-side streaming + late materialisation -- the joined frame is never
// built), and accumulates straight into a dense per-group accumulator. The
// final dimension's payload is the (factorised) group key code, so the group
// index g is produced by the chain lookups -- everything else reuses the DENSE
// accumulator logic. Inner-join semantics: a probe miss at any stage drops the
// row (no accumulation, seen not set).
//
// Dimension HTs are built once (build_ht_kernel) and probed read-only
// (probe_agg_kernel -- no atomics during probe). All keys/payloads are int64
// (int32 keys are promoted on the Python side). TPC-H join targets are primary
// keys, so dim keys are unique; build_ht_kernel keeps the first payload on a
// duplicate key (last-writer-wins would race) -- the Python side gates out
// non-unique dim keys.

// Build an open-addressing int64->int64 hash table from (key[p], payload[p]).
// atomicCAS on the key slot is the publish (same pattern as hash_kernel).
__global__ void build_ht_kernel(const long long *key, const long long *payload, int n,
                                long long *ht_key, long long *ht_payload, int capacity,
                                int *overflow) {
    for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < n; r += gridDim.x * blockDim.x) {
        long long mykey = key[r];
        if (mykey == HASH_EMPTY) { atomicExch(overflow, 1); continue; }
        unsigned long long h = (unsigned long long)mykey;
        h ^= h >> 33; h *= 0xff51afd7ed558ccdULL; h ^= h >> 33;
        h *= 0xc4ceb9fe1a85ec53ULL; h ^= h >> 33;
        int slot = (int)(h & (unsigned long long)(capacity - 1));
        bool placed = false;
        for (int probe = 0; probe < 64; probe++) {
            long long old = (long long)atomicCAS((unsigned long long *)&ht_key[slot],
                                                 (unsigned long long)HASH_EMPTY,
                                                 (unsigned long long)mykey);
            if (old == HASH_EMPTY) { ht_payload[slot] = payload[r]; placed = true; break; }
            if (old == mykey) { placed = true; break; }  // dup key: keep first payload
            slot = (slot + 1) & (capacity - 1);
        }
        if (!placed) atomicExch(overflow, 1);
    }
}

// Probe the chain of dimension HTs and accumulate per group (DENSE). Reuses
// pass_pred (no-op when n_pred==0) and eval_agg (reads fact cols at row i).
// ht_key/ht_payload are device arrays of per-join device pointers; ht_cap the
// per-join power-of-two capacity. first_probe_col is a fact DT_INT64 column.
__global__ void probe_agg_kernel(Plan p, const long long **ht_key, const long long **ht_payload,
                                 const int *ht_cap, int n_joins, int first_probe_col,
                                 double *acc, int *seen, int n_groups, int nagg) {
    extern __shared__ double sh[];
    int t = threadIdx.x;
    int nga = n_groups * nagg;
    for (int k = t; k < nga; k += blockDim.x) sh[k] = init_for_kind(p.agg_kind[k % nagg]);
    __syncthreads();
    for (int i = blockIdx.x * blockDim.x + t; i < p.n; i += gridDim.x * blockDim.x) {
        if (!pass_pred(p, i)) continue;
        long long key = ((const long long *)p.cols[first_probe_col])[i];
        bool dropped = false;
        for (int j = 0; j < n_joins; j++) {
            int cap = ht_cap[j];
            unsigned long long h = (unsigned long long)key;
            h ^= h >> 33; h *= 0xff51afd7ed558ccdULL; h ^= h >> 33;
            h *= 0xc4ceb9fe1a85ec53ULL; h ^= h >> 33;
            int slot = (int)(h & (unsigned long long)(cap - 1));
            int found = -1;
            for (int probe = 0; probe < 64; probe++) {
                long long k = ht_key[j][slot];  // read-only HT
                if (k == HASH_EMPTY) break;      // miss -> inner-join drop
                if (k == key) { found = slot; break; }
                slot = (slot + 1) & (cap - 1);
            }
            if (found < 0) { dropped = true; break; }
            key = ht_payload[j][found];  // carry payload -> next probe key
        }
        if (dropped) continue;
        long long g = key;  // final payload = group code
        if (g < 0 || g >= n_groups) continue;
        for (int a = 0; a < p.nagg; a++) {
            int kind = p.agg_kind[a];
            double *slot = &sh[g * nagg + a];
            if (kind == AGG_MIN) {
                atomic_min_d(slot, eval_agg(p, i, a));
            } else if (kind == AGG_MAX) {
                atomic_max_d(slot, eval_agg(p, i, a));
            } else {
                double val = kind == AGG_COUNT ? 1.0 : eval_agg(p, i, a);
                atomicAdd(slot, val);
            }
        }
        atomicMax(&seen[g], 1);
    }
    __syncthreads();
    for (int k = t; k < nga; k += blockDim.x) {
        int kind = p.agg_kind[k % nagg];
        if (kind == AGG_MIN) atomic_min_d(&acc[k], sh[k]);
        else if (kind == AGG_MAX) atomic_max_d(&acc[k], sh[k]);
        else atomicAdd(&acc[k], sh[k]);
    }
}

// ---------------- helpers ----------------
static void check(cudaError_t e, const char *what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

// Copy a host numpy array to a fresh device buffer; return device pointer.
template <typename T>
static T *to_dev(const T *host_ptr, size_t n) {
    T *d = nullptr;
    check(cudaMalloc(&d, sizeof(T) * n), "cudaMalloc");
    if (n) check(cudaMemcpy(d, host_ptr, sizeof(T) * n, cudaMemcpyHostToDevice), "cudaMemcpy H2D");
    return d;
}

template <typename T>
static T *np_dev(py::array_t<T> arr) {
    auto info = arr.request();
    return to_dev(static_cast<T *>(info.ptr), info.ndim > 0 ? info.shape[0] : 0);
}

// ---------------- host entry point ----------------
//
// Args (all numpy arrays unless noted):
//   col_ptrs   : int64 device pointers per column
//   col_dtypes : int32 DT_* per column
//   gkey_idx   : int32 column index per group key
//   gkey_stride: int64 row-major strides per group key (DENSE; empty for HASH)
//   pred_col, pred_op : int32 (n_pred)   pred_lit : float64 (n_pred)
//   agg_kind            : int32 (nagg)  0=COUNT,1=SUM
//   agg_tok_start,agg_tok_len : int32 (nagg)
//   tok_kind : int32 (ntok)  tok_col : int32 (ntok)  tok_lit : float64 (ntok)  tok_op : int32 (ntok)
//   acc_init : float64 (n_groups*nagg for DENSE; empty for HASH) per-slot init
//              (+inf for MIN, -inf for MAX, 0 otherwise). Empty -> memset 0.
//   strategy  : int (0=DENSE,1=HASH)
//   n_groups  : int (DENSE)   capacity : int (HASH, power of two)
//
// Returns a tuple (overflow:int, n_out:int, keys:py::list[int64 arrays],
//                  aggs:py::list[float64 arrays]).
//   overflow != 0 means the hash table filled -> caller falls back to cuDF.
//   keys[i] is the int64 code/value column for group key i (n_out rows);
//   aggs[a] is the float64 accumulator column for aggregate a (n_out rows).
//   nagg is the INTERNAL slot count: visible aggs followed by one hidden
//   AGG_COUNT slot when any AVG is present (the per-group passing-row count,
//   used as the AVG denominator at read-out; not emitted as an output column).
py::tuple fused_agg(py::array_t<long long> col_ptrs, py::array_t<int> col_dtypes,
                    py::array_t<int> gkey_idx, py::array_t<long long> gkey_stride,
                    py::array_t<int> pred_col, py::array_t<int> pred_op,
                    py::array_t<double> pred_lit, py::array_t<int> agg_kind,
                    py::array_t<int> agg_tok_start, py::array_t<int> agg_tok_len,
                    py::array_t<int> tok_kind, py::array_t<int> tok_col,
                    py::array_t<double> tok_lit, py::array_t<int> tok_op,
                    py::array_t<double> acc_init,
                    int strategy, int n_groups, int capacity, int n_rows) {
    int ncol = (int)col_ptrs.shape(0);
    int nagg = (int)agg_kind.shape(0);
    int ngkey = (int)gkey_idx.shape(0);
    int ntok = (int)tok_kind.shape(0);
    int n_pred = (int)pred_col.shape(0);

    // --- copy descriptors to device ---
    auto ptrs_info = col_ptrs.request();
    const void **d_cols = nullptr;
    check(cudaMalloc(&d_cols, sizeof(void *) * ncol), "malloc cols");
    // col_ptrs holds int64 device addresses; copy as raw bytes into void* array.
    check(cudaMemcpy(d_cols, ptrs_info.ptr, sizeof(void *) * ncol, cudaMemcpyHostToDevice),
          "memcpy cols");

    Plan p{};
    p.cols = d_cols;
    p.dtypes = np_dev(col_dtypes);
    p.gkey_idx = np_dev(gkey_idx);
    p.gkey_stride = np_dev(gkey_stride);
    p.ngkey = ngkey;
    p.n_pred = n_pred;
    p.pred_col = np_dev(pred_col);
    p.pred_op = np_dev(pred_op);
    p.pred_lit = np_dev(pred_lit);
    p.nagg = nagg;
    p.agg_kind = np_dev(agg_kind);
    p.agg_tok_start = np_dev(agg_tok_start);
    p.agg_tok_len = np_dev(agg_tok_len);
    p.ntok = ntok;
    p.tok_kind = np_dev(tok_kind);
    p.tok_col = np_dev(tok_col);
    p.tok_lit = np_dev(tok_lit);
    p.tok_op = np_dev(tok_op);
    p.n = n_rows;

    const int THREADS = 256;
    int blocks = (n_rows + THREADS - 1) / THREADS;
    if (blocks > 65535) blocks = 65535;  // grid cap; grid-stride loop covers the rest

    py::list keys_list;
    py::list aggs_list;
    int overflow = 0;
    int n_out = 0;

    if (strategy == STRAT_DENSE) {
        int nga = n_groups * nagg;
        double *acc = nullptr;
        int *seen = nullptr;
        check(cudaMalloc(&acc, sizeof(double) * nga), "malloc acc dense");
        check(cudaMalloc(&seen, sizeof(int) * n_groups), "malloc seen");
        // Initialise accumulators per agg kind (+inf/-inf for MIN/MAX, 0 else).
        // acc_init is n_groups*nagg; if empty (shouldn't happen for DENSE) fall
        // back to a zero memset.
        auto init_info = acc_init.request();
        if (init_info.ndim > 0 && init_info.shape[0] > 0) {
            check(cudaMemcpy(acc, init_info.ptr, sizeof(double) * nga, cudaMemcpyHostToDevice),
                  "cp acc_init");
        } else {
            check(cudaMemset(acc, 0, sizeof(double) * nga), "memset acc");
        }
        check(cudaMemset(seen, 0, sizeof(int) * n_groups), "memset seen");
        size_t shbytes = sizeof(double) * nga;
        dense_kernel<<<blocks, THREADS, shbytes>>>(p, acc, seen, n_groups, nagg);
        check(cudaGetLastError(), "dense_kernel launch");
        check(cudaDeviceSynchronize(), "dense sync");

        std::vector<double> h_acc(nga);
        std::vector<int> h_seen(n_groups);
        check(cudaMemcpy(h_acc.data(), acc, sizeof(double) * nga, cudaMemcpyDeviceToHost), "cp acc");
        check(cudaMemcpy(h_seen.data(), seen, sizeof(int) * n_groups, cudaMemcpyDeviceToHost), "cp seen");

        // Count occupied groups.
        for (int g = 0; g < n_groups; g++)
            if (h_seen[g]) n_out++;

        // Decode group index -> per-key codes (row-major: stride-major order).
        std::vector<std::vector<long long>> h_keys(ngkey, std::vector<long long>(n_out));
        std::vector<std::vector<double>> h_aggs(nagg, std::vector<double>(n_out));
        std::vector<long long> stride(ngkey);
        {
            auto sinfo = gkey_stride.request();
            long long *sp = static_cast<long long *>(sinfo.ptr);
            for (int j = 0; j < ngkey; j++) stride[j] = sp[j];
        }
        int row = 0;
        for (int g = 0; g < n_groups; g++) {
            if (!h_seen[g]) continue;
            long long rem = g;
            for (int j = 0; j < ngkey; j++) {
                h_keys[j][row] = rem / stride[j];
                rem = rem % stride[j];
            }
            for (int a = 0; a < nagg; a++) h_aggs[a][row] = h_acc[g * nagg + a];
            row++;
        }
        for (int j = 0; j < ngkey; j++)
            keys_list.append(py::array_t<long long>(n_out, h_keys[j].data()));
        for (int a = 0; a < nagg; a++)
            aggs_list.append(py::array_t<double>(n_out, h_aggs[a].data()));

        cudaFree(acc);
        cudaFree(seen);
    } else {  // HASH -- single int64 group key, lock-free atomicCAS-on-key
        long long *key = nullptr;   // hash table IS the key array (EMPTY=-1)
        double *acc = nullptr;
        int *distinct = nullptr, *ovf = nullptr;
        check(cudaMalloc(&key, sizeof(long long) * capacity), "malloc key hash");
        check(cudaMalloc(&acc, sizeof(double) * (size_t)capacity * nagg), "malloc acc hash");
        check(cudaMalloc(&distinct, sizeof(int)), "malloc distinct");
        check(cudaMalloc(&ovf, sizeof(int)), "malloc ovf");
        // 0xFF bytes -> every int64 slot = -1 = HASH_EMPTY.
        check(cudaMemset(key, 0xFF, sizeof(long long) * capacity), "memset key");
        check(cudaMemset(acc, 0, sizeof(double) * (size_t)capacity * nagg), "memset acc");
        check(cudaMemset(distinct, 0, sizeof(int)), "memset distinct");
        check(cudaMemset(ovf, 0, sizeof(int)), "memset ovf");

        hash_kernel<<<blocks, THREADS>>>(p, key, acc, capacity, nagg, distinct, ovf);
        check(cudaGetLastError(), "hash_kernel launch");
        check(cudaDeviceSynchronize(), "hash sync");

        int h_ovf = 0, h_distinct = 0;
        check(cudaMemcpy(&h_ovf, ovf, sizeof(int), cudaMemcpyDeviceToHost), "cp ovf");
        check(cudaMemcpy(&h_distinct, distinct, sizeof(int), cudaMemcpyDeviceToHost), "cp distinct");
        overflow = h_ovf;
        n_out = h_distinct;

        if (overflow == 0 && n_out > 0) {
            long long *out_keys = nullptr;
            double *out_acc = nullptr;
            int *counter = nullptr;
            check(cudaMalloc(&out_keys, sizeof(long long) * n_out), "malloc out_keys");
            check(cudaMalloc(&out_acc, sizeof(double) * (size_t)n_out * nagg), "malloc out_acc");
            check(cudaMalloc(&counter, sizeof(int)), "malloc counter");
            check(cudaMemset(counter, 0, sizeof(int)), "memset counter");
            int cblocks = (capacity + THREADS - 1) / THREADS;
            if (cblocks > 65535) cblocks = 65535;
            compact_kernel<<<cblocks, THREADS>>>(key, acc, capacity, nagg,
                                                 out_keys, out_acc, counter);
            check(cudaGetLastError(), "compact launch");
            check(cudaDeviceSynchronize(), "compact sync");

            std::vector<long long> h_keys(n_out);
            std::vector<double> h_acc2((size_t)n_out * nagg);
            check(cudaMemcpy(h_keys.data(), out_keys, sizeof(long long) * n_out,
                             cudaMemcpyDeviceToHost), "cp out_keys");
            check(cudaMemcpy(h_acc2.data(), out_acc, sizeof(double) * n_out * nagg,
                             cudaMemcpyDeviceToHost), "cp out_acc");
            keys_list.append(py::array_t<long long>(n_out, h_keys.data()));
            for (int a = 0; a < nagg; a++) {
                std::vector<double> col(n_out);
                for (int r = 0; r < n_out; r++) col[r] = h_acc2[(size_t)r * nagg + a];
                aggs_list.append(py::array_t<double>(n_out, col.data()));
            }
            cudaFree(out_keys);
            cudaFree(out_acc);
            cudaFree(counter);
        }
        cudaFree(key);
        cudaFree(acc);
        cudaFree(distinct);
        cudaFree(ovf);
    }

    // free descriptor device buffers
    cudaFree((void *)p.dtypes);
    cudaFree((void *)p.gkey_idx);
    cudaFree((void *)p.gkey_stride);
    cudaFree((void *)p.pred_col);
    cudaFree((void *)p.pred_op);
    cudaFree((void *)p.pred_lit);
    cudaFree((void *)p.agg_kind);
    cudaFree((void *)p.agg_tok_start);
    cudaFree((void *)p.agg_tok_len);
    cudaFree((void *)p.tok_kind);
    cudaFree((void *)p.tok_col);
    cudaFree((void *)p.tok_lit);
    cudaFree((void *)p.tok_op);
    cudaFree(d_cols);

    return py::make_tuple(overflow, n_out, keys_list, aggs_list);
}

// ---------------- host entry point: fused star-join + aggregate ----------------
//
// Streams the fact table, builds a dimension hash table per join, probes the
// chain, and accumulates per group (DENSE). The joined frame is never built.
//
// Args (all numpy arrays unless noted):
//   col_ptrs        : int64 device pointers per FACT column
//   col_dtypes      : int32 DT_* per fact column
//   first_probe_col : int -- fact column index of the first join's probe key (DT_INT64)
//   dim_key_ptrs    : int64 (n_joins,) device addresses of each dim KEY column (int64)
//   dim_payload_ptrs: int64 (n_joins,) device addresses of each dim PAYLOAD column (int64)
//   dim_n           : int32 (n_joins,) row counts per dimension
//   ht_cap          : int32 (n_joins,) power-of-two capacities per HT
//   pred_col,pred_op: int32 (n_pred)  pred_lit: float64 (n_pred)
//   agg_kind,agg_tok_start,agg_tok_len : int32 (nagg)
//   tok_kind:int32 (ntok) tok_col:int32 tok_lit:float64 tok_op:int32
//   acc_init : float64 (n_groups*nagg) per-slot init
//   n_groups : int (DENSE group count)   n_rows : int (fact row count)
//
// Returns (overflow:int, n_out:int, keys:py::list[int64 arrays (1 col, the codes)],
//          aggs:py::list[float64 arrays]). overflow!=0 -> caller falls back to cuDF.
py::tuple fused_join_agg(py::array_t<long long> col_ptrs, py::array_t<int> col_dtypes,
                         int first_probe_col,
                         py::array_t<long long> dim_key_ptrs,
                         py::array_t<long long> dim_payload_ptrs,
                         py::array_t<int> dim_n, py::array_t<int> ht_cap,
                         py::array_t<int> pred_col, py::array_t<int> pred_op,
                         py::array_t<double> pred_lit, py::array_t<int> agg_kind,
                         py::array_t<int> agg_tok_start, py::array_t<int> agg_tok_len,
                         py::array_t<int> tok_kind, py::array_t<int> tok_col,
                         py::array_t<double> tok_lit, py::array_t<int> tok_op,
                         py::array_t<double> acc_init,
                         int n_groups, int n_rows) {
    int ncol = (int)col_ptrs.shape(0);
    int nagg = (int)agg_kind.shape(0);
    int ntok = (int)tok_kind.shape(0);
    int n_pred = (int)pred_col.shape(0);
    int n_joins = (int)dim_n.shape(0);

    auto ptrs_info = col_ptrs.request();
    const void **d_cols = nullptr;
    check(cudaMalloc(&d_cols, sizeof(void *) * ncol), "malloc cols join");
    check(cudaMemcpy(d_cols, ptrs_info.ptr, sizeof(void *) * ncol, cudaMemcpyHostToDevice),
          "memcpy cols join");

    Plan p{};
    p.cols = d_cols;
    p.dtypes = np_dev(col_dtypes);
    p.ngkey = 0;  // group index comes from the join chain, not gkey_idx
    p.n_pred = n_pred;
    p.pred_col = np_dev(pred_col);
    p.pred_op = np_dev(pred_op);
    p.pred_lit = np_dev(pred_lit);
    p.nagg = nagg;
    p.agg_kind = np_dev(agg_kind);
    p.agg_tok_start = np_dev(agg_tok_start);
    p.agg_tok_len = np_dev(agg_tok_len);
    p.ntok = ntok;
    p.tok_kind = np_dev(tok_kind);
    p.tok_col = np_dev(tok_col);
    p.tok_lit = np_dev(tok_lit);
    p.tok_op = np_dev(tok_op);
    p.n = n_rows;

    auto dk_info = dim_key_ptrs.request();
    auto dp_info = dim_payload_ptrs.request();
    auto dn_info = dim_n.request();
    auto cap_info = ht_cap.request();
    long long *dk_host = static_cast<long long *>(dk_info.ptr);
    long long *dp_host = static_cast<long long *>(dp_info.ptr);
    int *dn_host = static_cast<int *>(dn_info.ptr);
    int *cap_host = static_cast<int *>(cap_info.ptr);

    const int THREADS = 256;
    int overflow = 0;
    int n_out = 0;
    py::list keys_list;
    py::list aggs_list;

    // --- build one HT per dimension ---
    std::vector<long long *> ht_key(n_joins, nullptr);
    std::vector<long long *> ht_payload(n_joins, nullptr);
    int *d_ovf = nullptr;
    check(cudaMalloc(&d_ovf, sizeof(int)), "malloc join ovf");
    check(cudaMemset(d_ovf, 0, sizeof(int)), "memset join ovf");

    for (int j = 0; j < n_joins; j++) {
        int cap = cap_host[j];
        int n = dn_host[j];
        check(cudaMalloc(&ht_key[j], sizeof(long long) * cap), "malloc ht_key");
        check(cudaMalloc(&ht_payload[j], sizeof(long long) * cap), "malloc ht_payload");
        check(cudaMemset(ht_key[j], 0xFF, sizeof(long long) * cap), "memset ht_key");  // EMPTY
        int blocks = (n + THREADS - 1) / THREADS;
        if (blocks > 65535) blocks = 65535;
        build_ht_kernel<<<blocks, THREADS>>>((const long long *)dk_host[j],
                                             (const long long *)dp_host[j], n,
                                             ht_key[j], ht_payload[j], cap, d_ovf);
        check(cudaGetLastError(), "build_ht_kernel launch");
    }
    check(cudaDeviceSynchronize(), "build sync");
    check(cudaMemcpy(&overflow, d_ovf, sizeof(int), cudaMemcpyDeviceToHost), "cp join ovf");

    if (overflow != 0) {
        for (int j = 0; j < n_joins; j++) { cudaFree(ht_key[j]); cudaFree(ht_payload[j]); }
        cudaFree(d_ovf);
        cudaFree((void *)p.dtypes); cudaFree((void *)p.pred_col); cudaFree((void *)p.pred_op);
        cudaFree((void *)p.pred_lit); cudaFree((void *)p.agg_kind); cudaFree((void *)p.agg_tok_start);
        cudaFree((void *)p.agg_tok_len); cudaFree((void *)p.tok_kind); cudaFree((void *)p.tok_col);
        cudaFree((void *)p.tok_lit); cudaFree((void *)p.tok_op); cudaFree(d_cols);
        return py::make_tuple(overflow, n_out, keys_list, aggs_list);
    }

    // --- device arrays of per-join HT pointers + capacities for the probe kernel ---
    long long **d_htkey = nullptr, **d_htpayload = nullptr;
    int *d_htcap = nullptr;
    check(cudaMalloc(&d_htkey, sizeof(void *) * n_joins), "malloc d_htkey");
    check(cudaMalloc(&d_htpayload, sizeof(void *) * n_joins), "malloc d_htpayload");
    check(cudaMemcpy(d_htkey, ht_key.data(), sizeof(void *) * n_joins, cudaMemcpyHostToDevice),
          "memcpy d_htkey");
    check(cudaMemcpy(d_htpayload, ht_payload.data(), sizeof(void *) * n_joins, cudaMemcpyHostToDevice),
          "memcpy d_htpayload");
    d_htcap = np_dev(ht_cap);

    // --- DENSE accumulator (single int64 group key = code 0..n_groups-1) ---
    int nga = n_groups * nagg;
    double *acc = nullptr;
    int *seen = nullptr;
    check(cudaMalloc(&acc, sizeof(double) * nga), "malloc join acc");
    check(cudaMalloc(&seen, sizeof(int) * n_groups), "malloc join seen");
    auto init_info = acc_init.request();
    if (init_info.ndim > 0 && init_info.shape[0] > 0) {
        check(cudaMemcpy(acc, init_info.ptr, sizeof(double) * nga, cudaMemcpyHostToDevice),
              "cp join acc_init");
    } else {
        check(cudaMemset(acc, 0, sizeof(double) * nga), "memset join acc");
    }
    check(cudaMemset(seen, 0, sizeof(int) * n_groups), "memset join seen");

    int blocks = (n_rows + THREADS - 1) / THREADS;
    if (blocks > 65535) blocks = 65535;
    size_t shbytes = sizeof(double) * nga;
    probe_agg_kernel<<<blocks, THREADS, shbytes>>>(
        p, (const long long **)d_htkey, (const long long **)d_htpayload, d_htcap,
        n_joins, first_probe_col, acc, seen, n_groups, nagg);
    check(cudaGetLastError(), "probe_agg_kernel launch");
    check(cudaDeviceSynchronize(), "probe sync");

    // --- read-out (single group key: code == group index g) ---
    std::vector<double> h_acc(nga);
    std::vector<int> h_seen(n_groups);
    check(cudaMemcpy(h_acc.data(), acc, sizeof(double) * nga, cudaMemcpyDeviceToHost), "cp join acc");
    check(cudaMemcpy(h_seen.data(), seen, sizeof(int) * n_groups, cudaMemcpyDeviceToHost), "cp join seen");
    for (int g = 0; g < n_groups; g++)
        if (h_seen[g]) n_out++;
    std::vector<long long> h_keys(n_out);
    std::vector<std::vector<double>> h_aggs(nagg, std::vector<double>(n_out));
    int row = 0;
    for (int g = 0; g < n_groups; g++) {
        if (!h_seen[g]) continue;
        h_keys[row] = g;  // code == group index
        for (int a = 0; a < nagg; a++) h_aggs[a][row] = h_acc[g * nagg + a];
        row++;
    }
    keys_list.append(py::array_t<long long>(n_out, h_keys.data()));
    for (int a = 0; a < nagg; a++)
        aggs_list.append(py::array_t<double>(n_out, h_aggs[a].data()));

    // --- free everything ---
    cudaFree(acc); cudaFree(seen); cudaFree(d_ovf);
    cudaFree(d_htkey); cudaFree(d_htpayload); cudaFree((void *)d_htcap);
    for (int j = 0; j < n_joins; j++) { cudaFree(ht_key[j]); cudaFree(ht_payload[j]); }
    cudaFree((void *)p.dtypes); cudaFree((void *)p.pred_col); cudaFree((void *)p.pred_op);
    cudaFree((void *)p.pred_lit); cudaFree((void *)p.agg_kind); cudaFree((void *)p.agg_tok_start);
    cudaFree((void *)p.agg_tok_len); cudaFree((void *)p.tok_kind); cudaFree((void *)p.tok_col);
    cudaFree((void *)p.tok_lit); cudaFree((void *)p.tok_op); cudaFree(d_cols);

    return py::make_tuple(overflow, n_out, keys_list, aggs_list);
}

// ============================================================================
// Phase 5: hand-rolled CUDA Parquet decoder fused with scan+filter+aggregate.
//
// Streams Parquet pages -> nvCOMP batched Snappy-decompress on GPU -> a decode
// kernel that folds decimal scale + date32->seconds, evaluates the predicate,
// and accumulates straight into the existing DENSE/HASH accumulator -- never
// materialising the 60M-row frame. Host-driven (nvCOMP is host-launched, not
// device-callable): one launch per row group into a PERSISTENT accumulator,
// then the existing read-out runs unchanged.
//
// v1 scope (certain cold wins): global aggregate (n_groups==1 DENSE, no group
// keys -> Q6 / scan_agg_full) and HASH with a single PLAIN int64 group key
// (high-card l_orderkey). Columns are PLAIN_RAW (read values at values_off) or
// PK_DICT_NUMERIC_ARG (decode RLE/bit-packed indices, gather the per-RG dict,
// fold). Dict-string DENSE group keys (Q1) are deferred: the dictionary is
// per-row-group so dict indices are local codes that don't map to a global
// DENSE accumulator without a per-RG local->global remap (Phase 5 step 2).
// ============================================================================

static constexpr int PK_PLAIN_RAW = 0;        // buf = decompressed data page; values at values_off
static constexpr int PK_DICT_NUMERIC_ARG = 1; // buf = int32 index array; dict = per-RG dict values
static constexpr int PHYS_I32 = 0;
static constexpr int PHYS_I64 = 1;
static constexpr int PQ_RUNS_CAP = 32768;     // max RLE/bit-packed runs per dict data page

// Per-column, per-row-group page source (device pointers), passed to kernels by value.
struct PageSrc {
    const void **buf;        // device: per-col ptr (plain: page uncomp buf; dict: int32 index array)
    const void **dict;       // device: per-col dict ptr (dict numeric; 0 otherwise)
    const int *kind;         // device: PK_* per col
    const int *phys;         // device: PHYS_* per col (plain read width / dict value width)
    const int *scale;        // device: decimal scale per col (0 if not decimal)
    const int *is_date;      // device: 1 -> int32 days *86400 -> seconds
};

// Byte offset of the values section within a PLAIN data page: the page is
// `[4B LE def-level-byte-len][def RLE bytes][values]`, so values start at
// 4 + deflen. deflen is read on-device from the page's first int -- this avoids
// a per-column D2H + host sync during the row-group loop.
__device__ inline int plain_values_off(const void *page) {
    return 4 + *(const int *)page;
}

__device__ inline double pow10d(int s) { double t = 1.0; for (int k = 0; k < s; k++) t *= 10.0; return t; }

// Unaligned little-endian loads. Parquet PLAIN values start at
// plain_values_off = 4 + deflen, and deflen (the def-level RLE block length) is
// data-dependent and NOT a multiple of 4/8, so the values are not guaranteed to
// be 4- or 8-byte aligned. A typed pointer dereference faults with
// cudaErrorMisalignedAddress; these byte-wise loads are safe at any offset.
__device__ inline int load_i32_una(const uint8_t *p) {
    return (int)p[0] | ((int)p[1] << 8) | ((int)p[2] << 16) | ((int)p[3] << 24);
}
__device__ inline long long load_i64_una(const uint8_t *p) {
    return (long long)p[0] | ((long long)p[1] << 8) | ((long long)p[2] << 16)
         | ((long long)p[3] << 24) | ((long long)p[4] << 32) | ((long long)p[5] << 40)
         | ((long long)p[6] << 48) | ((long long)p[7] << 56);
}

// Read a numeric column value for row i from the page source, folding decimal
// scale (INT64 / 10^scale) and date32 (INT32 days *86400 -> seconds) inline.
__device__ inline double page_col_val(const PageSrc &s, int c, int i) {
    double raw;
    if (s.kind[c] == PK_PLAIN_RAW) {
        const uint8_t *base = (const uint8_t *)s.buf[c] + plain_values_off(s.buf[c]);
        const uint8_t *vp = base + (size_t)i * (s.phys[c] == PHYS_I32 ? 4 : 8);
        raw = (s.phys[c] == PHYS_I32) ? (double)load_i32_una(vp) : (double)load_i64_una(vp);
    } else {  // PK_DICT_NUMERIC_ARG: gather dict[index]
        int idx = ((const int *)s.buf[c])[i];
        raw = (s.phys[c] == PHYS_I32) ? (double)((const int *)s.dict[c])[idx]
                                      : (double)((const long long *)s.dict[c])[idx];
    }
    if (s.is_date[c]) return raw * 86400.0;
    if (s.scale[c] > 0) return raw / pow10d(s.scale[c]);
    return raw;
}

// Raw stored integer for row i (PLAIN load or DICT gather), sign-extended to
// int64 with NO decimal/date/double fold. Used by the materialise kernel to
// write typed frame columns (int32/int64/datetime64) without the precision loss
// of page_col_val's double return (int64 > 2^53 would round). Uses the SAME
// kind discriminator as page_col_val.
__device__ inline long long page_col_raw64(const PageSrc &s, int c, int i) {
    if (s.kind[c] == PK_PLAIN_RAW) {
        const uint8_t *base = (const uint8_t *)s.buf[c] + plain_values_off(s.buf[c]);
        const uint8_t *vp = base + (size_t)i * (s.phys[c] == PHYS_I32 ? 4 : 8);
        return (s.phys[c] == PHYS_I32) ? (long long)load_i32_una(vp) : load_i64_una(vp);
    }
    // PK_DICT_NUMERIC_ARG: gather dict[index]
    int idx = ((const int *)s.buf[c])[i];
    return (s.phys[c] == PHYS_I32) ? (long long)((const int *)s.dict[c])[idx]
                                   : ((const long long *)s.dict[c])[idx];
}

__device__ inline bool page_pass_pred(const Plan &p, const PageSrc &s, int i) {
    for (int j = 0; j < p.n_pred; j++) {
        double v = page_col_val(s, p.pred_col[j], i);
        double lit = p.pred_lit[j];
        int op = p.pred_op[j];
        bool ok = op == OP_EQ ? v == lit : op == OP_NE ? v != lit
                       : op == OP_LT ? v < lit
                       : op == OP_LE ? v <= lit
                       : op == OP_GT ? v > lit
                                      : v >= lit;
        if (!ok) return false;
    }
    return true;
}

__device__ inline double page_eval_agg(const Plan &p, const PageSrc &s, int i, int agg) {
    int st = p.agg_tok_start[agg], len = p.agg_tok_len[agg];
    double stack[8];
    int sp = 0;
    for (int t = 0; t < len; t++) {
        int idx = st + t, kind = p.tok_kind[idx];
        if (kind == TK_COL) {
            stack[sp++] = page_col_val(s, p.tok_col[idx], i);
        } else if (kind == TK_LIT) {
            stack[sp++] = p.tok_lit[idx];
        } else {  // TK_OP
            double b = stack[--sp], a = stack[--sp];
            int op = p.tok_op[idx];
            double r = op == TOP_ADD ? a + b : op == TOP_SUB ? a - b
                            : op == TOP_MUL ? a * b
                                           : a / b;
            stack[sp++] = r;
        }
    }
    return sp > 0 ? stack[0] : 0.0;
}

// DENSE over pages. v1: ngkey == 0 (global, g == 0). Reuses the shared
// accumulator + cross-block reduce from dense_kernel; only the column source
// differs (PageSrc vs pre-materialised arrays).
__global__ void page_dense_kernel(Plan p, PageSrc s, double *acc, int *seen, int n_groups, int nagg) {
    extern __shared__ double sh[];
    int t = threadIdx.x;
    int nga = n_groups * nagg;
    for (int k = t; k < nga; k += blockDim.x) sh[k] = init_for_kind(p.agg_kind[k % nagg]);
    __syncthreads();
    for (int i = blockIdx.x * blockDim.x + t; i < p.n; i += gridDim.x * blockDim.x) {
        if (!page_pass_pred(p, s, i)) continue;
        long long g = 0;  // global: ngkey == 0
        for (int a = 0; a < p.nagg; a++) {
            int kind = p.agg_kind[a];
            double *slot = &sh[g * nagg + a];
            if (kind == AGG_MIN) atomic_min_d(slot, page_eval_agg(p, s, i, a));
            else if (kind == AGG_MAX) atomic_max_d(slot, page_eval_agg(p, s, i, a));
            else { double val = kind == AGG_COUNT ? 1.0 : page_eval_agg(p, s, i, a); atomicAdd(slot, val); }
        }
        atomicMax(&seen[g], 1);
    }
    __syncthreads();
    for (int k = t; k < nga; k += blockDim.x) {
        int kind = p.agg_kind[k % nagg];
        if (kind == AGG_MIN) atomic_min_d(&acc[k], sh[k]);
        else if (kind == AGG_MAX) atomic_max_d(&acc[k], sh[k]);
        else atomicAdd(&acc[k], sh[k]);
    }
}

// HASH over pages: single PLAIN int64 group key read at the page values offset
// (computed on-device from the def-level length). Reuses the atomicCAS-on-key
// insert from hash_kernel; only the column source differs.
__global__ void page_hash_kernel(Plan p, PageSrc s, long long *key, double *acc, int capacity,
                                 int nagg, int *distinct, int *overflow) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < p.n; i += gridDim.x * blockDim.x) {
        if (!page_pass_pred(p, s, i)) continue;
        int gc = p.gkey_idx[0];
        const uint8_t *kpage = (const uint8_t *)s.buf[gc];
        const uint8_t *kp = kpage + plain_values_off(kpage) + (size_t)i * 8;
        long long mykey = load_i64_una(kp);
        if (mykey == HASH_EMPTY) { atomicExch(overflow, 1); continue; }
        unsigned long long h = (unsigned long long)mykey;
        h ^= h >> 33; h *= 0xff51afd7ed558ccdULL; h ^= h >> 33;
        h *= 0xc4ceb9fe1a85ec53ULL; h ^= h >> 33;
        int slot = (int)(h & (unsigned long long)(capacity - 1));
        int gid = -1;
        for (int probe = 0; probe < 64; probe++) {
            long long old = (long long)atomicCAS((unsigned long long *)&key[slot],
                                                 (unsigned long long)HASH_EMPTY,
                                                 (unsigned long long)mykey);
            if (old == HASH_EMPTY) { atomicAdd(distinct, 1); gid = slot; break; }
            if (old == mykey) { gid = slot; break; }
            slot = (slot + 1) & (capacity - 1);
        }
        if (gid < 0) { atomicExch(overflow, 1); continue; }
        for (int a = 0; a < p.nagg; a++) {
            double val = p.agg_kind[a] == AGG_COUNT ? 1.0 : page_eval_agg(p, s, i, a);
            atomicAdd(&acc[(long long)gid * nagg + a], val);
        }
    }
}

// Materialise-gather: write every bound column's decoded value for this row
// group into a contiguous per-column device frame buffer (Python-owned cuDF
// Series, passed in as raw ptrs) at the RG's global row offset, so the scan
// path can populate Engine._scan_cache and warm repeats hit the GPU-resident
// frame. d_out_kind single-sources the store width so host _frame_dtype and the
// device store cannot diverge: 0=int32, 1=int64, 2=float64 (decimal scale folded
// here), 3=datetime64[s] (int64 seconds = days*86400). Reuses page_col_raw64
// (raw int, no precision loss) and the RG's already-wired PageSrc s.
__global__ void materialise_kernel(PageSrc s, const long long *d_frame_ptrs,
                                   const int *d_out_kind, const long long *d_row_off,
                                   int rg, int n, int ncol) {
    long long base = d_row_off[rg];
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        for (int c = 0; c < ncol; c++) {
            long long raw = page_col_raw64(s, c, i);
            int k = d_out_kind[c];
            char *p = (char *)d_frame_ptrs[c];
            if (k == 0) {
                ((int *)p)[base + i] = (int)raw;
            } else if (k == 2) {  // decimal -> float64
                double v = (double)raw;
                int sc = s.scale[c];
                if (sc > 0) v /= pow10d(sc);
                ((double *)p)[base + i] = v;
            } else {  // int64 (k==1) or datetime64[s] (k==3)
                ((long long *)p)[base + i] = (k == 3) ? raw * 86400LL : raw;
            }
        }
    }
}

// Parallel RLE/bit-packed dict-index decode (two-pass). The index stream is a
// sequence of runs; run headers are LEB128 varints with no outer length prefix.
struct RunEntry { int out_start; int count; int is_rle; int value; int data_off; };

// Batched dict-index decode (Phase 5 step 2). The per-RG versions launched
// scan_runs+apply_runs once per dict column per row group (~1467 serial launches
// for Q6 SF10), and the single-thread scan_runs serialised to ~180 ms. These two
// batched kernels collapse that into TWO launches: one block per dict data page.
// `blockIdx.x` (scan) / `blockIdx.y` (apply) is the per-page slot `s`; each
// slot has its own run table at `runs + s*PQ_RUNS_CAP`, count at `nruns[s]`,
// bitwidth at `bw[s]`, and int32 index output at `idxbig + s*max_n`.

// Pass 1 (one thread per dict page): walk that page's run headers into its
// per-page run-table slot. Reads def-level length + bit width straight from the
// page (`[4B deflen][def RLE][1B bitwidth][index runs]`) -- no D2H per page.
__global__ void scan_runs_batched_kernel(const void *const *pages, const int *nvals,
                                         RunEntry *runs, int *nruns, int *overflow,
                                         int *bw, size_t total) {
    if (threadIdx.x != 0) return;
    size_t slot = blockIdx.x;
    if (slot >= total) return;
    const uint8_t *page = (const uint8_t *)pages[slot];
    int num_values = nvals[slot];
    RunEntry *myruns = runs + slot * (size_t)PQ_RUNS_CAP;
    int deflen = *(const int *)page;
    int bitwidth = page[4 + deflen];
    int idx_off = 4 + deflen + 1;
    bw[slot] = bitwidth;
    int pos = idx_off, cur = 0, r = 0;
    int vw = (bitwidth + 7) / 8;
    while (cur < num_values) {
        unsigned long long h = 0; int sh = 0;
        for (int k = 0; k < 10; k++) {
            uint8_t b = page[pos++];
            h |= ((unsigned long long)(b & 0x7f)) << sh;
            if (!(b & 0x80)) break;
            sh += 7;
        }
        int type = (int)(h & 1ULL);
        long long cnt = (long long)(h >> 1);
        if (type == 0) {  // RLE run: count values, one width-rounded value
            int val = 0;
            for (int b = 0; b < vw; b++) val |= ((int)page[pos + b]) << (8 * b);
            pos += vw;
            myruns[r].out_start = cur; myruns[r].count = (int)cnt; myruns[r].is_rle = 1;
            myruns[r].value = val; myruns[r].data_off = 0;
            cur += (int)cnt;
        } else {  // bit-packed run: cnt groups of 8 values; cnt*bitwidth bytes
            int nvals_run = (int)cnt * 8;
            int nbytes = (int)cnt * bitwidth;
            myruns[r].out_start = cur; myruns[r].count = nvals_run; myruns[r].is_rle = 0;
            myruns[r].value = 0; myruns[r].data_off = pos;
            pos += nbytes; cur += nvals_run;
        }
        if (++r >= PQ_RUNS_CAP) { atomicExch(overflow, 1); nruns[slot] = r; return; }
    }
    nruns[slot] = r;
}

// Pass 2 (2D grid: blockIdx.y = page slot, blockIdx.x covers that page's values):
// each output value binary-searches its page's run table and writes the int32
// dict index. Reads nruns/bitwidth from the per-page device ints written by
// pass 1 (stream-ordered, no host sync). gridDim.x = ceil(max_n/THREADS) so the
// grid-stride loop covers every page's n (<= max_n).
__global__ void apply_runs_batched_kernel(const void *const *pages, const int *nvals,
                                          const RunEntry *runs, const int *nruns_p,
                                          const int *bw, int *idxbig, int max_n) {
    size_t slot = blockIdx.y;
    int num_values = nvals[slot];
    const uint8_t *page = (const uint8_t *)pages[slot];
    const RunEntry *myruns = runs + slot * (size_t)PQ_RUNS_CAP;
    int *out = idxbig + slot * (size_t)max_n;
    int nruns = nruns_p[slot];
    int bitwidth = bw[slot];
    unsigned mask = (bitwidth >= 32) ? 0xffffffffu : ((1u << bitwidth) - 1u);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_values; i += gridDim.x * blockDim.x) {
        int lo = 0, hi = nruns - 1, ans = 0;
        while (lo <= hi) {
            int mid = (lo + hi) / 2;
            if (myruns[mid].out_start <= i) { ans = mid; lo = mid + 1; } else hi = mid - 1;
        }
        const RunEntry &e = myruns[ans];
        if (e.is_rle) { out[i] = e.value; continue; }
        int local = i - e.out_start;
        int bit = local * bitwidth;
        int byteoff = e.data_off + bit / 8;
        int bitin = bit & 7;
        int val = ((int)page[byteoff]) >> bitin;
        if (bitin + bitwidth > 8) val |= ((int)page[byteoff + 1]) << (8 - bitin);
        out[i] = (int)((unsigned)val & mask);
    }
}

// Tiny guard kernel: OR a flag into `overflow` if any of the `npages` nvCOMP
// statuses in `stat` is not nvcompSuccess. Run stream-ordered after each row
// group's batched decompress so a decompress failure is caught without a
// per-row-group host sync.
__global__ void check_stat_kernel(const int *stat, int npages, int *overflow) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (int i = 0; i < npages; i++)
        if (stat[i] != 0) { atomicExch(overflow, 1); return; }
}

// --- async-materialise pending context registry ---
//
// When the cold scan path populates _scan_cache, the materialise-gather of the
// 60M-row frame runs on a NON-BLOCKING side stream so the host can return the
// aggregate immediately (recovering the cold win). The materialise reads device
// scratch that the scan produced (decompressed pages `ubig`, dict indices
// `d_idxbig`, the per-RG pointer arrays `d_buf_all`/`d_dict_all`, the row-offset
// / frame-descriptor arrays, and the static PageSrc `kind/phys/scale` arrays).
// That scratch must outlive the C++ call -- it is moved into a PendingMatCtx,
// keyed by an int id returned to Python, and freed later by fused_scan_finalize
// (the ONLY freer of this scratch) once the warm path has waited on E_mat. A
// missed transfer -> UAF -> sticky context fault (breaks the cuDF fallback too),
// so the registry is the single source of truth for these allocations.
struct PendingMatCtx {
    uint8_t *ubig = nullptr;
    uint8_t *d_idxbig = nullptr;
    const void **d_buf_all = nullptr;
    const void **d_dict_all = nullptr;
    long long *d_row_off = nullptr;
    long long *d_frame_ptrs = nullptr;
    int *d_out_kind = nullptr;
    int *d_kind = nullptr;
    int *d_phys = nullptr;
    int *d_scale = nullptr;
    cudaStream_t stream2 = nullptr;
    cudaEvent_t E_mat = nullptr;
};
static std::unordered_map<int, PendingMatCtx> g_pending;
static std::atomic<int> g_pending_next{1};

// Wait for the side-stream materialise to finish (no-op if already done) and
// free every allocation in the pending ctx. Idempotent: a missing/already-
// finalized id is a silent no-op. Called from Python on the first warm read of
// the cached frame and from clear_scan_cache.
void fused_scan_finalize(int pending_id) {
    if (pending_id == 0) return;
    auto it = g_pending.find(pending_id);
    if (it == g_pending.end()) return;
    PendingMatCtx c = it->second;  // copy; erase before (potentially slow) sync
    g_pending.erase(it);
    if (c.E_mat) { cudaEventSynchronize(c.E_mat); cudaEventDestroy(c.E_mat); }
    if (c.stream2) cudaStreamDestroy(c.stream2);
    cudaFree(c.ubig); cudaFree(c.d_idxbig);
    cudaFree(c.d_buf_all); cudaFree(c.d_dict_all);
    cudaFree(c.d_row_off); cudaFree(c.d_frame_ptrs); cudaFree(c.d_out_kind);
    cudaFree(c.d_kind); cudaFree(c.d_phys); cudaFree(c.d_scale);
}

// ---------------- host entry: fused scan + aggregate ----------------
//
// Args:
//   path       : str -- parquet file path (mmap'd read-only on the host)
//   ncol,nrg   : column / row-group counts
//   chunk_off  : int64 [ncol*nrg] file offset of each (col,row-group) chunk's first page
//   chunk_total: int32 [ncol*nrg] total_compressed_size per chunk
//   chunk_nvals: int32 [ncol*nrg] row count per (col,row-group) (col-invariant per RG)
//   col_kind   : int32 [ncol] PK_*  col_phys: int32 [ncol] PHYS_*
//   col_scale  : int32 [ncol]       col_is_date: int32 [ncol]
//   (Plan descriptors as in fused_agg, minus col_ptrs/col_dtypes which are per-RG)
//   acc_init   : float64 (n_groups*nagg for DENSE; empty for HASH)
//   strategy   : 0=DENSE (global, n_groups==1), 1=HASH (single plain int64 key)
//   n_groups   : DENSE group count   capacity : HASH power-of-two capacity
//
// Returns (overflow:int, n_out:int, keys:py::list, aggs:py::list, pending_id:int).
//   overflow != 0 -> caller falls back to the cuDF path (correctness never
//   depends on this extension).
//   pending_id != 0 -> the materialise-gather is still running on a non-blocking
//   side stream; Python holds the id + frame buffers and calls
//   fused_scan_finalize(id) on the first warm read (or clear). pending_id == 0
//   means no async materialise (sync path / no populate / overflow / async-setup
//   failure) -- the frame, if any, is already fully written.
py::tuple fused_scan_agg(std::string path, int ncol, int nrg,
                         py::array_t<long long> chunk_off, py::array_t<int> chunk_total,
                         py::array_t<int> chunk_nvals,
                         py::array_t<int> col_kind, py::array_t<int> col_phys,
                         py::array_t<int> col_scale, py::array_t<int> col_is_date,
                         py::array_t<int> gkey_idx, py::array_t<long long> gkey_stride,
                         py::array_t<int> pred_col, py::array_t<int> pred_op,
                         py::array_t<double> pred_lit, py::array_t<int> agg_kind,
                         py::array_t<int> agg_tok_start, py::array_t<int> agg_tok_len,
                         py::array_t<int> tok_kind, py::array_t<int> tok_col,
                         py::array_t<double> tok_lit, py::array_t<int> tok_op,
                         py::array_t<double> acc_init,
                         py::array_t<long long> frame_ptrs, py::array_t<int> out_kind,
                         int strategy, int n_groups, int capacity) {
    int nagg = (int)agg_kind.shape(0);
    int ngkey = (int)gkey_idx.shape(0);
    int ntok = (int)tok_kind.shape(0);
    int n_pred = (int)pred_col.shape(0);

    // --- mmap the parquet file read-only ---
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("fused_scan_agg: open " + path);
    struct stat fs; fstat(fd, &fs);
    size_t file_len = (size_t)fs.st_size;
    const uint8_t *file_data = (const uint8_t *)mmap(nullptr, file_len, PROT_READ, MAP_PRIVATE, fd, 0);
    if (file_data == MAP_FAILED) { close(fd); throw std::runtime_error("fused_scan_agg: mmap"); }
#ifdef RYUDB_SCAN_PROFILE
    _ScanTimer _tm;
    fprintf(stderr, "[profile] file_len=%zu nrg=%d ncol=%d\n", file_len, nrg, ncol);
#endif

    auto co = chunk_off.request();  long long *co_h = static_cast<long long *>(co.ptr);
    auto ct = chunk_total.request(); int *ct_h = static_cast<int *>(ct.ptr);
    auto cn = chunk_nvals.request(); int *cn_h = static_cast<int *>(cn.ptr);
    auto ck = col_kind.request();    int *ck_h = static_cast<int *>(ck.ptr);
    auto cp = col_phys.request();    int *cp_h = static_cast<int *>(cp.ptr);

    // --- Plan descriptors to device (static across row groups) ---
    Plan p{};
    p.cols = nullptr; p.dtypes = nullptr;
    p.gkey_idx = np_dev(gkey_idx);
    p.gkey_stride = np_dev(gkey_stride);
    p.ngkey = ngkey;
    p.n_pred = n_pred;
    p.pred_col = np_dev(pred_col); p.pred_op = np_dev(pred_op); p.pred_lit = np_dev(pred_lit);
    p.nagg = nagg;
    p.agg_kind = np_dev(agg_kind); p.agg_tok_start = np_dev(agg_tok_start);
    p.agg_tok_len = np_dev(agg_tok_len);
    p.ntok = ntok;
    p.tok_kind = np_dev(tok_kind); p.tok_col = np_dev(tok_col);
    p.tok_lit = np_dev(tok_lit);   p.tok_op = np_dev(tok_op);

    // --- PageSrc: static per-col arrays (buf/dict updated per RG on the stream) ---
    int *d_kind = np_dev(col_kind), *d_phys = np_dev(col_phys);
    int *d_scale = np_dev(col_scale), *d_isdate = np_dev(col_is_date);
    const void **d_buf = nullptr, **d_dict = nullptr;
    check(cudaMalloc(&d_buf, sizeof(void *) * ncol), "malloc d_buf");
    check(cudaMalloc(&d_dict, sizeof(void *) * ncol), "malloc d_dict");
    // Per-RG host staging for the page/dict pointer arrays (same reason as the
    // nvCOMP metadata: cudaMemcpyAsync from a host source requires the source
    // to stay valid until the DMA completes -- a shared vector is raced by the
    // next row group, corrupting d_buf/d_dict and faulting the page kernel).
    std::vector<std::vector<const void *>> buf_host(nrg), dict_host(nrg);
    for (int rg = 0; rg < nrg; rg++) { buf_host[rg].resize(ncol); dict_host[rg].resize(ncol); }
    PageSrc s{};
    s.buf = d_buf; s.dict = d_dict; s.kind = d_kind; s.phys = d_phys;
    s.scale = d_scale; s.is_date = d_isdate;

    const int THREADS = 256;
    int overflow = 0, n_out = 0;
    int pending_id = 0;  // >0 if the materialise gather is still running async
    py::list keys_list, aggs_list;

    // Unified device overflow flag: a bad nvCOMP status, a dict-index run cap
    // overflow, or a HASH empty-key all atomicExch into this. One D2H after the
    // single final sync -- no per-row-group sync needed.
    int *d_overflow = nullptr; check(cudaMalloc(&d_overflow, sizeof(int)), "malloc ovf");
    check(cudaMemset(d_overflow, 0, sizeof(int)), "memset ovf");

    // --- persistent accumulator (shared across all row-group launches) ---
    double *acc = nullptr; int *seen = nullptr;
    long long *key = nullptr; int *distinct = nullptr;
    if (strategy == STRAT_DENSE) {
        int nga = n_groups * nagg;
        check(cudaMalloc(&acc, sizeof(double) * nga), "malloc acc");
        check(cudaMalloc(&seen, sizeof(int) * n_groups), "malloc seen");
        auto ii = acc_init.request();
        if (ii.ndim > 0 && ii.shape[0] > 0)
            check(cudaMemcpy(acc, ii.ptr, sizeof(double) * nga, cudaMemcpyHostToDevice), "cp acc_init");
        else
            check(cudaMemset(acc, 0, sizeof(double) * nga), "memset acc");
        check(cudaMemset(seen, 0, sizeof(int) * n_groups), "memset seen");
    } else {
        check(cudaMalloc(&key, sizeof(long long) * capacity), "malloc key");
        check(cudaMalloc(&acc, sizeof(double) * (size_t)capacity * nagg), "malloc acc hash");
        check(cudaMalloc(&distinct, sizeof(int)), "malloc distinct");
        check(cudaMemset(key, 0xFF, sizeof(long long) * capacity), "memset key");  // EMPTY = -1
        check(cudaMemset(acc, 0, sizeof(double) * (size_t)capacity * nagg), "memset acc hash");
        check(cudaMemset(distinct, 0, sizeof(int)), "memset distinct");
    }

    // --- pre-parse every (col,row-group) chunk's pages ONCE on the host. Page
    // offsets are not in file metadata so each chunk is walked, but this runs a
    // single time and sizes all scratch to the max row group up front -- the key
    // change from the v0 loop, which cudaMalloc'd/cudaFree'd ~15 buffers and
    // cudaDeviceSynchronize'd ~4 times PER row group (~2000 syncs over SF10). ---
    struct HPage { int col; int role; size_t file_off; int comp; int uncomp; };
    std::vector<std::vector<HPage>> rg_pages(nrg);
    std::vector<int> rg_npages(nrg, 0);
    size_t max_comp = 0, max_uncomp_page = 0;  // max_comp = per-RG aligned comp total (pin-ring slot)
    int max_n = 0;
    // Page sub-buffers are carved out of one big device buffer, so each page's
    // offset is 8-byte aligned (page_col_val reads int/int64 at the page start
    // and at the def-level-derived values offset; an unaligned carve would
    // cudaErrorMisalignedAddress). align8 pads the stride, not the size handed
    // to nvCOMP (which gets the true comp/uncomp byte count).
    auto align8 = [](size_t x) { return (x + 7u) & ~size_t(7); };
    for (int rg = 0; rg < nrg && overflow == 0; rg++) {
        max_n = std::max(max_n, cn_h[rg]);
        size_t comp_total = 0;
        for (int c = 0; c < ncol; c++) {
            std::vector<ryudb_pq::PageDesc> pages;
            try {
                pages = ryudb_pq::parse_column_chunk_pages(file_data, file_len,
                        (size_t)co_h[c * nrg + rg], ct_h[c * nrg + rg]);
            } catch (std::exception &) { overflow = 1; break; }
            int n_data = 0, n_dict = 0;
            for (auto &pg : pages) {
                if (pg.type != ryudb_pq::PT_DATA && pg.type != ryudb_pq::PT_DICT) { overflow = 1; break; }
                if (pg.type == ryudb_pq::PT_DATA) n_data++; else n_dict++;
                int role = (pg.type == ryudb_pq::PT_DICT) ? 1 : 0;
                rg_pages[rg].push_back({c, role, (size_t)pg.payload_off, pg.comp_size, pg.uncomp_size});
                comp_total += align8((size_t)pg.comp_size);
                max_uncomp_page = std::max(max_uncomp_page, (size_t)pg.uncomp_size);
            }
            if (overflow) break;
            if (n_data != 1 || n_dict > 1) { overflow = 1; break; }  // v1: 1 data page, 0/1 dict
        }
        if (overflow) break;
        rg_npages[rg] = (int)rg_pages[rg].size();
        max_comp = std::max(max_comp, comp_total);
    }

    // --- flatten all row-group pages into one flat array with global aligned
    // dbig/ubig offsets, so a SINGLE batched nvCOMP call can decompress every
    // page at once. nvCOMP parallelizes across chunks; the old per-RG ~7-chunk
    // calls left most of the 82 SMs idle (3861ms = 97% of SF10 Q6). dbig/ubig
    // now span every page's aligned slot instead of one row group's. ---
    size_t total_pages = 0;
    for (int rg = 0; rg < nrg; rg++) total_pages += (size_t)rg_npages[rg];
    std::vector<int> rg_start(nrg + 1, 0);
    for (int rg = 0; rg < nrg; rg++) rg_start[rg + 1] = rg_start[rg] + rg_npages[rg];
    // Per-RG global row offset (prefix sum of cn_h) for the materialise kernel:
    // frame[c][row_off[rg] + i] is the global row for RG-local row i. int64: this
    // is a TABLE-GLOBAL row offset, so it must hold total_rows (which exceeds
    // INT32 at SF>=~500, ~2.1B rows) -- the per-RG cn_h[rg] stays int (< 2^31/RG).
    std::vector<long long> row_off_h(nrg + 1, 0);
    for (int rg = 0; rg < nrg; rg++) row_off_h[rg + 1] = row_off_h[rg] + cn_h[rg];
    std::vector<HPage> all_pages(total_pages);
    std::vector<size_t> g_comp_off(total_pages), g_up_off(total_pages);
    size_t sum_comp = 0, sum_uncomp = 0;
    for (int rg = 0; rg < nrg; rg++) {
        for (int i = 0; i < rg_npages[rg]; i++) {
            size_t gi = (size_t)rg_start[rg] + i;
            all_pages[gi] = rg_pages[rg][i];
            g_comp_off[gi] = sum_comp;    sum_comp   += align8((size_t)all_pages[gi].comp);
            g_up_off[gi]   = sum_uncomp;  sum_uncomp += align8((size_t)all_pages[gi].uncomp);
        }
    }

    // --- dict-job flatten: one decode job per (row group, dict column) data
    // page. v1 guarantees exactly 1 data page per col chunk, so each such page
    // gets a permanent index-array slot in d_idxbig and a permanent run-table
    // slot in d_runs, decoded ONCE by the two batched kernels (instead of
    // scan_runs+apply_runs launched per dict col per RG -- ~1467 serial launches
    // for Q6 SF10). dict_slot[rg*ncol+c] maps a (RG, dict col) to its slot (-1
    // for PLAIN cols); dict_data_upoff/dict_n_arr describe each job. ---
    std::vector<int> dict_slot((size_t)nrg * ncol, -1);
    std::vector<size_t> dict_data_upoff;
    std::vector<int> dict_n_arr;
    for (int rg = 0; rg < nrg; rg++) {
        for (int c = 0; c < ncol; c++) {
            if (ck_h[c] != PK_DICT_NUMERIC_ARG) continue;
            for (int i = 0; i < rg_npages[rg]; i++) {
                size_t gi = (size_t)rg_start[rg] + i;
                if (all_pages[gi].col == c && all_pages[gi].role == 0) {  // the dict DATA page (indices)
                    dict_slot[(size_t)rg * ncol + c] = (int)dict_data_upoff.size();
                    dict_data_upoff.push_back(g_up_off[gi]);
                    dict_n_arr.push_back(cn_h[rg]);
                    break;  // v1: one data page per col chunk
                }
            }
        }
    }
    size_t total_dict = dict_data_upoff.size();  // ~1467 for Q6 SF10
#ifdef RYUDB_SCAN_PROFILE
    _tm.mark("preparse+plan");
#endif

    // --- one stream + scratch allocated once to the max row group (no per-RG
    // malloc/free). Everything below is stream-ordered with a single sync. ---
    // BATCH bounds how many row groups stay in flight on the stream. It sizes
    // the gather-fallback pin ring (below) and the periodic drain sync. It is
    // declared here, before the scratch allocation, so the pin ring can be sized
    // off it.
    const int BATCH = 16;
    cudaStream_t stream = nullptr;
    uint8_t *dbig = nullptr, *ubig = nullptr, *d_idxbig = nullptr;
    void *pin = nullptr;  // BATCH-slot ring of pinned host staging (gather path)
    const void **d_cptrs = nullptr; size_t *d_cbytes = nullptr, *d_ubytes = nullptr, *d_actual = nullptr;
    void **d_uptrs = nullptr; int *d_stat = nullptr;
    void *d_temp = nullptr;
    RunEntry *d_runs = nullptr; int *d_nruns = nullptr, *d_bw = nullptr;
    const void **d_dict_data = nullptr; int *d_dict_n = nullptr;  // batched dict-decode job arrays
    long long *d_frame_ptrs = nullptr; int *d_out_kind = nullptr; long long *d_row_off = nullptr;  // materialise-gather
    // async materialise: side stream + events + stable per-RG pointer arrays.
    // stream2 is NON-BLOCKING so the default-stream read-out does not implicitly
    // wait for the materialise (legacy streams sync with the default stream --
    // that would re-serialize the cold return). d_buf_all/d_dict_all hold all
    // nrg per-RG pointer arrays so the post-loop gather reads stable slots
    // instead of the shared d_buf/d_dict (overwritten each RG).
    bool async_mat = false;
    cudaStream_t stream2 = nullptr;
    cudaEvent_t E_page = nullptr, E_mat = nullptr;
    const void **d_buf_all = nullptr, **d_dict_all = nullptr;
    bool pinned_mmap = false;
    nvcompBatchedSnappyDecompressOpts_t opts = nvcompBatchedSnappyDecompressDefaultOpts;

    if (overflow == 0) {
        check(cudaStreamCreate(&stream), "stream");
        // Pin the mmap'd file so per-page async H2D overlaps GPU decode. If the
        // registration fails (very large mappings, e.g. WSL2 pinning a 2.2GB
        // file), fall back to a pinned staging gather per row group. A failed
        // register leaves a sticky CUDA error -- drain it so a later check()
        // does not misreport it.
        pinned_mmap = (cudaHostRegister((void *)file_data, file_len,
                                         cudaHostRegisterDefault) == cudaSuccess);
        if (!pinned_mmap) {
            cudaGetLastError();
            // Ring of BATCH pinned slots: each in-flight row group gathers into
            // its own slot so an async H2D never reads a buffer the next row
            // group overwrites. The batch sync below guarantees slot rg % BATCH
            // is reused only after its H2D has completed.
            check(cudaMallocHost(&pin, (size_t)BATCH * max_comp), "malloc pin");
        }
        check(cudaMalloc(&dbig, sum_comp), "malloc dbig");
        check(cudaMalloc(&ubig, sum_uncomp), "malloc ubig");
        // Per-page dict-decode scratch: one index array + one run table per dict
        // data page (slot s at d_idxbig + s*max_n / d_runs + s*PQ_RUNS_CAP).
        if (total_dict > 0 && max_n > 0)
            check(cudaMalloc(&d_idxbig, total_dict * (size_t)max_n * sizeof(int)), "malloc idxbig");
        check(cudaMalloc(&d_cptrs, sizeof(void *) * total_pages), "m cptrs");
        check(cudaMalloc(&d_cbytes, sizeof(size_t) * total_pages), "m cbytes");
        check(cudaMalloc(&d_ubytes, sizeof(size_t) * total_pages), "m ubytes");
        check(cudaMalloc(&d_actual, sizeof(size_t) * total_pages), "m actual");
        check(cudaMalloc(&d_uptrs, sizeof(void *) * total_pages), "m uptrs");
        check(cudaMalloc(&d_stat, sizeof(int) * total_pages), "m stat");
        if (total_dict > 0) {
            check(cudaMalloc(&d_runs, total_dict * (size_t)PQ_RUNS_CAP * sizeof(RunEntry)), "malloc runs");
            check(cudaMalloc(&d_nruns, sizeof(int) * total_dict), "malloc nruns");
            check(cudaMalloc(&d_bw, sizeof(int) * total_dict), "malloc bw");
            check(cudaMalloc(&d_dict_data, sizeof(void *) * total_dict), "malloc dict_data");
            check(cudaMalloc(&d_dict_n, sizeof(int) * total_dict), "malloc dict_n");
        }
        // nvCOMP temp for the single batched call over ALL pages (host query).
        size_t max_temp = 0;
        if (nvcompBatchedSnappyDecompressGetTempSizeAsync(total_pages, max_uncomp_page,
                opts, &max_temp, sum_uncomp) != nvcompSuccess)
            max_temp = 0;
        if (max_temp) check(cudaMalloc(&d_temp, max_temp), "malloc temp");

        // --- materialise-gather frame buffers. d_row_off is always allocated
        // (cheap); d_frame_ptrs/d_out_kind only when the caller passed a
        // non-empty frame_ptrs (the scan path populates _scan_cache). Empty ->
        // no materialise launch, the path is identical to today. The frame
        // buffers themselves are Python-owned cuDF columns -- NOT freed here. ---
        check(cudaMalloc(&d_row_off, sizeof(long long) * (nrg + 1)), "malloc row_off");
        check(cudaMemcpyAsync(d_row_off, row_off_h.data(), sizeof(long long) * (nrg + 1),
                              cudaMemcpyHostToDevice, stream), "cp row_off");
        {
            auto fp = frame_ptrs.request();
            if (fp.ndim > 0 && fp.shape[0] > 0) {
                check(cudaMalloc(&d_frame_ptrs, sizeof(long long) * ncol), "malloc frame_ptrs");
                check(cudaMemcpyAsync(d_frame_ptrs, fp.ptr, sizeof(long long) * ncol,
                                      cudaMemcpyHostToDevice, stream), "cp frame_ptrs");
                auto okd = out_kind.request();
                check(cudaMalloc(&d_out_kind, sizeof(int) * ncol), "malloc out_kind");
                check(cudaMemcpyAsync(d_out_kind, okd.ptr, sizeof(int) * ncol,
                                      cudaMemcpyHostToDevice, stream), "cp out_kind");
            }
        }
        // Decide async materialise: only when populating the cache (d_frame_ptrs
        // allocated) and not disabled via RYUDB_ASYNC_MATERIALISE=0. Create a
        // non-blocking side stream + two sync-only events; on ANY failure drop to
        // the synchronous in-loop materialise (pending_id stays 0).
        if (d_frame_ptrs) {
            const char *am = std::getenv("RYUDB_ASYNC_MATERIALISE");
            bool want_async = !(am && am[0] == '0' && am[1] == '\0');
            if (want_async &&
                cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking) == cudaSuccess &&
                cudaEventCreateWithFlags(&E_page, cudaEventDisableTiming) == cudaSuccess &&
                cudaEventCreateWithFlags(&E_mat, cudaEventDisableTiming) == cudaSuccess) {
                async_mat = true;
            } else {
                if (stream2) { cudaStreamDestroy(stream2); stream2 = nullptr; }
                if (E_page)  { cudaEventDestroy(E_page);  E_page = nullptr; }
                if (E_mat)   { cudaEventDestroy(E_mat);   E_mat = nullptr; }
                cudaGetLastError();  // drain a sticky error so a later check() is clean
                async_mat = false;
            }
        }

        // --- global nvCOMP metadata: one flat array over ALL row-group pages,
        // built and H2D'd once. Replaces the per-RG hp/hc/hu/hup vectors: a
        // single batched decompress needs one set of device-resident ptr/size
        // arrays spanning every page, not 489 per-RG sets of ~7. ---
        std::vector<const void *> hp(total_pages);
        std::vector<size_t> hc(total_pages), hu(total_pages);
        std::vector<void *> hup(total_pages);
        for (int rg = 0; rg < nrg; rg++) {
            for (int i = 0; i < rg_npages[rg]; i++) {
                size_t gi = (size_t)rg_start[rg] + i;
                const HPage &pg = all_pages[gi];
                hp[gi]  = dbig + g_comp_off[gi];
                hc[gi]  = (size_t)pg.comp;     // true size handed to nvCOMP
                hu[gi]  = (size_t)pg.uncomp;   // capacity (<= align8 slot in ubig)
                hup[gi] = ubig + g_up_off[gi];
            }
        }
        check(cudaMemcpyAsync(d_cptrs, hp.data(),  sizeof(void *) * total_pages, cudaMemcpyHostToDevice, stream), "cp cptrs");
        check(cudaMemcpyAsync(d_cbytes, hc.data(), sizeof(size_t) * total_pages, cudaMemcpyHostToDevice, stream), "cp cbytes");
        check(cudaMemcpyAsync(d_ubytes, hu.data(), sizeof(size_t) * total_pages, cudaMemcpyHostToDevice, stream), "cp ubytes");
        check(cudaMemcpyAsync(d_uptrs, hup.data(), sizeof(void *) * total_pages, cudaMemcpyHostToDevice, stream), "cp uptrs");

        // --- H2D all compressed page payloads into dbig (stream-ordered; the
        // single nvCOMP is queued after on the same stream, so no sync is needed
        // between H2D and decompress). Sync every BATCH row groups only to keep
        // the pin-ring slot reuse safe (gather path) and the launch queue
        // bounded -- queuing ~3400 async copies with no drain exhausts the
        // driver descriptor pool (cudaErrorMemoryAllocation). ---
        for (int rg = 0; rg < nrg && overflow == 0; rg++) {
            int npages = rg_npages[rg];
            size_t base = (size_t)rg_start[rg];
            if (pinned_mmap) {
                for (int i = 0; i < npages; i++) {
                    size_t gi = base + i;
                    check(cudaMemcpyAsync(dbig + g_comp_off[gi], file_data + all_pages[gi].file_off,
                                          all_pages[gi].comp, cudaMemcpyHostToDevice, stream), "cp comp");
                }
            } else {
                // Gather this RG's compressed pages into its pin-ring slot, then
                // one async H2D into the RG's global dbig slice. The slot is
                // exclusive to in-flight RG rg % BATCH; the batch sync below
                // guarantees reuse only after the H2D has completed.
                uint8_t *pinslot = (uint8_t *)pin + (size_t)(rg % BATCH) * max_comp;
                size_t off = 0;
                for (int i = 0; i < npages; i++) {
                    size_t gi = base + i;
                    memcpy(pinslot + off, file_data + all_pages[gi].file_off, all_pages[gi].comp);
                    off += align8((size_t)all_pages[gi].comp);
                }
                check(cudaMemcpyAsync(dbig + g_comp_off[base], pinslot, off,
                                      cudaMemcpyHostToDevice, stream), "cp comp");
            }
            if ((rg + 1) % BATCH == 0) check(cudaStreamSynchronize(stream), "batch h2d sync");
        }
#ifdef RYUDB_SCAN_PROFILE
        check(cudaStreamSynchronize(stream), "prof h2d sync");
        _tm.mark("h2d-comp");
#endif

        // --- ONE batched Snappy decompress over every page, then a guard kernel
        // that ORs the per-page statuses into d_overflow (stream-ordered, no
        // host sync). This fills the GPU: ~3400 chunks instead of 489x7. ---
        if (overflow == 0) {
            if (nvcompBatchedSnappyDecompressAsync(d_cptrs, d_cbytes, d_ubytes, d_actual, total_pages,
                    d_temp, max_temp, d_uptrs, opts, (nvcompStatus_t *)d_stat, stream) != nvcompSuccess)
                overflow = 1;
            check_stat_kernel<<<1, 1, 0, stream>>>(d_stat, (int)total_pages, d_overflow);
        }
#ifdef RYUDB_SCAN_PROFILE
        check(cudaStreamSynchronize(stream), "prof nvcomp sync");
        _tm.mark("nvcomp");
#endif

        // --- ONE batched dict-index decode over all dict data pages (two launches
        // instead of ~1467 serial scan_runs+apply_runs pairs). Each dict page gets
        // a permanent index slot in d_idxbig; the per-RG page loop below just
        // points PageSrc.buf at the pre-decoded slot. Stream-ordered before the
        // page kernels, so no sync needed here. ---
        if (overflow == 0 && total_dict > 0) {
            std::vector<const void *> h_dict_data(total_dict);
            for (size_t s = 0; s < total_dict; s++) h_dict_data[s] = ubig + dict_data_upoff[s];
            check(cudaMemcpyAsync(d_dict_data, h_dict_data.data(), sizeof(void *) * total_dict, cudaMemcpyHostToDevice, stream), "cp dict_data");
            check(cudaMemcpyAsync(d_dict_n, dict_n_arr.data(), sizeof(int) * total_dict, cudaMemcpyHostToDevice, stream), "cp dict_n");
            scan_runs_batched_kernel<<<(int)total_dict, 1, 0, stream>>>(d_dict_data, d_dict_n, d_runs, d_nruns, d_overflow, d_bw, total_dict);
            int blocks_per_job = (max_n + THREADS - 1) / THREADS; if (blocks_per_job > 65535) blocks_per_job = 65535;
            apply_runs_batched_kernel<<<dim3(blocks_per_job, (int)total_dict), THREADS, 0, stream>>>(
                d_dict_data, d_dict_n, d_runs, d_nruns, d_bw, (int *)d_idxbig, max_n);
            check(cudaGetLastError(), "dict decode launch");
        }
#ifdef RYUDB_SCAN_PROFILE
        check(cudaStreamSynchronize(stream), "prof dict sync");
        _tm.mark("dict-decode");
#endif

        // ---------- per row group: page kernel only (stream-ordered) ----------
        // Dict index arrays were decoded ONCE above (batched), each into its own
        // permanent d_idxbig slot -- no per-RG dict decode, no d_runs reuse, no
        // race. This loop just wires PageSrc (buf = pre-decoded slot or PLAIN
        // page; dict = dict page) and launches the page kernel. Bound the queue
        // with the every-BATCH sync; the final cudaDeviceSynchronize handles tail.
        for (int rg = 0; rg < nrg && overflow == 0; rg++) {
            int n = cn_h[rg], npages = rg_npages[rg];
            size_t base = (size_t)rg_start[rg];

            // Build PageSrc per column. Dict cols get an on-device RLE/bit-packed
            // index decode (scan_runs writes bit width to d_bw, apply_runs reads
            // it -- stream-ordered, no D2H); PLAIN cols read values at the def-
            // level-derived offset computed inline in the page kernel. Each
            // page's decompressed buffer is its global ubig slice.
            for (int c = 0; c < ncol; c++) {
                const uint8_t *data_ptr = nullptr, *dict_ptr = nullptr;
                for (int i = 0; i < npages; i++) {
                    size_t gi = base + i;
                    if (all_pages[gi].col == c) {
                        if (all_pages[gi].role == 1) dict_ptr = ubig + g_up_off[gi];
                        else                         data_ptr = ubig + g_up_off[gi];
                    }
                }
                if (ck_h[c] == PK_PLAIN_RAW) {
                    buf_host[rg][c] = data_ptr; dict_host[rg][c] = nullptr;
                } else {  // PK_DICT_NUMERIC_ARG: index array pre-decoded by the
                          // batched scan_runs/apply_runs above -- just point at its slot.
                    int slot = dict_slot[(size_t)rg * ncol + c];
                    int *idx_arr = (int *)(d_idxbig + (size_t)slot * max_n * sizeof(int));
                    buf_host[rg][c] = idx_arr; dict_host[rg][c] = dict_ptr;
                }
            }
            check(cudaMemcpyAsync(d_buf, buf_host[rg].data(), sizeof(void *) * ncol, cudaMemcpyHostToDevice, stream), "cp buf");
            check(cudaMemcpyAsync(d_dict, dict_host[rg].data(), sizeof(void *) * ncol, cudaMemcpyHostToDevice, stream), "cp dict");
            p.n = n;
            int blocks = (n + THREADS - 1) / THREADS; if (blocks > 65535) blocks = 65535;
            if (strategy == STRAT_DENSE) {
                size_t shbytes = sizeof(double) * n_groups * nagg;
                page_dense_kernel<<<blocks, THREADS, shbytes, stream>>>(p, s, acc, seen, n_groups, nagg);
            } else {
                page_hash_kernel<<<blocks, THREADS, 0, stream>>>(p, s, key, acc, capacity, nagg, distinct, d_overflow);
            }
            check(cudaGetLastError(), "page launch");
            // Materialise this RG's decoded columns into the frame buffer at its
            // global row offset, so the scan path can cache the GPU-resident
            // frame for warm repeats. Reuses this RG's PageSrc s (d_buf/d_dict
            // copied earlier this same stream iteration; same-stream ordering
            // makes the reads safe). Skipped when no frame_ptrs were passed, and
            // when async_mat (the gather runs post-loop on the side stream).
            if (d_frame_ptrs && !async_mat) {
                materialise_kernel<<<blocks, THREADS, 0, stream>>>(
                    s, d_frame_ptrs, d_out_kind, d_row_off, rg, n, ncol);
                check(cudaGetLastError(), "materialise launch");
            }
            if ((rg + 1) % BATCH == 0) check(cudaStreamSynchronize(stream), "batch sync");
        }  // end row-group loop

        // --- async materialise: gather the frame on the non-blocking side stream
        // so the host can return the aggregate NOW and the warm path waits on
        // E_mat. The sync path already gathered in-loop above. Skipped on overflow
        // (never cache a partial frame -- Python returns None on overflow anyway).
        if (async_mat && overflow == 0) {
            // Stable per-RG pointer arrays: flatten all nrg buf_host/dict_host
            // slots into one H2D each on `stream`, so the side-stream gather reads
            // RG rg's pointers at d_buf_all + rg*ncol (not the shared d_buf, which
            // holds only the last RG by now). The page data (ubig/d_idxbig) and
            // the descriptors (d_row_off/d_frame_ptrs/d_out_kind/d_kind/d_phys/
            // d_scale) were all produced on `stream` -- E_page captures them.
            std::vector<const void *> flat_buf((size_t)nrg * ncol);
            std::vector<const void *> flat_dict((size_t)nrg * ncol);
            for (int rg = 0; rg < nrg; rg++)
                for (int c = 0; c < ncol; c++) {
                    flat_buf[(size_t)rg * ncol + c]  = buf_host[rg][c];
                    flat_dict[(size_t)rg * ncol + c] = dict_host[rg][c];
                }
            check(cudaMalloc(&d_buf_all,  sizeof(void *) * (size_t)nrg * ncol), "malloc buf_all");
            check(cudaMalloc(&d_dict_all, sizeof(void *) * (size_t)nrg * ncol), "malloc dict_all");
            check(cudaMemcpyAsync(d_buf_all,  flat_buf.data(),
                                  sizeof(void *) * (size_t)nrg * ncol, cudaMemcpyHostToDevice, stream), "cp buf_all");
            check(cudaMemcpyAsync(d_dict_all, flat_dict.data(),
                                  sizeof(void *) * (size_t)nrg * ncol, cudaMemcpyHostToDevice, stream), "cp dict_all");
            check(cudaEventRecord(E_page, stream), "rec E_page");
            check(cudaStreamWaitEvent(stream2, E_page, 0), "wait E_page");
            for (int rg = 0; rg < nrg; rg++) {
                int n = cn_h[rg];
                int blocks = (n + THREADS - 1) / THREADS; if (blocks > 65535) blocks = 65535;
                PageSrc s_rg = s;
                s_rg.buf  = (const void **)(d_buf_all)  + (size_t)rg * ncol;
                s_rg.dict = (const void **)(d_dict_all) + (size_t)rg * ncol;
                materialise_kernel<<<blocks, THREADS, 0, stream2>>>(
                    s_rg, d_frame_ptrs, d_out_kind, d_row_off, rg, n, ncol);
                check(cudaGetLastError(), "materialise async launch");
            }
            check(cudaEventRecord(E_mat, stream2), "rec E_mat");
            // Transfer ownership of every device allocation the gather reads to
            // the pending registry; null the locals so the cleanup below's
            // cudaFree(nullptr) skips them (the registry frees them in
            // fused_scan_finalize, the sole freer).
            PendingMatCtx ctx;
            ctx.ubig = ubig; ubig = nullptr;
            ctx.d_idxbig = d_idxbig; d_idxbig = nullptr;
            ctx.d_buf_all = d_buf_all; d_buf_all = nullptr;
            ctx.d_dict_all = d_dict_all; d_dict_all = nullptr;
            ctx.d_row_off = d_row_off; d_row_off = nullptr;
            ctx.d_frame_ptrs = d_frame_ptrs; d_frame_ptrs = nullptr;
            ctx.d_out_kind = d_out_kind; d_out_kind = nullptr;
            ctx.d_kind = d_kind; d_kind = nullptr;
            ctx.d_phys = d_phys; d_phys = nullptr;
            ctx.d_scale = d_scale; d_scale = nullptr;
            ctx.stream2 = stream2; stream2 = nullptr;
            ctx.E_mat = E_mat; E_mat = nullptr;
            pending_id = g_pending_next.fetch_add(1);
            g_pending[pending_id] = ctx;
        }

        // Sync ONLY the main scan stream in async mode (the materialise on the
        // non-blocking stream2 is allowed to continue past the return). In sync
        // mode a device sync waits for the in-loop materialise as before.
        if (async_mat) check(cudaStreamSynchronize(stream), "final stream sync");
        else            check(cudaDeviceSynchronize(), "final sync");
#ifdef RYUDB_SCAN_PROFILE
        _tm.mark("stream-loop+sync");
#endif
        if (overflow == 0) {
            int h_ovf = 0; check(cudaMemcpy(&h_ovf, d_overflow, sizeof(int), cudaMemcpyDeviceToHost), "cp ovf");
            overflow = h_ovf;
        }
    }

    // --- read-out (identical to fused_agg) ---
    if (overflow == 0) {
        if (strategy == STRAT_DENSE) {
            int nga = n_groups * nagg;
            std::vector<double> h_acc(nga); std::vector<int> h_seen(n_groups);
            check(cudaMemcpy(h_acc.data(), acc, sizeof(double) * nga, cudaMemcpyDeviceToHost), "cp acc ro");
            check(cudaMemcpy(h_seen.data(), seen, sizeof(int) * n_groups, cudaMemcpyDeviceToHost), "cp seen ro");
            for (int g = 0; g < n_groups; g++) if (h_seen[g]) n_out++;
            std::vector<std::vector<long long>> h_keys(ngkey, std::vector<long long>(n_out));
            std::vector<std::vector<double>> h_aggs(nagg, std::vector<double>(n_out));
            std::vector<long long> stride(ngkey);
            { auto sinfo = gkey_stride.request(); long long *sp = static_cast<long long *>(sinfo.ptr);
              for (int j = 0; j < ngkey; j++) stride[j] = sp[j]; }
            int row = 0;
            for (int g = 0; g < n_groups; g++) {
                if (!h_seen[g]) continue;
                long long rem = g;
                for (int j = 0; j < ngkey; j++) { h_keys[j][row] = rem / stride[j]; rem = rem % stride[j]; }
                for (int a = 0; a < nagg; a++) h_aggs[a][row] = h_acc[g * nagg + a];
                row++;
            }
            for (int j = 0; j < ngkey; j++)
                keys_list.append(py::array_t<long long>(n_out, h_keys[j].data()));
            for (int a = 0; a < nagg; a++)
                aggs_list.append(py::array_t<double>(n_out, h_aggs[a].data()));
        } else {
            // overflow was already read from d_overflow after the final sync.
            int h_distinct = 0;
            check(cudaMemcpy(&h_distinct, distinct, sizeof(int), cudaMemcpyDeviceToHost), "cp distinct ro");
            n_out = h_distinct;
            if (overflow == 0 && n_out > 0) {
                long long *out_keys = nullptr; double *out_acc = nullptr; int *counter = nullptr;
                check(cudaMalloc(&out_keys, sizeof(long long) * n_out), "malloc ok");
                check(cudaMalloc(&out_acc, sizeof(double) * (size_t)n_out * nagg), "malloc oa");
                check(cudaMalloc(&counter, sizeof(int)), "malloc ctr");
                check(cudaMemset(counter, 0, sizeof(int)), "memset ctr");
                int cblocks = (capacity + THREADS - 1) / THREADS; if (cblocks > 65535) cblocks = 65535;
                compact_kernel<<<cblocks, THREADS>>>(key, acc, capacity, nagg, out_keys, out_acc, counter);
                // Sync only the default stream (compact_kernel runs here). A
                // device sync would also wait for the async materialise on the
                // non-blocking stream2, re-serializing the cold return.
                if (async_mat) check(cudaStreamSynchronize(0), "compact stream sync");
                else            check(cudaDeviceSynchronize(), "compact sync");
                std::vector<long long> h_keys(n_out); std::vector<double> h_acc2((size_t)n_out * nagg);
                check(cudaMemcpy(h_keys.data(), out_keys, sizeof(long long) * n_out, cudaMemcpyDeviceToHost), "cp ok");
                check(cudaMemcpy(h_acc2.data(), out_acc, sizeof(double) * n_out * nagg, cudaMemcpyDeviceToHost), "cp oa");
                keys_list.append(py::array_t<long long>(n_out, h_keys.data()));
                for (int a = 0; a < nagg; a++) {
                    std::vector<double> col(n_out);
                    for (int r = 0; r < n_out; r++) col[r] = h_acc2[(size_t)r * nagg + a];
                    aggs_list.append(py::array_t<double>(n_out, col.data()));
                }
                cudaFree(out_keys); cudaFree(out_acc); cudaFree(counter);
            }
        }
    }

    // --- free everything (cudaFree(nullptr) / cudaFreeHost(nullptr) are no-ops) ---
#ifdef RYUDB_SCAN_PROFILE
    _tm.mark("readout");
#endif
    if (pinned_mmap) cudaHostUnregister((void *)file_data);
    if (stream) cudaStreamDestroy(stream);
    // async-materialise resources not transferred to the pending registry: in
    // async-success they were moved to the ctx (null here -> no-op); in
    // async-overflow or sync mode they're still local and must be freed here.
    // E_page is never registered (it only orders stream2 after `stream`).
    if (E_page)  cudaEventDestroy(E_page);
    if (stream2) cudaStreamDestroy(stream2);
    if (E_mat)   cudaEventDestroy(E_mat);
    cudaFreeHost(pin); cudaFree(dbig); cudaFree(ubig); cudaFree(d_idxbig);
    cudaFree(d_cptrs); cudaFree(d_cbytes); cudaFree(d_ubytes); cudaFree(d_actual);
    cudaFree(d_uptrs); cudaFree(d_stat); cudaFree(d_temp);
    cudaFree(d_runs); cudaFree(d_nruns); cudaFree(d_bw); cudaFree(d_dict_data); cudaFree(d_dict_n); cudaFree(d_overflow);
    cudaFree(d_frame_ptrs); cudaFree(d_out_kind); cudaFree(d_row_off);
    cudaFree(d_buf_all); cudaFree(d_dict_all);
    cudaFree(d_buf); cudaFree(d_dict);
    cudaFree((void *)d_kind); cudaFree((void *)d_phys);
    cudaFree((void *)d_scale); cudaFree((void *)d_isdate);
    if (acc) cudaFree(acc); if (seen) cudaFree(seen); if (key) cudaFree(key);
    if (distinct) cudaFree(distinct);
    cudaFree((void *)p.gkey_idx); cudaFree((void *)p.gkey_stride); cudaFree((void *)p.pred_col);
    cudaFree((void *)p.pred_op); cudaFree((void *)p.pred_lit); cudaFree((void *)p.agg_kind);
    cudaFree((void *)p.agg_tok_start); cudaFree((void *)p.agg_tok_len); cudaFree((void *)p.tok_kind);
    cudaFree((void *)p.tok_col); cudaFree((void *)p.tok_lit); cudaFree((void *)p.tok_op);
    munmap((void *)file_data, file_len); close(fd);

    return py::make_tuple(overflow, n_out, keys_list, aggs_list, pending_id);
}

// Testability hook: mmap `path`, parse the page headers of one column chunk
// (at file offset `chunk_off`, spanning `total_compressed_size` bytes), and
// return a list of (type, comp_size, num_values, value_encoding) tuples -- one
// per page. Lets the pytest suite assert the Thrift parser's invariants
// (sum(comp_size) == total_compressed_size, sum(data-page num_values) ==
// cc.num_values) without re-implementing the parser in Python. Throws on any
// parse error (mirrors how fused_scan_agg defers a bad chunk).
py::list pqpages_probe(std::string path, long long chunk_off, int total_compressed_size) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("pqpages_probe: open " + path);
    struct stat fs; fstat(fd, &fs);
    size_t file_len = (size_t)fs.st_size;
    const uint8_t *file_data = (const uint8_t *)mmap(nullptr, file_len, PROT_READ, MAP_PRIVATE, fd, 0);
    if (file_data == MAP_FAILED) { close(fd); throw std::runtime_error("pqpages_probe: mmap"); }
    std::vector<ryudb_pq::PageDesc> pages;
    try {
        pages = ryudb_pq::parse_column_chunk_pages(file_data, file_len,
                                                   (size_t)chunk_off, total_compressed_size);
    } catch (...) {
        munmap((void *)file_data, file_len); close(fd);
        throw;
    }
    munmap((void *)file_data, file_len); close(fd);
    py::list out;
    for (const auto &pg : pages)
        out.append(py::make_tuple(pg.type, pg.header_len, pg.comp_size,
                                  pg.num_values, pg.value_encoding));
    return out;
}

PYBIND11_MODULE(fused, m) {
    m.def("fused_agg", &fused_agg);
    m.def("fused_join_agg", &fused_join_agg);
    m.def("fused_scan_agg", &fused_scan_agg);
    m.def("fused_scan_finalize", &fused_scan_finalize);
    m.def("pqpages_probe", &pqpages_probe);
}