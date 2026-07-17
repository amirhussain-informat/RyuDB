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
#include <cstdint>

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

PYBIND11_MODULE(fused, m) {
    m.def("fused_agg", &fused_agg);
    m.def("fused_join_agg", &fused_join_agg);
}