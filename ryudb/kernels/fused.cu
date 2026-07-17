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
static constexpr int AGG_COUNT = 0, AGG_SUM = 1;
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
__global__ void dense_kernel(Plan p, double *acc, int *seen, int n_groups, int nagg) {
    extern __shared__ double sh[];
    int t = threadIdx.x;
    int nga = n_groups * nagg;
    for (int k = t; k < nga; k += blockDim.x) sh[k] = 0.0;
    __syncthreads();
    for (int i = blockIdx.x * blockDim.x + t; i < p.n; i += gridDim.x * blockDim.x) {
        if (!pass_pred(p, i)) continue;
        long long g = 0;
        for (int j = 0; j < p.ngkey; j++)
            g += ((const long long *)p.cols[p.gkey_idx[j]])[i] * p.gkey_stride[j];
        if (g < 0 || g >= n_groups) continue;
        for (int a = 0; a < p.nagg; a++) {
            double val = p.agg_kind[a] == AGG_COUNT ? 1.0 : eval_agg(p, i, a);
            atomicAdd(&sh[g * nagg + a], val);
        }
        atomicMax(&seen[g], 1);
    }
    __syncthreads();
    for (int k = t; k < nga; k += blockDim.x) atomicAdd(&acc[k], sh[k]);
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
//   strategy  : int (0=DENSE,1=HASH)
//   n_groups  : int (DENSE)   capacity : int (HASH, power of two)
//
// Returns a tuple (overflow:int, n_out:int, keys:py::list[int64 arrays],
//                  aggs:py::list[float64 arrays]).
//   overflow != 0 means the hash table filled -> caller falls back to cuDF.
//   keys[i] is the int64 code/value column for group key i (n_out rows);
//   aggs[a] is the float64 accumulator column for aggregate a (n_out rows).
py::tuple fused_agg(py::array_t<long long> col_ptrs, py::array_t<int> col_dtypes,
                    py::array_t<int> gkey_idx, py::array_t<long long> gkey_stride,
                    py::array_t<int> pred_col, py::array_t<int> pred_op,
                    py::array_t<double> pred_lit, py::array_t<int> agg_kind,
                    py::array_t<int> agg_tok_start, py::array_t<int> agg_tok_len,
                    py::array_t<int> tok_kind, py::array_t<int> tok_col,
                    py::array_t<double> tok_lit, py::array_t<int> tok_op,
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
        check(cudaMemset(acc, 0, sizeof(double) * nga), "memset acc");
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

PYBIND11_MODULE(fused, m) { m.def("fused_agg", &fused_agg); }