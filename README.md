# RyuDB

A GPU-powered HTAP RDBMS. Phase 1: a read-only GPU analytical SQL engine over
columnar Parquet storage, executing on NVIDIA GPUs via RAPIDS **cuDF**.

## Architecture

```
SQL Frontend (sqlglot)  ->  Logical Plan  ->  Rule-based Optimizer
                     ->  Physical Plan  ->  cuDF Executor (GPU)
                     ->  Parquet storage (columnar)
```

- **Parser:** `sqlglot` parses SQL into an AST we lower to relational algebra.
- **Planner/Optimizer:** custom. Predicate pushdown, projection pruning, join
  reordering by estimated cardinality.
- **Executor:** lowers the physical plan to `cudf` operations (scan, filter,
  join, groupby/aggregate, sort, limit) running on the GPU.
- **Storage:** columnar Parquet files; a catalog maps table names to files and
  Arrow schemas.

Phase 2 adds a CPU delta-store + WAL + MVCC for writes (HTAP), over an
**immutable Parquet base** (writes go to a delta-store merged at read; base
`.parquet` files are never mutated). Step 1 (done): the catalog is now **typed
and persistent** — `TableInfo` retains the full Arrow schema plus declarative
constraints (NOT NULL / PK / UNIQUE / DEFAULTs, stored not yet enforced), and
the catalog persists to `<data_dir>/ryudb.catalog.json` and reloads on restart.
The read path is unchanged. Phase 3 will add full SQL, a cost-based optimizer,
recovery, and C++/CUDA hot paths.

## Environment

Built and run inside **WSL2 Ubuntu** with an NVIDIA GPU. cuDF is installed via
the `rapidsai` conda channel.

```bash
# inside WSL2
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ryudb
python -c "import cudf; print(cudf.__version__)"
nvidia-smi   # should show the RTX 3090
```

## Usage

```bash
ryudb                         # start REPL
ryudb -e "SELECT count(*) FROM lineitem"
ryudb -f script.sql           # run a SQL script
```

Load a table from a directory of Parquet files:

```sql
CREATE TABLE lineitem FROM '/path/to/lineitem/*.parquet';
SELECT l_returnflag, sum(l_quantity) AS qty
  FROM lineitem
 WHERE l_shipdate <= date '1998-09-02'
 GROUP BY l_returnflag
 ORDER BY l_returnflag;
```

## SQL subset (Phase 1)

`SELECT` (columns / `*` / expressions), `WHERE`, inner equi-`JOIN`,
`GROUP BY` with `COUNT/SUM/AVG/MIN/MAX`, `ORDER BY`, `LIMIT`.

## Benchmark

```bash
python bench/run_bench.py --scale 1     # RyuDB (GPU) vs DuckDB (CPU) on TPC-H
python bench/run_bench.py --scale 10    # ~7 GB
```

Results are documented in [BENCHMARK.md](BENCHMARK.md). Short version: RyuDB is
correct (matches DuckDB) and ~27x faster than pandas on GPU, but does **not**
beat DuckDB end-to-end at SF≤10 — cuDF's Parquet reader is slower than DuckDB's
world-class reader and dominates scan-heavy queries. See the full analysis for
where the GPU wins and the Phase-3 path to close the gap.