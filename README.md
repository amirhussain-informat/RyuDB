# RyuDB



<img width="1254" height="1254" alt="ChatGPT Image Jul 18, 2026, 12_43_38 AM" src="https://github.com/user-attachments/assets/1d06adc7-1bb7-4327-81ff-3c07844fdaba" />




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

## Install

RyuDB is a normal Python package (`pyproject.toml`, console scripts
`ryudb` / `ryudb-server`). It is installed from source into the conda env
that already provides cuDF + nvcc:

```bash
conda activate ryudb
pip install .                    # editable: pip install -e .
ryudb build                     # compile the C++/CUDA fused kernel (nvcc + pybind11)
```

`ryudb build` (a.k.a. `python -m ryudb.kernels.build`) compiles
`ryudb/kernels/fused.cu` + `pqpages.cpp` to `fused.so` next to the sources,
linking `libnvcomp` (GPU Snappy) with an rpath to `$CONDA_PREFIX/lib`. The
kernel sources ship in the wheel (`package-data`), so the build works from a
non-editable install too. The fused kernel is **optional** — if it is absent or
stale (a source newer than the binary), the executor falls back to the
Numba/cuDF paths, so `ryudb build` is never required for correctness, only for
the fused star-join+aggregate hot path.

## Docker

The "single artifact" for a GPU app is a Docker image (a PyInstaller blob
bundling cuDF + RAPIDS + the CUDA runtime is large and fragile). The image is a
plain `ubuntu:22.04` base + a conda env (`docker/env.yml`, a `--no-builds` export
of the dev env: CUDA 13.3, cuDF 26.06, libnvcomp 5.2, pybind11 3.0.4) that brings
the full CUDA toolkit (nvcc + cudart + libnvcomp), plus the built `fused.so` and
the `ryudb-server` entrypoint. The NVIDIA driver is injected by the host via
`--gpus all` (NVIDIA Container Toolkit) — the image carries no driver.

```bash
docker build -t ryudb -f docker/Dockerfile .
# fat binary for a range of GPUs (default: Ampere/Ada/Hopper):
docker build --build-arg CUDA_ARCH=sm_80,sm_86,sm_89,sm_90 -t ryudb .

docker run --gpus all -p 5430:5430 -p 5432:5432 -v ryudb-data:/data ryudb
psql -h 127.0.0.1 -p 5432 -U ryudb ryudb        # Postgres wire front (no auth)
```

The container defaults to `RYUDB_HOST=0.0.0.0` (so host port mapping works),
`RYUDB_PG_PORT=5432` (PG wire front on), and `RYUDB_DATA=/data` (a named volume).
Override any `RYUDB_*` var with `-e`, or pass `ryudb-server` flags directly
(`docker run ryudb --data /data --port 6000`). A known command as the first arg
runs it instead of the server: `docker run -it ryudb ryudb -e "SELECT ..."` or
`docker run -it ryudb bash`.

> **No auth, no TLS.** The server authenticates nobody. Binding `0.0.0.0` is only
> safe on a trusted network or behind a proxy — do not expose the published ports
> to the public internet. The host GPU driver must be recent enough for the
> CUDA 13.3 userland (`nvidia-smi` should report >= 13.3).

`docker/smoke.sh` builds the image and runs a real SQL round-trip over the PG
wire (generate a parquet → `CREATE TABLE ... FROM` → `SELECT count(*)` via
pg8000). It needs Docker + the NVIDIA Container Toolkit on the host. *(The image
was not built in the dev environment — Docker is unavailable in the WSL distro —
so `docker/smoke.sh` is the validation step for a Docker-equipped host.)*

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

## Server

`ryudb-server` runs the engine as a server with **two** wire fronts sharing one
`Engine` behind a single worker thread (all access serializes):

- a **WebSocket** front (custom JSON + Arrow IPC) for the frontend / any client —
  results come back as Arrow IPC for zero-copy handoff to a dataframe UI; and
- an optional **Postgres v3 wire** front (`--pg-port`) so real drivers — `psql`,
  `psycopg`, `pg8000`, `asyncpg`, JDBC — connect to the same engine.

```bash
ryudb-server --data ./data --host 127.0.0.1 --port 5430 --pg-port 5432
# env overrides: RYUDB_DATA / RYUDB_HOST / RYUDB_PORT / RYUDB_MAX_ROWS /
#                RYUDB_PG_PORT / RYUDB_PG_MAX_ROWS / RYUDB_LOG_LEVEL
```

Ops: `sql`, `explain` (structured plan tree with a `fused` badge for the
star-join+aggregate shape), `catalog`, `table`, `sample`, `admin`
(register/drop/rename/alter/checkpoint/snapshot/restore/clear_cache), `cancel`
(drops pending requests), `history`. Parse errors carry a `position`; results
cap at `--max-rows` rows (the true `row_count` is always reported). See
[`ryudb/server/PROTOCOL.md`](ryudb/server/PROTOCOL.md) for the full wire format
(both fronts).

**Per-connection transactions (MVCC isolation):** each connection owns its own
in-flight transaction with a frozen snapshot — a commit on another connection
is invisible until this one commits (read-your-writes within a txn; a disconnect
rolls back the open txn). `checkpoint` / `restore` are refused while any
connection has an open txn. **Cooperative in-flight cancel:** `cancel` drops
pending requests and raises `CancelledByUser` at the next plan-node boundary of
an in-flight request (a single long cuDF call is not mid-call interruptible).
Binds to `127.0.0.1` with no auth (local console, not for exposing on a network).

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
