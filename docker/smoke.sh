#!/usr/bin/env bash
# Build the RyuDB image and smoke-test it on a Docker-equipped host.
#
# This was NOT run during development (Docker is unavailable in the dev WSL
# distro); it is the reproducible validation for a host that has Docker + the
# NVIDIA Container Toolkit (`nvidia-smi` + `docker run --rm --gpus all ...`).
#
#   bash docker/smoke.sh            # build + smoke
#   bash docker/smoke.sh --no-cache # force a clean build
#
# What it checks:
#   1. `ryudb-server --help` runs inside the image (console script + import OK).
#   2. `ryudb build --help` runs (the build subcommand is installed).
#   3. A real SQL round-trip over the Postgres wire front: generate a 3-row
#      parquet, mount it at /data, CREATE TABLE ... FROM, SELECT count(*) —
#      driven by pg8000 (installed in the image env).
set -euo pipefail

IMAGE="${RYUDB_IMAGE:-ryudb}"
BUILD_ARGS=("$@")   # pass-through, e.g. --no-cache

echo "==> docker build -t $IMAGE ${BUILD_ARGS[*]:-}"
docker build -t "$IMAGE" -f docker/Dockerfile "${BUILD_ARGS[@]}" .

echo "==> ryudb-server --help (console script + import)"
docker run --rm "$IMAGE" ryudb-server --help >/dev/null

echo "==> ryudb build --help (build subcommand present)"
docker run --rm "$IMAGE" ryudb build --help >/dev/null

# A tmp dir holding a generated parquet; mounted at /data in the container.
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"; docker rm -f ryudb-smoke >/dev/null 2>&1 || true' EXIT
mkdir -p "$WORK/t"

echo "==> generate a 3-row parquet on the host (needs pandas)"
python - "$WORK/t/0.parquet" <<'PY'
import sys, pandas as pd
path = sys.argv[1]
pd.DataFrame({"k": [1, 2, 3], "v": [10.0, 20.0, 30.0]}).to_parquet(path)
print("wrote", path)
PY

echo "==> start the server with the parquet mounted at /data"
docker run -d --name ryudb-smoke --gpus all \
    -p 5432:5432 -v "$WORK:/data" "$IMAGE" >/dev/null

# Wait for the PG wire front to accept connections (up to ~30s).
echo "==> drive a SQL round-trip over the Postgres wire (pg8000, in the image)"
docker exec ryudb-smoke /opt/conda/envs/ryudb/bin/python <<'PY'
import time, pg8000
last = None
for _ in range(30):
    try:
        conn = pg8000.connect(user="ryudb", host="127.0.0.1", port=5432,
                              database="ryudb", ssl_context=False, timeout=5)
        break
    except Exception as exc:  # noqa: BLE001
        last = exc
        time.sleep(1)
else:
    raise SystemExit(f"server did not come up: {last}")

conn.autocommit = True   # standalone statements; no implicit BEGIN around CREATE
cur = conn.cursor()
cur.execute("CREATE TABLE t FROM '/data/t/0.parquet'")
cur.execute("SELECT count(*) AS c FROM t")
rows = list(cur.fetchall())
print("rows:", rows)
assert rows == [[3]], f"expected [[3]], got {rows}"
conn.close()
print("SMOKE OK")
PY

echo "==> all smoke checks passed"